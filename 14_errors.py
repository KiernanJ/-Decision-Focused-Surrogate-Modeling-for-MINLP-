"""Error decomposition + figure, for whichever case QCAC_CASE selects.

Three errors against the QCAC-iterative reference:
  DISCRETE   % of generators with the wrong on/off status
  CONTINUOUS % of total generation misallocated, ||pg-pg*||_1 / sum(pg*), on
             the RECOVERED dispatch, not the fractional relaxation
  GAP        100*(cost-cost*)/cost*

Discrete and continuous are not redundant: a commitment can be wrong on cheap
marginal units and barely move cost, or right while the dispatch is
misallocated. The gap alone hides both.
"""
# %% setup
import os, sys, re, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp
exec(open(f"{HERE}/11_vcard.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

NTEST = int(os.environ.get("NTEST", "24"))
NET_V = os.environ.get("NET_V", "net_final_v.pt")
NET_VC = os.environ.get("NET_VC", "net_fix_vcard.pt")
NDOWN = int(os.environ.get("NDOWN", "6"))
set_cuts(8); set_down(NDOWN)
g, nl = grid(); NG = g.n_gen; S = Socp(g, nl, n_cuts=8)

POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
             for i in range(int(m.group(2))) if REF[i] is not None]
rng = np.random.default_rng(0); perm = rng.permutation(len(POOL))
te = [int(i) for i in perm[:NTEST]]; tr = [int(i) for i in perm[NTEST:]]
print(f"[setup] {CASE}: {g.n_bus} bus, {NG} gen | {len(POOL)} instances, "
      f"{len(te)} held out | downward repair {NDOWN}", flush=True)

PGS = {}
for j in te:
    r = S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))
    PGS[j] = r["pg"] if r else None
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
order = np.argsort(-g.pmax); cum = np.cumsum(g.pmax[order])
SRt = torch.tensor([float(np.searchsorted(cum, (1.0+RESERVE)*float(p.sum())) + 1)
                    for p, _, _, _ in POOL], dtype=torch.float32)


def deploy_full(pd_, qd_, Vr, Vi, A, b):
    t0 = time.time()
    r = S.solve(pd_, qd_, Vr, Vi, rho=1e6, A=A, b=b)
    if r is None:
        return None
    uf = r["u"]; z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0 + RESERVE)*float(pd_.sum())
    for k in o:
        if float(g.pmax @ z) >= need:
            break
        z[k] = 1.0
    best = None
    for _ in range(NG + 1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"], pr["pg"]); break
        off = [k for k in o if z[k] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, cost, pg = best
    for k in [t for t in np.argsort(uf) if z[t] > 0.5][:NDOWN]:
        w = z.copy(); w[k] = 0.0
        if float(g.pmax @ w) < need:
            continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w, pr["cost"], pr["pg"]
    return dict(u=z.astype(int), pg=pg, cost=cost, secs=time.time()-t0)


def arm(tag, ckpt=None, use_cut=False):
    net = None
    if ckpt and os.path.exists(f"{HERE}/results/{ckpt}"):
        net = VCardNet(g.n_bus, NG)
        net.load_state_dict(torch.load(f"{HERE}/results/{ckpt}")); net.eval()
        with torch.no_grad():
            vr, vi, lo, hi = net(Xn[te], SRt[te])
        vr, vi, lo, hi = [t.numpy().astype(float) for t in (vr, vi, lo, hi)]
    elif ckpt:
        print(f"  !! missing {ckpt}"); return tag, []
    rows = []
    for k, j in enumerate(te):
        pd_, qd_, u_s, c_s = POOL[j]
        if net is None:
            V_r, V_i, A, b = np.ones(g.n_bus), np.zeros(g.n_bus), None, None
        else:
            V_r, V_i = vr[k], vi[k]
            A, b = (card_rows(lo[k], hi[k], NG) if use_cut else (None, None))
        r = deploy_full(pd_, qd_, V_r, V_i, A, b)
        if r is None or PGS[j] is None:
            continue
        rows.append(dict(disc=100*float((r["u"] != u_s).mean()),
                         cont=100*float(np.abs(r["pg"]-PGS[j]).sum()/PGS[j].sum()),
                         gap=100*(r["cost"]-c_s)/c_s, secs=r["secs"]))
    return tag, rows


ARMS = [arm("flat V\nno cut", None), arm("V only", NET_V, False),
        arm("V + cardinality\ncut", NET_VC, True)]
print(f"\n{'arm':24s}{'n':>4s}{'discrete':>11s}{'continuous':>13s}{'gap':>10s}{'s/inst':>9s}")
for tag, rows in ARMS:
    if not rows:
        print(f"{tag.replace(chr(10),' '):24s}   -- no rows --"); continue
    d = lambda k: np.mean([x[k] for x in rows])
    print(f"{tag.replace(chr(10),' '):24s}{len(rows):>4d}{d('disc'):>10.2f}%"
          f"{d('cont'):>12.2f}%{d('gap'):>9.3f}%{d('secs'):>8.2f}s")

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
cols = ["#b0b0b0", "#4878a8", "#c44e52"]
for a_, (k, t_) in zip(ax[:3], [("disc", "discrete decision error  [% of generators]"),
                                ("cont", "continuous decision error  [% of generation]"),
                                ("gap", "optimality gap  [%]")]):
    data = [[x[k] for x in rows] for _, rows in ARMS if rows]
    bp = a_.boxplot(data, patch_artist=True, widths=.55, showfliers=False)
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(.65)
    for m in bp["medians"]:
        m.set_color("k")
    for i, dd in enumerate(data):
        a_.scatter(np.full(len(dd), i+1)+np.random.uniform(-.13, .13, len(dd)),
                   dd, s=9, color="k", alpha=.45, zorder=3)
    a_.set_xticks(range(1, len(data)+1))
    a_.set_xticklabels([t for t, r in ARMS if r], fontsize=8)
    a_.set_title(t_, fontsize=9)
for i, (tag, rows) in enumerate([x for x in ARMS if x[1]]):
    v = np.sort([x["gap"] for x in rows])
    ax[3].step(v, np.arange(1, len(v)+1)/len(v), where="post", color=cols[i],
               lw=1.8, label=tag.replace("\n", " "))
ax[3].set_xlabel("optimality gap [%]"); ax[3].set_ylabel("fraction of instances")
ax[3].set_title("gap CDF", fontsize=9); ax[3].legend(fontsize=7, loc="lower right")
ax[3].set_xscale("symlog", linthresh=.02)
fig.suptitle(f"{CASE}, {len(te)} held-out instances -- reference: QCAC iterative; "
             f"ONE continuous solve + restoration (downward repair {NDOWN})", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(f"{HERE}/results/errors_{CASE}.png", dpi=170)
print(f"\nwrote results/errors_{CASE}.png")

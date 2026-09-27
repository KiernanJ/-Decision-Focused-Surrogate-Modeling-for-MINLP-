"""Per-instance error decomposition, and the figure.

Three errors, all against the QCAC-iterative reference (which reproduces the
global MINLP optimum exactly):

  DISCRETE DECISION ERROR   % of generators whose on/off status is wrong,
                            100 * #{u_i != u*_i} / n_gen. This is the
                            complement of the commitment agreement.

  CONTINUOUS DECISION ERROR % of total generation misallocated,
                            100 * ||pg - pg*||_1 / sum(pg*). Reported on the
                            RECOVERED dispatch, not the relaxation's fractional
                            one, so it is the error a user would actually see.

  OPTIMALITY GAP            100 * (cost - cost*) / cost*, true AC cost of the
                            recovered commitment.

Discrete and continuous error are NOT redundant: a commitment can be wrong on
cheap marginal units and barely move cost, or right while the dispatch is
misallocated. Reporting only the gap hides both.
"""
# %% setup
import os, sys, re, time
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, RESERVE, grid, sample
from socp import Socp
sys.path.insert(0, HERE)
exec(open(f"{HERE}/11_vcard.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

g, nl = grid(); G = g.n_gen; S = Socp(g, nl, n_cuts=8)

POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
             for i in range(int(m.group(2))) if REF[i] is not None]
rng = np.random.default_rng(0)
perm = rng.permutation(len(POOL))
te = [int(i) for i in perm[:24]]; tr = [int(i) for i in perm[24:]]

# reference dispatch pg*: price the reference commitment
PGS = {}
for j in te:
    r = S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))
    PGS[j] = r["pg"]

base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
Xn = (X - mu)/sg
order = np.argsort(-g.pmax); cum = np.cumsum(g.pmax[order])
SRt = torch.tensor([float(np.searchsorted(cum, (1.0+RESERVE)*float(p.sum())) + 1)
                    for p, _, _, _ in POOL], dtype=torch.float32)


# %% deployment that also returns the recovered dispatch
def run(pd_, qd_, Vr, Vi, A=None, b=None):
    t0 = time.time()
    r = S.solve(pd_, qd_, Vr, Vi, rho=1e6, A=A, b=b)
    if r is None:
        return None
    uf = r["u"]; z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0 + RESERVE) * float(pd_.sum())
    for k in o:
        if float(g.pmax @ z) >= need:
            break
        z[k] = 1.0
    for _ in range(G + 1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            return dict(u=z.astype(int), pg=pr["pg"], cost=pr["cost"],
                        secs=time.time()-t0)
        off = [k for k in o if z[k] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    return None


def arm_rows(tag, ckpt=None, use_cut=False):
    net = None
    if ckpt:
        net = VCardNet(g.n_bus, G)
        net.load_state_dict(torch.load(f"{HERE}/results/{ckpt}"))
        net.eval()
        with torch.no_grad():
            vr, vi, lo, hi = net(Xn[te], SRt[te])
        vr, vi = vr.numpy().astype(float), vi.numpy().astype(float)
        lo, hi = lo.numpy().astype(float), hi.numpy().astype(float)
    out = []
    for k, j in enumerate(te):
        pd_, qd_, u_s, c_s = POOL[j]
        if net is None:
            V_r, V_i, A, b = np.ones(g.n_bus), np.zeros(g.n_bus), None, None
        else:
            V_r, V_i = vr[k], vi[k]
            A, b = (card_rows(lo[k], hi[k], G) if use_cut else (None, None))
        r = run(pd_, qd_, V_r, V_i, A, b)
        if r is None:
            continue
        pgs = PGS[j]
        out.append(dict(
            disc=100*float((r["u"] != u_s).mean()),
            cont=100*float(np.abs(r["pg"]-pgs).sum()/pgs.sum()),
            gap=100*(r["cost"]-c_s)/c_s,
            secs=r["secs"]))
    return tag, out


ARMS = [arm_rows("flat V\nno cut", None),
        arm_rows("V only", "net_final_v.pt", False),
        arm_rows("V + cardinality\ncut", "net_fix_vcard.pt", True)]

# %% report
print(f"\n{'arm':22s}{'discrete err':>14s}{'continuous err':>16s}{'gap':>10s}{'s/inst':>9s}")
for tag, rows in ARMS:
    d = lambda k: np.mean([x[k] for x in rows])
    print(f"{tag.replace(chr(10),' '):22s}{d('disc'):>13.2f}%{d('cont'):>15.2f}%"
          f"{d('gap'):>9.3f}%{d('secs'):>8.2f}s")

# %% figure
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
keys = [("disc", "discrete decision error  [% of generators]"),
        ("cont", "continuous decision error  [% of generation]"),
        ("gap",  "optimality gap  [%]")]
cols = ["#b0b0b0", "#4878a8", "#c44e52"]
labels = [t.replace("\n", " ") for t, _ in ARMS]
for a_, (k, title) in zip(ax[:3], keys):
    data = [[x[k] for x in rows] for _, rows in ARMS]
    bp = a_.boxplot(data, patch_artist=True, widths=.55, showfliers=False)
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(.65)
    for m in bp["medians"]:
        m.set_color("k")
    for i, dd in enumerate(data):
        a_.scatter(np.full(len(dd), i+1) + np.random.uniform(-.13, .13, len(dd)),
                   dd, s=9, color="k", alpha=.45, zorder=3)
    a_.set_xticks([1, 2, 3]); a_.set_xticklabels([t for t, _ in ARMS], fontsize=8)
    a_.set_title(title, fontsize=9)
for i, (tag, rows) in enumerate(ARMS):      # CDF of the gap
    v = np.sort([x["gap"] for x in rows])
    ax[3].step(v, np.arange(1, len(v)+1)/len(v), where="post",
               color=cols[i], lw=1.8, label=labels[i])
ax[3].set_xlabel("optimality gap [%]"); ax[3].set_ylabel("fraction of instances")
ax[3].set_title("gap CDF", fontsize=9); ax[3].legend(fontsize=7, loc="lower right")
ax[3].set_xscale("symlog", linthresh=.05)
fig.suptitle("case118, 24 held-out instances -- reference: QCAC iterative "
             "(= global MINLP optimum); one continuous solve per instance",
             fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(f"{HERE}/results/errors.png", dpi=170)
print(f"\nwrote results/errors.png")

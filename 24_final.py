"""Final table + figure for one case. Every arm, one split, one restoration.

The NN proxy is the baseline that matters -- it asks whether the optimiser is
needed at all. Comparing against "flat V, no cut" flatters the method and
settles nothing, so it appears only as a floor.

Arms (identical restoration and AC pricing throughout; only the source of the
commitment differs):

  relaxation only     flat V, no cut                          the floor
  proxy SUPERVISED    (pd,qd) -> u, trained on u*             uses LABELS
  proxy self-sup      (pd,qd) -> u, deployment cost only      no labels
  ours V only         predicted linearisation point           no labels
  ours V + penalty    plus a CONSTANT 1e4/generator penalty   no labels
  ours V + LEARNED    plus learned per-instance tiered cuts   no labels
       tiered cuts    (hard, cut_cap=0)

and for the learned-cut arm, the three controls that would refute it:
dummy band, random band, and a mean band carrying no per-instance information.
A missing checkpoint prints a NOTE and skips that arm rather than crashing.
"""
# %% setup
import os, sys, re, time, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp
exec(open(f"{HERE}/21_tiered_train.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())
exec(open(f"{HERE}/11_vcard.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

K = int(os.environ.get("KTIERS", "3")); K_ROWS = 2*K
NTEST = int(os.environ.get("NTEST", "48"))
# Restoration depth, identical for EVERY arm. Must match what the
# nets were trained with, or the network optimises a different
# pipeline than it is scored on.
NDOWN = int(os.environ.get("NDOWN", "6"))
NET_TIER = os.environ.get("NET_TIER", f"net_tier_k3full.pt")
NET_VC = os.environ.get("NET_VC", "net_c118_big_vcard.pt")
NET_V = os.environ.get("NET_V", "")
NET_PS = os.environ.get("NET_PS", f"net_proxyself_{CASE}.pt")
set_cuts(K_ROWS); set_down(NDOWN)
g, nl = grid(); NG = g.n_gen; S = Socp(g, nl, n_cuts=K_ROWS)
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
avg = (g.c2*g.pmax**2 + g.c1*g.pmax + nl)/np.maximum(g.pmax, 1e-6)
MERIT = np.argsort(avg); TIERS = [MERIT[i::K] for i in range(K)]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
U = torch.tensor(np.stack([u.astype(float) for _, _, u, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
order = np.argsort(-g.pmax); cum = np.cumsum(g.pmax[order])
SRt = torch.tensor([float(np.searchsorted(cum, (1.0+RESERVE)*float(p.sum()))+1)
                    for p, _, _, _ in POOL], dtype=torch.float32)
PGS = {j: S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))["pg"] for j in te}
print(f"[setup] {CASE}: {g.n_bus} bus, {NG} gen | {len(POOL)} inst, "
      f"{len(tr)} train / {len(te)} test | K={K} | hard cuts, {NDOWN} downward\n",
      flush=True)


def restore(j, uf):
    pd_, qd_ = POOL[j][0], POOL[j][1]
    z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0+RESERVE)*float(pd_.sum())
    for t_ in o:
        if float(g.pmax @ z) >= need:
            break
        z[t_] = 1.0
    best = None
    for _ in range(NG+1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"], pr["pg"]); break
        off = [t_ for t_ in o if z[t_] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, cost, pg = best
    for t_ in [t2 for t2 in np.argsort(uf) if z[t2] > 0.5][:NDOWN]:
        w = z.copy(); w[t_] = 0.0
        if float(g.pmax @ w) < need:
            continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w, pr["cost"], pr["pg"]
    return z, cost, pg


RES = {}


def run(name, role, labels, uf_of):
    rows = []
    for k, j in enumerate(te):
        uf = uf_of(k, j)
        if uf is None:
            continue
        out = restore(j, uf)
        if out is None:
            continue
        z, cost, pg = out
        rows.append(dict(disc=100*float((z.astype(int) != POOL[j][2]).mean()),
                         cont=100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()),
                         gap=100*(cost-POOL[j][3])/POOL[j][3]))
    if not rows:
        print(f"  NOTE: {name} produced no rows"); return
    RES[name] = (role, labels, rows)
    print(f"{name:34s}{labels:>8s}{len(rows):>4d}"
          f"{np.mean([x['disc'] for x in rows]):>10.2f}%"
          f"{np.mean([x['cont'] for x in rows]):>12.2f}%"
          f"{np.mean([x['gap'] for x in rows]):>9.3f}%", flush=True)


print(f"{'arm':34s}{'labels':>8s}{'n':>4s}{'discrete':>11s}{'continuous':>13s}{'gap':>10s}")
run("relaxation only (flat V)", "floor", "-",
    lambda k, j: (lambda r: r["u"] if r else None)(
        S.solve(POOL[j][0], POOL[j][1], np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e6)))

# supervised proxy -- trained here, it takes seconds
pn = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.SiLU(),
                         torch.nn.Linear(256, 256), torch.nn.SiLU(),
                         torch.nn.Linear(256, NG))
o_ = torch.optim.Adam(pn.parameters(), lr=2e-3)
sc_ = torch.optim.lr_scheduler.CosineAnnealingLR(o_, 8000)
lf = torch.nn.BCEWithLogitsLoss()
for _ in range(8000):
    l = lf(pn(Xn[tr]), U[tr]); o_.zero_grad(); l.backward(); o_.step(); sc_.step()
with torch.no_grad():
    PU = torch.sigmoid(pn(Xn[te])).numpy()
run("NN proxy (SUPERVISED on u*)", "proxy", "YES", lambda k, j: PU[k])

if os.path.exists(f"{HERE}/results/{NET_PS}"):
    sn = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.SiLU(),
                             torch.nn.Linear(256, 256), torch.nn.SiLU(),
                             torch.nn.Linear(256, NG))
    sn.load_state_dict(torch.load(f"{HERE}/results/{NET_PS}")); sn.eval()
    with torch.no_grad():
        SU = torch.sigmoid(sn(Xn[te])).numpy()
    run("NN proxy (self-supervised)", "proxy", "no", lambda k, j: SU[k])
else:
    print(f"  NOTE: {NET_PS} missing -- self-supervised proxy skipped")

if os.path.exists(f"{HERE}/results/{NET_VC}"):
    vn = VCardNet(g.n_bus, NG); vn.load_state_dict(torch.load(f"{HERE}/results/{NET_VC}"))
    vn.eval()
    with torch.no_grad():
        vr, vi, lo_, hi_ = vn(Xn[te], SRt[te])
    vr, vi = vr.numpy().astype(float), vi.numpy().astype(float)
    run("ours: V only", "ours", "no",
        lambda k, j: (lambda r: r["u"] if r else None)(
            S.solve(POOL[j][0], POOL[j][1], vr[k], vi[k], rho=1e6)))
    run("ours: V + constant penalty", "ours", "no",
        lambda k, j: (lambda r: r["u"] if r else None)(
            S.solve(POOL[j][0], POOL[j][1], vr[k], vi[k], rho=1e6,
                    A=card_rows(0., 0., NG, K_ROWS)[0],
                    b=card_rows(0., 0., NG, K_ROWS)[1], cut_cap=1e6)))
else:
    print(f"  NOTE: {NET_VC} missing -- V-only and constant-penalty skipped")

TN = f"{HERE}/results/{NET_TIER}"
if os.path.exists(TN):
    tn = TieredNet(g.n_bus, NG, [len(T) for T in TIERS])
    tn.load_state_dict(torch.load(TN)); tn.eval()
    with torch.no_grad():
        _out = tn(Xn[te])
    # TieredNet returns 4 values when the threshold head is on. Unpacking into
    # 3 crashed this script for BOTH cases after the training itself had
    # succeeded -- the numbers were in the log but no figure was written, and
    # the stale FINAL_*.png on disk still looked current.
    TV, TI, TC = [_out[i].numpy().astype(float) for i in range(3)]

    def tf(cnt):
        return lambda k, j: (lambda r: r["u"] if r else None)(
            S.solve(POOL[j][0], POOL[j][1], TV[k], TI[k], rho=1e6,
                    A=tier_rows(cnt(k), TIERS, NG, K_ROWS)[0],
                    b=tier_rows(cnt(k), TIERS, NG, K_ROWS)[1], cut_cap=0.0))
    run("ours: V + LEARNED tiered cuts", "ours", "no", tf(lambda k: TC[k]))
    run("CONTROL: dummy band [0,0]", "control", "no", tf(lambda k: np.zeros(K)))
    run("CONTROL: random band", "control", "no",
        tf(lambda k: rng.uniform(0, [len(T) for T in TIERS])))
    run("CONTROL: mean band", "control", "no", tf(lambda k: TC.mean(0)))
    print(f"\n  learned tier-count spread per instance: sd {TC.std(0).round(3)}"
          f"  (a constant band has sd ~ 0)")
else:
    print(f"  NOTE: {NET_TIER} missing -- tiered arm and controls skipped")
    TC = None

json.dump({k: dict(role=v[0], labels=v[1],
                   disc=float(np.mean([x['disc'] for x in v[2]])),
                   cont=float(np.mean([x['cont'] for x in v[2]])),
                   gap=float(np.mean([x['gap'] for x in v[2]])))
           for k, v in RES.items()},
          open(f"{HERE}/results/final_{CASE}.json", "w"), indent=1)

# %% figure -- colour encodes ROLE; hues unchanged from the validated palette
INK, INK2 = "#0b0b0b", "#52514e"
ROLE = {"floor": "#8a8a85", "proxy": "#eb6834", "ours": "#2a78d6", "control": "#e34948"}
SHORT = {"relaxation only (flat V)": "relaxation only",
         "NN proxy (SUPERVISED on u*)": "proxy (SUPERVISED)",
         "NN proxy (self-supervised)": "proxy (self-sup.)",
         "ours: V only": "V only",
         "ours: V + constant penalty": "V + const. penalty",
         "ours: V + LEARNED tiered cuts": "V + LEARNED cut",
         "CONTROL: dummy band [0,0]": "control: dummy",
         "CONTROL: random band": "control: random",
         "CONTROL: mean band": "control: mean"}
names = [n for n in SHORT if n in RES]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d7d7d2",
                     "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})
npan = 4 if TC is not None else 3
fig, ax = plt.subplots(1, npan, figsize=(4.3*npan, 4.9))
for a_, (key, title, unit) in zip(ax[:3], [
        ("disc", "Discrete decision error", "% of generators"),
        ("cont", "Continuous decision error", "% of generation"),
        ("gap", "Optimality gap", "%")]):
    a_.set_axisbelow(True); a_.grid(axis="y", color="#e8e8e3", lw=1)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    for i, n in enumerate(names):
        role, _, rows = RES[n]; v = np.array([x[key] for x in rows]); c = ROLE[role]
        bp = a_.boxplot([v], positions=[i], widths=.55, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=2),
                        whiskerprops=dict(color=c, lw=1.8), capprops=dict(color=c, lw=1.8))
        bp["boxes"][0].set(facecolor=c, alpha=.28, edgecolor=c, lw=1.8)
        a_.scatter(np.full(len(v), i)+rng.uniform(-.1, .1, len(v)), v, s=9,
                   color=c, alpha=.5, linewidths=.4, edgecolors="#fcfcfb", zorder=3)
        a_.annotate(f"{v.mean():.2f}", (i, v.mean()), ha="center", va="bottom",
                    fontsize=7.5, color=INK, fontweight="bold",
                    bbox=dict(fc="#fcfcfb", ec="none", pad=1))
    a_.set_xticks(range(len(names)))
    a_.set_xticklabels([SHORT[n] for n in names], fontsize=7.5, color=INK2,
                       rotation=38, ha="right", rotation_mode="anchor")
    a_.set_title(title, fontsize=9.5, color=INK, pad=8, loc="left")
    a_.set_ylabel(unit, fontsize=8)
if TC is not None:
    a_ = ax[3]
    a_.set_axisbelow(True); a_.grid(axis="y", color="#e8e8e3", lw=1)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    for t in range(K):
        a_.scatter(np.full(NTEST, t)+rng.uniform(-.16, .16, NTEST), TC[:, t], s=12,
                   color=ROLE["ours"], alpha=.6, linewidths=.4, edgecolors="#fcfcfb",
                   zorder=3, label="learned, per instance" if t == 0 else None)
        a_.hlines(TC[:, t].mean(), t-.3, t+.3, color=ROLE["control"], lw=2.5, zorder=4,
                  label="mean band (control)" if t == 0 else None)
        a_.annotate(f"sd {TC[:, t].std():.2f}", (t, TC[:, t].max()), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
    a_.set_xticks(range(K))
    a_.set_xticklabels(["all generators" if K == 1 else f"tier {t+1}"
                        for t in range(K)], fontsize=8, color=INK2)
    a_.set_ylabel("generators committed in tier", fontsize=8)
    a_.set_title("The learned cut is instance-specific", fontsize=9.5, color=INK,
                 pad=8, loc="left")
    a_.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.legend(handles=[Patch(fc=ROLE[r], alpha=.28, ec=ROLE[r], label=lab) for r, lab in
                    [("floor", "relaxation floor"), ("proxy", "NN proxy baseline"),
                     ("ours", "proposed method"), ("control", "refutation control")]],
           ncol=4, frameon=False, fontsize=8.5, loc="upper right",
           bbox_to_anchor=(.995, 1.035))
fig.suptitle(f"{CASE}: {NTEST} held-out instances, ONE continuous solve + restoration "
             f"| reference: QCAC iterative", fontsize=10.5, color=INK, x=.008,
             ha="left", y=1.035)
fig.tight_layout(rect=[0, 0, 1, .90])
fig.savefig(f"{HERE}/results/FINAL_{CASE}.png", dpi=170, bbox_inches="tight")
# Also write a TIMESTAMPED copy. FINAL_<case>.png and final_<case>.json are
# fixed filenames, so every rerun silently overwrites the previous result --
# including the figures behind a headline number. These copies are never
# overwritten, so a rerun can be compared against what it replaced.
import shutil, datetime
_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
_d = f"{HERE}/results/runs"; os.makedirs(_d, exist_ok=True)
for _src, _ext in ((f"{HERE}/results/FINAL_{CASE}.png", "png"),
                   (f"{HERE}/results/final_{CASE}.json", "json")):
    if os.path.exists(_src):
        shutil.copy2(_src, f"{_d}/{CASE}_{_ts}.{_ext}")
print(f"\nwrote results/FINAL_{CASE}.png and results/final_{CASE}.json")
print(f"      + timestamped copy results/runs/{CASE}_{_ts}.[png|json]")

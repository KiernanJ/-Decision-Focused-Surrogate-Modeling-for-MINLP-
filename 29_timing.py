"""Computational time: what ONE continuous solve actually buys.

The method's claim is speed -- one continuous solve replacing QCAC's 3-4
MIQCQP solves, each carrying every binary. That claim has never been plotted;
the fourth panel of the result figures shows the cut's per-instance spread
instead. This measures it.

Timed per instance, SEQUENTIALLY on an otherwise idle machine, because a
contended measurement is a wrong measurement:

  reference   QCAC iterative, as recorded when the references were generated.
              NOTE those ran 12-way parallel with 2 Gurobi threads each, so
              they are INFLATED by contention; a small sample is re-timed here
              single-process to give the honest number.
  relaxation  one continuous solve, flat V, no cut -- the floor
  ours        one continuous solve at the predicted V with the learned cut,
              plus rounding, reserve top-up, and upward/downward repair
  proxy       the supervised NN proxy: no relaxation, but the SAME restoration
"""
import os, sys, re, time, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp
exec(open(f"{HERE}/21_tiered_train.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

K = 1; NDOWN = int(os.environ.get("NDOWN", "6"))
NTEST = int(os.environ.get("NTEST", "48"))
NET_TIER = os.environ.get("NET_TIER", "net_tier_cons118.pt")
NRETIME = int(os.environ.get("NRETIME", "5"))
set_cuts(2*K); set_down(NDOWN)
g, nl = grid(); NG = g.n_gen; S = Socp(g, nl, n_cuts=2*K)

# POOL_SEEDS pins which reference sets are loaded. A net's input
# standardisation comes from ITS train split, so evaluating it against a
# pool that has since grown silently changes the test set AND corrupts the
# normalisation -- the net may even be scored on its own training
# instances. Adding 194 case300 references turned a pool of 151 into 345
# and moved the headline gap from +0.296% to +0.890% with no code change.
_seeds = os.environ.get("POOL_SEEDS")
_keep = set(_seeds.split(",")) if _seeds else None
POOL, REFSECS = [], []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m or (_keep is not None and m.group(1) not in _keep):
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    for i in range(int(m.group(2))):
        if REF[i] is not None:
            POOL.append((inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"]))
            REFSECS.append(float(REF[i]["secs"]))
rng = np.random.default_rng(0); perm = rng.permutation(len(POOL))
te = [int(i) for i in perm[:NTEST]]; tr = [int(i) for i in perm[NTEST:]]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
U = torch.tensor(np.stack([u.astype(float) for _, _, u, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
TIERS = [np.arange(NG)]
net = TieredNet(g.n_bus, NG, [NG], learn_thr=False)
net.load_state_dict(torch.load(f"{HERE}/results/{NET_TIER}")); net.eval()
with torch.no_grad():
    _o = net(Xn[te])
VR, VI, CNT = [_o[i].numpy().astype(float) for i in range(3)]
pn = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.SiLU(),
                         torch.nn.Linear(256, 256), torch.nn.SiLU(),
                         torch.nn.Linear(256, NG))
_op = torch.optim.Adam(pn.parameters(), lr=2e-3)
_sc = torch.optim.lr_scheduler.CosineAnnealingLR(_op, 8000)
_lf = torch.nn.BCEWithLogitsLoss()
for _ in range(8000):
    _l = _lf(pn(Xn[tr]), U[tr]); _op.zero_grad(); _l.backward(); _op.step(); _sc.step()
with torch.no_grad():
    PU = torch.sigmoid(pn(Xn[te])).numpy()


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
            best = (z.copy(), pr["cost"]); break
        off = [t_ for t_ in o if z[t_] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, c = best
    for t_ in [t2 for t2 in np.argsort(uf) if z[t2] > 0.5][:NDOWN]:
        w = z.copy(); w[t_] = 0.0
        if float(g.pmax @ w) < need:
            continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < c:
            z, c = w, pr["cost"]
    return z, c


T = {"relaxation only\n(1 solve, no restoration)": [], "ours: V + learned cut": [],
     "NN proxy (supervised)": []}
for k, j in enumerate(te):
    pd_, qd_ = POOL[j][0], POOL[j][1]
    t0 = time.perf_counter()
    S.solve(pd_, qd_, np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e6)
    T["relaxation only\n(1 solve, no restoration)"].append(time.perf_counter()-t0)
    t0 = time.perf_counter()
    A, b = tier_rows(CNT[k], TIERS, NG, 2*K)
    r = S.solve(pd_, qd_, VR[k], VI[k], rho=1e6, A=A, b=b, cut_cap=0.0)
    if r is not None and restore(j, r["u"]) is not None:
        T["ours: V + learned cut"].append(time.perf_counter()-t0)
    t0 = time.perf_counter()
    if restore(j, PU[k]) is not None:
        T["NN proxy (supervised)"].append(time.perf_counter()-t0)

# Reference timings. Prefer results/reftime_<case>.json, produced by
# 30_reftime.py with enough samples for the median to CONVERGE. A small
# in-line sample is not good enough here: QCAC iterative ranges 63-426s on
# case300, so n=4 gave medians of 120.4s and 345.1s for the same quantity --
# "25x" and "71x" for the same method. At n=15 the median drift over the last
# three samples is 0.0%.
_rt = f"{HERE}/results/reftime_{CASE}.json"
if os.path.exists(_rt):
    clean = json.load(open(_rt))
    print(f"  using {len(clean)} converged reference timings from {os.path.basename(_rt)}")
else:
    from acuc import solve_qcac_iterative
    clean = []
    for j in te[:NRETIME]:
        t0 = time.perf_counter()
        solve_qcac_iterative(g, nl, POOL[j][0], POOL[j][1], reserve=RESERVE, threads=12)
        clean.append(time.perf_counter()-t0)
    print(f"  WARNING: only {len(clean)} reference timings -- median may not have converged")

ref_all = np.array([REFSECS[j] for j in te])
print(f"\n{CASE}: {len(te)} held out, timed sequentially on an idle machine\n")
print(f"{'arm':44s}{'median':>10s}{'mean':>10s}{'max':>10s}")
for k_, v in T.items():
    v = np.array(v)
    print(f"{k_.replace(chr(10),' '):44s}{np.median(v):>9.2f}s{v.mean():>9.2f}s{v.max():>9.2f}s")
print(f"{'QCAC iterative (as recorded, 12-way contended)':44s}"
      f"{np.median(ref_all):>9.1f}s{ref_all.mean():>9.1f}s{ref_all.max():>9.1f}s")
print(f"{'QCAC iterative (converged, single process)':44s}"
      f"{np.median(clean):>9.1f}s{np.mean(clean):>9.1f}s{np.max(clean):>9.1f}s"
      f"   [n={len(clean)}, IQR {np.percentile(clean,25):.0f}-"
      f"{np.percentile(clean,75):.0f}s]")
sp = np.median(clean)/np.median(T['ours: V + learned cut'])
print(f"\n  speed-up of ours over the reference (median, like-for-like): {sp:.0f}x")

INK, INK2 = "#0b0b0b", "#52514e"
COL = {"relaxation only\n(1 solve, no restoration)": "#8a8a85",
       "NN proxy (supervised)": "#eb6834", "ours: V + learned cut": "#2a78d6"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d7d7d2",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.set_axisbelow(True); ax.grid(color="#e8e8e3", lw=1)
for sp_ in ("top", "right"):
    ax.spines[sp_].set_visible(False)
for k_, v in T.items():
    v = np.sort(np.array(v))
    ax.step(v, 100*np.arange(1, len(v)+1)/len(v), where="post", lw=2.2,
            color=COL[k_], label=k_.replace("\n", " "))
v = np.sort(np.array(clean))
ax.step(v, 100*np.arange(1, len(v)+1)/len(v), where="post", lw=2.2, color="#e34948",
        label=f"QCAC iterative (reference)")
ax.set_xscale("log")
ax.set_xlabel("solution time per instance [s, log scale]")
ax.set_ylabel("% of instances solved")
ax.set_title(f"{CASE}: computational time, {len(te)} held-out instances",
             fontsize=10.5, color=INK, loc="left", pad=10)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(f"{HERE}/results/TIME_{CASE}.png", dpi=170, bbox_inches="tight")
json.dump({k_: list(map(float, v)) for k_, v in T.items()} |
          {"qcac_iterative_clean": list(map(float, clean))},
          open(f"{HERE}/results/time_{CASE}.json", "w"), indent=1)
print(f"\nwrote results/TIME_{CASE}.png")

"""case300 component ablation (learned V vs learned cuts), mirroring the case118
ablation in ../case118_ablation. Read-only on the repo; outputs to ./out.

Every arm uses the paper's case300 restoration (cheapest-first MERIT order, reserve
top-up, upward repair, NDOWN=40 downward trials; src/restore.py, identical to the
local restore in acuc/04_eval_case300.py) and differs only in how the fractional
commitment uf is produced:
  flat_nocut   uncut relaxation at flat V (no learning; seed-independent)
  pred_nocut   relaxation at the network's V, cuts disabled
  pred_cut     relaxation at the network's V with its cuts (the paper's method)
  flat_cut     relaxation at flat V with the network's cuts (anchor point)
  proxy        supervised NN proxy (exact copy of acuc/06_nn_proxy.py training)
Networks use the SELECTED-epoch weights ck["best"] (the paper evaluated the last
epoch; see AUDIT.md item 2). Cuts are anchored at the flat-V uncut relaxation, as
in training.

Thresholds: a scalar rounding threshold from {0.5, 0.65, 0.8}. 0.8 is the paper's
(selected on test, AUDIT.md item 3). The LABEL-FREE choice is made per arm and seed
on the 24 validation instances by mean deployed cost / uncut relaxation cost.

  python abl300.py val  ARM SEED            # deployed cost on validation at each threshold
  python abl300.py test ARM SEED THR        # 48 test instances at one threshold
  python abl300.py refs                     # reference prices for the converged-only analysis
  add --limit N to run on the first N instances only (smoke test)
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "audit"))
import _common
from _common import setup, load_net, ref_slack, RHO
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
GRID = (0.5, 0.65, 0.8)
NDOWN = 40

args = [a for a in sys.argv[1:] if not a.startswith("--limit")]
LIMIT = None
for i, a in enumerate(sys.argv):
    if a == "--limit":
        LIMIT = int(sys.argv[i + 1]); args.remove(sys.argv[i + 1])
mode = args[0]

E = setup("case300")
g, S, POOL, Xn, nl = E["g"], E["S"], E["POOL"], E["Xn"], E["nl"]
te, va, tr = E["te"], E["va"], E["tr"]
from restore import restore
from cutnet_k import rows_torch
NB, NG = g.n_bus, g.n_gen
V1, V0 = np.ones(NB), np.zeros(NB)
MERIT = np.argsort(g.c1 + g.c2*g.pmax + nl/np.maximum(g.pmax, 1e-9))

_RLX = {}
def rlx(j):
    if j not in _RLX:
        _RLX[j] = S.solve(POOL[j][0], POOL[j][1], V1, V0, rho=RHO)
    return _RLX[j]


def proxy_net(seed):
    """Exact replica of acuc/06_nn_proxy.py training (seed, 400 full-batch epochs)."""
    torch.manual_seed(seed)
    Y = torch.tensor(np.stack([POOL[i][2] for i in tr]), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(2*NB, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, NG))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = nn.MSELoss()
    for _ in range(400):
        opt.zero_grad(); loss = lf(torch.sigmoid(net(Xn[tr])), Y); loss.backward(); opt.step()
    net.eval()
    return net


def relaxed_u(arm, seed, ids):
    """uf for each instance (None if the relaxation fails)."""
    if arm == "proxy":
        pn = proxy_net(seed)
        with torch.no_grad():
            return list(torch.sigmoid(pn(Xn[ids])).numpy().astype(float))
    if arm == "flat_nocut":
        return [None if rlx(j) is None else rlx(j)["u"] for j in ids]
    net = load_net("case300", seed, "best", g)
    W = torch.tensor(np.stack([np.r_[rlx(j)["pg"], rlx(j)["u"]] for j in ids]), dtype=torch.float32)
    with torch.no_grad():
        Vr_t, Vi_t, a, al, a2, d_, d2, _ = net(Xn[ids])
        A_t, b_t = rows_torch(a, al, a2, d_, d2, W)
    Vr, Vi = Vr_t.numpy().astype(float), Vi_t.numpy().astype(float)
    A, b = A_t.numpy().astype(float), b_t.numpy().astype(float)
    out = []
    for k, j in enumerate(ids):
        vr, vi = (Vr[k], Vi[k]) if arm.startswith("pred") else (V1, V0)
        kw = dict(A=A[k], b=b[k]) if arm.endswith("_cut") else {}
        r = S.solve(POOL[j][0], POOL[j][1], vr, vi, rho=RHO, **kw)
        out.append(None if r is None else r["u"])
    return out


def deploy_rows(arm, seed, ids, thrs):
    UF = relaxed_u(arm, seed, ids)
    res = {t: [] for t in thrs}
    for k, j in enumerate(ids):
        for t in thrs:
            t0 = time.time()
            if UF[k] is None:
                res[t].append(dict(j=j, fail="relaxation")); continue
            c, d = restore(S, g, POOL[j][0], POOL[j][1], UF[k], thr=t, order=MERIT, ndown=NDOWN)
            if d is None:
                res[t].append(dict(j=j, fail="restore")); continue
            res[t].append(dict(j=j, cost=float(c), scale=float(rlx(j)["cost"]),
                               gap=100*(c - POOL[j][3])/POOL[j][3],
                               disc=100*float((np.asarray(d["u"]) != POOL[j][2]).mean()),
                               n_on=int(np.sum(d["u"])), uf_sum=float(np.sum(UF[k])),
                               secs=time.time()-t0))
    return res


if mode == "val":
    arm, seed = args[1], int(args[2])
    ids = va[:LIMIT] if LIMIT else va
    res = deploy_rows(arm, seed, ids, GRID)
    score = {}
    for t in GRID:
        sc = [10.0 if "fail" in r else r["cost"]/r["scale"] for r in res[t]]
        score[t] = float(np.mean(sc))
    sel = min(GRID, key=lambda t: score[t])
    print(f"val {arm:10s} s{seed} label-free score " + "  ".join(f"thr {t}: {score[t]:.6f}" for t in GRID)
          + f"  -> selected {sel}", flush=True)
    json.dump(dict(arm=arm, seed=seed, score={str(t): v for t, v in score.items()}, selected=sel,
                   rows={str(t): v for t, v in res.items()}),
              open(f"{OUT}/val_{arm}_s{seed}{'_lim' if LIMIT else ''}.json", "w"))

elif mode == "test":
    arm, seed, thr = args[1], int(args[2]), float(args[3])
    ids = te[:LIMIT] if LIMIT else te
    rows = deploy_rows(arm, seed, ids, (thr,))[thr]
    ok = [r for r in rows if "gap" in r]
    G = np.array([r["gap"] for r in ok])
    print(f"test {arm:10s} s{seed} thr {thr}: n={len(ok)}/{len(ids)} |gap| {np.abs(G).mean():.4f}% "
          f"signed {G.mean():+.4f}%  disc {np.mean([r['disc'] for r in ok]):.2f}%  "
          f"on {np.mean([r['n_on'] for r in ok]):.1f}  {np.mean([r['secs'] for r in ok]):.1f}s/inst", flush=True)
    json.dump(rows, open(f"{OUT}/test_{arm}_s{seed}_t{thr}{'_lim' if LIMIT else ''}.json", "w"))

elif mode == "refs":
    sl = ref_slack("case300")
    ids = te[:LIMIT] if LIMIT else te
    out = {}
    for j in ids:
        pr = S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))
        out[j] = dict(stored_cost=float(POOL[j][3]), stored_slack=float(sl[j]),
                      priced_cost=None if pr is None else float(pr["cost"]),
                      priced_slack=None if pr is None else float(pr["slack"]))
    n_conv = sum(v["stored_slack"] <= 1e-3 for v in out.values())
    print(f"refs: {len(out)} test instances, {n_conv} converged (stored slack <= 1e-3)", flush=True)
    json.dump(out, open(f"{OUT}/refs.json", "w"), indent=1)

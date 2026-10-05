# %% One SHARD of the per-generator threshold search (phase 1: SEARCH ONLY).
#
#    Why this exists: the relaxation leaves 41 of 54 case118 units fractional
#    and we collapse all of it at a hard-coded 0.5. Oracle probe: a perfect
#    per-generator threshold gives 1.39% discrete / 2.87% continuous against
#    5.71% / 15.93% shipped. This fits one WITHOUT labels and WITHOUT a
#    gradient (the analytic dL/dthr failed its finite-difference gate).
#
#        thr_k = 0.5 + c0 + c1*z(pmax_k) + c2*z(mc_k) + c3*z(nl_k)
#
#    ranked on mean DEPLOYED COST over training instances only. A threshold can
#    only change the rounding -- restoration still runs after it -- so a bad
#    one costs money, never feasibility.
import os, sys, json, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(1)

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="case118")
ap.add_argument("--pool", default="wide")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-fit", type=int, default=32)
ap.add_argument("--n-cfg", type=int, default=24)
ap.add_argument("--kcuts", type=int, default=2)
ap.add_argument("--down", type=int, default=6)
# case300 needs the cheapest-first MERIT order; case118 needs the
# network's confidence. Fitting under one and deploying under the
# other would tune the threshold against the wrong pipeline.
ap.add_argument("--order", default="conf", choices=["conf", "merit"])
# case118 UNDER-commits, so its search was biased DOWNWARD
# (c0 in [-0.30, 0.05]) -- lower threshold, commit more. case300
# OVER-commits by 10.2 units, so it needs the OPPOSITE direction.
# Reusing case118's range would search the wrong half of the space.
ap.add_argument("--c0-lo", type=float, default=-0.30)
ap.add_argument("--c0-hi", type=float, default=0.05)
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case; os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); QC = None   # wide pools live in results/wide
from pipeline import grid, RESERVE, set_cuts, set_down, deploy
from socp import Socp
from cutnet_k import CutNetK, rows_torch
import poolload

K = a.kcuts; ROWS = 2*K; RHO = 1e6
NT, NV, NTR = 48, 24, 144
set_cuts(ROWS); set_down(a.down)
g, nl = grid(a.case); NG, NB = g.n_gen, g.n_bus
S = Socp(g, nl, n_cuts=ROWS); V1, V0 = np.ones(NB), np.zeros(NB)
poolload.set_grid(g)
POOL = poolload.load_pool(a.case, a.pool, QC)
perm = np.random.default_rng(0).permutation(len(POOL))
te = [int(i) for i in perm[:NT]]
_tr = [int(i) for i in perm[NT:NT+NTR+NV]]; tr = _tr[NV:]
# normalisation must match the trainer, which used the FULL train split
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
# SPEEDUP: only the instances this shard actually deploys need a relaxation
fit = tr[:a.n_fit]
need = fit if os.environ.get("PHASE", "1") == "1" else te
W = {}
for j in need:
    r = S.solve(POOL[j][0], POOL[j][1], V1, V0, rho=RHO)
    if r is not None:
        W[j] = np.concatenate([r["pg"], r["u"]])
ids = [j for j in need if j in W]
WR = torch.tensor(np.stack([W[j] for j in ids]), dtype=torch.float32)
ck = f"checkpoints/{a.case}_s{a.seed}.pt"
net = CutNetK(NB, NG, k_rows=K, v_scale=0.5, seed=a.seed)
net.load_state_dict(torch.load(ck, weights_only=False)["net"], strict=False)
net.eval()
with torch.no_grad():
    Vr_t, Vi_t, aa, al, a2, d, d2, _ = net(Xn[ids])
    A_t, b_t = rows_torch(aa, al, a2, d, d2, WR)
A = A_t.numpy().astype(float); b = b_t.numpy().astype(float)
Vr = Vr_t.numpy().astype(float); Vi = Vi_t.numpy().astype(float)
# LABEL-FREE normaliser. This previously divided by POOL[j][3], the GUROBI
# reference cost: that never reveals which units to commit, but it is derived
# from the optimal solution and it reweights instances in the mean, so the fit
# was not label-free. The uncut relaxation cost at flat V serves the same
# scaling purpose and is the normaliser the network trainer already uses.
SCALE = np.array([float(S.solve(POOL[j][0], POOL[j][1], V1, V0, rho=RHO)["cost"])
                  for j in ids])

ORD = (np.argsort(g.c1 + g.c2*g.pmax + nl/np.maximum(g.pmax, 1e-9))
       if a.order == 'merit' else None)
zz = lambda v: (v - v.mean())/(v.std() + 1e-12)
FEAT = np.stack([zz(g.pmax), zz(g.c1 + g.c2*g.pmax), zz(nl)])
thr_of = lambda c: np.clip(0.5 + c[0] + c[1]*FEAT[0] + c[2]*FEAT[1] + c[3]*FEAT[2], 0.05, 0.95)

def score(c):
    th = None if c is None else thr_of(c)
    tot = 0.0
    for k, j in enumerate(ids):
        cst, dd = deploy(S, g, POOL[j][0], POOL[j][1], Vr[k], Vi[k],
                         A=A[k], b=b[k], rho=RHO, thr=th, order=ORD)
        tot += 10.0 if (cst >= 1e8 or dd is None) else cst/SCALE[k]
    return tot/len(ids)

import time
t0 = time.time()
base_sc = score(None)
print(f"[s{a.seed} sh{a.shard}] {len(ids)} fit instances, shipped {base_sc:.6f} "
      f"({time.time()-t0:.0f}s/eval-set)", flush=True)
rng = np.random.default_rng(7000 + 97*a.seed + a.shard)
best = (base_sc, np.zeros(4))
for t in range(a.n_cfg):
    c = np.array([0.5*(a.c0_lo+a.c0_hi), 0, 0, 0]) \
        if (t == 0 and a.shard == 0) else \
        np.r_[rng.uniform(a.c0_lo, a.c0_hi), rng.normal(0, 0.12, 3)]
    sc = score(c)
    if sc < best[0]:
        best = (sc, c)
    print(f"  [{t:3d}] {sc:.6f}{'  *' if sc == best[0] else ''}  [{time.time()-t0:.0f}s]",
          flush=True)
json.dump(dict(seed=a.seed, shard=a.shard, base=base_sc,
               best=best[0], c=best[1].tolist()),
          open(f"results/thr_lf/sh_{a.case}_s{a.seed}_{a.shard}.json", "w"))
print(f"[s{a.seed} sh{a.shard}] BEST {best[0]:.6f} base {base_sc:.6f} "
      f"c={np.round(best[1],4).tolist()}  {time.time()-t0:.0f}s", flush=True)

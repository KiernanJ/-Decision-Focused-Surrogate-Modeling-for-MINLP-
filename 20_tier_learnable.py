"""Can tier counts be predicted accurately enough to be worth cutting on?

19_tiered showed the ceiling: with ORACLE tier counts and hard cuts, K=3 tiers
reaches 0.184% against global cardinality's 0.489% and 0.923% with no cut. So
the family carries real information -- a constant cannot imitate it.

It also showed the danger: WIDENING each tier band by +-1 collapses the benefit
(0.290% -> 0.795%). A tiered cut only pays when tight, so a learned one has to
be accurate. Two things decide whether that is achievable:

  A. ROBUSTNESS TO BEING WRONG. A widened band is not the same failure a network
     makes -- a network puts the band in the WRONG PLACE. Shift each tier count
     by +-1 (tight, but misplaced) and see what survives. If a 1-unit error per
     tier is fatal, the family is unlearnable however good the ceiling looks.
  B. PREDICTABILITY. Global n* correlates 0.926 with total demand and a linear
     fit gets it to within 0.63 units. If per-tier counts are similarly
     predictable, the errors in A are within reach.
"""
# %% setup
import os, sys, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp

rng = np.random.default_rng(0)
set_cuts(24); set_down(6)
g, nl = grid(); NG = g.n_gen
S = Socp(g, nl, n_cuts=24)
POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
             for i in range(int(m.group(2))) if REF[i] is not None]
avg = (g.c2*g.pmax**2 + g.c1*g.pmax + nl)/np.maximum(g.pmax, 1e-6)
MERIT = np.argsort(avg)
tiers = lambda K: [MERIT[i::K] for i in range(K)]

# ---- B. predictability of per-tier counts ---------------------------------
print("B. how predictable are tier counts from total demand?")
D = np.array([float(p.sum()) for p, _, _, _ in POOL])
A_ = np.c_[D, np.ones(len(D))]
for K in [1, 3, 5, 8]:
    errs = []
    for T in tiers(K):
        y = np.array([float(u[T].sum()) for _, _, u, _ in POOL])
        coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
        errs.append(np.abs(A_ @ coef - y))
    E = np.stack(errs)
    print(f"  K={K}: per-tier |error| mean {E.mean():.2f}  max {E.max():.2f}   "
          f"total-count |error| {np.abs(E.sum(0)).mean():.2f} units")

# ---- A. robustness to a MISPLACED (not widened) band ----------------------
perm = rng.permutation(len(POOL)); te = [int(i) for i in perm[:24]]
VS = {j: (lambda r: np.r_[r["vr"], r["vi"]])(
    S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))) for j in te}
PGS = {j: S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))["pg"] for j in te}


def rows(K, u_star, err_sd, rg):
    A, b = np.zeros((24, 2*NG)), np.ones(24)*1e3
    k = 0
    for T in tiers(K):
        n_t = float(u_star[T].sum())
        if err_sd > 0:
            n_t = float(np.clip(round(n_t + rg.normal(0, err_sd)), 0, len(T)))
        A[k, NG + T] = -1.0; b[k] = -n_t; k += 1
        A[k, NG + T] = 1.0;  b[k] = n_t;  k += 1
    return A, b


def deploy(j, V, A, b):
    pd_, qd_ = POOL[j][0], POOL[j][1]
    r = S.solve(pd_, qd_, V[:g.n_bus], V[g.n_bus:], rho=1e6, A=A, b=b, cut_cap=0.0)
    if r is None:
        return None
    uf = r["u"]; z = (uf > 0.5).astype(float); o = np.argsort(-uf)
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
    for t_ in [t2 for t2 in np.argsort(uf) if z[t2] > 0.5][:6]:
        w_ = z.copy(); w_[t_] = 0.0
        if float(g.pmax @ w_) < need:
            continue
        pr = S.price(pd_, qd_, w_)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w_, pr["cost"], pr["pg"]
    return z.astype(int), cost, pg


print(f"\nA. tier counts TIGHT but MISPLACED (the error a network makes)")
print(f"   moderate V error, hard cuts")
print(f"{'family / tier-count error':34s}{'n':>4s}{'discrete':>11s}{'continuous':>13s}{'gap':>10s}")
for K, sd_err in [(0, 0.0), (1, 0.0), (1, 0.7), (3, 0.0), (3, 0.5), (3, 0.7),
                  (3, 1.0), (5, 0.0), (5, 0.5), (5, 0.7)]:
    rg = np.random.default_rng(7)
    Dd, Cc, Gg = [], [], []
    for j in te:
        V = VS[j] + rng.normal(0, 0.02, 2*g.n_bus)
        A, b = (None, None) if K == 0 else rows(K, POOL[j][2].astype(float), sd_err, rg)
        out = deploy(j, V, A, b)
        if out is None:
            continue
        z, cost, pg = out
        Dd.append(100*float((z != POOL[j][2]).mean()))
        Cc.append(100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()))
        Gg.append(100*(cost-POOL[j][3])/POOL[j][3])
    name = "no cut" if K == 0 else f"K={K}  tier error sd={sd_err:.1f}"
    print(f"{name:34s}{len(Gg):>4d}{np.mean(Dd):>10.2f}%{np.mean(Cc):>12.2f}%"
          f"{np.mean(Gg):>9.3f}%", flush=True)

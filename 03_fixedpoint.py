"""Is the LABEL-FREE fixed point as good as the oracle?

02_ceiling showed the oracle linearisation point -- taken with u fixed at the
reference commitment -- deploys at -0.002% and 100% agreement. That target needs
the reference, so it cannot train anything by itself.

The self-supervised alternative is the fixed point of the RELAXATION map, u
free:  V <- Solve_relaxed(V).  No labels, no reference, no binaries. If that
lands in the same place, it is a training target the network can chase, and the
loss ||Solve(V) - V||^2 is SMOOTH -- exact gradients, one solve per step instead
of the five the smoothed estimator needs.

This measures three things per instance:
  * how many iterations the label-free map takes to settle
  * how far its fixed point sits from the oracle point
  * what deploying from it actually costs
"""
# %% setup
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, grid, sample, deploy
from socp import Socp

SEED, N, NTEST = 0, 32, 8
g, nl = grid()
inst = sample(g, N, seed=SEED)
REF = list(np.load(f"{HERE}/results/ref_{CASE}_s{SEED}_n{N}.npz",
                   allow_pickle=True)["ref"])
te = [i for i in range(N-NTEST, N) if REF[i] is not None]
S = Socp(g, nl, n_cuts=8)


# %% the label-free map:  V <- Solve_relaxed(V), u FREE
def fixed_point(pd_, qd_, iters=12, tol=1e-6, rho=1e6):
    Vr, Vi = np.ones(g.n_bus), np.zeros(g.n_bus)
    for it in range(iters):
        r = S.solve(pd_, qd_, Vr, Vi, rho=rho)
        if r is None:
            return Vr, Vi, it, np.nan
        d = float(np.linalg.norm(np.r_[r["vr"]-Vr, r["vi"]-Vi]))
        Vr, Vi = r["vr"], r["vi"]
        if d < tol:
            return Vr, Vi, it+1, d
    return Vr, Vi, iters, d


# %% run
rows = []
for i in te:
    pd_, qd_ = inst[i]
    ref_c, ref_u = REF[i]["cost"], REF[i]["u"]
    t0 = time.time()
    Vr, Vi, nit, res = fixed_point(pd_, qd_)
    t_fp = time.time() - t0
    c_fp, d_fp = deploy(S, g, pd_, qd_, Vr, Vi)
    pr = S.price(pd_, qd_, ref_u.astype(float))          # oracle point
    dist = float(np.linalg.norm(np.r_[Vr-pr["vr"], Vi-pr["vi"]]))
    rows.append(dict(
        gap=100*(c_fp-ref_c)/ref_c if c_fp < 1e8 else np.nan,
        agr=100*float((d_fp["u"] == ref_u).mean()) if d_fp else np.nan,
        nit=nit, res=res, dist=dist, secs=t_fp))

# %% report
f = lambda k: np.nanmean([x[k] for x in rows])
print(f"\ncase118, seed {SEED}, {len(te)} held-out instances")
print(f"{'linearisation point':30s}{'gap':>9s}{'agree':>9s}")
print(f"{'flat V (baseline)':30s}{10.885:>8.3f}%{56.9:>8.1f}%")
print(f"{'learned V (smoothed, 60 ep)':30s}{1.388:>8.3f}%{86.1:>8.1f}%")
print(f"{'LABEL-FREE fixed point':30s}{f('gap'):>8.3f}%{f('agr'):>8.1f}%")
print(f"{'ORACLE V (needs reference)':30s}{-0.002:>8.3f}%{100.0:>8.1f}%")
print(f"\n  label-free map: {f('nit'):.1f} iterations to settle, "
      f"residual {f('res'):.2e}, {f('secs'):.2f}s")
print(f"  distance from its fixed point to the oracle point: {f('dist'):.4f}")
print(f"""
  If the LABEL-FREE row matches the ORACLE row, ||Solve(V)-V||^2 is a valid
  self-supervised target worth ~0% gap -- smooth, exact-gradient, one solve a
  step. The network's job becomes predicting that point in ONE shot.""")

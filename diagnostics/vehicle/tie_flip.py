"""No-learning baseline: uncut relaxation -> qz projection -> flip near-tie steps.
After the L1 projection, try moving each step whose relaxed z is within 0.15 of a
half-integer to its other rounding; keep the flip if the priced cost drops; repeat
until no flip helps. Label-free (labels only score). Validation + 48 train instances.
Usage: python tie_flip.py   (~5 s, needs Gurobi)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

R = make_relax(2)


def price(i, z):
    r = R.solve(D[i], z_lo=z, z_hi=z)
    return None if r is None else r["cost"]


for name, ids in (("val", va), ("train[:48]", tr[:48])):
    g0, g1, nsolve, t0 = [], [], [], time.time()
    for i in ids:
        zf = R.solve(D[i])["z"]; zq = qz_restore(D[i], zf); c = price(i, zq)
        g0.append(100*(c - REF[i]["cost"])/abs(REF[i]["cost"]))
        fr = zf - np.floor(zf)
        ties = [t for t in np.argsort(np.abs(fr - 0.5)) if abs(fr[t] - 0.5) < 0.15]
        n, improved = 0, True
        while improved:
            improved = False
            for t in ties:
                w = zq.copy(); w[t] = np.floor(zf[t]) if zq[t] > zf[t] else np.ceil(zf[t])
                cw = price(i, w); n += 1
                if cw is not None and cw < c - 1e-9:
                    zq, c, improved = w, cw, True
        g1.append(100*(c - REF[i]["cost"])/abs(REF[i]["cost"])); nsolve.append(n)
    g0, g1 = np.array(g0), np.array(g1)
    print(f"{name:10s} uncut+qz |gap| {np.abs(g0).mean():.4f}% (exact {np.sum(np.abs(g0) < 1e-3)}/{len(ids)})  ->  "
          f"+ tie-flip |gap| {np.abs(g1).mean():.4f}% (exact {np.sum(np.abs(g1) < 1e-3)}/{len(ids)}, max {g1.max():.3f}%)  "
          f"extra convex solves/inst {np.mean(nsolve):.1f}  {time.time() - t0:.0f}s")

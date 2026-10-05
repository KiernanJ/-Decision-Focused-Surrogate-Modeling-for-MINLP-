"""Feasibility restoration shared by EVERY arm, so arms differ only in how the
fractional commitment `uf` is produced. Byte-for-byte the post-processing of
pipeline.deploy: threshold -> reserve top-up in `order` -> upward repair until
the AC price is feasible -> NDOWN downward trials in the mirrored order."""
from __future__ import annotations
import numpy as np


def restore(S, g, pd_, qd_, uf, thr=0.5, order=None, ndown=6, reserve=0.10):
    uf = np.asarray(uf, float)
    z = (uf > np.asarray(thr, float)).astype(float)
    order = np.argsort(-uf) if order is None else np.asarray(order, int)
    need = (1.0 + reserve)*float(pd_.sum())
    for k in order:
        if float(g.pmax @ z) >= need:
            break
        z[k] = 1.0
    best = None
    for _ in range(g.n_gen + 1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"], pr["pg"]); break
        off = [k for k in order if z[k] < 0.5]
        if not off:
            return np.inf, None
        z[off[0]] = 1.0
    if best is None:
        return np.inf, None
    z, cost, pg = best
    for k in [t for t in order[::-1] if z[t] > 0.5][:ndown]:
        w = z.copy(); w[k] = 0.0
        if float(g.pmax @ w) < need:
            continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w, pr["cost"], pr["pg"]
    return cost, dict(u=z.astype(int), pg=pg)

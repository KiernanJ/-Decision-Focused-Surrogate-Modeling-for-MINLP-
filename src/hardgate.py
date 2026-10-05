"""Worker module for the HEADROOM GATE.

The question the gate answers: on the current instance distribution the
majority-vote CONSTANT commitment is within +0.35% (case118) / +0.22% (case300)
of optimal, so no learned map has more than that to win.  Does hardening the
distribution create headroom?  Three axes, each varied ALONE:

  B1 OUTAGE  each non-always-on unit independently unavailable w.p. 0.05
  B2 WIDE    system scale s ~ U[0.55,1.30] instead of U[0.80,1.15]
  B3 BOTH

An outage is applied by zeroing pmax/qmax/qmin on a COPY of the grid, which
makes the unit unable to produce anything; pmin is left alone so that
pg >= pmin*u with pg <= 0 forces u = 0 without any extra constraint.
Lives in a real module because multiprocessing pickles workers by module path.
"""
import os, copy
import numpy as np

_W = {}


def init(case):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["QCAC_CASE"] = case
    from pipeline import grid
    g, nl = grid(case)
    _W["g"], _W["nl"], _W["case"] = g, nl, case


def sample_hard(g, n, seed, lo, hi, p_out, spread=0.05, pf=0.05):
    """(pd, qd, avail) triples. avail=1 means the unit is available."""
    rng = np.random.default_rng(seed)
    elig = ~np.asarray(g.always_on, bool) if getattr(g, "always_on", None) is not None \
        else np.ones(g.n_gen, bool)
    out = []
    for _ in range(n):
        s = rng.uniform(lo, hi)
        pd_ = g.pd * s * rng.uniform(1-spread, 1+spread, g.n_bus)
        qd_ = g.qd * s * rng.uniform(1-spread-pf, 1+spread+pf, g.n_bus)
        av = np.ones(g.n_gen)
        if p_out > 0:
            av[elig & (rng.random(g.n_gen) < p_out)] = 0.0
        out.append((pd_, qd_, av))
    return out


def work(args):
    i, pd_, qd_, av = args
    import time
    from acuc import solve_qcac_iterative
    g = copy.deepcopy(_W["g"])
    off = av < 0.5
    g.pmax = g.pmax.copy(); g.qmax = g.qmax.copy(); g.qmin = g.qmin.copy()
    g.pmax[off] = 0.0; g.qmax[off] = 0.0; g.qmin[off] = 0.0
    t0 = time.time()
    try:
        r = solve_qcac_iterative(g, _W["nl"], pd_, qd_, reserve=0.10, threads=1)
    except Exception as e:
        return i, None, str(e)[:120]
    if r is None:
        return i, None, "no solution"
    return i, dict(u=np.asarray(r["u"]), cost=float(r["cost"]),
                   secs=time.time()-t0, iters=int(r.get("iters", -1)),
                   slack=float(r.get("slack", np.nan))), None

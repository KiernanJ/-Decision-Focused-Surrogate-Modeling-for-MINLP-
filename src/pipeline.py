"""Shared pipeline: grid, instance sampling, deployment, and the pool workers.

These live in a real importable module because multiprocessing must PICKLE the
worker functions, which it does by module path. `01_run.py` starts with a digit
so it cannot be imported, and exec'ing it produces functions whose module does
not exist -- the pool then fails with "Can't pickle <function _ref>". Anything a
worker touches belongs here.
"""
from __future__ import annotations

import os
import time

import numpy as np

from acopf_data import load
from qcac import no_load_cost
from socp import Socp

CASE, RESERVE = os.environ.get("QCAC_CASE", "case118"), 0.10

# Per-case voltage limits. pandapower ships 0/2 placeholders on both cases --
# not operating limits. case118 runs at MATPOWER's 0.94/1.06; case300 is
# INFEASIBLE there and needs 0.90/1.10, which is its own MATPOWER range.
VBOUNDS = {"case118": (0.94, 1.06), "case300": (0.90, 1.10)}
NCUTS = 8
NDOWN = 0          # downward repair trials; 0 reproduces upward-only restoration


def set_down(k):
    global NDOWN
    NDOWN = int(k)


def set_cuts(k):
    global NCUTS
    NCUTS = int(k)


# %% grid
def grid(case=None):
    case = case or CASE
    g = load(case)
    lo, hi = VBOUNDS.get(case, (0.94, 1.06))
    g.vmin = np.full(g.n_bus, lo)
    g.vmax = np.full(g.n_bus, hi)
    return g, no_load_cost(g, 0.15)


# %% instances -- ONE correlated system draw, not i.i.d. per load.
# i.i.d. per-load sampling averages out the aggregate variation the reserve
# constraint and the merit order respond to: it gave under 1% demand CV on
# case118 and 1-2 distinct optimal commitments out of 32. This gives ~10% CV
# and 16 distinct.
def sample(g, n_inst, seed=0, lo=0.80, hi=1.15, spread=0.05, pf=0.05):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_inst):
        s = rng.uniform(lo, hi)
        out.append((g.pd * s * rng.uniform(1-spread, 1+spread, g.n_bus),
                    g.qd * s * rng.uniform(1-spread-pf, 1+spread+pf, g.n_bus)))
    return out


# %% deployment: one solve -> round -> reserve top-up -> price -> repair up
def deploy(S, g, pd_, qd_, Vr, Vi, A=None, b=None, rho=1e6, fail=1e9):
    r = S.solve(pd_, qd_, Vr, Vi, rho=rho, A=A, b=b)
    if r is None:
        return fail, None
    uf = r["u"]
    z = (uf > 0.5).astype(float)
    order = np.argsort(-uf)
    need = (1.0 + RESERVE) * float(pd_.sum())
    for k in order:
        if float(g.pmax @ z) >= need:
            break
        z[k] = 1.0
    best = None
    for _ in range(g.n_gen + 1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"]); break
        off = [k for k in order if z[k] < 0.5]
        if not off:
            return fail, None
        z[off[0]] = 1.0
    if best is None:
        return fail, None
    z, cost = best

    # Downward repair. Without it restoration can only switch generators ON, so
    # over-commitment is unrecoverable and the network is pushed into biasing
    # its cardinality band low to leave room for the upward climb. Trying the
    # least-confident ON units -- the order the reserve top-up forced them on --
    # removes that asymmetry: measured 0.103% -> 0.029% at 6 trials.
    for k in [t for t in np.argsort(uf) if z[t] > 0.5][:NDOWN]:
        w = z.copy(); w[k] = 0.0
        if float(g.pmax @ w) < need:
            continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost = w, pr["cost"]
    return cost, dict(u=z.astype(int), u_frac=uf, iters=-1)


# %% pool workers
_W = {}
INST = []


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    g, nl = grid()
    _W["g"], _W["nl"], _W["S"] = g, nl, Socp(g, nl, n_cuts=NCUTS)


def _ref(args):
    """Reference commitment: QCAC iterative in Gurobi, 54 binaries."""
    i, pd_, qd_ = args
    from acuc import solve_qcac_iterative
    t0 = time.time()
    r = solve_qcac_iterative(_W["g"], _W["nl"], pd_, qd_, reserve=RESERVE,
                             threads=2)
    if r is None:
        return i, None
    return i, dict(u=r["u"], cost=float(r["cost"]), secs=time.time()-t0,
                   iters=int(r.get("iters", -1)))


def _dep(args):
    i, pd_, qd_, Vr, Vi, A, b = args
    t0 = time.time()
    c, d = deploy(_W["S"], _W["g"], pd_, qd_, Vr, Vi, A, b)
    return i, c, (d["u"] if d else None), time.time()-t0


"""
Rung D: unit commitment over the QCAC convex approximation of AC-OPF.

QCAC (Constante-Flores & Li) rewrites AC-OPF in lifted rectangular coordinates,
expresses the nonconvex voltage products as differences of convex functions, and
linearizes ONLY the concave part around a base voltage (V_re, V_im). Given that
base point the result is a convex QCQP.

Add on/off decisions for the generators and you get a convex MIQCQP:

    min  sum_k  c2_k pg_k^2 + c1_k pg_k + c0_k u_k      + rho * slacks
    s.t. power balance                                  (1b),(1c)
         branch flows                                   (1d)-(1g)
         thermal limits, voltage bounds                 (1h),(1i)
         u_k pmin_k <= pg_k <= u_k pmax_k               commitment-linked
         u_k qmin_k <= qg_k <= u_k qmax_k
         QCAC convexified voltage constraints           (4b)-(4g)
         u_k in {0,1}

That is precisely the problem class the project targets: convex MINLP. The
continuous relaxation (u in [0,1]) is a convex QCQP and is what the learned cuts
are added to.

The branch-flow equations use the sign convention validated in acopf_data.py
against a converged AC power flow (machine precision on case5/14/30). The
opposite convention -- PowerModels' wi = -s_ij -- is what the earlier project
code used, and it is wrong by ~4 pu on case14.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acopf_data import Grid, load



# Penalty on the QCAC consistency slacks, CALIBRATED rather than assumed.
#
# rho prices the slack on the convex relaxation of the voltage products -- not
# on power balance -- so too small a value lets the network fabricate power by
# violating the relaxation, and generating becomes unnecessary. At the rho=1e3
# this project used everywhere, the relaxation served almost none of the load:
#
#     fraction of load the uncut relaxation actually generates
#       rho     case118   case300
#       1e3        0.8%      0.0%     <- the historical default
#       1e4       80.6%     50.5%
#       1e5      105.1%     98.3%
#       1e6      106.2%    100.6%     <- converged; excess is transmission loss
#       1e7      106.2%    101.2%
#
# Both the exact MIQCQP and the surrogate returned pg=0 at 1e3 and paid the
# penalty, so they agreed with each other on a problem where nothing generated.
# Gap, discrete error and the re-pricing sanity check all looked healthy; only
# the continuous dispatch error exposed it. build_references.py now gates on
# this. Measured 2026-09-25 by src/../logs/rho_sweep.log.
DEFAULT_RHO = 1e6

@dataclass
class QCACSpec:
    grid: Grid
    Vre: np.ndarray          # convexification point, real part
    Vim: np.ndarray          # convexification point, imaginary part
    rho: float = DEFAULT_RHO  # see DEFAULT_RHO above
    reserve: float = 0.10    # committed capacity margin over demand
    nl_frac: float = 0.15    # no-load cost as a fraction of cost at pmax


def base_point(grid: Grid, flat=False):
    """
    Convexification point for the difference-of-convex linearization.

    Measured: at a FLAT start the QCAC constraint set needs large slacks -- hard
    bounding them at 1e-3 makes case30 infeasible -- which is the situation the
    paper describes as the base point rendering an empty constraint set. The
    remedy it recommends is a realistic operating point: "a reliable
    linearization point can typically be obtained from the most recent solution
    of the state estimator."

    So the default is the converged AC power flow at NOMINAL demand, computed
    once and held fixed across instances. That is available offline and does not
    peek at the per-instance answer -- using each instance's own AC solution
    would be leaking the thing we are trying to predict.

    This base point is also the object the QCAC paper proposes learning
    (advantage #3), so it is the natural parametric handle for this case study.
    """
    if flat:
        return np.ones(grid.n_bus), np.zeros(grid.n_bus)
    import pandapower as pp
    import pandapower.networks as pn
    net = getattr(pn, grid.name)()
    pp.runpp(net, calculate_voltage_angles=True, init="flat")
    bus = net._ppc["bus"]
    vm, va = bus[:, 7], np.deg2rad(bus[:, 8])
    return vm * np.cos(va), vm * np.sin(va)


def no_load_cost(grid: Grid, frac=0.15):
    """
    Commitment cost. MATPOWER cases carry no on/off cost, so without one the UC
    is trivial (every unit on) -- the same degeneracy the vehicle problem showed
    below T=30. Set it to a fraction of each unit's cost at full output.
    """
    nl = getattr(grid, "nl_override", None)
    if nl is not None:
        return np.asarray(nl, float)      # real no-load cost from the case file
    at_max = grid.c2 * grid.pmax**2 + grid.c1 * grid.pmax + grid.c0
    return frac * np.maximum(at_max, 1e-6)


# ---------------------------------------------------------------------------
# shared constraint construction (cvxpy expressions)
# ---------------------------------------------------------------------------

def _qcac_constraints(cp, spec, vr, vi, cii, cij, sij, xi_c, xi_ij, xi_s):
    """QCAC convexified voltage constraints, paper equations (4b)-(4g)."""
    g = spec.grid
    i, j = g.f_bus, g.t_bus
    Vr, Vi = spec.Vre, spec.Vim
    C = []
    # 4b: c_ii >= |v_i|^2   (convex, kept exactly)
    C.append(cii >= cp.square(vr) + cp.square(vi))
    # 4c: the concave side, linearized around (Vr, Vi)
    C.append(cii <= 2 * (cp.multiply(Vr, vr) + cp.multiply(Vi, vi))
             - (Vr**2 + Vi**2) + xi_c)
    # 4d / 4e: c_ij
    C.append(cp.square(vr[i] + vr[j]) + cp.square(vi[i] + vi[j]) - 4 * cij
             <= xi_ij
             + 2 * cp.multiply(Vr[i] - Vr[j], vr[i] - vr[j])
             + 2 * cp.multiply(Vi[i] - Vi[j], vi[i] - vi[j])
             - ((Vr[i] - Vr[j])**2 + (Vi[i] - Vi[j])**2))
    C.append(cp.square(vr[i] - vr[j]) + cp.square(vi[i] - vi[j]) + 4 * cij
             <= xi_ij
             + 2 * cp.multiply(Vr[i] + Vr[j], vr[i] + vr[j])
             + 2 * cp.multiply(Vi[i] + Vi[j], vi[i] + vi[j])
             - ((Vr[i] + Vr[j])**2 + (Vi[i] + Vi[j])**2))
    # 4f / 4g: s_ij
    C.append(cp.square(vr[i] - vi[j]) + cp.square(vr[j] + vi[i]) + 4 * sij
             <= xi_s
             + 2 * cp.multiply(Vr[i] + Vi[j], vr[i] + vi[j])
             + 2 * cp.multiply(Vr[j] - Vi[i], vr[j] - vi[i])
             - ((Vr[i] + Vi[j])**2 + (Vr[j] - Vi[i])**2))
    C.append(cp.square(vr[i] + vi[j]) + cp.square(vr[j] - vi[i]) - 4 * sij
             <= xi_s
             + 2 * cp.multiply(Vr[i] - Vi[j], vr[i] - vi[j])
             + 2 * cp.multiply(Vr[j] + Vi[i], vr[j] + vi[i])
             - ((Vr[i] - Vi[j])**2 + (Vr[j] + Vi[i])**2))
    return C


def _flows(spec, cii, cij, sij):
    """Branch flows, paper (1d)-(1g), in the validated sign convention."""
    g = spec.grid
    i, j = g.f_bus, g.t_bus
    gg, bb, bsh, T, tm2 = g.g, g.b, g.b_fr, g.tap, g.tm2
    # NOTE: `numpy_array * cvxpy_expr` means MATRIX multiplication in cvxpy, which
    # silently contracts these per-branch vectors into scalars. Elementwise
    # products of constants with expressions must go through cp.multiply.
    import cvxpy as cp
    m = cp.multiply
    p_fr = m(gg / tm2, cii[i]) - m(gg / T, cij) + m(bb / T, sij)
    p_to = m(gg, cii[j]) - m(gg / T, cij) - m(bb / T, sij)
    q_fr = -m((bb + bsh) / tm2, cii[i]) + m(gg / T, sij) + m(bb / T, cij)
    q_to = -m(bb + bsh, cii[j]) - m(gg / T, sij) + m(bb / T, cij)
    return p_fr, q_fr, p_to, q_to


def _balance(cp, spec, pg, qg, cii, p_fr, q_fr, p_to, q_to, pd, qd):
    """Nodal power balance (1b),(1c) assembled with incidence matrices."""
    g = spec.grid
    n, L, G = g.n_bus, g.n_branch, g.n_gen
    Ag = np.zeros((n, G));  Ag[g.gen_bus, np.arange(G)] = 1.0
    Af = np.zeros((n, L));  Af[g.f_bus, np.arange(L)] = 1.0
    At = np.zeros((n, L));  At[g.t_bus, np.arange(L)] = 1.0
    return [
        Ag @ pg - cp.multiply(g.gs, cii) - Af @ p_fr - At @ p_to == pd,
        Ag @ qg + cp.multiply(g.bs, cii) - Af @ q_fr - At @ q_to == qd,
    ]


def build_relaxation(spec, pd, qd, relax_u=True, u_fixed=None, extra_cuts=None):
    """
    Continuous QCAC-UC model as a cvxpy Problem.

    relax_u=True  -> u in [0,1], the convex relaxation the cuts attach to
    u_fixed given -> commitment held fixed (used for integrality recovery)
    extra_cuts    -> (A, b) cvxpy Parameters adding A @ w <= b on w = [pg; u]
    """
    import cvxpy as cp
    g = spec.grid
    n, L, G = g.n_bus, g.n_branch, g.n_gen

    vr = cp.Variable(n); vi = cp.Variable(n)
    cii = cp.Variable(n); cij = cp.Variable(L); sij = cp.Variable(L)
    pg = cp.Variable(G); qg = cp.Variable(G)
    xi_c = cp.Variable(n, nonneg=True)
    xi_ij = cp.Variable(L, nonneg=True); xi_s = cp.Variable(L, nonneg=True)

    if u_fixed is not None:
        u = np.asarray(u_fixed, dtype=float)
    else:
        u = cp.Variable(G)

    cons = []
    if u_fixed is None:
        cons += [u >= 0, u <= 1]
    cons += _qcac_constraints(cp, spec, vr, vi, cii, cij, sij, xi_c, xi_ij, xi_s)
    cons += [cii >= g.vmin**2, cii <= g.vmax**2]
    cons += [vi[g.ref] == 0, vr[g.ref] >= 0]
    # commitment-linked generator limits
    cons += [pg >= cp.multiply(g.pmin, u), pg <= cp.multiply(g.pmax, u),
             qg >= cp.multiply(g.qmin, u), qg <= cp.multiply(g.qmax, u)]

    p_fr, q_fr, p_to, q_to = _flows(spec, cii, cij, sij)
    cons += _balance(cp, spec, pg, qg, cii, p_fr, q_fr, p_to, q_to, pd, qd)
    fin = np.isfinite(g.rate)
    if fin.any():
        cons += [cp.square(p_fr[fin]) + cp.square(q_fr[fin]) <= g.rate[fin]**2,
                 cp.square(p_to[fin]) + cp.square(q_to[fin]) <= g.rate[fin]**2]

    if extra_cuts is not None:
        A, b = extra_cuts
        w = cp.hstack([pg, u]) if u_fixed is None else cp.hstack([pg])
        cons += [A @ w <= b]

    nl = no_load_cost(g, spec.nl_frac)
    obj = (g.c2 @ cp.square(pg) + g.c1 @ pg + nl @ u
           + spec.rho * (cp.sum(xi_c) + cp.sum(xi_ij) + cp.sum(xi_s)))
    return cp.Problem(cp.Minimize(obj), cons), dict(
        vr=vr, vi=vi, cii=cii, cij=cij, sij=sij, pg=pg, qg=qg, u=u,
        xi=(xi_c, xi_ij, xi_s))


def solve_exact(spec, pd, qd, solver="GUROBI", time_limit=None, verbose=False,
                threads=None):
    """Exact convex MIQCQP: QCAC + binary commitments."""
    import cvxpy as cp
    g = spec.grid
    n, L, G = g.n_bus, g.n_branch, g.n_gen

    vr = cp.Variable(n); vi = cp.Variable(n)
    cii = cp.Variable(n); cij = cp.Variable(L); sij = cp.Variable(L)
    pg = cp.Variable(G); qg = cp.Variable(G); u = cp.Variable(G, boolean=True)
    xi_c = cp.Variable(n, nonneg=True)
    xi_ij = cp.Variable(L, nonneg=True); xi_s = cp.Variable(L, nonneg=True)

    cons = _qcac_constraints(cp, spec, vr, vi, cii, cij, sij, xi_c, xi_ij, xi_s)
    cons += [cii >= g.vmin**2, cii <= g.vmax**2, vi[g.ref] == 0, vr[g.ref] >= 0]
    cons += [pg >= cp.multiply(g.pmin, u), pg <= cp.multiply(g.pmax, u),
             qg >= cp.multiply(g.qmin, u), qg <= cp.multiply(g.qmax, u)]
    # The slack bus is always energised: pandapower's ext_grid is not a
    # dispatchable unit and its capacity is the synthetic 1.5*load_tot the
    # loader substitutes for the sentinel, which alone satisfies the reserve
    # constraint. Left free, the optimum was "commit only the phantom unit"
    # on 7/32 case300 instances. See Grid.always_on.
    ao = getattr(g, "always_on", None)
    if ao is not None and np.any(ao):
        cons += [u[np.asarray(ao, bool)] == 1]
    cons += [g.pmax @ u >= (1.0 + spec.reserve) * float(np.sum(pd))]
    p_fr, q_fr, p_to, q_to = _flows(spec, cii, cij, sij)
    cons += _balance(cp, spec, pg, qg, cii, p_fr, q_fr, p_to, q_to, pd, qd)
    fin = np.isfinite(g.rate)
    if fin.any():
        cons += [cp.square(p_fr[fin]) + cp.square(q_fr[fin]) <= g.rate[fin]**2,
                 cp.square(p_to[fin]) + cp.square(q_to[fin]) <= g.rate[fin]**2]

    nl = no_load_cost(g, spec.nl_frac)
    obj = (g.c2 @ cp.square(pg) + g.c1 @ pg + nl @ u
           + spec.rho * (cp.sum(xi_c) + cp.sum(xi_ij) + cp.sum(xi_s)))
    prob = cp.Problem(cp.Minimize(obj), cons)
    kw = {"verbose": verbose}
    if time_limit is not None:
        kw["TimeLimit"] = time_limit
    # Gurobi has its OWN thread pool and defaults to every core it can see.
    # OMP_NUM_THREADS does not touch it. Running N solves in a process pool
    # therefore asks for N x ncores threads: with 11 workers on 12 cores that
    # is 132 threads, and every solve is starved. That -- not problem hardness
    # -- is why reference builds kept hitting their time limit (15/32 proved,
    # 0/32 proved, a 9.2% max MIP gap, and a case300 instance that "timed out"
    # at 1801 s and then solved optimally in 748 s when given room). Callers
    # running a pool MUST pass threads = cores // workers.
    if threads is not None:
        kw["Threads"] = int(threads)
    try:
        prob.solve(solver=solver, **kw)
    except Exception as e:
        return {"feasible": False, "err": str(e)[:200]}
    if prob.status not in ("optimal", "optimal_inaccurate", "user_limit") or u.value is None:
        return {"feasible": False, "status": prob.status}
    slack = float(np.sum(xi_c.value) + np.sum(xi_ij.value) + np.sum(xi_s.value))
    # Record the remaining MIP gap, not just the binary proved flag. A
    # time-limited solve still returns an incumbent, and whether that incumbent
    # is usable as a reference depends entirely on how far the bound is from it
    # -- 0.001% is a fine reference, 5% is not. With only `proved` recorded
    # there was no way to tell, which matters now that capping the phantom slack
    # made the MIQCQP hard enough that 17/32 case300 instances hit the limit.
    mip_gap = None
    try:
        mip_gap = float(prob.solver_stats.extra_stats.MIPGap)
    except Exception:
        pass
    return {"feasible": True, "status": prob.status, "cost": float(prob.value),
            "mip_gap": mip_gap,
            "u": np.round(np.asarray(u.value).ravel()),
            "pg": np.asarray(pg.value).ravel(),
            "slack": slack,
            "proved": prob.status in ("optimal", "optimal_inaccurate")}


def sample_demand(grid: Grid, n, seed=0, lo=0.85, hi=1.15):
    """Instance parameter: a multiplicative load scaling per bus."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(lo, hi, size=(n, grid.n_bus))
    return grid.pd[None, :] * f, grid.qd[None, :] * f


# ---------------------------------------------------------------------------
# Per-instance convexification point
# ---------------------------------------------------------------------------

def make_instance(case: str, factors: np.ndarray, base_from_pf: bool = True):
    """
    Build one instance: scale every load by `factors`, then take the
    convexification point from an AC POWER FLOW at that demand.

    Measured motivation: a flat base point leaves the QCAC constraint set needing
    large slacks (hard-bounding at 1e-3 makes case30 infeasible), and a base
    point from NOMINAL demand still leaves slack ~4.5e-2 once demand moves +/-15%,
    with CLARABEL and SCS reporting `optimal_inaccurate`. Linearizing at the
    instance's own operating point should drive slacks toward zero, which is both
    what the paper prescribes and what the conditioning needs.

    This does not leak the answer. A power flow is not an OPF -- it solves the
    network at a GIVEN dispatch (the case's nominal generator setpoints), costs
    milliseconds, and knows nothing about which units the optimizer will commit
    or how it will dispatch them.

    Returns (pd, qd, Vre, Vim), all per-unit, buses in Grid order.
    """
    import pandapower as pp
    import pandapower.networks as pn

    net = getattr(pn, case)()
    f = np.asarray(factors, dtype=float)
    if f.size == 1:
        f = np.full(len(net.load), float(f))
    if f.size != len(net.load):
        raise ValueError(f"{case} has {len(net.load)} loads, got {f.size} factors")
    net.load = net.load.copy()
    net.load["p_mw"] = net.load["p_mw"].values * f
    net.load["q_mvar"] = net.load["q_mvar"].values * f

    try:
        pp.runpp(net, calculate_voltage_angles=True, init="flat")
    except Exception:
        return None

    ppc = net._ppc
    base = ppc["baseMVA"]
    bus = ppc["bus"]
    pd_ = bus[:, 2] / base
    qd_ = bus[:, 3] / base
    vm, va = bus[:, 7], np.deg2rad(bus[:, 8])
    if base_from_pf:
        Vre, Vim = vm * np.cos(va), vm * np.sin(va)
    else:
        Vre, Vim = np.ones(len(bus)), np.zeros(len(bus))
    return pd_, qd_, Vre, Vim


def sample_instances(case: str, n: int, seed: int = 0, lo=0.85, hi=1.15):
    """n instances with per-load scalings, each carrying its own base point."""
    import pandapower.networks as pn
    net = getattr(pn, case)()
    n_load = len(net.load)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        f = rng.uniform(lo, hi, size=n_load)
        inst = make_instance(case, f)
        if inst is not None:
            out.append(inst + (f,))     # (pd, qd, Vre, Vim, factors)
    return out

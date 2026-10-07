"""Worker side of the CombOptNet-style learned cuts for AC-UC (post-audit).

Restoration = small MILP: the L1-closest commitment to the relaxed u_hat that meets
reserve and the learned cut rows (A u <= b + s, s priced), then AC pricing (QCAC
re-linearisation with u fixed) with upward repair if pricing fails. A = None is the
no-cut baseline. The gradient signal is a feasible discrete steepest descent over
single-unit flips, priced with the same evaluator.
Lives in a module because macOS spawn pickles workers by module path.
"""
import os
import numpy as np

_W = {}


def init(case):
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["QCAC_CASE"] = case
    import gurobipy as gp
    from pipeline import grid, RESERVE, set_cuts
    from socp import Socp
    import data2
    set_cuts(4)
    g, nl = grid(case)
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.setParam("Threads", 1); env.start()
    S = Socp(g, nl, n_cuts=4)
    from conelayer import ConeLayer
    _W["L2"] = ConeLayer(S.prob, [S.p_ulo, S.p_uhi, S.p_Vr0, S.p_Vi0, S.p_V0sq, S.p_VVc, S.p_VVs],
                         [S.v["pg"], S.v["u"]], reg=1e-10)
    _W["L2_built"] = False
    _W.update(g=g, nl=nl, S=S, env=env, inst=data2.instances(g, case),
              need_f=1.0 + RESERVE,
              merit=np.argsort(g.c1 + g.c2*g.pmax + nl/np.maximum(g.pmax, 1e-9), kind="stable"))


def relax(i, V=None):
    """uncut relaxation at flat V (or at a given/learned V) -> (u_hat, J_R)."""
    if isinstance(i, tuple):
        i, V = i
    g, S = _W["g"], _W["S"]; p, q = _W["inst"][i]
    vr, vi = (np.ones(g.n_bus), np.zeros(g.n_bus)) if V is None else V
    r = S.solve(p, q, vr, vi, rho=1e6)
    return i, (None if r is None else (r["u"], float(r["cost"])))


def _project(i, uhat, A, b):
    import gurobipy as gp
    from gurobipy import GRB
    g = _W["g"]; G = g.n_gen; p = _W["inst"][i][0]
    m = gp.Model(env=_W["env"])
    u = m.addVars(G, vtype=GRB.BINARY)
    m.addConstr(gp.quicksum(float(g.pmax[k])*u[k] for k in range(G)) >= _W["need_f"]*float(p.sum()))
    obj = gp.quicksum((1 - 2*float(uhat[k]))*u[k] for k in range(G))       # = |u - uhat|_1 + const
    if A is not None:
        s = m.addVars(len(b), lb=0)
        for r in range(len(b)):
            m.addConstr(gp.quicksum(float(A[r, k])*u[k] for k in range(G)) <= float(b[r]) + s[r])
        obj = obj + 1e3*gp.quicksum(s[r] for r in range(len(b)))
    m.setObjective(obj, GRB.MINIMIZE); m.optimize()
    return np.array([round(u[k].X) for k in range(G)], float)


def _price(i, u):
    g, S = _W["g"], _W["S"]; p, q = _W["inst"][i]
    if float(g.pmax @ u) < _W["need_f"]*float(p.sum()):
        return None
    pr = S.price(p, q, u)
    return None if (pr is None or pr["slack"] >= 1e-4) else (float(pr["cost"]), pr["pg"], pr["vr"], pr["vi"])


def deploy(i, uhat, A=None, b=None):
    """-> (u, cost, pg) with upward repair in merit order; None if impossible."""
    u = _project(i, uhat, A, b)
    for _ in range(_W["g"].n_gen + 1):
        r = _price(i, u)
        if r is not None:
            return u, r[0], r[1], r[2], r[3]
        off = [k for k in _W["merit"] if u[k] < 0.5]
        if not off:
            return None
        u[off[0]] = 1.0
    return None


def _rank_flips(i, u, pg, vr, vi):
    """EXACT conic layer: d(cost)/d(u) at the priced commitment, from one KKT solve at the
    converged linearisation point of the pricing. Ranks units by the predicted cost change
    of flipping them (most negative first); only these are priced exactly."""
    from framework import v_params
    g, S, L2 = _W["g"], _W["S"], _W["L2"]; p, q = _W["inst"][i]
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
    VP = v_params(vr, vi, fb, tb)
    S.solve(p, q, vr, vi, rho=1e6, u_lo=u, u_hi=u)
    if not _W["L2_built"]:
        L2.build_structure([u, u] + VP); _W["sl"] = L2.var_slices(); _W["L2_built"] = True
    x, st = L2.forward([u, u] + VP, tol=1e-10)
    if x is None:
        return list(np.random.default_rng(i).permutation(len(u)))
    dc = np.zeros(x.size)
    dc[_W["sl"][S.v["pg"].id]] = 2*g.c2*pg + g.c1
    dc[_W["sl"][S.v["u"].id]] = _W["nl"]
    try:
        gr = L2.vjp(st, dc); du = gr[0] + gr[1]
    except RuntimeError:
        return list(np.random.default_rng(i).permutation(len(u)))
    pred = du*(1.0 - 2.0*u)                       # flip on: +du ; flip off: -du
    need = _W["need_f"]*float(p.sum()); cap = float(g.pmax @ u)
    ok = [k for k in np.argsort(pred) if u[k] < 0.5 or cap - g.pmax[k] >= need]   # reserve-feasible flips
    return ok


def improve(i, u, c0, pg, vr, vi, k=6, rounds=8):
    """Exact-layer-guided local search: rank single flips by the conic layer's d cost/d u,
    price the top-k, accept the first improvement, re-rank; stop when none improves."""
    u = u.copy()
    for _ in range(rounds):
        moved = False
        for kk in _rank_flips(i, u, pg, vr, vi)[:k]:
            w = u.copy(); w[kk] = 1.0 - w[kk]
            r = _price(i, w)
            if r is not None and r[0] < c0 - 1e-9:
                u, c0, pg, vr, vi = w, r[0], r[1], r[2], r[3]; moved = True; break
        if not moved:
            break
    return u, c0, pg


def down_pass(i, u, c0, pg, order, ndown=25):
    """old pipeline's downward repair: try switching off up to ndown committed units in
    the given priority order (reverse confidence), keep each that lowers the AC price."""
    g = _W["g"]; need = _W["need_f"]*float(_W["inst"][i][0].sum()); u = u.copy()
    for k in [t for t in order if u[t] > 0.5][:ndown]:
        w = u.copy(); w[k] = 0.0
        if float(g.pmax @ w) < need: continue
        r = _price(i, w)
        if r is not None and r[0] < c0:
            u, c0, pg = w, r[0], r[1]
    return u, c0, pg


def final2(args):
    """variants of the post-projection search, for the diagnostic."""
    i, uhat, A, b, mode = args
    d = deploy(i, uhat, A, b)
    if d is None: return i, None
    u, c, pg, vr, vi = d
    if mode in ("down", "down+guided"):
        u, c, pg = down_pass(i, u, c, pg, list(np.argsort(uhat)))
        if mode == "down+guided":
            r = _price(i, u); u, c, pg = improve(i, u, c, pg, r[2], r[3], k=15, rounds=40)
    elif mode == "guided15":
        u, c, pg = improve(i, u, c, pg, vr, vi, k=15, rounds=40)
    return i, dict(u=u, cost=c, cost_proj=d[1])


def final(args):
    """deployment used at test time: cut-constrained projection + guided local search."""
    i, uhat, A, b, k, rounds = args
    d = deploy(i, uhat, A, b)
    if d is None:
        return i, None
    u, c, pg = improve(i, *d, k=k, rounds=rounds)
    return i, dict(u=u, cost=c, pg=pg, cost_proj=d[1])


def work(args):
    """Deploy, then the CombOptNet signal: feasible single-flip descent among k_cand
    candidates, and feasibility of cumulative neighbours along the descent order."""
    i, uhat, A, b, k_cand, k_moves = args
    d = deploy(i, uhat, A, b)
    if d is None:
        return i, None
    u, c0, pg, vr, vi = d
    desc = np.zeros(len(u))
    cand = _rank_flips(i, u, pg, vr, vi)[:k_cand] if k_cand else []
    for k in cand:
        w = u.copy(); w[k] = 1.0 - w[k]
        r = _price(i, w)
        if r is not None and r[0] < c0:
            desc[k] = (w[k] - u[k])*(c0 - r[0])
    order = [k for k in np.argsort(-np.abs(desc)) if abs(desc[k]) > 0][:k_moves]
    neigh, w = [], u.copy()
    for k in order:
        w = w.copy(); w[k] = 1.0 - w[k]
        neigh.append((w.copy(), _price(i, w) is not None))
    return i, dict(u=u, cost=c0, pg=pg, desc=desc, order=order, neigh=neigh, n_priced=len(cand))


def price_ref(args):
    i, u = args
    r = _price(i, np.asarray(u, float))
    return i, (None if r is None else (r[0], r[1]))

"""F3 audit worker helpers. Every function delegates to the repo's own combopt_ac / selfsup code (unchanged);
the only addition is reading the Gurobi Work of the model(s) built inside the call, by wrapping acuc.build
(a plain Python function, looked up at call time by `from acuc import build` inside combopt_ac.miqcqp)."""
import time
import numpy as np
import combopt_ac as C

_M = []


def _patch():
    import acuc
    if getattr(acuc, "_f3_patched", False):
        return
    orig = acuc.build

    def build_rec(*a, **k):
        out = orig(*a, **k)
        _M.append(out[0])
        return out
    acuc.build = build_rec
    acuc._f3_patched = True


def _work():
    w = 0.0
    for m in _M:
        try:
            w += float(m.Work)
        except Exception:
            pass
    for m in _M:
        try:
            m.dispose()
        except Exception:
            pass
    _M.clear()
    return w


def miqcqp_w(args):
    """combopt_ac.miqcqp unchanged + Gurobi work of its model."""
    _patch(); _M.clear()
    i, R = C.miqcqp(args)
    w = _work()
    return i, (None if R is None else dict(R, work=w))


def deploy_strict(args):
    """miqcqp_w then the SAME strict pricing as 55 (proxy_deploy on the miqcqp commitment)."""
    i, R = miqcqp_w(args)
    if R is None:
        return i, None
    _, P = C.proxy_deploy((i, R["u"]))
    return i, dict(sol=R, op=P)


def proxy_polish(args):
    """A10: proxy commitment -> strict price (with upward repair) -> combopt_ac.improve (label-free local search)
    -> strict price of the result (proxy_deploy). Returns the final priced point + timings."""
    i, u0, k, rounds, swaps = args
    t0 = time.time()
    _, r = C.price_strict((i, np.asarray(u0, float)))
    if r is None:
        return i, None
    cost, pg, vr, vi, nrep, u1 = r
    u2, c2, pg2 = C.improve(i, np.asarray(u1, float), float(cost), pg, vr, vi, k=k, rounds=rounds, swaps=swaps)
    _, P = C.proxy_deploy((i, u2))
    if P is None:
        return i, None
    return i, dict(P, secs_total=time.time() - t0, n_moves=int(np.sum(np.asarray(u2) != np.asarray(u1))), cost_start=float(cost))


def teacher_method(args):
    """A11: the label generator of 60_selfteacher.py as a deployment method: own W MIQCQP (no cuts) + improve,
    and the two restorations at V; best under the strict evaluator."""
    import selfsup
    i, V, W, rho, k, rounds, swaps = args
    t0 = time.time()
    _patch(); _M.clear()
    _, SI = selfsup.deploy_improve((i, V, None, None, W, rho, k, rounds, swaps))
    w = _work()
    _, RC = selfsup.restore_candidates((i, V))
    cands = ([("W+improve", SI["u"])] if SI else []) + list(RC["cands"].items())
    best = None
    for nm, u in cands:
        _, r = C.price_strict((i, u))
        if r is not None and (best is None or r[0] < best[1]):
            best = (nm, r[0], r)
    if best is None:
        return i, None
    nm, cost, r = best
    return i, dict(cost=r[0], pg=r[1], u=r[5], repaired=r[4], winner=nm, secs=time.time() - t0, work=w,
                   fallback=bool(SI.get("fallback")) if SI else True)


def relax_flat(i):
    return C.relax(i)


def miqcqp_hard(args):
    """AUDIT VARIANT of combopt_ac.miqcqp: the per-unit fixings are imposed as HARD variable bounds (u_k = 1 / u_k = 0)
    instead of soft rows priced 1e6; optional extra rows (general rows, 2G columns) stay soft exactly as in miqcqp.
    Everything else (model, W, MIP gap, pricing with upward merit repair) is copied from combopt_ac.miqcqp.
    If the hard-fixed model yields no incumbent within W (e.g. reserve infeasible under the fixings), falls back to
    the ORIGINAL combopt_ac.miqcqp with all fixings as soft rows (flag hard_fallback)."""
    from gurobipy import GRB
    import gurobipy as gp
    import acuc
    i, V, on, off, A, b, work, rho, Asoft, bsoft = args
    g = C._W["g"]; p, q = C._W["inst"][i]; t0 = time.time()
    vr, vi = (np.ones(g.n_bus), np.zeros(g.n_bus)) if V is None else V
    m, X, cost = acuc.build(g, C._W["nl"], p, q, GRB.BINARY, exact=False, rho=rho, Vr0=vr, Vi0=vi, threads=1)
    _M.clear()
    for k in on: X["u"][k].LB = 1.0
    for k in off: X["u"][k].UB = 0.0
    if A is not None:
        s = m.addVars(len(b), lb=0)
        for r in range(len(b)):
            lhs = (gp.quicksum(float(A[r, k])*X["pg"][k] for k in range(g.n_gen))
                   + gp.quicksum(float(A[r, g.n_gen + k])*X["u"][k] for k in range(g.n_gen)))
            m.addConstr(lhs <= float(b[r]) + s[r])
        m.setObjective(m.getObjective() + 1e6*gp.quicksum(s[r] for r in range(len(b))))
    m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
    m.optimize()
    w = float(m.Work)
    if m.SolCount == 0:
        m.dispose()
        _, R = miqcqp_w((i, V, Asoft, bsoft, work, rho))
        if R is None:
            return i, None
        return i, dict(R, work=R["work"] + w, hard_fallback=True, secs=time.time() - t0)
    u = np.array([round(X["u"][k].X) for k in range(g.n_gen)], float)
    mg = float(m.MIPGap); t_mip = time.time() - t0
    m.dispose()
    for _ in range(g.n_gen + 1):
        r = C._price(i, u)
        if r is not None:
            return i, dict(u=u, cost=r[0], pg=r[1], vr=r[2], vi=r[3], secs=time.time()-t0, t_mip=t_mip, mipgap=mg,
                           fallback=False, work=w, hard_fallback=False)
        offl = [k for k in C._W["merit"] if u[k] < 0.5]
        if not offl: return i, None
        u[offl[0]] = 1.0
    return i, None


def deploy_strict_hard(args):
    i, R = miqcqp_hard(args)
    if R is None:
        return i, None
    _, P = C.proxy_deploy((i, R["u"]))
    return i, dict(sol=R, op=P)

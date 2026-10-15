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


def init(case, fresh=False, inst=None):
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
    _W.update(g=g, nl=nl, S=S, env=env,
              inst=inst if inst is not None else (data2.all_instances(g, case) if fresh else data2.instances(g, case)),
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


def _pred(i, u, pg, vr, vi):
    """exact-layer predicted cost change of flipping each unit (None if the layer fails)."""
    from framework import v_params
    g, S, L2 = _W["g"], _W["S"], _W["L2"]; p, q = _W["inst"][i]
    VP = v_params(vr, vi, g.f_bus.astype(int), g.t_bus.astype(int))
    S.solve(p, q, vr, vi, rho=1e6, u_lo=u, u_hi=u)
    if not _W["L2_built"]:
        L2.build_structure([u, u] + VP); _W["sl"] = L2.var_slices(); _W["L2_built"] = True
    x, st = L2.forward([u, u] + VP, tol=1e-10)
    if x is None: return None
    dc = np.zeros(x.size); dc[_W["sl"][S.v["pg"].id]] = 2*g.c2*pg + g.c1; dc[_W["sl"][S.v["u"].id]] = _W["nl"]
    try:
        gr = L2.vjp(st, dc)
    except RuntimeError:
        return None
    return (gr[0] + gr[1])*(1.0 - 2.0*u)


def improve(i, u, c0, pg, vr, vi, k=6, rounds=8, swaps=4):
    """Exact-layer-guided local search: rank single flips by the conic layer's d cost/d u,
    price the top-k, accept the first improvement, re-rank. When no single flip improves,
    try SWAPS (one unit off + one unit on, reserve-feasible), ranked by the summed predicted
    change, top swaps x swaps priced. Stop when neither improves."""
    u = u.copy(); g = _W["g"]; need = _W["need_f"]*float(_W["inst"][i][0].sum())
    for _ in range(rounds):
        moved = False
        for kk in _rank_flips(i, u, pg, vr, vi)[:k]:
            w = u.copy(); w[kk] = 1.0 - w[kk]
            r = _price(i, w)
            if r is not None and r[0] < c0 - 1e-9:
                u, c0, pg, vr, vi = w, r[0], r[1], r[2], r[3]; moved = True; break
        if not moved and swaps:
            pr = _pred(i, u, pg, vr, vi)
            if pr is not None:
                offs = [j for j in np.argsort(pr) if u[j] > 0.5][:swaps]; ons = [j for j in np.argsort(pr) if u[j] < 0.5][:swaps]
                pairs = sorted([(pr[a_] + pr[b_], a_, b_) for a_ in offs for b_ in ons])
                for _, a_, b_ in pairs:
                    w = u.copy(); w[a_] = 0.0; w[b_] = 1.0
                    if float(g.pmax @ w) < need: continue
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


def old_restore(i, uf, thr, oname, ndown=40):
    """the audit pipeline's restoration: round at thr -> reserve top-up in order (merit or
    confidence) -> upward repair -> ndown downward trials in the mirrored order."""
    g = _W["g"]; pd_ = _W["inst"][i][0]; uf = np.asarray(uf, float)
    z = (uf > thr).astype(float)
    order = _W["merit"] if oname == "merit" else np.argsort(-uf, kind="stable")
    need = _W["need_f"]*float(pd_.sum())
    for k in order:
        if float(g.pmax @ z) >= need: break
        z[k] = 1.0
    best = None
    for _ in range(g.n_gen + 1):
        r = _price(i, z)
        if r is not None:
            best = (z.copy(), r[0], r[1]); break
        off = [k for k in order if z[k] < 0.5]
        if not off: return None
        z[off[0]] = 1.0
    if best is None: return None
    z, cost, pg = best
    for k in [t for t in order[::-1] if z[t] > 0.5][:ndown]:
        w = z.copy(); w[k] = 0.0
        if float(g.pmax @ w) < need: continue
        r = _price(i, w)
        if r is not None and r[0] < cost:
            z, cost, pg = w, r[0], r[1]
    return z, cost, pg


def final_best(args):
    """deployment: candidates = (cut projection + guided search), (merit restoration thr 0.8),
    (confidence restoration thr 0.5); keep the cheapest PRICED one (label-free)."""
    i, uhat, A, b, k, rounds = args
    cands = {}
    d = deploy(i, uhat, A, b)
    if d is not None:
        u, c, pg = improve(i, *d, k=k, rounds=rounds, swaps=0); cands["proj+search"] = (u, c, pg)
    for name, thr, o in (("merit0.8", 0.8, "merit"), ("conf0.5", 0.5, "conf")):
        r = old_restore(i, uhat, thr, o)
        if r is not None: cands[name] = r
    if not cands: return i, None
    name = min(cands, key=lambda n: cands[n][1]); u, c, pg = cands[name]
    return i, dict(u=u, cost=c, pg=pg, winner=name, costs={n: v[1] for n, v in cands.items()})


def target_old(args):
    """self-supervised training target: cheaper of the two fast label-free restorations."""
    i, uhat = args
    best = None
    for thr, o in ((0.8, "merit"), (0.5, "conf")):
        r = old_restore(i, uhat, thr, o)
        if r is not None and (best is None or r[1] < best[1]): best = r
    return i, (None if best is None else best[0])


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


def proj_only(args):
    """training step: cut-constrained projection only (no pricing)."""
    i, uhat, A, b = args
    return i, _project(i, uhat, A, b)


def miqcqp(args):
    """deployment: ONE time-limited MIQCQP of the QCAC UC linearised at V (learned or flat),
    with optional learned cut rows A u <= b (hard; priced slack fallback), then AC pricing
    of the commitment (upward merit repair if pricing fails)."""
    import time
    from gurobipy import GRB
    import gurobipy as gp
    from acuc import build
    i, V, A, b, work, rho = args
    g = _W["g"]; p, q = _W["inst"][i]; t0 = time.time()
    vr, vi = (np.ones(g.n_bus), np.zeros(g.n_bus)) if V is None else V
    m, X, cost = build(g, _W["nl"], p, q, GRB.BINARY, exact=False, rho=rho, Vr0=vr, Vi0=vi, threads=1)
    if A is not None:
        s = m.addVars(len(b), lb=0)
        for r in range(len(b)):
            # BUG FIX 2026-10-10: rows may be over u only (G columns) OR over [pg; u] (2G columns, the general/integer
            # cut categories). Before, 2G-column rows were read as u-only: their pg coefficients were applied to u.
            if A.shape[1] == 2*g.n_gen:
                lhs = (gp.quicksum(float(A[r, k])*X["pg"][k] for k in range(g.n_gen))
                       + gp.quicksum(float(A[r, g.n_gen + k])*X["u"][k] for k in range(g.n_gen)))
            else:
                assert A.shape[1] == g.n_gen, A.shape
                lhs = gp.quicksum(float(A[r, k])*X["u"][k] for k in range(g.n_gen))
            m.addConstr(lhs <= float(b[r]) + s[r])
        m.setObjective(m.getObjective() + 1e6*gp.quicksum(s[r] for r in range(len(b))))
    m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
    m.optimize()
    if m.SolCount == 0:
        # no incumbent within the work limit: label-free fallback (same for every arm)
        uh = relax((i, V))[1]
        best = None
        if uh is not None:
            for thr, o in ((0.8, "merit"), (0.5, "conf")):
                r = old_restore(i, uh[0], thr, o)
                if r is not None and (best is None or r[1] < best[1]): best = r
        if best is None:
            return i, None
        r = _price(i, best[0])
        return i, dict(u=best[0], cost=best[1], pg=best[2], vr=r[2] if r else vr, vi=r[3] if r else vi,
                       secs=time.time()-t0, t_mip=time.time()-t0, mipgap=np.nan, fallback=True)
    u = np.array([round(X["u"][k].X) for k in range(g.n_gen)], float)
    t_mip = time.time() - t0
    for _ in range(g.n_gen + 1):
        r = _price(i, u)
        if r is not None:
            return i, dict(u=u, cost=r[0], pg=r[1], vr=r[2], vi=r[3], secs=time.time()-t0, t_mip=t_mip, mipgap=float(m.MIPGap), fallback=False)
        off = [k for k in _W["merit"] if u[k] < 0.5]
        if not off: return i, None
        u[off[0]] = 1.0
    return i, None


def miqcqp_fix(args):
    """time-limited MIQCQP at V with learned FIXING cuts: u_k = 1 for k in on_set, u_k = 0
    for k in off_set (the simplest cuts; Xavier et al.'s affine subspace)."""
    i, V, on_set, off_set, work, rho = args
    G = _W["g"].n_gen
    rows, rhs = [], []
    for k in on_set:  e = np.zeros(G); e[k] = -1.0; rows.append(e); rhs.append(-1.0)   # -u_k <= -1
    for k in off_set: e = np.zeros(G); e[k] = 1.0; rows.append(e); rhs.append(0.0)     #  u_k <= 0
    A = np.array(rows) if rows else None; b = np.array(rhs) if rows else None
    return miqcqp((i, V, A, b, work, rho))


def miqcqp_fix_grad(args):
    """deploy with fixing cuts, then the EXACT conic layer's d(cost)/d(u) at the deployed,
    AC-priced commitment (converged linearisation point of the pricing)."""
    from framework import v_params
    i, V, on_set, off_set, work, rho = args
    _, R = miqcqp_fix((i, V, on_set, off_set, work, rho))
    if R is None: return i, None
    g, S, L2 = _W["g"], _W["S"], _W["L2"]; p, q = _W["inst"][i]; u = R["u"]
    VP = v_params(R["vr"], R["vi"], g.f_bus.astype(int), g.t_bus.astype(int))
    S.solve(p, q, R["vr"], R["vi"], rho=1e6, u_lo=u, u_hi=u)
    if not _W["L2_built"]:
        L2.build_structure([u, u] + VP); _W["sl"] = L2.var_slices(); _W["L2_built"] = True
    x, st = L2.forward([u, u] + VP, tol=1e-10)
    du = None
    if x is not None:
        dc = np.zeros(x.size); dc[_W["sl"][S.v["pg"].id]] = 2*g.c2*R["pg"] + g.c1; dc[_W["sl"][S.v["u"].id]] = _W["nl"]
        try:
            gr = L2.vjp(st, dc); du = gr[0] + gr[1]
        except RuntimeError:
            du = None
    R = dict(R); R["du"] = du
    return i, R


def miqcqp_warm(args):
    """cut-guided warm start: stage 1 = MIQCQP WITH learned cuts (work w1); stage 2 = MIQCQP
    WITHOUT cuts (work w2) started from stage 1's commitment; return the cheaper priced one.
    Total work w1 + w2 = the no-cut arm's budget."""
    import time
    from gurobipy import GRB
    import gurobipy as gp
    from acuc import build
    i, V, A, b, w1, w2, rho = args
    t0 = time.time()
    _, r1 = miqcqp((i, V, A, b, w1, rho))
    g = _W["g"]; p, q = _W["inst"][i]
    vr, vi = (np.ones(g.n_bus), np.zeros(g.n_bus)) if V is None else V
    m, X, cost = build(g, _W["nl"], p, q, GRB.BINARY, exact=False, rho=rho, Vr0=vr, Vi0=vi, threads=1)
    m.Params.TimeLimit = 1e6; m.Params.WorkLimit = w2; m.Params.MIPGap = 1e-4
    if r1 is not None:
        for k in range(g.n_gen): X["u"][k].Start = float(r1["u"][k])
    m.optimize()
    cands = [] if r1 is None else [r1]
    if m.SolCount:
        u = np.array([round(X["u"][k].X) for k in range(g.n_gen)], float)
        pr = _price(i, u)
        if pr is not None:
            cands.append(dict(u=u, cost=pr[0], pg=pr[1], vr=pr[2], vi=pr[3], fallback=False))
    if not cands: return i, None
    best = min(cands, key=lambda d: d["cost"]); best = dict(best); best["secs"] = time.time() - t0
    return i, best


def price_ref(args):
    i, u = args
    r = _price(i, np.asarray(u, float))
    return i, (None if r is None else (r[0], r[1]))


def relax_t(i):
    """relax() plus its wall time (the safeguard needs this solve, so it is timed)."""
    import time
    t0 = time.time(); _, r = relax(i)
    return i, (None if r is None else (r[0], r[1], time.time() - t0))


def _fallback_or_price(i, u, V, t0, extra):
    """shared ending: price u with upward merit repair, or the label-free fallback when no
    commitment was found (identical to miqcqp's, so every arm ends the same way)."""
    import time
    g = _W["g"]
    if u is None:
        uh = relax((i, V))[1]
        best = None
        if uh is not None:
            for thr, o in ((0.8, "merit"), (0.5, "conf")):
                r = old_restore(i, uh[0], thr, o)
                if r is not None and (best is None or r[1] < best[1]): best = r
        if best is None:
            return i, None
        return i, dict(u=best[0], cost=best[1], pg=best[2], secs=time.time()-t0, fallback=True, **extra)
    u = u.copy()
    for _ in range(g.n_gen + 1):
        r = _price(i, u)
        if r is not None:
            return i, dict(u=u, cost=r[0], pg=r[1], secs=time.time()-t0, fallback=False, **extra)
        off = [k for k in _W["merit"] if u[k] < 0.5]
        if not off: return i, None
        u[off[0]] = 1.0
    return i, None


def seq_budget(args):
    """BASELINE: sequential QCAC (Constante-Flores & Li) exactly as the reference -- flat
    start, rho0=1e3, x4 per iteration, stop at slack <= 1e-3, MIP gap 1e-3, warm start from
    the full previous iterate -- but with a TOTAL Gurobi work budget shared by all
    iterations. Returns the last iterate's commitment, priced like every other arm."""
    import time
    from gurobipy import GRB
    from acuc import build
    i, work = args
    g, nl = _W["g"], _W["nl"]; p, q = _W["inst"][i]; n, L = g.n_bus, g.n_branch
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
    t0 = time.time(); Vr, Vi = np.ones(n), np.zeros(n); rho = 1e3
    prev, last_u, left, it, slack = None, None, float(work), 0, np.nan
    keys = ("vr", "vi", "pg", "qg", "u", "cii", "cij", "sij", "wv", "zv")
    while left > 1e-3 and it < 30:
        m, V, cost = build(g, nl, p, q, GRB.BINARY, exact=False, rho=rho, Vr0=Vr, Vi0=Vi,
                           reserve=0.10, threads=1)
        m.Params.TimeLimit = 1e6; m.Params.MIPGap = 1e-3; m.Params.WorkLimit = left
        if prev is not None:
            pvr, pvi = prev["vr"], prev["vi"]
            lin_ii = 2*Vr*pvr + 2*Vi*pvi - Vr**2 - Vi**2
            lc = (Vr[tb]*pvr[fb] + Vr[fb]*pvr[tb] + Vi[tb]*pvi[fb] + Vi[fb]*pvi[tb]
                  - Vr[fb]*Vr[tb] - Vi[fb]*Vi[tb])
            ls = (Vi[tb]*pvr[fb] - Vi[fb]*pvr[tb] - Vr[tb]*pvi[fb] + Vr[fb]*pvi[tb]
                  - Vr[fb]*Vi[tb] + Vr[tb]*Vi[fb])
            xis = np.r_[np.abs(prev["cii"]-lin_ii), np.abs(prev["cij"]-lc), np.abs(prev["sij"]-ls)]
            for key in keys:
                for k, v in enumerate(prev[key]):
                    V[key][k].Start = float(v)
            for k, v in enumerate(xis):
                V["xi"][k].Start = float(v)
        m.optimize(); left -= float(m.Work); it += 1
        if m.SolCount == 0:
            break
        slack = float(sum(V["xi"][k].X for k in range(n + 2*L)))
        last_u = np.array([round(V["u"][k].X) for k in range(g.n_gen)], float)
        prev = {key: np.array([V[key][k].X for k in range(len(V[key]))]) for key in keys}
        if slack <= 1e-3:
            break
        Vr = prev["vr"].copy(); Vi = prev["vi"].copy(); rho = min(4.0*rho, 1e10)
    return _fallback_or_price(i, last_u, None, t0, dict(iters=it, slack=slack))


def ref_tight(args):
    """REFERENCE for the fresh test set: sequential QCAC to convergence with a 10x tighter
    MIP gap (1e-4) and a larger per-iteration work cap, so the reference's own tolerance
    is below the differences being reported. Stores every iteration's MIP gap."""
    import time, copy
    from acuc import solve_qcac_iterative
    i, mip_gap, tl = args
    p, q = _W["inst"][i]; t0 = time.time()
    try:
        r = solve_qcac_iterative(copy.deepcopy(_W["g"]), _W["nl"], p, q, reserve=0.10,
                                 threads=1, mip_gap=mip_gap, tl=tl)
    except Exception as e:
        return i, dict(err=str(e)[:200], converged=False)
    if r is None:
        return i, dict(err="no solution", converged=False, secs=time.time()-t0)
    return i, dict(u=np.asarray(r["u"]), cost=float(r["cost"]), secs=time.time()-t0,
                   iters=int(r.get("iters", -1)), slack=float(r.get("slack", np.nan)),
                   status=r.get("status", "?"), converged=bool(r.get("converged", False)),
                   mip_gaps=[h.get("mip_gap", np.nan) for h in r.get("hist", [])], hist=r.get("hist", []))


def ref_polish(args):
    """REFERENCE for the fresh test set: sequential QCAC to convergence with the ORIGINAL
    reference settings (MIP gap 1e-3, 120 work units per iteration), then the FINAL
    iteration's MIQCQP -- same linearisation point, same rho -- re-solved to MIP gap gap2
    (work cap tl2), warm-started from the converged solution. The polished commitment is kept
    only if its QCAC objective is no worse. Stores both MIP gaps and the bound."""
    import time, copy
    from gurobipy import GRB
    from acuc import solve_qcac_iterative, build
    i, gap2, tl2 = args
    g, nl = _W["g"], _W["nl"]; p, q = _W["inst"][i]; t0 = time.time()
    try:
        r = solve_qcac_iterative(copy.deepcopy(g), nl, p, q, reserve=0.10, threads=1)
    except Exception as e:
        return i, dict(err=str(e)[:200], converged=False)
    if r is None:
        return i, dict(err="no solution", converged=False, secs=time.time()-t0)
    t1 = time.time()
    out = dict(u=np.asarray(r["u"]), cost=float(r["cost"]), iters=int(r.get("iters", -1)),
               slack=float(r.get("slack", np.nan)), status=r.get("status", "?"),
               converged=bool(r.get("converged", False)),
               mip_gaps=[h.get("mip_gap", np.nan) for h in r.get("hist", [])], gap_before=float(r["mip_gap"]))
    if out["converged"]:
        m, V, cost = build(g, nl, p, q, GRB.BINARY, exact=False, rho=r["lin_rho"], Vr0=r["lin_vr"],
                           Vi0=r["lin_vi"], reserve=0.10, threads=1)
        m.Params.TimeLimit = 1e6; m.Params.MIPGap = gap2; m.Params.WorkLimit = tl2
        for key, val in r["start"].items():
            for k, v in enumerate(val): V[key][k].Start = float(v)
        for k, v in enumerate(r["xi"]): V["xi"][k].Start = float(v)
        m.optimize()
        if m.SolCount and m.ObjVal <= r["obj"] + 1e-9*abs(r["obj"]):
            slack = float(sum(V["xi"][k].X for k in range(len(r["xi"]))))
            if slack <= 1e-3:          # (audit 2026-10-11) a polish that breaks convergence is discarded: keep the converged solution
                out.update(u=np.array([round(V["u"][k].X) for k in range(g.n_gen)]), cost=float(cost.getValue()), slack=slack)
            else:
                out.update(polish_rejected_slack=slack)
        out.update(gap_after=float(m.MIPGap) if m.SolCount else np.nan, bound=float(m.ObjBound),
                   changed=bool(np.any(out["u"] != np.asarray(r["u"]))), secs_polish=time.time()-t1)
        out["mip_gaps"] = out["mip_gaps"] + [out["gap_after"]]
    out["secs"] = time.time() - t0
    return i, out


def oracle_fix(args):
    """HEADROOM CHECK: the same fixing cuts at the ORACLE linearisation point -- the AC
    voltages of the converged reference commitment (from pricing it). No method can use this;
    it bounds how much ANY learned V could gain over flat V."""
    i, u_ref, on_set, off_set, work, rho = args
    r = _price(i, np.asarray(u_ref, float))
    if r is None:
        return i, None
    _, R = miqcqp_fix((i, (r[2], r[3]), on_set, off_set, work, rho))
    return i, (None if R is None else dict(R, jref=r[0], vref=(r[2], r[3])))


def _phys(i, r, res_tol=1e-3, load_tol=1.001):
    """independent physics check of a priced point: P/Q balance rebuilt from the raw voltages
    (acuc.verify) <= res_tol pu and every line <= load_tol x rating."""
    from acuc import verify
    g = _W["g"]; p, q = _W["inst"][i]
    v = verify(g, _W["nl"], p, q, dict(vr=r["vr"], vi=r["vi"], pg=r["pg"], qg=r["qg"], u=r["u"]))
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int); vr, vi = r["vr"], r["vi"]
    cii = vr**2 + vi**2; cij = vr[fb]*vr[tb] + vi[fb]*vi[tb]; sij = vr[fb]*vi[tb] - vr[tb]*vi[fb]
    gg, bb, bsh, T, tm2 = g.g, g.b, g.b_fr, g.tap, g.tm2
    pf = gg/tm2*cii[fb] - gg/T*cij + bb/T*sij; qf = -(bb+bsh)/tm2*cii[fb] + gg/T*sij + bb/T*cij
    pt = gg*cii[tb] - gg/T*cij - bb/T*sij;     qt = -(bb+bsh)*cii[tb] - gg/T*sij + bb/T*cij
    ld = float(np.max(np.maximum(np.hypot(pf, qf), np.hypot(pt, qt)) / g.rate))
    return max(v["p_balance_max_resid"], v["q_balance_max_resid"]) <= res_tol and ld <= load_tol


def price_strict(args):
    """STRICT evaluator (2026-10-09): QCAC re-linearisation with u fixed, iterated until the point
    passes the physics check; upward merit repair otherwise. -> (cost, pg, vr, vi, n_repaired)."""
    i, u = args
    g, S = _W["g"], _W["S"]; p, q = _W["inst"][i]; u = np.asarray(u, float).copy()
    for nrep in range(g.n_gen + 1):
        if float(g.pmax @ u) >= _W["need_f"]*float(p.sum()):
            r = S.price(p, q, u, max_iter=200, check=lambda r_: _phys(i, r_))   # stops at the first passing iterate
            if r is not None and r["slack"] <= 1e-4 and _phys(i, r):
                return i, (float(r["cost"]), r["pg"], r["vr"], r["vi"], nrep, u)
        off = [k for k in _W["merit"] if u[k] < 0.5]
        if not off: break
        u[off[0]] = 1.0
    return i, None


def proxy_deploy(args):
    """supervised NN proxy deployment: rounded prediction -> strict pricing with upward merit repair
    (reserve top-up included), timed."""
    import time
    i, u0 = args
    t0 = time.time(); _, r = price_strict((i, u0))
    return i, (None if r is None else dict(cost=r[0], pg=r[1], u=r[5], repaired=r[4], secs=time.time() - t0))


def vdescent(args):
    """LABEL-FREE V by the EXACT conic layer (2026-10-09). Loss = relaxed QCAC objective at V
    (generation cost + rho * linearisation slack, u in [0,1]); its total derivative w.r.t. V comes from
    the ConeLayer VJP (implicit differentiation of the KKT) chained through v_params. Adam on (Vr, Vi);
    the reference bus angle stays 0. Returns V at the requested steps and the loss trajectory."""
    from framework import v_params, v_chain
    i, steps, lr, rho, keep = args[:5]
    g, S, L2 = _W["g"], _W["S"], _W["L2"]; p, q = _W["inst"][i]; n, G = g.n_bus, g.n_gen
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
    Vr, Vi = np.ones(n), np.zeros(n); m1 = np.zeros(2*n); m2 = np.zeros(2*n); b1, b2 = 0.9, 0.999
    # optional 6th arg (on_set, off_set): the LEARNED FIXING CUTS -> descend on the cut-constrained relaxation
    lo, hi = np.zeros(G), np.ones(G)
    if len(args) > 5 and args[5] is not None:
        on, off = args[5]; lo[list(on)] = 1.0; hi[list(off)] = 0.0
    out, hist = {}, []
    for t in range(steps + 1):
        if t in keep:
            out[t] = (Vr.copy(), Vi.copy())
        if t == steps:
            break
        r = S.solve(p, q, Vr, Vi, rho=rho, u_lo=lo, u_hi=hi)     # sets every other parameter (load, rho)
        if r is None:
            break
        VP = v_params(Vr, Vi, fb, tb)
        if not _W["L2_built"]:
            L2.build_structure([lo, hi] + VP); _W["sl"] = L2.var_slices(); _W["L2_built"] = True
        x, st = L2.forward([lo, hi] + VP, tol=1e-10)
        if x is None:
            break
        sl = _W["sl"]
        if "sl_xi" not in _W:                                    # xi is not a layer output: read its offset
            import cvxpy as cp
            _, _, inv = S.prob.get_problem_data(cp.CLARABEL)
            off = next(el.var_offsets for el in inv if getattr(el, "var_offsets", None))
            o = int(off[S.v["xi"].id]); _W["sl_xi"] = slice(o, o + int(np.prod(S.v["xi"].shape)))
        sx = _W["sl_xi"]; pg = x[sl[S.v["pg"].id]]
        dc = np.zeros(x.size)
        dc[sl[S.v["pg"].id]] = 2*g.c2*pg + g.c1
        dc[sl[S.v["u"].id]] = _W["nl"]
        dc[sx] = rho
        loss = float(g.c2 @ pg**2 + g.c1 @ pg + _W["nl"] @ x[sl[S.v["u"].id]] + rho*x[sx].sum())
        hist.append(loss)
        try:
            gr = L2.vjp(st, dc)
        except RuntimeError:
            break
        dVr, dVi = v_chain(gr[2:7], Vr, Vi, fb, tb)
        dVi[int(g.ref)] = 0.0
        gv = np.r_[dVr, dVi]
        m1 = b1*m1 + (1-b1)*gv; m2 = b2*m2 + (1-b2)*gv**2
        step = lr*(m1/(1-b1**(t+1)))/(np.sqrt(m2/(1-b2**(t+1))) + 1e-12)
        Vr = Vr - step[:n]; Vi = Vi - step[n:]
    return i, dict(V=out, loss=hist)


def vgrad(args):
    """ONE exact-layer gradient for V-net training (label-free): loss = cut-constrained relaxed QCAC
    objective at the given V (cost + rho * slack); returns (loss, dL/dVr, dL/dVi) via the ConeLayer VJP."""
    from framework import v_params, v_chain
    i, Vr, Vi, rho, cuts = args
    g, S, L2 = _W["g"], _W["S"], _W["L2"]; p, q = _W["inst"][i]; G = g.n_gen
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
    lo, hi = np.zeros(G), np.ones(G)
    if cuts is not None:
        on, off = cuts; lo[list(on)] = 1.0; hi[list(off)] = 0.0
    if S.solve(p, q, Vr, Vi, rho=rho, u_lo=lo, u_hi=hi) is None:
        return i, None
    VP = v_params(Vr, Vi, fb, tb)
    if not _W["L2_built"]:
        L2.build_structure([lo, hi] + VP); _W["sl"] = L2.var_slices(); _W["L2_built"] = True
    x, st = L2.forward([lo, hi] + VP, tol=1e-10)
    if x is None:
        return i, None
    sl = _W["sl"]
    if "sl_xi" not in _W:
        import cvxpy as cp
        _, _, inv = S.prob.get_problem_data(cp.CLARABEL)
        off_ = next(el.var_offsets for el in inv if getattr(el, "var_offsets", None))
        o = int(off_[S.v["xi"].id]); _W["sl_xi"] = slice(o, o + int(np.prod(S.v["xi"].shape)))
    sx = _W["sl_xi"]; pg = x[sl[S.v["pg"].id]]
    dc = np.zeros(x.size); dc[sl[S.v["pg"].id]] = 2*g.c2*pg + g.c1; dc[sl[S.v["u"].id]] = _W["nl"]; dc[sx] = rho
    loss = float(g.c2 @ pg**2 + g.c1 @ pg + _W["nl"] @ x[sl[S.v["u"].id]] + rho*x[sx].sum())
    try:
        gr = L2.vjp(st, dc)
    except RuntimeError:
        return i, None
    dVr, dVi = v_chain(gr[2:7], Vr, Vi, fb, tb); dVi[int(g.ref)] = 0.0
    return i, (loss, dVr, dVi)


def vgrad_env(args):
    """CORRECTED exact-layer gradient for V-net training (2026-10-10). The loss is the conic program's
    OPTIMAL VALUE (cut-constrained relaxed QCAC: cost + rho * slack), so its exact derivative w.r.t. the
    parameters is the envelope/Danskin sensitivity  1/2 x'dP x + dq'x + y'(dA x - db)  evaluated with the
    layer's own canonical solution x, duals y and per-parameter directions (ConeLayer structure). The VJP of
    vgrad() was wrong off flat V (degenerate KKT; FD corr 0.01-0.5 at trained V); this matched FD to 0.2%.
    Returns (loss, dL/dVr, dL/dVi) and, for the cuts, dL/du_lo and dL/du_hi (fixing sensitivities)."""
    from framework import v_params, v_chain
    i, Vr, Vi, rho, cuts = args
    g, S, L2 = _W["g"], _W["S"], _W["L2"]; p, q = _W["inst"][i]; G = g.n_gen
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
    lo, hi = np.zeros(G), np.ones(G)
    if cuts is not None:
        on, off = cuts; lo[list(on)] = 1.0; hi[list(off)] = 0.0
    r = S.solve(p, q, Vr, Vi, rho=rho, u_lo=lo, u_hi=hi)
    if r is None:
        return i, None
    VP = v_params(Vr, Vi, fb, tb)
    if not _W["L2_built"]:
        L2.build_structure([lo, hi] + VP); _W["sl"] = L2.var_slices(); _W["L2_built"] = True
    x, st = L2.forward([lo, hi] + VP, tol=1e-10)
    if x is None:
        return i, None
    y = st["y"]
    env = [np.array([0.5*x @ (D["dP"] @ x) + D["dq"] @ x + y @ (D["dA"] @ x - D["db"]) for D in L2._struct["dirs"][pi]])
           for pi in range(7)]
    dVr, dVi = v_chain(env[2:7], Vr, Vi, fb, tb); dVi[int(g.ref)] = 0.0
    loss = float(r["cost"] + rho*r["slack"])
    return i, (loss, dVr, dVi, env[0], env[1])


def usens_env(args):
    """EXACT per-unit commitment sensitivity at a priced commitment (2026-10-10): the pricing problem
    (u fixed: u_lo = u_hi = u) at the priced point's V; envelope gradient of its optimal value w.r.t. the
    bound parameters, dJ/du_k = dJ/du_lo_k + dJ/du_hi_k. Predicted change of flipping unit k: (1 - 2u_k) dJ/du_k."""
    from framework import v_params
    i, u, vr, vi = args
    g, S, L2 = _W["g"], _W["S"], _W["L2"]; p, q = _W["inst"][i]
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int); u = np.asarray(u, float)
    if S.solve(p, q, vr, vi, rho=1e6, u_lo=u, u_hi=u) is None:
        return i, None
    VP = v_params(vr, vi, fb, tb)
    if not _W["L2_built"]:
        L2.build_structure([u, u] + VP); _W["sl"] = L2.var_slices(); _W["L2_built"] = True
    x, st = L2.forward([u, u] + VP, tol=1e-10)
    if x is None:
        return i, None
    y = st["y"]
    du = [np.array([0.5*x @ (D["dP"] @ x) + D["dq"] @ x + y @ (D["dA"] @ x - D["db"]) for D in L2._struct["dirs"][pi]]) for pi in (0, 1)]
    return i, (du[0] + du[1])*(1.0 - 2.0*u)


def uncut_point(args):
    """uncut relaxation at V -> [pg; u] (anchor for the cut rows)."""
    i, Vr, Vi = args
    r = _W["S"].solve(*_W["inst"][i], Vr, Vi, rho=1e6)
    return i, (None if r is None else np.r_[r["pg"], r["u"]])


def surrogate(args):
    """SURROGATE deployment (2026-10-10): ONE convex solve = QCAC relaxation at V with the learned cut rows (A, b),
    then a fixed parameter-free decision: round u at 0.5 -> strict pricing (reserve top-up + upward merit repair).
    Returns the relaxed u (to measure integrality), the priced decision and how many units the repair added."""
    import time
    i, Vr, Vi, A, b = args
    t0 = time.time()
    r = _W["S"].solve(*_W["inst"][i], Vr, Vi, rho=1e6, A=A, b=b, cut_cap=0.0)
    if r is None:
        return i, None
    ur = r["u"]; u0 = (ur >= 0.5).astype(float)
    _, pr = price_strict((i, u0))
    return i, dict(u_relax=ur, integral=bool(np.all(np.minimum(ur, 1 - ur) < 1e-3)), u=None if pr is None else pr[5],
                   cost=None if pr is None else pr[0], pg=None if pr is None else pr[1], repaired=None if pr is None else pr[4],
                   secs=time.time() - t0)

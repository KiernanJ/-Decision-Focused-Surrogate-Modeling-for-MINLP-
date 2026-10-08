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
            m.addConstr(gp.quicksum(float(A[r, k])*X["u"][k] for k in range(g.n_gen)) <= float(b[r]) + s[r])
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

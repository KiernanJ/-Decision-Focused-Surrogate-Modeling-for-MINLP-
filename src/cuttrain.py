"""Pool worker for training the TWO-CATEGORY cuts through the smoothed exact layer (2026-10-10).
Lives in a module because macOS spawn pickles workers by module path."""
import os
import numpy as np

_W = {}


def init(case, gamma, k_rows=2):
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["QCAC_CASE"] = case
    import data2
    from pipeline import grid, RESERVE, set_cuts
    from socp import Socp
    from framework import DiffDeploy
    set_cuts(2*k_rows)
    g, nl = grid(case)
    S = Socp(g, nl, n_cuts=2*k_rows); S1 = Socp(g, nl, n_cuts=2*k_rows, u_smooth=gamma)
    _W.update(g=g, S=S, inst=data2.all_instances(g, case),
              DD=DiffDeploy(S, g, nl, reserve=RESERVE, tau=0.08, thr=0.5, rho=1e6, reg=1e-10, reg2=1e-10, S1=S1))


def anchor(args):
    """uncut relaxation at the learned V -> w_rlx = [pg; u] (constant anchor), and the flat-V cost (normaliser)."""
    i, Vr, Vi = args
    S, n = _W["S"], _W["g"].n_bus; p, q = _W["inst"][i]
    r = S.solve(p, q, Vr, Vi, rho=1e6); rf = S.solve(p, q, np.ones(n), np.zeros(n), rho=1e6)
    return i, (None if r is None or rf is None else (np.r_[r["pg"], r["u"]], float(rf["cost"])))


def step(args):
    i, Vr, Vi, A, b, scale = args
    p, q = _W["inst"][i]
    L, gA, gb, info = _W["DD"](p, q, Vr, Vi, A, b, scale)
    return i, (None if gA is None else (L, gA, gb))


def init2(case, gamma, k_rows=2, n_rows=None):
    """worker for the TIGHTENING objective: layer 1 only (smoothed relaxation with cuts), exact VJP.
    n_rows: total number of cut rows (default 2*k_rows: general + integer); integer-only runs pass n_rows=K."""
    n_rows = 2*k_rows if n_rows is None else n_rows
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["QCAC_CASE"] = case
    import data2
    from pipeline import grid, set_cuts
    from socp import Socp
    from conelayer import ConeLayer
    set_cuts(n_rows)
    g, nl = grid(case)
    S1 = Socp(g, nl, n_cuts=n_rows, u_smooth=gamma)
    VP = [S1.p_Vr0, S1.p_Vi0, S1.p_V0sq, S1.p_VVc, S1.p_VVs]
    _W.update(g=g, S1=S1, inst=data2.all_instances(g, case), built=False,
              L1=ConeLayer(S1.prob, [S1.p_A, S1.p_b] + VP, [S1.v["pg"], S1.v["u"]], reg=1e-10))


def track(args):
    """tracking loss ||u_relax(A,b) - u_own||^2 / G through the smoothed exact layer; returns (loss, dA, db).
    With want_grad False: loss only (for finite differences)."""
    from framework import v_params
    i, Vr, Vi, A, b, u_own, want_grad = args
    g, S1, L1 = _W["g"], _W["S1"], _W["L1"]; p, q = _W["inst"][i]; G = g.n_gen
    r = S1.solve(p, q, Vr, Vi, rho=1e6, A=A, b=b, cut_cap=0.0)
    if r is None:
        return i, None
    loss = float(((r["u"] - u_own)**2).sum()/G)
    if not want_grad:
        return i, (loss,)
    VP = v_params(Vr, Vi, g.f_bus.astype(int), g.t_bus.astype(int))
    if not _W["built"]:
        L1.build_structure([np.asarray(A, float), np.asarray(b, float)] + VP); _W["sl"] = L1.var_slices(); _W["built"] = True
    x, st = L1.forward([np.asarray(A, float), np.asarray(b, float)] + VP, tol=1e-10)
    if x is None:
        return i, None
    u = x[_W["sl"][S1.v["u"].id]]
    dx = np.zeros(x.size); dx[_W["sl"][S1.v["u"].id]] = 2.0*(u - u_own)/G
    try:
        gr = L1.vjp(st, dx)
    except RuntimeError:
        return i, None
    return i, (float(((u - u_own)**2).sum()/G), gr[0], gr[1])


def init3(case, gamma):
    """worker for PER-UNIT integer cuts u_k >= l_k, u_k <= h_k: the smoothed relaxation with the bounds as
    the layer's differentiable parameters (Socp p_ulo / p_uhi), exact VJP."""
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["QCAC_CASE"] = case
    import data2
    from pipeline import grid
    from socp import Socp
    from conelayer import ConeLayer
    g, nl = grid(case)
    S1 = Socp(g, nl, n_cuts=4, u_smooth=gamma)
    VP = [S1.p_Vr0, S1.p_Vi0, S1.p_V0sq, S1.p_VVc, S1.p_VVs]
    _W.update(g=g, S1=S1, inst=data2.all_instances(g, case), built3=False,
              L3=ConeLayer(S1.prob, [S1.p_ulo, S1.p_uhi] + VP, [S1.v["pg"], S1.v["u"]], reg=1e-10))


def track_bounds(args):
    """tracking loss ||u_relax(l, h) - u_own||^2 / G through the smoothed layer; returns (loss, d/dl, d/dh)."""
    from framework import v_params
    i, Vr, Vi, lo, hi, u_own, want_grad = args
    g, S1, L3 = _W["g"], _W["S1"], _W["L3"]; p, q = _W["inst"][i]; G = g.n_gen
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    r = S1.solve(p, q, Vr, Vi, rho=1e6, u_lo=lo, u_hi=hi)
    if r is None:
        return i, None
    if not want_grad:
        return i, (float(((r["u"] - u_own)**2).sum()/G),)
    VP = v_params(Vr, Vi, g.f_bus.astype(int), g.t_bus.astype(int))
    if not _W["built3"]:
        L3.build_structure([lo, hi] + VP); _W["sl3"] = L3.var_slices(); _W["built3"] = True
    x, st = L3.forward([lo, hi] + VP, tol=1e-10)
    if x is None:
        return i, None
    u = x[_W["sl3"][S1.v["u"].id]]
    dx = np.zeros(x.size); dx[_W["sl3"][S1.v["u"].id]] = 2.0*(u - u_own)/G
    try:
        gr = L3.vjp(st, dx)
    except RuntimeError:
        return i, None
    return i, (float(((u - u_own)**2).sum()/G), gr[0], gr[1])


def init4(case, gamma):
    """worker for per-unit integer cuts as SOFT cut rows (same as deployment: violation priced 1e6):
    rows 0..G-1:  -u_k <= -l_k   (active only where l_k > 0; else disabled A=0, b=1)
    rows G..2G-1:  u_k <=  h_k   (active only where h_k < 1)
    smoothed relaxation (gamma), exact VJP w.r.t. b."""
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["QCAC_CASE"] = case
    import data2
    from pipeline import grid, set_cuts
    from socp import Socp
    from conelayer import ConeLayer
    g, nl = grid(case); G = g.n_gen
    set_cuts(2*G)
    S1 = Socp(g, nl, n_cuts=2*G, cut_price=1e6, u_smooth=gamma)
    VP = [S1.p_Vr0, S1.p_Vi0, S1.p_V0sq, S1.p_VVc, S1.p_VVs]
    _W.update(g=g, S1=S1, inst=data2.all_instances(g, case), built4=False,
              L4=ConeLayer(S1.prob, [S1.p_A, S1.p_b] + VP, [S1.v["pg"], S1.v["u"]], reg=1e-10))


def unit_rows(lo, hi, G):
    A = np.zeros((2*G, 2*G)); b = np.ones(2*G)
    for k in range(G):
        if lo[k] > 0: A[k, G + k] = -1.0; b[k] = -lo[k]
        if hi[k] < 1: A[G + k, G + k] = 1.0; b[G + k] = hi[k]
    return A, b


def track_soft(args):
    """tracking loss through the soft-row layer; returns (loss, dL/dl, dL/dh) (zero where the row is disabled)."""
    from framework import v_params
    i, Vr, Vi, lo, hi, u_own, want_grad = args
    g, S1, L4 = _W["g"], _W["S1"], _W["L4"]; p, q = _W["inst"][i]; G = g.n_gen
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); A, b = unit_rows(lo, hi, G)
    r = S1.solve(p, q, Vr, Vi, rho=1e6, A=A, b=b, cut_cap=1e6, soft_fallback=False)
    if r is None:
        return i, None
    if not want_grad:
        return i, (float(((r["u"] - u_own)**2).sum()/G),)
    VP = v_params(Vr, Vi, g.f_bus.astype(int), g.t_bus.astype(int))
    if not _W["built4"]:
        L4.build_structure([A, b] + VP); _W["sl4"] = L4.var_slices(); _W["built4"] = True
    x, st = L4.forward([A, b] + VP, tol=1e-10)
    if x is None:
        return i, None
    u = x[_W["sl4"][S1.v["u"].id]]
    dx = np.zeros(x.size); dx[_W["sl4"][S1.v["u"].id]] = 2.0*(u - u_own)/G
    try:
        gr = L4.vjp(st, dx)
    except RuntimeError:
        return i, None
    gb = gr[1]; dl = np.where(lo > 0, -gb[:G], 0.0); dh = np.where(hi < 1, gb[G:], 0.0)
    return i, (float(((u - u_own)**2).sum()/G), dl, dh)


def init5(case, gamma):
    """worker for the COST-AWARE loss: DiffDeploy (layer 1 = smoothed relaxation with SOFT per-unit cut rows,
    soft rounding, layer 2 = pricing) with the per-unit integer cuts u_k >= l_k, u_k <= h_k."""
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["QCAC_CASE"] = case
    import data2
    from pipeline import grid, RESERVE, set_cuts
    from socp import Socp
    from framework import DiffDeploy
    g, nl = grid(case); G = g.n_gen
    set_cuts(2*G)
    S = Socp(g, nl, n_cuts=2*G, cut_price=1e6); S1 = Socp(g, nl, n_cuts=2*G, cut_price=1e6, u_smooth=gamma)
    _W.update(g=g, inst=data2.all_instances(g, case),
              DD5=DiffDeploy(S, g, nl, reserve=RESERVE, tau=0.08, thr=0.5, rho=1e6, reg=1e-10, reg2=1e-10, S1=S1, cut_cap=1e6))


def dd_unit(args):
    """priced-cost loss of the per-unit cuts; returns (loss, dL/dl, dL/dh) or (loss,) without gradient."""
    i, Vr, Vi, lo, hi, scale, want_grad = args
    G = _W["g"].n_gen; p, q = _W["inst"][i]
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); A, b = unit_rows(lo, hi, G)
    L, gA, gb, info = _W["DD5"](p, q, Vr, Vi, A, b, scale, want_grad=want_grad)
    if L is None:
        return i, None
    if not want_grad:
        return i, (L,)
    if gb is None:
        return i, None
    return i, (L, np.where(lo > 0, -gb[:G], 0.0), np.where(hi < 1, gb[G:], 0.0))


def track_value(args):
    """loss = ||u_relax - u_own||^2/G + beta * (relaxed optimal value)/scale, per-unit integer cuts as SOFT rows
    (init4 worker). Tracking gradient: exact VJP; value gradient: exact envelope w.r.t. the cut RHS b."""
    from framework import v_params
    i, Vr, Vi, lo, hi, u_own, scale, beta, want_grad = args
    g, S1, L4 = _W["g"], _W["S1"], _W["L4"]; p, q = _W["inst"][i]; G = g.n_gen
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); A, b = unit_rows(lo, hi, G)
    r = S1.solve(p, q, Vr, Vi, rho=1e6, A=A, b=b, cut_cap=1e6, soft_fallback=False)
    if r is None:
        return i, None
    val = float(S1.prob.value)/scale
    if not want_grad:
        return i, (float(((r["u"] - u_own)**2).sum()/G) + beta*val,)
    VP = v_params(Vr, Vi, g.f_bus.astype(int), g.t_bus.astype(int))
    if not _W["built4"]:
        L4.build_structure([A, b] + VP); _W["sl4"] = L4.var_slices(); _W["built4"] = True
    x, st = L4.forward([A, b] + VP, tol=1e-10)
    if x is None:
        return i, None
    u = x[_W["sl4"][S1.v["u"].id]]
    dx = np.zeros(x.size); dx[_W["sl4"][S1.v["u"].id]] = 2.0*(u - u_own)/G
    try:
        gr = L4.vjp(st, dx)
    except RuntimeError:
        return i, None
    y = st["y"]
    env_b = np.array([0.5*x @ (D["dP"] @ x) + D["dq"] @ x + y @ (D["dA"] @ x - D["db"]) for D in L4._struct["dirs"][1]])/scale
    gb = gr[1] + beta*env_b
    return i, (float(((u - u_own)**2).sum()/G) + beta*val, np.where(lo > 0, -gb[:G], 0.0), np.where(hi < 1, gb[G:], 0.0))


def init6(case, gamma, k_gen):
    """worker: smoothed relaxation with [per-unit integer cut rows (frozen) ; k_gen GENERAL rows (trainable)], all SOFT."""
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["QCAC_CASE"] = case
    import data2
    from pipeline import grid, set_cuts
    from socp import Socp
    from conelayer import ConeLayer
    g, nl = grid(case); G = g.n_gen
    set_cuts(2*G + k_gen)
    S1 = Socp(g, nl, n_cuts=2*G + k_gen, cut_price=1e6, u_smooth=gamma)
    VP = [S1.p_Vr0, S1.p_Vi0, S1.p_V0sq, S1.p_VVc, S1.p_VVs]
    _W.update(g=g, S1=S1, inst=data2.all_instances(g, case), built6=False, kgen=k_gen,
              L6=ConeLayer(S1.prob, [S1.p_A, S1.p_b] + VP, [S1.v["pg"], S1.v["u"]], reg=1e-10))


def _rows6(lo, hi, Ag, bg):
    G = _W["g"].n_gen; A_u, b_u = unit_rows(lo, hi, G)
    if Ag is None:
        Ag = np.zeros((_W["kgen"], 2*G)); bg = np.ones(_W["kgen"])
    return np.vstack([A_u, Ag]), np.r_[b_u, bg]


def anchor6(args):
    """relaxation with the integer cuts only (general rows disabled) at V -> [pg; u] (anchor of the general rows)."""
    i, Vr, Vi, lo, hi = args
    A, b = _rows6(lo, hi, None, None)
    r = _W["S1"].solve(*_W["inst"][i], Vr, Vi, rho=1e6, A=A, b=b, cut_cap=1e6, soft_fallback=False)
    return i, (None if r is None else np.r_[r["pg"], r["u"]])


def track_gen(args):
    """tracking loss ||u_relax - u_own||^2/G with integer cuts + general rows; returns (loss, dA_gen, db_gen)."""
    from framework import v_params
    i, Vr, Vi, lo, hi, Ag, bg, u_own, want_grad = args
    g, S1, L6 = _W["g"], _W["S1"], _W["L6"]; p, q = _W["inst"][i]; G = g.n_gen
    A, b = _rows6(lo, hi, Ag, bg)
    r = S1.solve(p, q, Vr, Vi, rho=1e6, A=A, b=b, cut_cap=1e6, soft_fallback=False)
    if r is None:
        return i, None
    if not want_grad:
        return i, (float(((r["u"] - u_own)**2).sum()/G),)
    VP = v_params(Vr, Vi, g.f_bus.astype(int), g.t_bus.astype(int))
    if not _W["built6"]:
        L6.build_structure([A, b] + VP); _W["sl6"] = L6.var_slices(); _W["built6"] = True
    x, st = L6.forward([A, b] + VP, tol=1e-10)
    if x is None:
        return i, None
    u = x[_W["sl6"][S1.v["u"].id]]
    dx = np.zeros(x.size); dx[_W["sl6"][S1.v["u"].id]] = 2.0*(u - u_own)/G
    try:
        gr = L6.vjp(st, dx)
    except RuntimeError:
        return i, None
    return i, (float(((u - u_own)**2).sum()/G), gr[0].reshape(A.shape)[2*G:], gr[1][2*G:])


def anchor6_t(args):
    """anchor6 + its wall time (audit 2026-10-11: the anchor solve is part of our deployment time)."""
    import time
    t0 = time.time(); i, w = anchor6(args)
    return i, (w, time.time() - t0)


# ---------------------------------------------------------------------------------------------------------------------
# (2026-10-12, Parth: option A) VALUE-BASED cut losses with the EXACT ENVELOPE gradient. The tracking losses above need a
# KKT solve that is singular on most AC instances (regularised fallback, gradient 40-85% off; audit 2026-10-12). Here:
#   loss = -(min(J, J_fix) - J_base) / (J_fix - J_base)        (fraction of the relaxation gap closed, capped)
#   J      = optimal value of the relaxation with the learned cut rows (soft rows priced 1e6; no smoothing: init with gamma=0)
#   J_base = the same relaxation without the rows being trained;  J_fix = the relaxation with u fixed to OUR OWN best-of answer
#   d J / d(params) = 1/2 x'dP x + dq'x + y'(dA x - db)  (Danskin; verified exact on a test SOCP), from ONE canonical forward
#   solve -- no KKT system. J >= J_fix means the rows cut off our own answer: no value reward (validity term handles it).
# Signatures match track_soft / track_gen so 46 / 52 can swap them in. Per-instance constants are cached in the worker.
def _env(L, st, x, pi, idx):
    """envelope derivative of the optimal value w.r.t. parameter pi, entries idx (flattened C-order)."""
    y = st["y"]; D_ = L._struct["dirs"][pi]
    return np.array([0.5*x @ (D_[j]["dP"] @ x) + D_[j]["dq"] @ x + y @ (D_[j]["dA"] @ x - D_[j]["db"]) for j in idx])


def value_soft(args):
    """per-unit integer cuts (init4 worker, gamma=0). -> (loss,) or (loss, dloss/dl, dloss/dh)."""
    from framework import v_params
    i, Vr, Vi, lo, hi, u_own, want_grad = args
    g, S1, L4 = _W["g"], _W["S1"], _W["L4"]; p, q = _W["inst"][i]; G = g.n_gen
    C = _W.setdefault("vcache", {})
    if i not in C:
        r0 = S1.solve(p, q, Vr, Vi, rho=1e6); J0 = None if r0 is None else float(S1.prob.value)
        uo = np.asarray(u_own, float); rf = S1.solve(p, q, Vr, Vi, rho=1e6, u_lo=uo, u_hi=uo); Jf = None if rf is None else float(S1.prob.value)
        C[i] = (J0, Jf)
    J0, Jf = C[i]
    if J0 is None or Jf is None:
        return i, None
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); A, b = unit_rows(lo, hi, G)
    r = S1.solve(p, q, Vr, Vi, rho=1e6, A=A, b=b, cut_cap=1e6, soft_fallback=False)
    if r is None:
        return i, None
    J = float(S1.prob.value); den = max(Jf - J0, 1e-9*abs(J0) + 1e-9); loss = -(min(J, Jf) - J0)/den
    if not want_grad:
        return i, (loss,)
    if J >= Jf:
        return i, (loss, np.zeros(G), np.zeros(G))
    VP = v_params(Vr, Vi, g.f_bus.astype(int), g.t_bus.astype(int))
    if not _W["built4"]:
        L4.build_structure([A, b] + VP); _W["sl4"] = L4.var_slices(); _W["built4"] = True
    x, st = L4.forward([A, b] + VP, tol=1e-10)
    if x is None:
        return i, None
    eb = _env(L4, st, x, 1, range(2*G))                               # dJ/db, rows 0..G-1: b_k = -l_k ; rows G..2G-1: b = h_k
    dJdl = np.where(lo > 0, -eb[:G], 0.0); dJdh = np.where(hi < 1, eb[G:], 0.0)
    return i, (loss, -dJdl/den, -dJdh/den)


def value_gen(args):
    """K general rows on top of frozen integer cuts (init6 worker, gamma=0). -> (loss,) or (loss, dloss/dA_gen (K x 2G), dloss/db_gen)."""
    from framework import v_params
    i, Vr, Vi, lo, hi, Ag, bg, u_own, want_grad = args
    g, S1, L6 = _W["g"], _W["S1"], _W["L6"]; p, q = _W["inst"][i]; G = g.n_gen; K = _W["kgen"]
    C = _W.setdefault("gcache", {})
    if i not in C:
        Ab, bb = _rows6(lo, hi, None, None)
        r0 = S1.solve(p, q, Vr, Vi, rho=1e6, A=Ab, b=bb, cut_cap=1e6, soft_fallback=False); J0 = None if r0 is None else float(S1.prob.value)
        uo = np.asarray(u_own, float); rf = S1.solve(p, q, Vr, Vi, rho=1e6, u_lo=uo, u_hi=uo); Jf = None if rf is None else float(S1.prob.value)
        C[i] = (J0, Jf)
    J0, Jf = C[i]
    if J0 is None or Jf is None:
        return i, None
    A, b = _rows6(lo, hi, Ag, bg)
    r = S1.solve(p, q, Vr, Vi, rho=1e6, A=A, b=b, cut_cap=1e6, soft_fallback=False)
    if r is None:
        return i, None
    J = float(S1.prob.value); den = max(Jf - J0, 1e-9*abs(J0) + 1e-9); loss = -(min(J, Jf) - J0)/den
    if not want_grad:
        return i, (loss,)
    if J >= Jf:
        return i, (loss, np.zeros((K, 2*G)), np.zeros(K))
    VP = v_params(Vr, Vi, g.f_bus.astype(int), g.t_bus.astype(int))
    if not _W["built6"]:
        L6.build_structure([A, b] + VP); _W["sl6"] = L6.var_slices(); _W["built6"] = True
    x, st = L6.forward([A, b] + VP, tol=1e-10)
    if x is None:
        return i, None
    nA = 2*G                                                           # p_A is (2G+K, 2G), flattened C-order: row r, col j -> r*2G + j
    gA = _env(L6, st, x, 0, [r*nA + j for r in range(2*G, 2*G + K) for j in range(nA)]).reshape(K, nA)
    gb = _env(L6, st, x, 1, range(2*G, 2*G + K))
    return i, (loss, -gA/den, -gb/den)


# ---------------------------------------------------------------------------------------------------------------------
# (2026-10-13) HARDENED value losses (audit of the value loss, 2026-10-12): the plain value loss ALWAYS rewards tighter cuts and nothing pushes back at the
# validity boundary, so cuts drift past our own answer and deployment turns them into hard fixings. Hardening:
#   * bounds clipped to [0, 1] (float32 round-off such as lo = 1.0000001 made rows look violated with dual 1e6);
#   * the value gradient is ZERO on every coordinate / row that already cuts off our own answer (then only the validity term acts there);
#   * instances whose BASE relaxation already exceeds the fixed-answer value (J_fix - J_base <= 1e-3 |J_base|) carry no signal -> zero loss, zero gradient.
def value_soft_h(args):
    """per-unit integer cuts (init4 worker, gamma=0). -> (loss,) or (loss, dloss/dl, dloss/dh)."""
    from framework import v_params
    i, Vr, Vi, lo, hi, u_own, want_grad = args
    g, S1, L4 = _W["g"], _W["S1"], _W["L4"]; p, q = _W["inst"][i]; G = g.n_gen
    C = _W.setdefault("vcache_h", {})
    if i not in C:
        r0 = S1.solve(p, q, Vr, Vi, rho=1e6); J0 = None if r0 is None else float(S1.prob.value)
        uo = np.asarray(u_own, float); rf = S1.solve(p, q, Vr, Vi, rho=1e6, u_lo=uo, u_hi=uo); Jf = None if rf is None else float(S1.prob.value)
        C[i] = (J0, Jf)
    J0, Jf = C[i]
    if J0 is None or Jf is None:
        return i, None
    if Jf - J0 <= 1e-3*abs(J0):
        return i, ((0.0,) if not want_grad else (0.0, np.zeros(G), np.zeros(G)))
    lo = np.clip(np.asarray(lo, float), 0.0, 1.0); hi = np.clip(np.asarray(hi, float), 0.0, 1.0); A, b = unit_rows(lo, hi, G)
    r = S1.solve(p, q, Vr, Vi, rho=1e6, A=A, b=b, cut_cap=1e6, soft_fallback=False)
    if r is None:
        return i, None
    J = float(S1.prob.value); den = Jf - J0; loss = -(min(J, Jf) - J0)/den
    if not want_grad:
        return i, (loss,)
    if J >= Jf:
        return i, (loss, np.zeros(G), np.zeros(G))
    VP = v_params(Vr, Vi, g.f_bus.astype(int), g.t_bus.astype(int))
    if not _W["built4"]:
        L4.build_structure([A, b] + VP); _W["sl4"] = L4.var_slices(); _W["built4"] = True
    x, st = L4.forward([A, b] + VP, tol=1e-10)
    if x is None:
        return i, None
    uo = np.asarray(u_own, float); eb = _env(L4, st, x, 1, range(2*G))
    dJdl = np.where(lo > 0, -eb[:G], 0.0); dJdh = np.where(hi < 1, eb[G:], 0.0)
    dJdl = np.where(lo > uo + 1e-6, 0.0, dJdl)        # cut already cuts off our own answer: no tightening reward (validity acts alone)
    dJdh = np.where(hi < uo - 1e-6, 0.0, dJdh)
    dJdl = np.where(lo >= 1.0 - 1e-5, 0.0, dJdl)      # SATURATED rows (l = 1 / h = 0 exactly): the soft-row dual is degenerate (any value in [0, 1e6]); the
    dJdh = np.where(hi <= 1e-5, 0.0, dJdh)            # gradient into the cut-net through a saturated sigmoid is ~0 anyway -> no signal, no noise
    return i, (loss, -dJdl/den, -dJdh/den)


def value_gen_h(args):
    """K general rows on top of frozen integer cuts (init6 worker, gamma=0). -> (loss,) or (loss, dloss/dA_gen (K x 2G), dloss/db_gen)."""
    from framework import v_params
    i, Vr, Vi, lo, hi, Ag, bg, u_own, want_grad = args
    g, S1, L6 = _W["g"], _W["S1"], _W["L6"]; p, q = _W["inst"][i]; G = g.n_gen; K = _W["kgen"]
    C = _W.setdefault("gcache_h", {})
    if i not in C:
        Ab, bb = _rows6(lo, hi, None, None)
        r0 = S1.solve(p, q, Vr, Vi, rho=1e6, A=Ab, b=bb, cut_cap=1e6, soft_fallback=False); J0 = None if r0 is None else float(S1.prob.value)
        uo = np.asarray(u_own, float); rf = S1.solve(p, q, Vr, Vi, rho=1e6, u_lo=uo, u_hi=uo)
        Jf = None if rf is None else float(S1.prob.value); pgf = None if rf is None else np.asarray(rf["pg"], float)
        C[i] = (J0, Jf, pgf)
    J0, Jf, pgf = C[i]
    if J0 is None or Jf is None:
        return i, None
    if Jf - J0 <= 1e-3*abs(J0):
        return i, ((0.0,) if not want_grad else (0.0, np.zeros((K, 2*G)), np.zeros(K)))
    A, b = _rows6(lo, hi, Ag, bg)
    r = S1.solve(p, q, Vr, Vi, rho=1e6, A=A, b=b, cut_cap=1e6, soft_fallback=False)
    if r is None:
        return i, None
    J = float(S1.prob.value); den = Jf - J0; loss = -(min(J, Jf) - J0)/den
    if not want_grad:
        return i, (loss,)
    if J >= Jf:
        return i, (loss, np.zeros((K, 2*G)), np.zeros(K))
    VP = v_params(Vr, Vi, g.f_bus.astype(int), g.t_bus.astype(int))
    if not _W["built6"]:
        L6.build_structure([A, b] + VP); _W["sl6"] = L6.var_slices(); _W["built6"] = True
    x, st = L6.forward([A, b] + VP, tol=1e-10)
    if x is None:
        return i, None
    nA = 2*G
    gA = _env(L6, st, x, 0, [r_*nA + j for r_ in range(2*G, 2*G + K) for j in range(nA)]).reshape(K, nA); gb = _env(L6, st, x, 1, range(2*G, 2*G + K))
    w_fix = np.r_[pgf, np.asarray(u_own, float)]                         # rows that already cut off our own answer: no tightening reward
    bad = (np.asarray(Ag, float) @ w_fix) > np.asarray(bg, float) + 1e-6
    gA[bad] = 0.0; gb[bad] = 0.0
    return i, (loss, -gA/den, -gb/den)

# worker module for B (row activity) and E (smoothing distortion); spawn-picklable
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "src"))
import numpy as np
_X = {}

def init(case, K):
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["QCAC_CASE"] = case
    import combopt_ac, cuttrain
    from socp import Socp
    combopt_ac.init(case, True)
    cuttrain.init6(case, 1000.0, K)                       # anchor relaxation exactly as 55 (gamma=1000, rho=1e6 inside anchor6)
    g = combopt_ac._W["g"]; nl = combopt_ac._W["nl"]; G = g.n_gen
    _X.update(case=case, K=K, g=g, nl=nl, inst=combopt_ac._W["inst"], env=combopt_ac._W["env"],
              S0=Socp(g, nl, n_cuts=2*G + K, cut_price=1e6, u_smooth=0.0))   # unsmoothed, same soft-row structure

def _miqcqp_x(i, V, A, b, work, rho):
    """combopt_ac.miqcqp's model, returning the MIQCQP's own (pg, u, s) before pricing."""
    from gurobipy import GRB
    import gurobipy as gp
    from acuc import build
    g = _X["g"]; p, q = _X["inst"][i]; G = g.n_gen
    m, X, cost = build(g, _X["nl"], p, q, GRB.BINARY, exact=False, rho=rho, Vr0=V[0], Vi0=V[1], threads=1)
    s = m.addVars(len(b), lb=0)
    for r in range(len(b)):
        lhs = gp.quicksum(float(A[r, k])*X["pg"][k] for k in range(G)) + gp.quicksum(float(A[r, G + k])*X["u"][k] for k in range(G))
        m.addConstr(lhs <= float(b[r]) + s[r])
    m.setObjective(m.getObjective() + 1e6*gp.quicksum(s[r] for r in range(len(b))))
    m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
    m.optimize()
    if m.SolCount == 0: return None
    pg = np.array([X["pg"][k].X for k in range(G)]); u = np.array([round(X["u"][k].X) for k in range(G)], float)
    return dict(pg=pg, u=u, s=np.array([s[r].X for r in range(len(b))]), obj=float(m.ObjVal), gap=float(m.MIPGap), cost=float(cost.getValue()))

def work(args):
    """-> per-instance diagnostics. args: i, V, LO, HI, rowsA (K x 2G) or None if anchor failed, rows_b, rho, W, do_milp"""
    import cuttrain
    i, V, LO, HI, Ar, br, rho, W, do_milp = args
    g = _X["g"]; G = g.n_gen; K = _X["K"]; p, q = _X["inst"][i]; out = dict(i=i)
    # anchor (as deployed): smoothed gamma=1000, fractional unit rows u_k >= l_k / u_k <= h_k, rho 1e6
    w_s = cuttrain.anchor6((i, V[0], V[1], LO, HI))[1]
    # deploy-like relaxation: unsmoothed, HARD-rounded fixings (as the MIQCQP sees them), deployment rho, no general rows
    lo = (LO > 0).astype(float); hi = np.where(HI < 1, 0.0, 1.0)
    S0 = _X["S0"]
    r_d = S0.solve(p, q, V[0], V[1], rho=rho, u_lo=lo, u_hi=hi)
    # same relaxation, fractional rows as in the anchor but gamma=0 (isolates the smoothing effect)
    Au, bu = cuttrain.unit_rows(LO, HI, G); Afull = np.vstack([Au, np.zeros((K, 2*G))]); bfull = np.r_[bu, np.ones(K)]
    r_f0 = S0.solve(p, q, V[0], V[1], rho=1e6, A=Afull, b=bfull, cut_cap=1e6, soft_fallback=False)
    if w_s is not None and r_d is not None and r_f0 is not None:
        u_s = w_s[G:]; u_d = r_d["u"]; u_f0 = r_f0["u"]
        out.update(du_smooth=float(np.abs(u_s - u_f0).sum()), du_anchor_vs_deploy=float(np.abs(u_s - u_d).sum()),
                   n_half_smooth=int(((u_s > .5) != (u_f0 > .5)).sum()), frac_s=int(((u_s > 1e-3) & (u_s < 1 - 1e-3)).sum()), frac_f0=int(((u_f0 > 1e-3) & (u_f0 < 1 - 1e-3)).sum()))
    if Ar is not None and w_s is not None:
        out["depth"] = (Ar @ w_s - br)                  # = depth by construction
        if r_d is not None: out["row_at_deployrelax"] = Ar @ np.r_[r_d["pg"], r_d["u"]] - br
    if do_milp:
        Abase = np.zeros((2*G, 2*G)); bbase = np.ones(2*G)
        for k in range(G):
            if LO[k] > 0: Abase[k, G + k] = -1.0; bbase[k] = -1.0
            if HI[k] < 1: Abase[G + k, G + k] = 1.0; bbase[G + k] = 0.0
        t0 = time.time(); off = _miqcqp_x(i, V, Abase, bbase, W, rho); out["t_off"] = time.time() - t0
        if Ar is not None:
            t0 = time.time(); on = _miqcqp_x(i, V, np.vstack([Abase, Ar]), np.r_[bbase, br], W, rho); out["t_on"] = time.time() - t0
        else:
            on = off
        if on is not None and off is not None:
            out.update(u_changed=int((on["u"] != off["u"]).sum()), obj_on=on["cost"], obj_off=off["cost"], gap_on=on["gap"], gap_off=off["gap"],
                       s_rows_on=on["s"][2*G:] if Ar is not None else None, s_fix_on=float(on["s"][:2*G].sum()), s_fix_off=float(off["s"].sum()))
            if Ar is not None:
                out["row_on"] = Ar @ np.r_[on["pg"], on["u"]] - br
                out["row_off"] = Ar @ np.r_[off["pg"], off["u"]] - br
    return out

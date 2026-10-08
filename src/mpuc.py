"""Multi-period QCAC unit commitment on IEEE RTS-24 (T = 24 h).

Units: all 32 RTS-24 generating units (pandapower gen + sgen + ext_grid; the synchronous
condenser is kept as a reactive source only). Unit-commitment data per unit type from RTS-GMLC
(GridMod/RTS-GMLC, gen.csv): min up/down, ramp rate, hot-start heat x fuel price. U100 and
U197 are not in RTS-GMLC: they take U155 (coal steam) values -- ASSUMPTION, stated in the paper.
Fuel cost curves and no-load costs: pandapower poly_cost. Voltage bounds 0.95-1.05.
The AC network in every hour is the QCAC linearisation (acuc.build), at an hourly point V_t.
"""
from __future__ import annotations
import numpy as np
import gurobipy as gp
from gurobipy import GRB

T = 24


def set_T(t):
    """horizon length (hours); instances are an hourly window of the daily profile."""
    global T
    T = int(t)


# type: (min_up_h, min_down_h, ramp_MW_per_h, start_cost_$)   start = hot heat [MBTU] x fuel [$/MBTU]
UCDATA = {
    "U12":  (4, 2, 60.0, 38.0*10.3494), "U20": (1, 1, 180.0, 5.0*10.3494),
    "U50":  (0, 0, 3000.0, 0.0),        "U76": (8, 4, 120.0, 3379.4*2.11399),
    "U100": (8, 8, 180.0, 6892.1*2.11399), "U155": (8, 8, 180.0, 6892.1*2.11399),
    "U197": (8, 8, 180.0, 6892.1*2.11399), "U350": (24, 48, 240.0, 9768.2*2.11399),
    "U400": (24, 48, 1200.0, 9999.0*0.81035),
}
# RTS-79 hourly load shapes (% of daily peak): winter weekday / weekend, summer weekday / weekend
SHAPES = np.array([
    [67, 63, 60, 59, 59, 60, 74, 86, 95, 96, 96, 95, 95, 95, 93, 94, 99, 100, 100, 96, 91, 83, 73, 63],
    [78, 72, 68, 66, 64, 65, 66, 70, 80, 88, 90, 91, 90, 88, 87, 87, 91, 100, 99, 97, 94, 92, 87, 81],
    [64, 60, 58, 56, 56, 58, 64, 76, 87, 95, 99, 100, 99, 100, 100, 97, 96, 96, 93, 92, 92, 93, 87, 72],
    [74, 70, 66, 65, 64, 62, 62, 66, 81, 86, 91, 93, 93, 92, 91, 91, 92, 94, 95, 95, 100, 93, 88, 80],
], float)/100.0


def load_system():
    import pandapower as pp, pandapower.networks as pn
    from acopf_data import load
    g = load("case24_ieee_rts")
    net = pn.case24_ieee_rts(); pp.runpp(net)
    base = net._ppc["baseMVA"]; bl = net._pd2ppc_lookups["bus"]
    units = []                      # (bus_row, pmin, pmax, qmin, qmax, type, et, idx)
    def typ(pmax):
        return {12: "U12", 20: "U20", 50: "U50", 76: "U76", 100: "U100", 155: "U155",
                197: "U197", 350: "U350", 400: "U400"}[int(round(pmax))]
    for et, df in (("gen", net.gen), ("sgen", net.sgen), ("ext_grid", net.ext_grid)):
        for idx, r in df.iterrows():
            pmax = float(r.get("max_p_mw", 0.0))
            if pmax <= 0: continue        # synchronous condenser
            units.append((int(bl[int(r.bus)]), float(r.get("min_p_mw", 0.0)), pmax,
                          float(r.get("min_q_mvar", -50.0)), float(r.get("max_q_mvar", 50.0)), typ(pmax), et, int(idx)))
    pc = {(str(r.et), int(r.element)): r for r in net.poly_cost.itertuples()}
    U = len(units)
    c2 = np.array([pc[(u[6], u[7])].cp2_eur_per_mw2*base**2 for u in units])
    c1 = np.array([pc[(u[6], u[7])].cp1_eur_per_mw*base for u in units])
    c0 = np.array([pc[(u[6], u[7])].cp0_eur for u in units])
    g.gen_bus = np.array([u[0] for u in units])
    g.pmin = np.array([u[1] for u in units])/base; g.pmax = np.array([u[2] for u in units])/base
    g.qmin = np.minimum(np.array([u[3] for u in units])/base, 0.0); g.qmax = np.maximum(np.array([u[4] for u in units])/base, 0.0)
    g.c2, g.c1, g.c0 = c2, c1, c0
    g.utype = [u[5] for u in units]
    g.min_up = np.array([UCDATA[t][0] for t in g.utype]); g.min_dn = np.array([UCDATA[t][1] for t in g.utype])
    g.ramp = np.array([UCDATA[t][2] for t in g.utype])/base; g.su_cost = np.array([UCDATA[t][3] for t in g.utype])
    # condenser: fixed reactive source at its bus (Q range), always on, no cost
    cond = net.gen[net.gen.max_p_mw <= 0]
    g.cond_bus = np.array([int(bl[int(b)]) for b in cond.bus]); g.cond_qmin = cond.min_q_mvar.values/base; g.cond_qmax = cond.max_q_mvar.values/base
    # base loads per bus (net.load), the sgens are generators here, not negative load
    pd0 = np.zeros(g.n_bus); qd0 = np.zeros(g.n_bus)
    for _, r in net.load.iterrows():
        pd0[bl[int(r.bus)]] += r.p_mw/base; qd0[bl[int(r.bus)]] += r.q_mvar/base
    g.pd0, g.qd0 = pd0, qd0
    g.vmin = np.full(g.n_bus, 0.95); g.vmax = np.full(g.n_bus, 1.05)
    return g


def sample(g, n, seed=0, lo=0.55, hi=1.0):
    """[(pd (T, n_bus), qd (T, n_bus))]: shape family x peak level x peak shift +-2 h x smooth
    hourly noise x per-bus jitter."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        sh = SHAPES[rng.integers(len(SHAPES))]
        sh = np.roll(sh, rng.integers(-2, 3))
        noise = np.convolve(rng.normal(0, 0.03, 28), np.ones(5)/5, "valid")
        prof = rng.uniform(lo, hi)*sh*(1 + noise)
        if T < 24:                                   # window over a load ramp: start 04:00-14:00
            s0 = int(rng.integers(4, 15)); prof = np.roll(prof, -s0)[:T]
        jit = rng.uniform(0.95, 1.05, g.n_bus)
        out.append((np.outer(prof, g.pd0*jit), np.outer(prof, g.qd0*jit*rng.uniform(0.9, 1.1))))
    return out


def initial_status(g, pd):
    """units on at t=0: cheapest-first (average cost at full output) until hour-0 load x1.1
    is covered; nuclear and the 350 MW coal unit always start on. Long history assumed."""
    avg = g.c1 + g.c2*g.pmax + g.c0/np.maximum(g.pmax, 1e-9)
    u0 = np.zeros(g.n_gen); need = 1.1*pd[0].sum()
    for k in np.argsort(avg):
        if g.pmax @ u0 >= need: break
        u0[k] = 1
    for k, t in enumerate(g.utype):
        if t in ("U400", "U350"): u0[k] = 1
    return u0


def build(g, nl_unused, pd, qd, V=None, rho=1e3, integer=True, ufix=None, reserve=0.10,
          threads=1, cuts=None, target=None, soc=False):
    """multi-period QCAC UC. V: (Vr (T,n), Vi (T,n)) linearisation points (flat if None).
    integer=False -> continuous relaxation; ufix -> commitment fixed (pricing); target ->
    L1 projection objective onto the UC-feasible set (with optional cuts (A, b) on u)."""
    n, L, G = g.n_bus, g.n_branch, g.n_gen
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = threads
    vt = GRB.BINARY if (integer and ufix is None) else GRB.CONTINUOUS
    u = m.addVars(T, G, vtype=vt, lb=0, ub=1); su = m.addVars(T, G, lb=0, ub=1); sd = m.addVars(T, G, lb=0, ub=1)
    pg = m.addVars(T, G, lb=0); qg = m.addVars(T, G, lb=-GRB.INFINITY)
    u0 = initial_status(g, pd)
    for t in range(T):
        for k in range(G):
            prev = u[t-1, k] if t else float(u0[k])
            m.addConstr(u[t, k] - prev == su[t, k] - sd[t, k])
            if ufix is not None: m.addConstr(u[t, k] == float(ufix[t, k]))
            m.addConstr(pg[t, k] <= g.pmax[k]*u[t, k]); m.addConstr(pg[t, k] >= g.pmin[k]*u[t, k])
            m.addConstr(qg[t, k] <= g.qmax[k]*u[t, k]); m.addConstr(qg[t, k] >= g.qmin[k]*u[t, k])
            if t:   # ramping
                m.addConstr(pg[t, k] - pg[t-1, k] <= g.ramp[k] + g.pmax[k]*su[t, k])
                m.addConstr(pg[t-1, k] - pg[t, k] <= g.ramp[k] + g.pmax[k]*sd[t, k])
            UT, DT = int(g.min_up[k]), int(g.min_dn[k])
            if UT > 1: m.addConstr(gp.quicksum(su[s, k] for s in range(max(0, t-UT+1), t+1)) <= u[t, k])
            if DT > 1: m.addConstr(gp.quicksum(sd[s, k] for s in range(max(0, t-DT+1), t+1)) <= 1 - u[t, k])
        m.addConstr(gp.quicksum(g.pmax[k]*u[t, k] for k in range(G)) >= (1+reserve)*float(pd[t].sum()))
    xi_all = []
    cl, ch = g.vmin**2, g.vmax**2; ph = g.vmax[fb]*g.vmax[tb]
    for t in range(T):
        Vr = np.ones(n) if V is None else V[0][t]; Vi = np.zeros(n) if V is None else V[1][t]
        vr = m.addVars(n, lb=-g.vmax, ub=g.vmax); vi = m.addVars(n, lb=-g.vmax, ub=g.vmax)
        cii = m.addVars(n, lb=cl, ub=ch); cij = m.addVars(L, lb=-ph, ub=ph); sij = m.addVars(L, lb=-ph, ub=ph)
        xi = m.addVars(n + 2*L, lb=0, ub=(0.0 if soc else GRB.INFINITY)); xi_all.append(xi)
        for i in (range(0) if soc else range(n)):
            lin = 2*Vr[i]*vr[i] + 2*Vi[i]*vi[i] - Vr[i]**2 - Vi[i]**2
            m.addConstr(cii[i] - lin <= xi[i]); m.addConstr(lin - cii[i] <= xi[i])
        for k in range(L):
            i, j = fb[k], tb[k]
            if soc:     # Jabr SOC relaxation: only the rotated-cone link between cii, cij, sij
                w = m.addVar(lb=cl[i]+cl[j], ub=ch[i]+ch[j]); z = m.addVar(lb=cl[i]-ch[j], ub=ch[i]-cl[j])
                m.addConstr(w == cii[i] + cii[j]); m.addConstr(z == cii[i] - cii[j])
                m.addQConstr(4*cij[k]*cij[k] + 4*sij[k]*sij[k] + z*z <= w*w); continue
            lc = Vr[j]*vr[i] + Vr[i]*vr[j] + Vi[j]*vi[i] + Vi[i]*vi[j] - Vr[i]*Vr[j] - Vi[i]*Vi[j]
            ls = Vi[j]*vr[i] - Vi[i]*vr[j] - Vr[j]*vi[i] + Vr[i]*vi[j] - Vr[i]*Vi[j] + Vr[j]*Vi[i]
            m.addConstr(cij[k] - lc <= xi[n+k]); m.addConstr(lc - cij[k] <= xi[n+k])
            m.addConstr(sij[k] - ls <= xi[n+L+k]); m.addConstr(ls - sij[k] <= xi[n+L+k])
            w = m.addVar(lb=cl[i]+cl[j], ub=ch[i]+ch[j]); z = m.addVar(lb=cl[i]-ch[j], ub=ch[i]-cl[j])
            m.addConstr(w == cii[i] + cii[j]); m.addConstr(z == cii[i] - cii[j])
            m.addQConstr(4*cij[k]*cij[k] + 4*sij[k]*sij[k] + z*z <= w*w)
        m.addConstr(vi[int(g.ref)] == 0); m.addConstr(vr[int(g.ref)] >= 0)
        gg, bb, bsh, Tp, tm2 = g.g, g.b, g.b_fr, g.tap, g.tm2
        pf = [gg[k]/tm2[k]*cii[fb[k]] - gg[k]/Tp[k]*cij[k] + bb[k]/Tp[k]*sij[k] for k in range(L)]
        pt = [gg[k]*cii[tb[k]] - gg[k]/Tp[k]*cij[k] - bb[k]/Tp[k]*sij[k] for k in range(L)]
        qf = [-(bb[k]+bsh[k])/tm2[k]*cii[fb[k]] + gg[k]/Tp[k]*sij[k] + bb[k]/Tp[k]*cij[k] for k in range(L)]
        qt = [-(bb[k]+bsh[k])*cii[tb[k]] - gg[k]/Tp[k]*sij[k] + bb[k]/Tp[k]*cij[k] for k in range(L)]
        qc = {int(b): m.addVar(lb=float(lo), ub=float(hi)) for b, lo, hi in zip(g.cond_bus, g.cond_qmin, g.cond_qmax)}
        for i in range(n):
            gen_i = [k for k in range(G) if g.gen_bus[k] == i]
            m.addConstr(gp.quicksum(pg[t, k] for k in gen_i) - float(pd[t][i]) - g.gs[i]*cii[i]
                        == gp.quicksum(pf[k] for k in range(L) if fb[k] == i) + gp.quicksum(pt[k] for k in range(L) if tb[k] == i))
            m.addConstr(gp.quicksum(qg[t, k] for k in gen_i) + (qc[i] if i in qc else 0) - float(qd[t][i]) + g.bs[i]*cii[i]
                        == gp.quicksum(qf[k] for k in range(L) if fb[k] == i) + gp.quicksum(qt[k] for k in range(L) if tb[k] == i))
        fin = np.isfinite(g.rate)
        for k in range(L):
            if fin[k]:
                m.addQConstr(pf[k]*pf[k] + qf[k]*qf[k] <= g.rate[k]**2); m.addQConstr(pt[k]*pt[k] + qt[k]*qt[k] <= g.rate[k]**2)
        m._v = getattr(m, "_v", []) + [(vr, vi)]
    cost = gp.quicksum(g.c2[k]*pg[t, k]*pg[t, k] + g.c1[k]*pg[t, k] + g.c0[k]*u[t, k] + g.su_cost[k]*su[t, k]
                       for t in range(T) for k in range(G))
    pen = rho*gp.quicksum(x[j] for x in xi_all for j in x)
    if cuts is not None:
        A, b = cuts; s = m.addVars(len(b), lb=0)
        for r in range(len(b)):
            m.addConstr(gp.quicksum(float(A[r][t, k])*u[t, k] for t in range(T) for k in range(G) if A[r][t, k] != 0) <= float(b[r]) + s[r])
        pen = pen + 1e5*gp.quicksum(s[r] for r in range(len(b)))
    if target is not None:
        m.setObjective(gp.quicksum((1 - 2*float(target[t, k]))*u[t, k] for t in range(T) for k in range(G)) + 1e-3*pen, GRB.MINIMIZE)
    else:
        m.setObjective(cost + pen, GRB.MINIMIZE)
    m.update()
    return m, dict(u=u, su=su, pg=pg, xi=xi_all), cost


def solve_iterative(g, pd, qd, rho0=1e3, mu=4.0, eps=1e-3, max_iter=20, work=300., threads=1, mip_gap=1e-3):
    """QCAC iterative reference for the multi-period UC (re-linearise every hour at the solution)."""
    import time
    n = g.n_bus; Vr = np.ones((T, n)); Vi = np.zeros((T, n)); rho = rho0; out = None; t0 = time.time(); hist = []
    for it in range(max_iter):
        m, X, cost = build(g, None, pd, qd, V=(Vr, Vi), rho=rho, threads=threads)
        m.Params.WorkLimit = work; m.Params.MIPGap = mip_gap
        if out is not None:
            for t in range(T):
                for k in range(g.n_gen): X["u"][t, k].Start = float(out["u"][t, k])
        m.optimize()
        if m.SolCount == 0: break
        slack = float(sum(x[j].X for x in X["xi"] for j in x))
        out = dict(u=np.array([[round(X["u"][t, k].X) for k in range(g.n_gen)] for t in range(T)]),
                   cost=float(cost.getValue()), slack=slack, mip_gap=float(m.MIPGap))
        hist.append((it, rho, slack, out["cost"]))
        if slack <= eps: break
        Vr = np.array([[v.X for v in m._v[t][0].values()] for t in range(T)])
        Vi = np.array([[v.X for v in m._v[t][1].values()] for t in range(T)])
        rho = min(mu*rho, 1e10)
    if out is not None:
        out.update(secs=time.time()-t0, iters=len(hist), hist=hist, converged=bool(hist and hist[-1][2] <= eps))
    return out

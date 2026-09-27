"""
AC unit commitment: one model builder, two solvers, one verifier.

ONE builder on purpose. The previous codebase kept parallel copies of the same
model and they drifted -- a fix applied to one never reached the other, and the
difference was invisible until results disagreed. Everything here goes through
`build`, with flags selecting exact-AC or QCAC.

  build(exact=True)   true nonconvex AC: the voltage products are EQUALITIES.
  build(exact=False)  QCAC: products linearised around (Vr0, Vi0), residual
                      taken up by a slack penalised at rho.

  solve_global()          spatial branch-and-bound on the exact model
  solve_qcac_iterative()  Constante-Flores & Li: solve, re-linearise, escalate
                          rho, repeat until the slack vanishes
  verify()                independent feasibility check of a returned solution,
                          recomputing power balance from the raw voltages

Two modelling points that are not optional:

  * The lifted variables MUST be bounded. |c_ij|, |s_ij| <= vmax_i vmax_j and
    c_ii in [vmin^2, vmax^2]. Spatial branch-and-bound converges by splitting
    variable domains; with an unbounded domain the bound never moves.
  * pandapower ships case118 with vmin=0, vmax=2 on every bus. Those are
    placeholders, not operating limits (MATPOWER uses 0.94/1.06). vmin=0 lets
    the model de-energise a bus and inflates the search domain 16x in c_ii.
"""
from __future__ import annotations

import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def build(g, nl, pd_, qd_, u_type=GRB.BINARY, exact=True, rho=None,
          Vr0=None, Vi0=None, reserve=0.10, threads=12, cii_lo=None,
          cii_hi=None):
    """Returns (model, vars, cost_expression). `cost` EXCLUDES any penalty."""
    n, L, G = g.n_bus, g.n_branch, g.n_gen
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
    cl = g.vmin ** 2 if cii_lo is None else cii_lo
    ch = g.vmax ** 2 if cii_hi is None else cii_hi
    ph = g.vmax[fb] * g.vmax[tb]

    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Threads = threads
    vr = m.addVars(n, lb=-g.vmax, ub=g.vmax)
    vi = m.addVars(n, lb=-g.vmax, ub=g.vmax)
    pg = m.addVars(G, lb=0.0, ub=g.pmax)
    qg = m.addVars(G, lb=g.qmin, ub=g.qmax)
    u = m.addVars(G, vtype=u_type, lb=0.0, ub=1.0)
    cii = m.addVars(n, lb=cl, ub=ch)
    cij = m.addVars(L, lb=-ph, ub=ph)
    sij = m.addVars(L, lb=-ph, ub=ph)
    xi = None

    if exact:
        m.Params.NonConvex = 2
        for i in range(n):
            m.addQConstr(cii[i] == vr[i] * vr[i] + vi[i] * vi[i])
        for k in range(L):
            i, j = fb[k], tb[k]
            m.addQConstr(cij[k] == vr[i] * vr[j] + vi[i] * vi[j])
            m.addQConstr(sij[k] == vr[i] * vi[j] - vr[j] * vi[i])
    else:
        # First-order expansion of each product about (Vr0, Vi0), with the
        # residual absorbed by a nonnegative slack priced at rho.
        xi = m.addVars(n + 2 * L, lb=0.0)
        for i in range(n):
            lin = 2 * Vr0[i] * vr[i] + 2 * Vi0[i] * vi[i] - Vr0[i]**2 - Vi0[i]**2
            m.addConstr(cii[i] - lin <= xi[i])
            m.addConstr(lin - cii[i] <= xi[i])
        for k in range(L):
            i, j = fb[k], tb[k]
            lc = (Vr0[j]*vr[i] + Vr0[i]*vr[j] + Vi0[j]*vi[i] + Vi0[i]*vi[j]
                  - Vr0[i]*Vr0[j] - Vi0[i]*Vi0[j])
            ls = (Vi0[j]*vr[i] - Vi0[i]*vr[j] - Vr0[j]*vi[i] + Vr0[i]*vi[j]
                  - Vr0[i]*Vi0[j] + Vr0[j]*Vi0[i])
            m.addConstr(cij[k] - lc <= xi[n + k])
            m.addConstr(lc - cij[k] <= xi[n + k])
            m.addConstr(sij[k] - ls <= xi[n + L + k])
            m.addConstr(ls - sij[k] <= xi[n + L + k])

    # SOC strengthening. Implied by the exact definitions (relaxed rank-1), so
    # valid in BOTH modes -- and load-bearing in both. Dropping it from the
    # linearised path looks harmless (slack still goes to 1e-10) but is not:
    # at zero slack the linearisation only forces cii to equal the FIRST-ORDER
    # EXPANSION of |V|^2, not |V|^2, so (cii, cij, sij) drifts off the rank-1
    # manifold. Measured on case118 inst0, removing it returned 161,352.6 with
    # 16 units -- BELOW the proven optimum of 164,101.4, i.e. not physical.
    #
    # Written as a rotated cone, `cij^2 + sij^2 <= cii_i cii_j`, Gurobi
    # recognises 178 of case118's 186 and leaves 8 bilinear; 8 is enough to
    # make it report "Continuous model is non-convex -- solving as a MIP" and
    # run spatial branch-and-bound on a relaxation with NO integer variables
    # (60 s time limit at a 0.5% gap, versus 0.38 s to optimality).
    #
    # The substitution w = cii_i + cii_j, z = cii_i - cii_j gives
    # cii_i cii_j = (w^2 - z^2)/4, so the same set is
    #     4 cij^2 + 4 sij^2 + z^2 <= w^2,   w >= 2 vmin^2 > 0,
    # a PLAIN second-order cone that Gurobi always recognises. Exact algebraic
    # rewrite -- identical feasible set, nothing relaxed.
    wv = m.addVars(L, lb=cl[fb] + cl[tb], ub=ch[fb] + ch[tb])
    zv = m.addVars(L, lb=cl[fb] - ch[tb], ub=ch[fb] - cl[tb])
    for k in range(L):
        m.addConstr(wv[k] == cii[fb[k]] + cii[tb[k]])
        m.addConstr(zv[k] == cii[fb[k]] - cii[tb[k]])
        m.addQConstr(4*cij[k]*cij[k] + 4*sij[k]*sij[k] + zv[k]*zv[k]
                     <= wv[k]*wv[k])

    m.addConstr(vi[int(g.ref)] == 0.0)
    m.addConstr(vr[int(g.ref)] >= 0.0)

    gg, bb, bsh, T, tm2 = g.g, g.b, g.b_fr, g.tap, g.tm2
    p_fr = [gg[k]/tm2[k]*cii[fb[k]] - gg[k]/T[k]*cij[k] + bb[k]/T[k]*sij[k] for k in range(L)]
    p_to = [gg[k]*cii[tb[k]] - gg[k]/T[k]*cij[k] - bb[k]/T[k]*sij[k] for k in range(L)]
    q_fr = [-(bb[k]+bsh[k])/tm2[k]*cii[fb[k]] + gg[k]/T[k]*sij[k] + bb[k]/T[k]*cij[k] for k in range(L)]
    q_to = [-(bb[k]+bsh[k])*cii[tb[k]] - gg[k]/T[k]*sij[k] + bb[k]/T[k]*cij[k] for k in range(L)]
    frm, to = {i: [] for i in range(n)}, {i: [] for i in range(n)}
    for k in range(L):
        frm[fb[k]].append(k); to[tb[k]].append(k)
    gat = {i: [] for i in range(n)}
    for gi, b_ in enumerate(g.gen_bus):
        gat[int(b_)].append(gi)
    for i in range(n):
        m.addConstr(gp.quicksum(pg[x] for x in gat[i]) - pd_[i] - g.gs[i]*cii[i]
                    == gp.quicksum(p_fr[k] for k in frm[i]) + gp.quicksum(p_to[k] for k in to[i]))
        m.addConstr(gp.quicksum(qg[x] for x in gat[i]) - qd_[i] + g.bs[i]*cii[i]
                    == gp.quicksum(q_fr[k] for k in frm[i]) + gp.quicksum(q_to[k] for k in to[i]))
    for x in range(G):
        m.addConstr(pg[x] <= g.pmax[x]*u[x]); m.addConstr(pg[x] >= g.pmin[x]*u[x])
        m.addConstr(qg[x] <= g.qmax[x]*u[x]); m.addConstr(qg[x] >= g.qmin[x]*u[x])
    fin = np.isfinite(g.rate)
    for k in range(L):
        if fin[k]:
            m.addQConstr(p_fr[k]*p_fr[k] + q_fr[k]*q_fr[k] <= g.rate[k]**2)
            m.addQConstr(p_to[k]*p_to[k] + q_to[k]*q_to[k] <= g.rate[k]**2)
    m.addConstr(gp.quicksum(g.pmax[x]*u[x] for x in range(G))
                >= (1.0 + reserve) * float(pd_.sum()))

    cost = gp.quicksum(g.c2[x]*pg[x]*pg[x] + g.c1[x]*pg[x] + nl[x]*u[x]
                       for x in range(G))
    pen = rho * gp.quicksum(xi[t] for t in range(n + 2*L)) if xi is not None else 0
    m.setObjective(cost + pen, GRB.MINIMIZE)
    m.update()
    return m, dict(vr=vr, vi=vi, pg=pg, qg=qg, u=u, cii=cii, cij=cij,
                   sij=sij, xi=xi), cost


def _extract(V, g, cost, m):
    n, G = g.n_bus, g.n_gen
    return dict(u=np.array([round(V["u"][x].X) for x in range(G)], dtype=int),
                pg=np.array([V["pg"][x].X for x in range(G)]),
                qg=np.array([V["qg"][x].X for x in range(G)]),
                vr=np.array([V["vr"][i].X for i in range(n)]),
                vi=np.array([V["vi"][i].X for i in range(n)]),
                cost=float(cost.getValue()), obj=float(m.ObjVal),
                bound=float(m.ObjBound), mip_gap=float(m.MIPGap))


def solve_global(g, nl, pd_, qd_, reserve=0.10, time_limit=600., threads=12,
                 mip_gap=1e-4, u_start=None, pg_start=None, start=None):
    """Spatial branch-and-bound on the exact nonconvex model."""
    t0 = time.time()
    m, V, cost = build(g, nl, pd_, qd_, GRB.BINARY, exact=True,
                       reserve=reserve, threads=threads)
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.MIPFocus = 2
    # A COMPLETE start. Setting only u and pg leaves Gurobi to solve for
    # vr/vi/qg and the lifted variables, which under NonConvex=2 is itself a
    # hard nonconvex completion -- it can fail, and then the model has no
    # incumbent at all even though one was handed to it. Instance 2 returned
    # zero solutions in 300s + a 150s fallback for exactly this reason, while a
    # feasible point of cost 154,957.0 was already in hand.
    if start is not None:
        n, L = g.n_bus, g.n_branch
        fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
        vr_s, vi_s = np.asarray(start["vr"]), np.asarray(start["vi"])
        cii_s = vr_s**2 + vi_s**2
        cij_s = vr_s[fb]*vr_s[tb] + vi_s[fb]*vi_s[tb]
        sij_s = vr_s[fb]*vi_s[tb] - vr_s[tb]*vi_s[fb]
        for i in range(n):
            V["vr"][i].Start = float(vr_s[i]); V["vi"][i].Start = float(vi_s[i])
            V["cii"][i].Start = float(cii_s[i])
        for k in range(L):
            V["cij"][k].Start = float(cij_s[k]); V["sij"][k].Start = float(sij_s[k])
        for x in range(g.n_gen):
            V["u"][x].Start = float(start["u"][x])
            V["pg"][x].Start = float(start["pg"][x])
            V["qg"][x].Start = float(start["qg"][x])
    else:
        if u_start is not None:
            for x in range(g.n_gen):
                V["u"][x].Start = float(u_start[x])
        if pg_start is not None:
            for x in range(g.n_gen):
                V["pg"][x].Start = float(pg_start[x])
    m.optimize()
    if m.SolCount == 0:
        # MIPFocus=2 prioritises the bound and will happily finish with no
        # incumbent. Retry once biased toward finding a feasible solution.
        m.Params.MIPFocus = 1
        m.Params.TimeLimit = max(60.0, time_limit / 2)
        m.optimize()
    if m.SolCount == 0:
        return None
    r = _extract(V, g, cost, m)
    r.update(secs=time.time() - t0, status=int(m.Status))
    return r


def solve_qcac_iterative(g, nl, pd_, qd_, reserve=0.10, rho0=1e3, mu=4.0,
                         rho_max=1e10, eps=1e-3, max_iter=20, tl=120.,
                         threads=12, mip_gap=1e-3):
    """Constante-Flores & Li: linearise, penalise, re-linearise, escalate rho."""
    n, L = g.n_bus, g.n_branch
    Vr, Vi = np.ones(n), np.zeros(n)        # flat start
    rho, hist, out = rho0, [], None
    t0 = time.time()
    for t in range(max_iter):
        m, V, cost = build(g, nl, pd_, qd_, GRB.BINARY, exact=False, rho=rho,
                           Vr0=Vr, Vi0=Vi, reserve=reserve, threads=threads)
        m.Params.TimeLimit = tl
        m.Params.MIPGap = mip_gap
        m.optimize()
        if m.SolCount == 0:
            break
        slack = float(sum(V["xi"][k].X for k in range(n + 2*L)))
        out = _extract(V, g, cost, m)
        hist.append(dict(it=t+1, rho=rho, slack=slack, cost=out["cost"],
                         n_on=int(out["u"].sum())))
        if slack <= eps:
            break
        Vr = np.array([V["vr"][i].X for i in range(n)])
        Vi = np.array([V["vi"][i].X for i in range(n)])
        rho = min(mu * rho, rho_max)
    if out is not None:
        out.update(secs=time.time() - t0, iters=len(hist), hist=hist,
                   slack=hist[-1]["slack"] if hist else np.nan)
    return out


def cost_commitment(g, nl, pd_, qd_, u_fixed, reserve=0.10, time_limit=300.,
                    threads=12):
    """TRUE AC cost of a given commitment: fix u in the exact model and solve.

    The QCAC objective contains the penalty term and is not a cost, so a
    commitment produced by the relaxation must be re-costed here before being
    compared against anything.
    """
    m, V, cost = build(g, nl, pd_, qd_, GRB.CONTINUOUS, exact=True,
                       reserve=reserve, threads=threads)
    for x in range(g.n_gen):
        V["u"][x].lb = V["u"][x].ub = float(u_fixed[x])
    m.Params.TimeLimit = time_limit
    m.optimize()
    if m.SolCount == 0:
        return None
    return _extract(V, g, cost, m)


def verify(g, nl, pd_, qd_, sol, reserve=0.10):
    """Independent feasibility check, recomputed from the raw voltages.

    Deliberately does NOT reuse the model's own expressions -- it rebuilds the
    power balance from vr/vi with numpy, so a sign error or a mis-indexed
    incidence in the builder would show up here rather than being confirmed by
    its own arithmetic.
    """
    n, L = g.n_bus, g.n_branch
    fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
    vr, vi, pg, qg, u = sol["vr"], sol["vi"], sol["pg"], sol["qg"], sol["u"]
    cii = vr**2 + vi**2
    cij = vr[fb]*vr[tb] + vi[fb]*vi[tb]
    sij = vr[fb]*vi[tb] - vr[tb]*vi[fb]
    gg, bb, bsh, T, tm2 = g.g, g.b, g.b_fr, g.tap, g.tm2
    p_fr = gg/tm2*cii[fb] - gg/T*cij + bb/T*sij
    p_to = gg*cii[tb] - gg/T*cij - bb/T*sij
    q_fr = -(bb+bsh)/tm2*cii[fb] + gg/T*sij + bb/T*cij
    q_to = -(bb+bsh)*cii[tb] - gg/T*sij + bb/T*cij
    pinj = np.zeros(n); qinj = np.zeros(n)
    np.add.at(pinj, fb, p_fr); np.add.at(pinj, tb, p_to)
    np.add.at(qinj, fb, q_fr); np.add.at(qinj, tb, q_to)
    pgen = np.zeros(n); qgen = np.zeros(n)
    np.add.at(pgen, g.gen_bus.astype(int), pg)
    np.add.at(qgen, g.gen_bus.astype(int), qg)
    p_res = np.abs(pgen - pd_ - g.gs*cii - pinj).max()
    q_res = np.abs(qgen - qd_ + g.bs*cii - qinj).max()
    vm = np.sqrt(cii)
    return dict(
        p_balance_max_resid=float(p_res),
        q_balance_max_resid=float(q_res),
        vmin=float(vm.min()), vmax=float(vm.max()),
        v_ok=bool(vm.min() >= g.vmin.min() - 1e-6 and vm.max() <= g.vmax.max() + 1e-6),
        pg_within_u=bool(np.all(pg <= g.pmax*u + 1e-6) and np.all(pg >= g.pmin*u - 1e-6)),
        reserve_ok=bool(float(g.pmax @ u) >= (1+reserve)*float(pd_.sum()) - 1e-6),
        cost_recomputed=float(g.c2 @ pg**2 + g.c1 @ pg + nl @ u),
    )

# %% [markdown]
# # QCAC iterative method (Constante-Flores & Li) vs the global MINLP
#
# Runs the published iterative QCAC algorithm on the SAME instance that
# `02_global_minlp.py` solved globally, and compares.
#
# The algorithm: linearise the voltage products around (Vre_t, Vim_t), relax the
# exact equalities to inequalities with a penalised slack, solve the resulting
# MIQCQP, re-linearise at the solution's own voltages, escalate rho, repeat
# until the slack is negligible.
#
#     V_0 = flat/power-flow point,  rho_0 small
#     for t = 1..T:
#         solve QCAC MIQCQP linearised at V_t with penalty rho_t
#         V_{t+1} <- solution voltages        (re-linearise)
#         rho_{t+1} <- min(mu * rho_t, rho_max)
#         stop when sum(slack) <= eps
#
# The comparison that matters is NOT the QCAC objective -- that includes the
# penalty term and is not a cost. It is: take the commitment the method returns,
# fix it in the TRUE nonconvex AC model, and see what it actually costs. That is
# the number to put against the global optimum.

# %% imports
import os, sys, time
import numpy as np
sys.path.insert(0, "src")
from acopf_data import load
from qcac import no_load_cost
import gurobipy as gp
from gurobipy import GRB

CASE, RESERVE, NL_FRAC = "case118", 0.10, 0.15
VMIN, VMAX = 0.94, 1.06        # same physical limits as the global solve
THREADS = 12

# %% the SAME instance the global solve used
ref = np.load("results/global_minlp_case118_inst0.npz")
pd_, qd_ = ref["pd"], ref["qd"]
U_GLOBAL, OBJ_GLOBAL = ref["u"], float(ref["obj"])
BOUND_GLOBAL, GAP_GLOBAL = float(ref["bound"]), float(ref["mip_gap"])

grid = load(CASE); g = grid
g.vmin = np.full(g.n_bus, VMIN); g.vmax = np.full(g.n_bus, VMAX)
nl = no_load_cost(g, NL_FRAC)
n, L, G = g.n_bus, g.n_branch, g.n_gen
fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)

print(f"{CASE} instance 0: load {pd_.sum():.2f} pu")
print(f"GLOBAL MINLP reference: {OBJ_GLOBAL:,.1f}  "
      f"({int(U_GLOBAL.sum())} units on, MIP gap {100*GAP_GLOBAL:.3f}%)")

# %% model builder -- QCAC (relaxed + penalised) or exact AC
def build(u_type, rho=None, Vr0=None, Vi0=None, exact=False):
    """exact=True  -> true AC equalities, no slack (for honest costing)
       exact=False -> QCAC: products relaxed, violation penalised by rho and
                      linearised around (Vr0, Vi0)."""
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = THREADS
    cl, ch = g.vmin**2, g.vmax**2
    ph = g.vmax[fb]*g.vmax[tb]
    vr = m.addVars(n, lb=-g.vmax, ub=g.vmax)
    vi = m.addVars(n, lb=-g.vmax, ub=g.vmax)
    pg = m.addVars(G, lb=0.0, ub=g.pmax); qg = m.addVars(G, lb=g.qmin, ub=g.qmax)
    u  = m.addVars(G, vtype=u_type, lb=0.0, ub=1.0)
    cii = m.addVars(n, lb=cl, ub=ch)
    cij = m.addVars(L, lb=-ph, ub=ph); sij = m.addVars(L, lb=-ph, ub=ph)
    xi = None

    if exact:
        m.Params.NonConvex = 2
        for i in range(n):
            m.addQConstr(cii[i] == vr[i]*vr[i] + vi[i]*vi[i])
        for k in range(L):
            i, j = fb[k], tb[k]
            m.addQConstr(cij[k] == vr[i]*vr[j] + vi[i]*vi[j])
            m.addQConstr(sij[k] == vr[i]*vi[j] - vr[j]*vi[i])
    else:
        # QCAC: linearise the products around (Vr0, Vi0) and let the residual
        # be taken up by a penalised slack. This is the relaxation the method
        # operates on; rho decides how hard the residual is punished.
        xi = m.addVars(n + 2*L, lb=0.0)
        for i in range(n):
            m.addConstr(cii[i] - (2*Vr0[i]*vr[i] + 2*Vi0[i]*vi[i]
                                  - Vr0[i]**2 - Vi0[i]**2) <= xi[i])
            m.addConstr((2*Vr0[i]*vr[i] + 2*Vi0[i]*vi[i]
                         - Vr0[i]**2 - Vi0[i]**2) - cii[i] <= xi[i])
        for k in range(L):
            i, j = fb[k], tb[k]
            lin_c = (Vr0[j]*vr[i] + Vr0[i]*vr[j] + Vi0[j]*vi[i] + Vi0[i]*vi[j]
                     - Vr0[i]*Vr0[j] - Vi0[i]*Vi0[j])
            lin_s = (Vi0[j]*vr[i] - Vi0[i]*vr[j] - Vr0[j]*vi[i] + Vr0[i]*vi[j]
                     - Vr0[i]*Vi0[j] + Vr0[j]*Vi0[i])
            m.addConstr(cij[k] - lin_c <= xi[n+k]);  m.addConstr(lin_c - cij[k] <= xi[n+k])
            m.addConstr(sij[k] - lin_s <= xi[n+L+k]); m.addConstr(lin_s - sij[k] <= xi[n+L+k])
    # SOC strengthening is valid either way
    for k in range(L):
        m.addQConstr(cij[k]*cij[k] + sij[k]*sij[k] <= cii[fb[k]]*cii[tb[k]])
    m.addConstr(vi[int(g.ref)] == 0.0); m.addConstr(vr[int(g.ref)] >= 0.0)

    gg, bb, bsh, T, tm2 = g.g, g.b, g.b_fr, g.tap, g.tm2
    p_fr=[gg[k]/tm2[k]*cii[fb[k]] - gg[k]/T[k]*cij[k] + bb[k]/T[k]*sij[k] for k in range(L)]
    p_to=[gg[k]*cii[tb[k]] - gg[k]/T[k]*cij[k] - bb[k]/T[k]*sij[k] for k in range(L)]
    q_fr=[-(bb[k]+bsh[k])/tm2[k]*cii[fb[k]] + gg[k]/T[k]*sij[k] + bb[k]/T[k]*cij[k] for k in range(L)]
    q_to=[-(bb[k]+bsh[k])*cii[tb[k]] - gg[k]/T[k]*sij[k] + bb[k]/T[k]*cij[k] for k in range(L)]
    frm={i:[] for i in range(n)}; to={i:[] for i in range(n)}
    for k in range(L): frm[fb[k]].append(k); to[tb[k]].append(k)
    gat={i:[] for i in range(n)}
    for gi,b_ in enumerate(g.gen_bus): gat[int(b_)].append(gi)
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
    m.addConstr(gp.quicksum(g.pmax[x]*u[x] for x in range(G)) >= (1+RESERVE)*float(pd_.sum()))

    cost = gp.quicksum(g.c2[x]*pg[x]*pg[x] + g.c1[x]*pg[x] + nl[x]*u[x] for x in range(G))
    m.setObjective(cost + (rho*gp.quicksum(xi[t] for t in range(n+2*L))
                           if xi is not None else 0), GRB.MINIMIZE)
    m.update()
    return m, dict(vr=vr, vi=vi, pg=pg, qg=qg, u=u, cii=cii, xi=xi), cost

# %% run the QCAC iterative algorithm  (EXPENSIVE)
RHO0, MU, RHO_MAX, EPS, MAX_IT, TL = 1e3, 4.0, 1e10, 1e-3, 20, 120.0
Vr_t = np.ones(n); Vi_t = np.zeros(n)          # flat start
rho_t = RHO0
hist = []
t_start = time.time()

for t in range(MAX_IT):
    m, V, cost_expr = build(GRB.BINARY, rho=rho_t, Vr0=Vr_t, Vi0=Vi_t, exact=False)
    m.Params.TimeLimit = TL; m.Params.MIPGap = 1e-3
    m.optimize()
    if m.SolCount == 0:
        print(f"  it {t+1}: no solution"); break
    slack = sum(V["xi"][k].X for k in range(n + 2*L))
    u_t = np.array([round(V["u"][x].X) for x in range(G)], dtype=int)
    cost_t = cost_expr.getValue()
    hist.append(dict(it=t+1, rho=rho_t, slack=slack, cost=cost_t,
                     n_on=int(u_t.sum()), secs=time.time()-t_start))
    print(f"  it {t+1:>2}  rho {rho_t:>9.1e}  slack {slack:>11.6f}  "
          f"cost {cost_t:>12,.1f}  units {int(u_t.sum()):>3}  "
          f"{time.time()-t_start:>6.0f}s", flush=True)
    if slack <= EPS:
        break
    Vr_t = np.array([V["vr"][i].X for i in range(n)])
    Vi_t = np.array([V["vi"][i].X for i in range(n)])
    rho_t = min(MU*rho_t, RHO_MAX)

u_qcac = u_t
secs_qcac = time.time() - t_start
print(f"\nQCAC iterative finished: {len(hist)} iterations, {secs_qcac:.0f}s, "
      f"final slack {hist[-1]['slack']:.2e}, {int(u_qcac.sum())} units on")

# %% cost the QCAC commitment HONESTLY in the true AC model
# The QCAC objective contains the penalty term and is not a cost. Fix its
# commitment in the exact nonconvex AC model and see what it really costs.
t0 = time.time()
me, Ve, cost_e = build(GRB.CONTINUOUS, exact=True)
for x in range(G):
    Ve["u"][x].lb = Ve["u"][x].ub = float(u_qcac[x])
me.Params.TimeLimit = 300; me.optimize()
true_cost_qcac = cost_e.getValue() if me.SolCount else np.nan
print(f"true AC cost of the QCAC commitment: {true_cost_qcac:,.1f}  "
      f"({time.time()-t0:.0f}s, status {me.Status})")

# %% COMPARISON
print("\n" + "="*64)
print(f"{'method':<28}{'cost':>14}{'gap':>10}{'units':>7}{'time':>8}")
print("-"*64)
print(f"{'GLOBAL MINLP (reference)':<28}{OBJ_GLOBAL:>14,.1f}{'--':>10}"
      f"{int(U_GLOBAL.sum()):>7}{'600s':>8}")
if np.isfinite(true_cost_qcac):
    gap = 100*(true_cost_qcac - OBJ_GLOBAL)/OBJ_GLOBAL
    print(f"{'QCAC iterative (Can Li)':<28}{true_cost_qcac:>14,.1f}"
          f"{gap:>9.3f}%{int(u_qcac.sum()):>7}{secs_qcac:>7.0f}s")
print("="*64)
agree = int(np.sum(u_qcac == np.round(U_GLOBAL).astype(int)))
print(f"commitment agreement: {agree}/{G} generators "
      f"({100*agree/G:.1f}%)   QCAC {int(u_qcac.sum())} on vs global "
      f"{int(U_GLOBAL.sum())} on")
np.savez("results/qcac_iterative_case118_inst0.npz", u=u_qcac,
         true_cost=true_cost_qcac, hist=np.array([list(h.values()) for h in hist]),
         secs=secs_qcac)

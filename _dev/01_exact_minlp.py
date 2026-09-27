# %% [markdown]
# # Exact AC unit commitment (MINLP) -- one case118 instance
#
# The TRUE problem, solved directly with Gurobi. No QCAC relaxation, no slack
# variables, no penalty weight rho.
#
# That last point is the reason this file exists. The previous pipeline scored
# everything on `generation + rho * slack`, where the slack sits on a convex
# relaxation of the voltage products. There is no good rho:
#
#   rho = 1e3 -> the relaxation is loose enough to fabricate power. Measured:
#                the relaxation served 0.8% of load, and both the exact solve
#                and the surrogate returned pg = 0 and simply paid the penalty.
#   rho = 1e6 -> the relaxation binds, but rho*slack becomes 97% of the score.
#                Fuel is 1.5-2.3%, no-load 0.7%. Committing six extra
#                generators costs +0.038%, so a constant over-committed
#                schedule beats a method that tracks the true optimum.
#
# Solving the MINLP exactly removes the question: the objective is fuel plus
# no-load cost, the AC physics are equality constraints, and the optimum is the
# optimum. It is the reference every surrogate should be measured against.
#
# Formulation (standard rectangular-voltage AC-UC):
#
#   min  sum_g c2_g pg_g^2 + c1_g pg_g + nl_g u_g
#   s.t. c_ii = vr_i^2 + vi_i^2                      (exact, nonconvex)
#        c_ij = vr_i vr_j + vi_i vi_j                (exact, nonconvex)
#        s_ij = vr_i vi_j - vr_j vi_i                (exact, nonconvex)
#        nodal power balance
#        branch MVA limits
#        pmin_g u_g <= pg_g <= pmax_g u_g            (binary u)
#        vmin^2 <= c_ii <= vmax^2
#        reserve: sum_g pmax_g u_g >= (1+r) sum_i pd_i
#
# The QCAC relaxation replaces the three equalities with inequalities plus a
# penalised slack; here they are enforced exactly and Gurobi handles the
# nonconvexity with NonConvex=2.

# %% imports
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), "src"))
sys.path.insert(0, "src")

from acopf_data import load                      # noqa: E402
from qcac import no_load_cost                    # noqa: E402

CASE = "case118"
RESERVE = 0.10        # spinning-reserve margin on committed capacity
NL_FRAC = 0.15        # no-load cost as a fraction of cost-at-pmax
TIME_LIMIT = 600.0    # seconds
MIP_GAP = 1e-4

# %% load the grid
grid = load(CASE)
nl = no_load_cost(grid, NL_FRAC)

print(f"{CASE}: {grid.n_bus} buses, {grid.n_branch} branches, {grid.n_gen} generators")
print(f"  pmax  [{grid.pmax.min():.2f}, {grid.pmax.max():.2f}] pu   "
      f"total {grid.pmax.sum():.1f}")
print(f"  c1    [{grid.c1.min():.0f}, {grid.c1.max():.0f}]   "
      f"c2 [{grid.c2.min():.1f}, {grid.c2.max():.1f}]")
print(f"  no-load cost  [{nl.min():.0f}, {nl.max():.0f}]")
print(f"  nominal load  {grid.pd.sum():.2f} pu")

# %% build ONE instance
# One correlated system-wide demand level plus a small per-bus jitter. A single
# scalar per instance is deliberate: drawing an independent factor per load (the
# old sampler) averages the AGGREGATE demand out to under 1% variation, and the
# commitment barely responds to anything else.
rng = np.random.default_rng(0)
level = 1.00
pd_ = grid.pd * level * rng.uniform(0.97, 1.03, grid.n_bus)
qd_ = grid.qd * level * rng.uniform(0.95, 1.05, grid.n_bus)

print(f"instance: load {pd_.sum():.2f} pu active, {qd_.sum():.2f} pu reactive")
print(f"  reserve requirement {(1 + RESERVE) * pd_.sum():.2f} pu "
      f"vs {grid.pmax.sum():.1f} pu installed")

# %% build the MINLP in gurobipy
import gurobipy as gp                            # noqa: E402
from gurobipy import GRB                         # noqa: E402

g = grid
n, L, G = g.n_bus, g.n_branch, g.n_gen
fb, tb = g.f_bus, g.t_bus

m = gp.Model("ac_uc_minlp")
m.Params.OutputFlag = 0
m.Params.NonConvex = 2                 # exact bilinear/quadratic equalities
m.Params.TimeLimit = TIME_LIMIT
m.Params.MIPGap = MIP_GAP

vr = m.addVars(n, lb=-g.vmax, ub=g.vmax, name="vr")
vi = m.addVars(n, lb=-g.vmax, ub=g.vmax, name="vi")
pg = m.addVars(G, lb=0.0, ub=g.pmax, name="pg")
qg = m.addVars(G, lb=g.qmin, ub=g.qmax, name="qg")
u = m.addVars(G, vtype=GRB.BINARY, name="u")

cii = m.addVars(n, lb=g.vmin ** 2, ub=g.vmax ** 2, name="cii")
cij = m.addVars(L, lb=-GRB.INFINITY, name="cij")
sij = m.addVars(L, lb=-GRB.INFINITY, name="sij")

# exact voltage-product definitions -- the nonconvex core of the problem
for i in range(n):
    m.addQConstr(cii[i] == vr[i] * vr[i] + vi[i] * vi[i], name=f"cii_{i}")
for k in range(L):
    i, j = int(fb[k]), int(tb[k])
    m.addQConstr(cij[k] == vr[i] * vr[j] + vi[i] * vi[j], name=f"cij_{k}")
    m.addQConstr(sij[k] == vr[i] * vi[j] - vr[j] * vi[i], name=f"sij_{k}")

# reference bus angle
m.addConstr(vi[int(g.ref)] == 0.0, name="ref_angle")
m.addConstr(vr[int(g.ref)] >= 0.0, name="ref_sign")

# %% flows, balance, limits
gg, bb, bsh, T, tm2 = g.g, g.b, g.b_fr, g.tap, g.tm2

p_fr = [gg[k] / tm2[k] * cii[int(fb[k])] - gg[k] / T[k] * cij[k] + bb[k] / T[k] * sij[k]
        for k in range(L)]
p_to = [gg[k] * cii[int(tb[k])] - gg[k] / T[k] * cij[k] - bb[k] / T[k] * sij[k]
        for k in range(L)]
q_fr = [-(bb[k] + bsh[k]) / tm2[k] * cii[int(fb[k])] + gg[k] / T[k] * sij[k]
        + bb[k] / T[k] * cij[k] for k in range(L)]
q_to = [-(bb[k] + bsh[k]) * cii[int(tb[k])] - gg[k] / T[k] * sij[k]
        + bb[k] / T[k] * cij[k] for k in range(L)]

gen_at = {i: [] for i in range(n)}
for gi, bus in enumerate(g.gen_bus):
    gen_at[int(bus)].append(gi)

for i in range(n):
    out = gp.quicksum(p_fr[k] for k in range(L) if int(fb[k]) == i) + \
          gp.quicksum(p_to[k] for k in range(L) if int(tb[k]) == i)
    outq = gp.quicksum(q_fr[k] for k in range(L) if int(fb[k]) == i) + \
           gp.quicksum(q_to[k] for k in range(L) if int(tb[k]) == i)
    m.addConstr(gp.quicksum(pg[gi] for gi in gen_at[i]) - pd_[i]
                - g.gs[i] * cii[i] == out, name=f"pbal_{i}")
    m.addConstr(gp.quicksum(qg[gi] for gi in gen_at[i]) - qd_[i]
                + g.bs[i] * cii[i] == outq, name=f"qbal_{i}")

# generator limits gated by the commitment
for gi in range(G):
    m.addConstr(pg[gi] <= g.pmax[gi] * u[gi], name=f"pmax_{gi}")
    m.addConstr(pg[gi] >= g.pmin[gi] * u[gi], name=f"pmin_{gi}")
    m.addConstr(qg[gi] <= g.qmax[gi] * u[gi], name=f"qmax_{gi}")
    m.addConstr(qg[gi] >= g.qmin[gi] * u[gi], name=f"qmin_{gi}")

# branch MVA limits where finite
fin = np.isfinite(g.rate)
for k in range(L):
    if fin[k]:
        m.addQConstr(p_fr[k] * p_fr[k] + q_fr[k] * q_fr[k] <= g.rate[k] ** 2,
                     name=f"rate_fr_{k}")
        m.addQConstr(p_to[k] * p_to[k] + q_to[k] * q_to[k] <= g.rate[k] ** 2,
                     name=f"rate_to_{k}")

# spinning reserve
m.addConstr(gp.quicksum(g.pmax[gi] * u[gi] for gi in range(G))
            >= (1.0 + RESERVE) * float(pd_.sum()), name="reserve")

# %% objective: fuel + no-load. No slack, no rho.
m.setObjective(
    gp.quicksum(g.c2[gi] * pg[gi] * pg[gi] + g.c1[gi] * pg[gi] + nl[gi] * u[gi]
                for gi in range(G)),
    GRB.MINIMIZE)

print(f"model: {m.NumVars} vars ({G} binary), {m.NumConstrs} linear, "
      f"{m.NumQConstrs} quadratic")

# %% solve  (EXPENSIVE -- this is the cell that costs minutes)
t0 = time.time()
m.optimize()
elapsed = time.time() - t0

status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time limit",
          GRB.INFEASIBLE: "INFEASIBLE", GRB.SUBOPTIMAL: "suboptimal"}.get(
              m.Status, str(m.Status))
print(f"status {status}   {elapsed:.1f}s   solutions found {m.SolCount}")

# %% results
if m.SolCount == 0:
    print("no feasible solution found")
else:
    u_opt = np.array([round(u[gi].X) for gi in range(G)], dtype=int)
    pg_opt = np.array([pg[gi].X for gi in range(G)])
    fuel = float(g.c2 @ pg_opt ** 2 + g.c1 @ pg_opt)
    noload = float(nl @ u_opt)

    print(f"\nOBJECTIVE          {m.ObjVal:,.1f}")
    print(f"  fuel             {fuel:,.1f}  ({100*fuel/m.ObjVal:.1f}%)")
    print(f"  no-load          {noload:,.1f}  ({100*noload/m.ObjVal:.1f}%)")
    print(f"  MIP gap          {100*m.MIPGap:.4f}%   bound {m.ObjBound:,.1f}")
    print(f"\nCOMMITMENT         {u_opt.sum()} of {G} generators on")
    print(f"  generation       {pg_opt.sum():.2f} pu")
    print(f"  load             {pd_.sum():.2f} pu")
    print(f"  losses           {pg_opt.sum() - pd_.sum():.2f} pu "
          f"({100*(pg_opt.sum()-pd_.sum())/pd_.sum():.2f}%)")
    print(f"  committed cap    {float(g.pmax @ u_opt):.2f} pu  "
          f"(reserve needs {(1+RESERVE)*pd_.sum():.2f})")
    vm = np.array([np.hypot(vr[i].X, vi[i].X) for i in range(n)])
    print(f"  |V|              [{vm.min():.4f}, {vm.max():.4f}] pu")
    print(f"\n  u* = {u_opt.tolist()}")

    np.savez(os.path.join("results", f"exact_minlp_{CASE}_inst0.npz"),
             u=u_opt, pg=pg_opt, obj=m.ObjVal, bound=m.ObjBound,
             mip_gap=m.MIPGap, secs=elapsed, pd=pd_, qd=qd_)
    print(f"\nsaved results/exact_minlp_{CASE}_inst0.npz")

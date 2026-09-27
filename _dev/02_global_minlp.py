# %% [markdown]
# # Global AC unit commitment, done properly
#
# `01_exact_minlp.py` handed the raw nonconvex model to Gurobi and waited. That
# does not work, and the reason is a formulation error rather than solver
# weakness: `cij` and `sij` were declared with infinite bounds. Spatial
# branch-and-bound converges by splitting variable DOMAINS and tightening the
# convex underestimators on each piece -- with an unbounded domain there is
# nothing to split, so the bound never improves.
#
# This file applies the standard machinery for global AC-OPF:
#
#  1. TIGHT BOUNDS on the lifted variables.  |c_ij|, |s_ij| <= vmax_i vmax_j,
#     and c_ii in [vmin^2, vmax^2]. These are what spatial B&B branches on.
#
#  2. SOC STRENGTHENING.  c_ij^2 + s_ij^2 <= c_ii c_jj is implied by the exact
#     definitions (it is the rank-1 condition relaxed), so adding it is VALID
#     and it is the single most effective cut for this problem class -- it is
#     what makes the QC/SOC relaxations of Coffrin et al. tight.
#
#  3. A LOWER BOUND from the convex relaxation, solved first and cheaply. It
#     brackets the optimum and tells us how much of any remaining gap is real.
#
#  4. A WARM START.  Round the relaxation's commitment, fix it, and solve the
#     resulting continuous nonconvex AC-OPF -- far easier than the full MINLP.
#     That feasible point goes in as a MIP start, so the search begins with a
#     good incumbent instead of hunting for one.
#
#  5. OBBT (optimality-based bound tightening) on the voltage magnitudes: for
#     each bus, minimise and maximise c_ii over the convex relaxation and keep
#     the tightened range. Costs 2n small convex solves and shrinks the domain
#     the B&B has to cover.
#
# Each is a separate cell so its individual effect on the bound is visible.

# %% imports
import os, sys, time
import numpy as np
sys.path.insert(0, "src")
from acopf_data import load
from qcac import no_load_cost

import gurobipy as gp
from gurobipy import GRB

CASE, RESERVE, NL_FRAC = "case118", 0.10, 0.15
TIME_LIMIT, MIP_GAP = 600.0, 1e-4
THREADS = 12

# %% grid + one instance
grid = load(CASE); g = grid
nl = no_load_cost(g, NL_FRAC)
n, L, G = g.n_bus, g.n_branch, g.n_gen
fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)

rng = np.random.default_rng(0)
pd_ = g.pd * rng.uniform(0.97, 1.03, n)
qd_ = g.qd * rng.uniform(0.95, 1.05, n)
print(f"{CASE}: {n} bus / {L} branch / {G} gen   load {pd_.sum():.2f} pu")

# %% bounds on the lifted variables  -- step 1
# Without these the spatial B&B has an unbounded domain and cannot converge.
cii_lo, cii_hi = g.vmin**2, g.vmax**2
prod_hi = g.vmax[fb] * g.vmax[tb]          # |v_i||v_j| upper bound
cij_lo, cij_hi = -prod_hi, prod_hi
sij_lo, sij_hi = -prod_hi, prod_hi
print(f"lifted-variable bounds:  c_ii in [{cii_lo.min():.3f}, {cii_hi.max():.3f}]   "
      f"|c_ij|,|s_ij| <= {prod_hi.max():.3f}")

# %% shared model builder
def build(u_type, cii_l, cii_u, add_soc=True, relax_products=False):
    """u_type: GRB.BINARY or GRB.CONTINUOUS.
    relax_products=True drops the exact equalities, leaving the CONVEX
    relaxation (SOC only) -- used for the lower bound, OBBT and the warm start.
    """
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.Threads = THREADS
    vr = m.addVars(n, lb=-g.vmax, ub=g.vmax)
    vi = m.addVars(n, lb=-g.vmax, ub=g.vmax)
    pg = m.addVars(G, lb=0.0, ub=g.pmax); qg = m.addVars(G, lb=g.qmin, ub=g.qmax)
    u  = m.addVars(G, vtype=u_type, lb=0.0, ub=1.0)
    cii = m.addVars(n, lb=cii_l, ub=cii_u)
    cij = m.addVars(L, lb=cij_lo, ub=cij_hi)
    sij = m.addVars(L, lb=sij_lo, ub=sij_hi)

    if not relax_products:
        m.Params.NonConvex = 2
        for i in range(n):
            m.addQConstr(cii[i] == vr[i]*vr[i] + vi[i]*vi[i])
        for k in range(L):
            i, j = fb[k], tb[k]
            m.addQConstr(cij[k] == vr[i]*vr[j] + vi[i]*vi[j])
            m.addQConstr(sij[k] == vr[i]*vi[j] - vr[j]*vi[i])
        m.addConstr(vi[int(g.ref)] == 0.0); m.addConstr(vr[int(g.ref)] >= 0.0)

    # step 2: SOC strengthening -- valid for the exact problem, and the cut that
    # makes this relaxation class tight
    if add_soc:
        for k in range(L):
            m.addQConstr(cij[k]*cij[k] + sij[k]*sij[k] <= cii[fb[k]]*cii[tb[k]])

    gg, bb, bsh, T, tm2 = g.g, g.b, g.b_fr, g.tap, g.tm2
    p_fr = [gg[k]/tm2[k]*cii[fb[k]] - gg[k]/T[k]*cij[k] + bb[k]/T[k]*sij[k] for k in range(L)]
    p_to = [gg[k]*cii[tb[k]] - gg[k]/T[k]*cij[k] - bb[k]/T[k]*sij[k] for k in range(L)]
    q_fr = [-(bb[k]+bsh[k])/tm2[k]*cii[fb[k]] + gg[k]/T[k]*sij[k] + bb[k]/T[k]*cij[k] for k in range(L)]
    q_to = [-(bb[k]+bsh[k])*cii[tb[k]] - gg[k]/T[k]*sij[k] + bb[k]/T[k]*cij[k] for k in range(L)]
    frm = {i: [] for i in range(n)}; to = {i: [] for i in range(n)}
    for k in range(L): frm[fb[k]].append(k); to[tb[k]].append(k)
    gat = {i: [] for i in range(n)}
    for gi, b_ in enumerate(g.gen_bus): gat[int(b_)].append(gi)
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
    m.setObjective(gp.quicksum(g.c2[x]*pg[x]*pg[x] + g.c1[x]*pg[x] + nl[x]*u[x]
                               for x in range(G)), GRB.MINIMIZE)
    m.update()
    return m, dict(vr=vr, vi=vi, pg=pg, qg=qg, u=u, cii=cii, cij=cij, sij=sij)

# %% step 3: convex relaxation -> LOWER BOUND (fast)
t0 = time.time()
mr, V = build(GRB.CONTINUOUS, cii_lo, cii_hi, add_soc=True, relax_products=True)
mr.Params.TimeLimit = 120; mr.optimize()
LB0 = mr.ObjVal if mr.SolCount else np.nan
u_rlx = np.array([V["u"][x].X for x in range(G)]) if mr.SolCount else None
print(f"convex relaxation  LOWER BOUND {LB0:,.1f}   {time.time()-t0:.1f}s  "
      f"({mr.NumVars} vars, {mr.NumQConstrs} quadratic)")

# %% step 1b: REALISTIC voltage bounds
# pandapower ships case118 with vmin=0, vmax=2 on every bus -- placeholders, not
# operating limits (MATPOWER's own case118 uses 0.94/1.06). vmin=0 lets the
# model de-energise a bus, which is unphysical AND makes the spatial B&B domain
# 16x wider in c_ii than it needs to be. This is the single cheapest tightening
# available.
VMIN, VMAX = 0.94, 1.06
g.vmin = np.full(n, VMIN); g.vmax = np.full(n, VMAX)
cii_lo, cii_hi = g.vmin**2, g.vmax**2
prod_hi = g.vmax[fb]*g.vmax[tb]
cij_lo, cij_hi = -prod_hi, prod_hi
sij_lo, sij_hi = -prod_hi, prod_hi
print(f"tightened:  c_ii in [{cii_lo[0]:.4f}, {cii_hi[0]:.4f}]   "
      f"|c_ij|,|s_ij| <= {prod_hi[0]:.4f}")

t0 = time.time()
mr2, V2 = build(GRB.CONTINUOUS, cii_lo, cii_hi, add_soc=True, relax_products=True)
mr2.Params.TimeLimit = 120; mr2.optimize()
LB1 = mr2.ObjVal if mr2.SolCount else np.nan
u_rlx = np.array([V2["u"][x].X for x in range(G)]) if mr2.SolCount else u_rlx
print(f"  LOWER BOUND {LB1:,.1f}  (was {LB0:,.1f}, +{100*(LB1-LB0)/LB0:.2f}%)"
      f"   {time.time()-t0:.1f}s")

# %% step 5: OBBT on the bus voltages
# Minimise and maximise each c_ii over the convex relaxation and keep the
# tightened interval. 2n small convex solves, and every unit of width removed
# here is width the spatial B&B never has to branch on.
t0 = time.time()
mo, Vo = build(GRB.CONTINUOUS, cii_lo, cii_hi, add_soc=True, relax_products=True)
mo.Params.TimeLimit = 5
lo_new, hi_new = cii_lo.copy(), cii_hi.copy()
tightened = 0
for i in range(n):
    for sense, arr in ((GRB.MINIMIZE, lo_new), (GRB.MAXIMIZE, hi_new)):
        mo.setObjective(Vo["cii"][i], sense); mo.optimize()
        if mo.SolCount:
            val = Vo["cii"][i].X
            if sense == GRB.MINIMIZE and val > lo_new[i] + 1e-6:
                lo_new[i] = val; tightened += 1
            elif sense == GRB.MAXIMIZE and val < hi_new[i] - 1e-6:
                hi_new[i] = val; tightened += 1
width0 = float((cii_hi - cii_lo).mean()); width1 = float((hi_new - lo_new).mean())
print(f"OBBT: {tightened} bounds tightened, mean c_ii width "
      f"{width0:.4f} -> {width1:.4f} ({100*(1-width1/width0):.1f}% narrower)"
      f"   {time.time()-t0:.0f}s")

# %% step 4: warm start -- repair upward until the AC-OPF is FEASIBLE
# Rounding the relaxation and topping up to the reserve margin is not enough:
# with 11 units the reserve constraint holds but the AC-OPF is infeasible,
# because reactive limits and branch flows need generation in the right PLACES,
# not just enough megawatts. So add units by descending relaxed u until a
# feasible AC point exists -- the same upward repair the surrogate uses.
t0 = time.time()
z = (u_rlx > 0.5).astype(float)
order = np.argsort(-u_rlx)
need = (1 + RESERVE) * float(pd_.sum())
for k in order:
    if float(g.pmax @ z) >= need:
        break
    z[k] = 1.0

UB0, pg_w, z_w = np.nan, None, None
for attempt in range(12):
    mw, Vw = build(GRB.CONTINUOUS, lo_new, hi_new, add_soc=True, relax_products=False)
    for x in range(G):
        Vw["u"][x].lb = Vw["u"][x].ub = z[x]
    mw.Params.TimeLimit = 120
    mw.optimize()
    if mw.SolCount:
        UB0 = mw.ObjVal
        pg_w = np.array([Vw["pg"][x].X for x in range(G)])
        z_w = z.copy()
        print(f"  attempt {attempt}: {int(z.sum())} units -> FEASIBLE, "
              f"obj {UB0:,.1f}  ({time.time()-t0:.0f}s)")
        break
    off = [k for k in order if z[k] < 0.5]
    if not off:
        break
    # add the three most-wanted uncommitted units and retry
    for k in off[:3]:
        z[k] = 1.0
    print(f"  attempt {attempt}: {int(z.sum())-3} units infeasible, "
          f"adding 3 -> {int(z.sum())}", flush=True)

if np.isfinite(UB0):
    print(f"warm start: {int(z_w.sum())} units on -> UPPER BOUND {UB0:,.1f}   "
          f"{time.time()-t0:.0f}s")
    print(f"\nBRACKET:  [{LB1:,.1f}, {UB0:,.1f}]   gap {100*(UB0-LB1)/UB0:.2f}%")
else:
    print("warm start failed: no commitment gave a feasible AC point")

# %% step 6: the full MINLP with everything applied  (EXPENSIVE)
# Tight lifted bounds + realistic voltage limits + OBBT ranges + SOC cuts +
# a feasible MIP start. Without the bounds this model cannot converge at all;
# without the start Gurobi spends its budget hunting for a first incumbent,
# which we already know takes 17 committed units to find.
t0 = time.time()
mf, Vf = build(GRB.BINARY, lo_new, hi_new, add_soc=True, relax_products=False)
mf.Params.TimeLimit = TIME_LIMIT
mf.Params.MIPGap = MIP_GAP
mf.Params.MIPFocus = 2            # prove optimality rather than hunt incumbents
if z_w is not None:
    for x in range(G):
        Vf["u"][x].Start = float(z_w[x])
    if pg_w is not None:
        for x in range(G):
            Vf["pg"][x].Start = float(pg_w[x])
mf.optimize()
secs = time.time() - t0

print(f"\nFULL MINLP  status {mf.Status}  {secs:.0f}s  solutions {mf.SolCount}")
if mf.SolCount:
    u_opt = np.array([round(Vf["u"][x].X) for x in range(G)], dtype=int)
    pg_opt = np.array([Vf["pg"][x].X for x in range(G)])
    fuel = float(g.c2 @ pg_opt**2 + g.c1 @ pg_opt); noload = float(nl @ u_opt)
    print(f"  INCUMBENT   {mf.ObjVal:,.1f}   (warm start was {UB0:,.1f}, "
          f"{100*(mf.ObjVal-UB0)/UB0:+.2f}%)")
    print(f"  BOUND       {mf.ObjBound:,.1f}   (relaxation gave {LB1:,.1f})")
    print(f"  MIP GAP     {100*mf.MIPGap:.3f}%")
    print(f"  units on    {u_opt.sum()} of {G}")
    print(f"  fuel {fuel:,.1f} ({100*fuel/mf.ObjVal:.1f}%)  "
          f"no-load {noload:,.1f} ({100*noload/mf.ObjVal:.1f}%)")
    print(f"  generation  {pg_opt.sum():.2f} pu vs load {pd_.sum():.2f} pu  "
          f"(losses {100*(pg_opt.sum()-pd_.sum())/pd_.sum():.2f}%)")
    np.savez("results/global_minlp_case118_inst0.npz", u=u_opt, pg=pg_opt,
             obj=mf.ObjVal, bound=mf.ObjBound, mip_gap=mf.MIPGap,
             lb_relax=LB1, ub_warm=UB0, secs=secs, pd=pd_, qd=qd_)
    print("  saved results/global_minlp_case118_inst0.npz")
else:
    print(f"  no incumbent; bound {mf.ObjBound:,.1f}")

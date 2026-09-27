# %% [markdown]
# # Re-run instances 2 and 3 with the corrected warm start
#
# Their global solves returned NO solution in the previous run. That was my
# bug, twice over:
#   1. the MIP start came from QCAC's RELAXED point, which violates the exact
#      voltage equalities by ~1e-5 and is rejected;
#   2. MIPFocus=2 tells Gurobi to prioritise the bound, so after rejecting the
#      start it never went looking for an incumbent.
# Both fixed in src/acuc.py -- but the fixes landed while the run was already
# in flight, and Python had imported the old module, so instances 2 and 3 ran
# the unfixed code anyway. Hence this file.

# %% imports
import sys
import numpy as np
sys.path.insert(0, "src")
from acopf_data import load
from qcac import no_load_cost
from acuc import solve_global, solve_qcac_iterative, cost_commitment, verify

CASE, RESERVE, NL_FRAC, VMIN, VMAX = "case118", 0.10, 0.15, 0.94, 1.06
GLOBAL_TL = 300.0

g = load(CASE)
g.vmin = np.full(g.n_bus, VMIN); g.vmax = np.full(g.n_bus, VMAX)
nl = no_load_cost(g, NL_FRAC)

def make_instance(seed):
    r = np.random.default_rng(seed)
    lvl = r.uniform(0.90, 1.10)
    return (g.pd*lvl*r.uniform(0.97, 1.03, g.n_bus),
            g.qd*lvl*r.uniform(0.95, 1.05, g.n_bus))

# %% run  (EXPENSIVE ~8 min per instance)
rows = []
for i in (2, 3):
    pd_, qd_ = make_instance(i)
    print(f"\n{'='*66}\ninstance {i}  (load {pd_.sum():.2f} pu)", flush=True)
    q = solve_qcac_iterative(g, nl, pd_, qd_, reserve=RESERVE)
    qt = cost_commitment(g, nl, pd_, qd_, q["u"], reserve=RESERVE)
    print(f"  QCAC iterative : {q['iters']} its, slack {q['slack']:.2e}, "
          f"{int(q['u'].sum())} units, {q['secs']:.0f}s -> true AC cost "
          f"{qt['cost']:,.1f}", flush=True)
    gl = solve_global(g, nl, pd_, qd_, reserve=RESERVE, time_limit=GLOBAL_TL,
                      start=qt)
    if gl is None:
        print("  global STILL failed"); continue
    agree = int(np.sum(q["u"] == gl["u"]))
    gap = 100*(qt["cost"] - gl["cost"])/gl["cost"]
    vq, vg = verify(g, nl, pd_, qd_, qt), verify(g, nl, pd_, qd_, gl)
    print(f"  global MINLP   : {int(gl['u'].sum())} units, {gl['secs']:.0f}s, "
          f"MIP gap {100*gl['mip_gap']:.3f}% -> cost {gl['cost']:,.1f}")
    print(f"  agreement {agree}/{g.n_gen}   QCAC gap {gap:+.4f}%")
    print(f"  verify: QCAC p-resid {vq['p_balance_max_resid']:.2e} V ok {vq['v_ok']} | "
          f"global p-resid {vg['p_balance_max_resid']:.2e} V ok {vg['v_ok']}")
    rows.append(dict(inst=i, load=float(pd_.sum()), q_cost=qt["cost"],
                     g_cost=gl["cost"], gap=gap, agree=agree,
                     q_secs=q["secs"], g_secs=gl["secs"],
                     g_mipgap=gl["mip_gap"], q_units=int(q["u"].sum()),
                     g_units=int(gl["u"].sum())))
np.save("results/rerun_2_3.npy", rows, allow_pickle=True)
print(f"\nsaved results/rerun_2_3.npy  ({len(rows)} instances)")

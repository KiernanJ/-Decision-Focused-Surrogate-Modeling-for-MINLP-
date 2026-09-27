# %% [markdown]
# # Global MINLP vs QCAC iterative, 5 instances
#
# Instance 0 gave a perfect match (164,101.4 both, 54/54 generators, 42s vs
# 600s). One instance proves nothing, so this repeats it on five more and
# verifies every solution independently.
#
# Both methods come from ONE model builder (`src/acuc.py`). The old codebase
# kept parallel copies that drifted apart, so a fix applied to one never reached
# the other -- here `build(exact=True/False)` is the only implementation and the
# two solvers differ solely in that flag.
#
# What gets checked per instance:
#   * QCAC's commitment re-costed in the TRUE AC model. The QCAC objective
#     includes its penalty term and is not a cost, so it cannot be compared
#     directly to anything.
#   * `verify()` recomputes power balance from the raw voltages with numpy,
#     not from the model's own expressions -- a sign error in the builder would
#     otherwise be confirmed by its own arithmetic.

# %% imports
import os, sys, time
import numpy as np
sys.path.insert(0, "src")
from acopf_data import load
from qcac import no_load_cost
from acuc import solve_global, solve_qcac_iterative, cost_commitment, verify

CASE, RESERVE, NL_FRAC = "case118", 0.10, 0.15
VMIN, VMAX = 0.94, 1.06
N_INST = 3
GLOBAL_TL = 300.0

grid = load(CASE); g = grid
g.vmin = np.full(g.n_bus, VMIN); g.vmax = np.full(g.n_bus, VMAX)
nl = no_load_cost(g, NL_FRAC)
print(f"{CASE}: {g.n_bus} bus / {g.n_gen} gen, V in [{VMIN}, {VMAX}]")

# %% instances 1..5  (instance 0 was done in 02/03)
def make_instance(seed):
    r = np.random.default_rng(seed)
    lvl = r.uniform(0.90, 1.10)          # one correlated system level
    return (g.pd * lvl * r.uniform(0.97, 1.03, g.n_bus),
            g.qd * lvl * r.uniform(0.95, 1.05, g.n_bus))

insts = [make_instance(s) for s in range(1, N_INST + 1)]
for i, (p, q) in enumerate(insts, 1):
    print(f"  instance {i}: load {p.sum():6.2f} pu")

# %% run both methods on every instance  (EXPENSIVE ~ 15 min per instance)
rows = []
for i, (pd_, qd_) in enumerate(insts, 1):
    print(f"\n{'='*66}\ninstance {i}  (load {pd_.sum():.2f} pu)")

    q = solve_qcac_iterative(g, nl, pd_, qd_, reserve=RESERVE)
    if q is None:
        print("  QCAC failed"); continue
    qt = cost_commitment(g, nl, pd_, qd_, q["u"], reserve=RESERVE)
    print(f"  QCAC iterative : {q['iters']} its, slack {q['slack']:.2e}, "
          f"{int(q['u'].sum())} units, {q['secs']:.0f}s  "
          f"-> true AC cost {qt['cost']:,.1f}" if qt else "  QCAC re-cost failed")

    # warm start from the EXACT-feasible re-costed point, not QCAC's relaxed
    # one -- a start that violates the equalities is rejected, and MIPFocus=2
    # will then not go looking for an incumbent (instance 2, 300s, 0 solutions)
    gl = solve_global(g, nl, pd_, qd_, reserve=RESERVE, time_limit=GLOBAL_TL,
                      u_start=(qt["u"] if qt else q["u"]),
                      pg_start=(qt["pg"] if qt else q["pg"]))
    if gl is None:
        print("  global failed"); continue
    print(f"  global MINLP   : {int(gl['u'].sum())} units, {gl['secs']:.0f}s, "
          f"MIP gap {100*gl['mip_gap']:.3f}%  -> cost {gl['cost']:,.1f}")

    vq = verify(g, nl, pd_, qd_, qt) if qt else None
    vg = verify(g, nl, pd_, qd_, gl)
    agree = int(np.sum(q["u"] == gl["u"]))
    gap = 100*(qt["cost"] - gl["cost"])/gl["cost"] if qt else np.nan
    print(f"  agreement {agree}/{g.n_gen} gens   QCAC gap vs global {gap:+.4f}%")
    print(f"  verify(global): p-resid {vg['p_balance_max_resid']:.2e}  "
          f"q-resid {vg['q_balance_max_resid']:.2e}  V ok {vg['v_ok']}  "
          f"reserve ok {vg['reserve_ok']}")
    if vq:
        print(f"  verify(QCAC)  : p-resid {vq['p_balance_max_resid']:.2e}  "
              f"q-resid {vq['q_balance_max_resid']:.2e}  V ok {vq['v_ok']}  "
              f"reserve ok {vq['reserve_ok']}")
    rows.append(dict(inst=i, load=float(pd_.sum()),
                     q_cost=qt["cost"] if qt else np.nan, g_cost=gl["cost"],
                     gap=gap, agree=agree, q_units=int(q["u"].sum()),
                     g_units=int(gl["u"].sum()), q_secs=q["secs"],
                     g_secs=gl["secs"], g_mipgap=gl["mip_gap"],
                     q_iters=q["iters"], q_slack=q["slack"],
                     vq_p=vq["p_balance_max_resid"] if vq else np.nan,
                     vg_p=vg["p_balance_max_resid"]))

# %% summary
print("\n" + "="*84)
print(f"{'inst':>5}{'load':>8}{'QCAC cost':>13}{'global cost':>13}{'gap':>9}"
      f"{'agree':>8}{'QCAC s':>8}{'glob s':>8}{'MIPgap':>9}")
print("-"*84)
for r in rows:
    print(f"{r['inst']:>5}{r['load']:>8.2f}{r['q_cost']:>13,.1f}"
          f"{r['g_cost']:>13,.1f}{r['gap']:>8.4f}%{r['agree']:>5}/{g.n_gen}"
          f"{r['q_secs']:>8.0f}{r['g_secs']:>8.0f}{100*r['g_mipgap']:>8.3f}%")
print("="*84)
if rows:
    gaps = np.array([r["gap"] for r in rows], float)
    ag = np.array([r["agree"] for r in rows])
    print(f"QCAC vs global: mean gap {np.nanmean(gaps):+.4f}%  "
          f"max |gap| {np.nanmax(np.abs(gaps)):.4f}%")
    print(f"commitment agreement: {ag.min()}-{ag.max()} of {g.n_gen} "
          f"({100*ag.mean()/g.n_gen:.1f}% mean)")
    print(f"speed: QCAC {np.mean([r['q_secs'] for r in rows]):.0f}s vs "
          f"global {np.mean([r['g_secs'] for r in rows]):.0f}s "
          f"({np.mean([r['g_secs'] for r in rows])/np.mean([r['q_secs'] for r in rows]):.1f}x)")
    print(f"max power-balance residual: QCAC {np.nanmax([r['vq_p'] for r in rows]):.2e}, "
          f"global {np.nanmax([r['vg_p'] for r in rows]):.2e}")
    np.save("results/compare_5_instances.npy", rows, allow_pickle=True)
    print("\nsaved results/compare_5_instances.npy")

# %% Reference pool for the hybrid-vehicle convex MINLP (original MINLP_Project
#    parameters): T=30 periods, z_t in {0,1,2,3} engine modes (S_MODES=3),
#    E_max=10, E_init~U(5,8) (drawn once), P_max=2, P_batt_max=3, tau=1,
#    gamma=0.1, eta=2, alpha_t~U(1,3), beta_t~U(0.5,1.5) (drawn once, fixed
#    across the pool). The instance parameter is the demand profile
#    D_t ~ U(0.5,2.5). References are solved to optimality by Gurobi.
#    Usage (sharded):  python 01_make_pool.py --i0 0 --i1 30 --out results/pool/pool_0.json
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json, time, argparse
import numpy as np, gurobipy as gp
from gurobipy import GRB
ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=30); ap.add_argument("--S", type=int, default=3)
ap.add_argument("--n", type=int, default=260); ap.add_argument("--tl", type=float, default=300.)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--i0", type=int, default=0); ap.add_argument("--i1", type=int, default=10**9)
ap.add_argument("--out", default="results/pool/pool.json")
a = ap.parse_args()
T, S = a.T, a.S
env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()

VEH = dict(eta=2.0, tau=1.0, gamma=0.1, E_max=10.0,
           E_init=float(np.random.default_rng(42).uniform(5.0, 8.0)),
           P_max=2.0, P_batt_max=3.0, S_MODES=S, T=T)
rng0 = np.random.default_rng(42)                      # vehicle, fixed across pool
VEH["alpha"] = rng0.uniform(1.0, 3.0, T).tolist()
VEH["beta"] = rng0.uniform(0.5, 1.5, T).tolist()
D = np.random.default_rng(a.seed).uniform(0.5, 2.5, (a.n, T))   # instance parameter
AL = np.array(VEH["alpha"]); BE = np.array(VEH["beta"])

def build(Dp, z_fixed=None, tl=300.):
    m = gp.Model(env=env); m.Params.TimeLimit = tl; m.Params.MIPGap = 1e-4
    E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"])
    Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = ({t: float(z_fixed[t]) for t in range(T)} if z_fixed is not None
         else m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER))
    m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S))
        m.addConstr(Pb[t] + Pe[t] >= Dp[t])
    m.setObjective(gp.quicksum(AL[t]*Pe[t]*Pe[t] + BE[t]*z[t] for t in range(T))
                   + VEH["eta"]*(VEH["E_max"]-E[T]), GRB.MINIMIZE)
    return m, z, E, Pe, Pb

REF, t0 = {}, time.time()
lo, hi = a.i0, min(a.i1, a.n)
for i in range(lo, hi):
    m, z, E, Pe, Pb = build(D[i], tl=a.tl); m.optimize()
    if m.SolCount == 0:
        REF[i] = None
        print(f"  [{i}] NO SOLUTION", flush=True); continue
    REF[i] = dict(cost=float(m.ObjVal), z=[int(round(z[t].X)) for t in range(T)],
                  E=[float(E[t].X) for t in range(T+1)],
                  Peng=[float(Pe[t].X) for t in range(T)],
                  Pbatt=[float(Pb[t].X) for t in range(T)],
                  secs=float(m.Runtime), mipgap=float(m.MIPGap))
    if (i-lo+1) % 20 == 0:
        print(f"  [{time.time()-t0:5.0f}s] {i-lo+1}/{hi-lo}", flush=True)
json.dump(dict(veh=VEH, D=D[lo:hi].tolist(), i0=lo, i1=hi,
               ref={str(k): v for k, v in REF.items()}), open(a.out, "w"))
good = [r for r in REF.values() if r]
sec = np.array([r["secs"] for r in good])
print(f"[{lo}:{hi}] {len(good)}/{hi-lo} solved  mean {sec.mean():.2f}s "
      f"median {np.median(sec):.2f}s max {sec.max():.1f}s  wall {time.time()-t0:.0f}s")

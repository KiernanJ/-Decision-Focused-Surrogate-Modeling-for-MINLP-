# %% FRESH VEHICLE TEST POOL (2026-10-11, audit #4): the old test ids were scored in earlier studies (02/03/04 MILP, vehicle/02-07 MINLP), so the
#    final vehicle method is scored ONCE on new profiles. MILP (Dixit sampler u ~ {100..2000}/1000): 100 NEW u per vehicle, drawn WITHOUT replacement
#    from the values NOT already in that vehicle's pool (the old pool's sampler draws with replacement from 1901 values, so identical profiles could
#    sit in train and test). MINLP: 48 new demand profiles D ~ U(0.5, 2.5) (new seed), same vehicle. References: exact solves, Threads=1,
#    TimeLimit 120 s (as the original pools), MIPGap 1e-4.   Usage: python dixit_vehicle/17_fresh_test_pool.py
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, multiprocessing as mp
VIDS, N_MILP, N_MINLP, TL = (0, 1, 2, 4, 6, 7, 8, 9), 100, 48, 120.0     # TL = the original pools' setting (600 s was 4x slower under load)
OUT = os.path.join(HERE, "results", "pool_fresh"); os.makedirs(OUT, exist_ok=True)


def milp_job(args):
    vid, u = args
    from dixit_vehicle import Vehicle
    V = Vehicle(vid); D = Vehicle.demand(u); ex = V.solve(D, tl=TL)
    return vid, dict(u=float(u), cost=ex["cost"], z=ex["z"].round().tolist(), Pe=ex["Pe"].tolist(), secs=ex["secs"], mipgap=ex["gap"])


def minlp_job(args):
    i, Dp, VEH = args
    import gurobipy as gp
    from gurobipy import GRB
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.setParam("Threads", 1); env.start()
    T, S = VEH["T"], VEH["S_MODES"]; AL = np.array(VEH["alpha"]); BE = np.array(VEH["beta"])
    m = gp.Model(env=env); m.Params.TimeLimit = TL; m.Params.MIPGap = 1e-4
    E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"]); Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER); m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0); m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t] + Pe[t] >= Dp[t])
    m.setObjective(gp.quicksum(AL[t]*Pe[t]*Pe[t] + BE[t]*z[t] for t in range(T)) + VEH["eta"]*(VEH["E_max"]-E[T]), GRB.MINIMIZE); m.optimize()
    if m.SolCount == 0: return i, None
    return i, dict(cost=float(m.ObjVal), z=[int(round(z[t].X)) for t in range(T)], Peng=[float(Pe[t].X) for t in range(T)], secs=float(m.Runtime), mipgap=float(m.MIPGap))


if __name__ == "__main__":
    t0 = time.time(); jobs = []; fresh_u = {}
    for vid in VIDS:
        P = json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json"))); used = {int(round(p["u"]*1000)) for p in P}
        avail = np.array(sorted(set(range(100, 2001)) - used)); rng = np.random.default_rng(90000 + vid)
        fresh_u[vid] = (rng.choice(avail, N_MILP, replace=False)/1000).tolist(); jobs += [(vid, u) for u in fresh_u[vid]]
        print(f"[MILP v{vid}] old pool has {len(used)} distinct u of 224 draws; {len(avail)} unused values; fresh u drawn from those", flush=True)
    from src_hv import load_pool
    VEH, Dm, RL = load_pool(os.path.join(HERE, "..", "vehicle", "results", "pool", "pool_*.json"))
    Dn = np.random.default_rng(20261011).uniform(0.5, 2.5, (N_MINLP, VEH["T"]))
    print(f"[MINLP] fresh D: min L-inf distance to the old pool {min(np.abs(Dm - d).max(1).min() for d in Dn):.3f}", flush=True)
    with mp.get_context("spawn").Pool(6) as pool:
        R = {}
        for vid in VIDS:
            R[vid] = [r for _, r in pool.imap(milp_job, [j for j in jobs if j[0] == vid])]     # saved per vehicle as it completes
            json.dump(R[vid], open(os.path.join(OUT, f"vehicle_{vid}.json"), "w"))
            g = np.array([r["mipgap"] for r in R[vid]]); s = np.array([r["secs"] for r in R[vid]])
            print(f"[MILP v{vid}] {len(R[vid])} refs, MIPGap>1e-4: {(g > 1e-4 + 1e-12).sum()}, median {np.median(s):.1f}s max {s.max():.0f}s  [{time.time()-t0:.0f}s]", flush=True)
        M = dict(pool.imap(minlp_job, [(k, Dn[k], VEH) for k in range(N_MINLP)]))
    json.dump(dict(D=Dn.tolist(), ref={str(k): v for k, v in M.items()}), open(os.path.join(OUT, "minlp.json"), "w"))
    good = [v for v in M.values() if v]; print(f"[MINLP] {len(good)}/{N_MINLP} solved, MIPGap>1e-4: {sum(v['mipgap'] > 1e-4 + 1e-12 for v in good)}, "
                                             f"median {np.median([v['secs'] for v in good]):.1f}s  [{time.time()-t0:.0f}s]", flush=True)

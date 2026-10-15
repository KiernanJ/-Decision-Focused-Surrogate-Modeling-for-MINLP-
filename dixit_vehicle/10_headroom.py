# %% HEADROOM CHECK (2026-10-10, VALIDATION instances only), BOTH hybrid-vehicle versions:
#    MILP  = Dixit et al.'s vehicle (src/dixit_vehicle.py), vehicles given on the command line;
#    MINLP = our convex MIQCQP vehicle (vehicle/01_make_pool.py model: quadratic battery loss + quadratic fuel cost).
#    Question: does a SHORT-BUDGET solve (no cuts, deterministic Gurobi WorkLimit) leave room for learned cuts?
#    Metrics vs the exact pool reference: gap mean / median, discrete error, continuous error, time, no-solution count.
#    Usage: python dixit_vehicle/10_headroom.py 1,2,4,9
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, gurobipy as gp
from gurobipy import GRB
from dixit_vehicle import Vehicle, T, SW, TAU, E0, EMAX, PMAX
WORKS = (0.01, 0.05, 0.2, 1.0)
ENV = gp.Env(empty=True); ENV.setParam("OutputFlag", 0); ENV.setParam("Threads", 1); ENV.start()


def milp_short(V, D, work):
    m = gp.Model(env=ENV); m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
    Pb = m.addVars(T, lb=-GRB.INFINITY); Pe = m.addVars(T, lb=0, ub=PMAX); E = m.addVars(T+1, lb=0, ub=EMAX)
    z = m.addVars(T, lb=0, ub=SW, vtype=GRB.INTEGER); m.addConstr(E[0] == E0)
    for i in range(T):
        m.addConstr(E[i+1] == E[i] - TAU*Pb[i]); m.addConstr(Pb[i] + Pe[i] >= D[i]); m.addConstr(Pe[i] <= z[i]*PMAX/SW)
    cost = (-V.cT*(EMAX - E[T]) + gp.quicksum(float(V.cE[i])*E[i] for i in range(T)) + gp.quicksum(float(V.cP[i])*Pe[i] + float(V.cZ[i])*z[i] for i in range(T)))
    m.setObjective(cost, GRB.MINIMIZE); t0 = time.time(); m.optimize()
    if m.SolCount == 0: return None
    return float(cost.getValue()), np.array([round(z[i].X) for i in range(T)]), np.array([Pe[i].X for i in range(T)]), time.time() - t0


def minlp_short(VEH, Dp, work):
    Tn, S = VEH["T"], VEH["S_MODES"]; AL = np.array(VEH["alpha"]); BE = np.array(VEH["beta"])
    m = gp.Model(env=ENV); m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
    E = m.addVars(Tn+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(Tn, lb=0, ub=VEH["P_max"]); Pb = m.addVars(Tn, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(Tn, lb=0, ub=S, vtype=GRB.INTEGER); m.addConstr(E[0] == VEH["E_init"])
    for t in range(Tn):
        m.addQConstr(E[t+1] - E[t] + VEH["tau"]*Pb[t] + VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t] + Pe[t] >= Dp[t])
    obj = gp.quicksum(AL[t]*Pe[t]*Pe[t] + BE[t]*z[t] for t in range(Tn)) + VEH["eta"]*(VEH["E_max"] - E[Tn])
    m.setObjective(obj, GRB.MINIMIZE); t0 = time.time(); m.optimize()
    if m.SolCount == 0: return None
    return float(m.ObjVal), np.array([round(z[t].X) for t in range(Tn)]), np.array([Pe[t].X for t in range(Tn)]), time.time() - t0


def report(label, ref_ex_ms, rows_by_w, n):
    print(f"== {label}: exact median {ref_ex_ms:.0f} ms ({n} validation instances)", flush=True)
    for w, (rows, nos) in rows_by_w.items():
        v = np.array(rows) if rows else np.full((1, 4), np.nan)
        print(f"   W={w:5.2f}: gap mean {np.nanmean(v[:,0]):.3f}% median {np.nanmedian(v[:,0]):.3f}% | discrete {np.nanmean(v[:,1]):.2f}% | "
              f"continuous {np.nanmean(v[:,2]):.2f}% | time median {1000*np.nanmedian(v[:,3]):.0f} ms | no solution {nos}/{n}", flush=True)


def metrics(r, J, zt, pt):
    c, z, pe, s = r
    return (100*abs(c - J)/abs(J), 100*np.abs(z - zt).sum()/max(zt.sum(), 1e-9), 100*np.abs(pe - pt).sum()/max(pt.sum(), 1e-9), s)


if __name__ == "__main__":
    # ---- MILP (Dixit vehicle)
    va = list(range(100, 124))
    for vid in [int(v) for v in sys.argv[1].split(",")]:
        P = json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json"))); V = Vehicle(vid, env=ENV)
        out = {}
        for w in WORKS:
            rows, nos = [], 0
            for i in va:
                r = milp_short(V, Vehicle.demand(P[i]["u"]), w)
                if r is None: nos += 1; continue
                rows.append(metrics(r, P[i]["cost"], np.array(P[i]["z"]), np.array(P[i]["Pe"])))
            out[w] = (rows, nos)
        report(f"MILP Dixit vehicle {vid}", 1000*np.median([P[i]["secs"] for i in va]), out, len(va))
    # ---- MINLP (our convex MIQCQP vehicle)
    from src_hv import load_pool, splits
    VEH, D, REF = load_pool(os.path.join(HERE, "..", "vehicle", "results", "pool", "pool_*.json"))
    te, vaM, trM = splits(len(REF)); vaM = [i for i in vaM if REF[i] is not None]
    out = {}
    for w in WORKS:
        rows, nos = [], 0
        for i in vaM:
            r = minlp_short(VEH, D[i], w)
            if r is None: nos += 1; continue
            rows.append(metrics(r, REF[i]["cost"], np.array(REF[i]["z"]), np.array(REF[i]["Peng"])))
        out[w] = (rows, nos)
    report("MINLP (convex MIQCQP) vehicle", 1000*np.median([REF[i]["secs"] for i in vaM]), out, len(vaM))

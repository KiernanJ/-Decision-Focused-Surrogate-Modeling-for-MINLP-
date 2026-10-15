# %% VEHICLE stage 1 (2026-10-10, VALIDATION only): IMITATION-trained per-step INTEGER cuts (label-free teacher = our own
#    W=1.0 solves on training profiles, 11_teacher.py). Cut-net: MLP D -> softmax over levels {0..3} per step; steps with
#    max prob >= c get the integer cut z_t = level (both bounds). c in {0.999,0.99,0.95,0.9,0.8} chosen by a LABEL-FREE
#    validation score (deployed cost / relaxation cost). Deploy = one short solve, W = 0.01. Both MILP (8 vehicles) and
#    MINLP. Four metrics on validation are REPORT ONLY.  Usage: python dixit_vehicle/12_fixcuts_stage1.py
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn, gurobipy as gp
from gurobipy import GRB
from importlib import import_module
H = import_module("10_headroom")
import runlog
from dixit_vehicle import Vehicle, T, SW, TAU, E0, EMAX, PMAX
W, CS = 0.01, (0.999, 0.99, 0.95, 0.9, 0.8)


def milp(V, D, work, fix=None, relax=False, bnd=None, rows=None):
    m = gp.Model(env=H.ENV); m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
    Pb = m.addVars(T, lb=-GRB.INFINITY); Pe = m.addVars(T, lb=0, ub=PMAX); E = m.addVars(T+1, lb=0, ub=EMAX)
    z = m.addVars(T, lb=0, ub=SW, vtype=GRB.CONTINUOUS if relax else GRB.INTEGER); m.addConstr(E[0] == E0)
    for i in range(T):
        m.addConstr(E[i+1] == E[i] - TAU*Pb[i]); m.addConstr(Pb[i] + Pe[i] >= D[i]); m.addConstr(Pe[i] <= z[i]*PMAX/SW)
        if fix is not None and fix[i] >= 0: z[i].lb = z[i].ub = float(fix[i])          # integer cut z_t = level
        if bnd is not None: z[i].lb, z[i].ub = float(bnd[0][i]), float(bnd[1][i])       # integer cuts lb_t <= z_t <= ub_t
    if rows is not None:                                                        # general cut rows A[Pe; z] <= b, SOFT (priced 1e3)
        sr = m.addVars(len(rows[1]), lb=0)
        for r in range(len(rows[1])): m.addConstr(gp.quicksum(float(rows[0][r, j])*Pe[j] + float(rows[0][r, T + j])*z[j] for j in range(T)) <= float(rows[1][r]) + sr[r])
    cost = (-V.cT*(EMAX - E[T]) + gp.quicksum(float(V.cE[i])*E[i] for i in range(T)) + gp.quicksum(float(V.cP[i])*Pe[i] + float(V.cZ[i])*z[i] for i in range(T)))
    m.setObjective(cost + (1e3*sr.sum() if rows is not None else 0), GRB.MINIMIZE); t0 = time.time(); m.optimize()
    if m.SolCount == 0: return None
    return float(cost.getValue()), np.array([round(z[i].X) for i in range(T)]), np.array([Pe[i].X for i in range(T)]), time.time() - t0


def minlp(VEH, Dp, work, fix=None, relax=False, bnd=None, rows=None):
    Tn, S = VEH["T"], VEH["S_MODES"]; AL = np.array(VEH["alpha"]); BE = np.array(VEH["beta"])
    m = gp.Model(env=H.ENV); m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
    E = m.addVars(Tn+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(Tn, lb=0, ub=VEH["P_max"]); Pb = m.addVars(Tn, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(Tn, lb=0, ub=S, vtype=GRB.CONTINUOUS if relax else GRB.INTEGER); m.addConstr(E[0] == VEH["E_init"])
    for t in range(Tn):
        m.addQConstr(E[t+1] - E[t] + VEH["tau"]*Pb[t] + VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t] + Pe[t] >= Dp[t])
        if fix is not None and fix[t] >= 0: z[t].lb = z[t].ub = float(fix[t])
        if bnd is not None: z[t].lb, z[t].ub = float(bnd[0][t]), float(bnd[1][t])
    obj = gp.quicksum(AL[t]*Pe[t]*Pe[t] + BE[t]*z[t] for t in range(Tn)) + VEH["eta"]*(VEH["E_max"] - E[Tn])
    if rows is not None:
        sr = m.addVars(len(rows[1]), lb=0)
        for r in range(len(rows[1])): m.addConstr(gp.quicksum(float(rows[0][r, j])*Pe[j] + float(rows[0][r, Tn + j])*z[j] for j in range(Tn)) <= float(rows[1][r]) + sr[r])
    m.setObjective(obj + (1e3*sr.sum() if rows is not None else 0), GRB.MINIMIZE); t0 = time.time(); m.optimize()
    if m.SolCount == 0: return None
    return float(obj.getValue()), np.array([round(z[t].X) for t in range(Tn)]), np.array([Pe[t].X for t in range(Tn)]), time.time() - t0


def cutnet(Xtr, Ztr, seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, Ztr.shape[1]*(SW+1)))
    o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); Y = torch.tensor(Ztr, dtype=torch.long)
    for _ in range(1500):
        o.zero_grad(); l = nn.functional.cross_entropy(m(Xtr).view(-1, SW+1), Y.view(-1)); l.backward(); o.step()
    return m.eval()


def run_case(label, solve, Dall, tr, va, TG, REFc, REFz, REFp):
    X = torch.tensor(Dall, dtype=torch.float32); mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    trk = [i for i in tr if TG.get(i) is not None]
    net = cutnet(Xn[trk], np.stack([TG[i]["z"] for i in trk]))
    with torch.no_grad(): Pr = torch.softmax(net(Xn).view(len(Dall), -1, SW+1), -1).numpy()
    SCv = {i: abs(solve(Dall[i], 1e6, relax=True)[0]) for i in va}               # label-free normaliser (relaxation)
    fixv = lambda i, c: np.where(Pr[i].max(-1) >= c, Pr[i].argmax(-1), -1)
    def deploy(c):
        return {i: solve(Dall[i], W, fix=None if c is None else fixv(i, c)) for i in va}
    scores = {}
    for c in (None,) + CS:
        R = deploy(c); scores[c] = (float(np.mean([R[i][0]/SCv[i] if R[i] else 10.0 for i in va])), R)
    cbest = min(CS, key=lambda c: scores[c][0])
    def four(R):
        v = np.array([(100*abs(R[i][0]-REFc[i])/abs(REFc[i]), 100*np.abs(R[i][1]-REFz[i]).sum()/max(REFz[i].sum(), 1e-9),
                       100*np.abs(R[i][2]-REFp[i]).sum()/max(REFp[i].sum(), 1e-9), R[i][3]) if R[i] else (np.nan,)*4 for i in va])
        return v
    out = {}
    for nm, c in (("no cuts", None), (f"imitation integer cuts (c={cbest}, label-free)", cbest)):
        v = four(scores[c][1]); nfix = np.mean([(fixv(i, c) >= 0).sum() for i in va]) if c else 0
        out[nm] = (np.nanmean(v[:, 0]), np.nanmedian(v[:, 0]), np.nanmean(v[:, 1]), np.nanmean(v[:, 2]), 1000*np.nanmedian(v[:, 3]), nfix)
    print(f"== {label} (validation {len(va)}, W={W})", flush=True)
    for nm, r in out.items():
        print(f"   {nm:44s} | gap mean {r[0]:.3f}% | median {r[1]:.3f}% | discrete {r[2]:.2f}% | continuous {r[3]:.2f}% | time {r[4]:.0f} ms | steps fixed {r[5]:.1f}/{T}", flush=True)
    return out


if __name__ == "__main__":
    RUN = runlog.start("vehicle_stage1_fixcuts"); allout = {}
    for vid in (0, 1, 2, 4, 6, 7, 8, 9):
        P = json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json"))); V = Vehicle(vid, env=H.ENV)
        TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", f"milp_v{vid}_w1.json"))).items()}
        Dall = np.stack([Vehicle.demand(p["u"]) for p in P]); va, tr = list(range(100, 124)), list(range(124, 224))
        allout[f"milp_v{vid}"] = run_case(f"MILP vehicle {vid}", lambda D, w, fix=None, relax=False: milp(V, D, w, fix, relax), Dall, tr, va, TG,
                                          {i: P[i]["cost"] for i in va}, {i: np.array(P[i]["z"]) for i in va}, {i: np.array(P[i]["Pe"]) for i in va})
    from src_hv import load_pool, splits
    VEH, Dm, REF = load_pool(os.path.join(HERE, "..", "vehicle", "results", "pool", "pool_*.json")); te, va, tr = splits(len(REF)); va = [i for i in va if REF[i]]
    TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", "minlp_w1.json"))).items()}
    allout["minlp"] = run_case("MINLP vehicle", lambda D, w, fix=None, relax=False: minlp(VEH, D, w, fix, relax), Dm, tr, va, TG,
                               {i: REF[i]["cost"] for i in va}, {i: np.array(REF[i]["z"]) for i in va}, {i: np.array(REF[i]["Peng"]) for i in va})
    runlog.save(RUN, "summary.pkl", allout)
    for k, o in allout.items():
        for nm, r in o.items():
            runlog.add_result(RUN, f"VEHICLE {k} VALIDATION W={W} {nm}: gap mean | median | disc | cont | time = {r[0]:.3f}% | {r[1]:.3f}% | {r[2]:.2f}% | {r[3]:.2f}% | {r[4]:.0f} ms")

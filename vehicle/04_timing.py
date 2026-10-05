# %% Per-instance deployment time for the hybrid-vehicle case (time-CDF panel):
#    network forward -> relaxation with the learned cut -> L1 projection ->
#    continuous re-solve, single thread. Reference = Gurobi solve times stored
#    in the pool. Uses checkpoints/vehicle_s{seed}.pt when present; the paper's
#    timing used freshly initialised nets of the same size (solve time does not
#    depend on the weights' values).
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import os, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import gurobipy as gp
from gurobipy import GRB
from src_hv import load_pool, splits, HVRelax
from cutnet_hv import CutNetHV, rows_torch

# %% data, relaxation, splits
VEH, D, REF = load_pool("results/pool/pool_*.json"); T, S = VEH["T"], VEH["S_MODES"]
R = HVRelax(VEH, n_cuts=4)
env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
te, va, tr = splits(len(REF)); te = [i for i in te if REF[i]]
X = torch.tensor(D/D.mean(), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
W = {}
for i in te:
    r = R.solve(D[i])
    if r is not None:
        W[i] = np.concatenate([r["E"], r["Peng"], r["Pbatt"], r["z"]])
ids = [i for i in te if i in W]
WR = torch.tensor(np.stack([W[i] for i in ids]), dtype=torch.float32)


# %% L1 projection restoration (same as 02_train.py)
def qz(Dp, zf, tl=60.):
    m = gp.Model(env=env); m.Params.TimeLimit = tl
    E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"])
    Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER); w = m.addVars(T, lb=0)
    m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t]+Pe[t] >= Dp[t])
        m.addConstr(w[t] >= z[t]-float(zf[t])); m.addConstr(w[t] >= float(zf[t])-z[t])
    m.setObjective(gp.quicksum(w[t] for t in range(T)), GRB.MINIMIZE); m.optimize()
    return None if m.SolCount == 0 else np.array([round(z[t].X) for t in range(T)], float)


# %% time the full deployment, 4 seeds
TT = []
for seed in (0, 1, 2, 3):
    net = CutNetHV(T, k_rows=2, hidden=16, seed=seed, learn_thr=False)
    ck = f"checkpoints/vehicle_s{seed}.pt"
    if os.path.exists(ck):
        net.load_state_dict(torch.load(ck, weights_only=False)["net"])
    net.eval()
    for k, i in enumerate(ids):
        t0 = time.time()
        with torch.no_grad():
            hh = net(Xn[[i]]); A, b = rows_torch(*hh[:7], WR[[k]])
        r1 = R.solve(D[i], A=A[0].numpy().astype(float), b=b[0].numpy().astype(float), cut_cap=0.0)
        if r1 is None: continue
        zq = qz(D[i], r1["z"])
        if zq is None: continue
        R.solve(D[i], z_lo=zq, z_hi=zq)
        TT.append(time.time()-t0)
ex = [r["secs"] for r in REF if r]
json.dump(dict(deploy=TT, reference=ex), open("results/timing_hv.json", "w"))
print(f"deploy median {np.median(TT):.4f}s | exact MINLP median {np.median(ex):.3f}s "
      f"-> {np.median(ex)/np.median(TT):.0f}x")

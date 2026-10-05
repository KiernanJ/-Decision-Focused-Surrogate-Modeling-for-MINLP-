# %% NN PROXY baseline for the hybrid-vehicle case study.
#    Standard optimization proxy: D -> net -> z directly, no optimizer in the
#    loop, then the SAME L1 projection restoration our method uses, so the only
#    difference is whether a convex solve sits inside the map.
#
#    SUPERVISED on z* (MSE), exactly as run_hybrid_vehicle.py's std_nn arm --
#    which makes it a STRONGER baseline than ours, since our method never sees
#    a label. That asymmetry is stated rather than hidden.
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from src_hv import load_pool, splits, HVRelax
ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="results/pool/pool_*.json")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--hidden", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-2)
a = ap.parse_args()
torch.manual_seed(a.seed)
VEH, D, REF = load_pool(a.pool); T, S = VEH["T"], VEH["S_MODES"]
R = HVRelax(VEH, n_cuts=4)
env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
te, va, tr = splits(len(REF))
te = [i for i in te if REF[i]]; tr = [i for i in tr if REF[i]]

def qz(Dp, zf, tl=60.):
    m = gp.Model(env=env); m.Params.TimeLimit = tl
    E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"])
    Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER); w = m.addVars(T, lb=0)
    m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S))
        m.addConstr(Pb[t] + Pe[t] >= Dp[t])
        m.addConstr(w[t] >= z[t] - float(zf[t])); m.addConstr(w[t] >= float(zf[t]) - z[t])
    m.setObjective(gp.quicksum(w[t] for t in range(T)), GRB.MINIMIZE); m.optimize()
    return None if m.SolCount == 0 else np.array([round(z[t].X) for t in range(T)], float)

X = torch.tensor(D/D.mean(), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
Y = torch.tensor(np.array([REF[i]["z"] for i in tr]), dtype=torch.float32)
net = nn.Sequential(nn.Linear(T, a.hidden), nn.ReLU(),
                    nn.Linear(a.hidden, a.hidden), nn.ReLU(), nn.Linear(a.hidden, T))
opt = torch.optim.Adam(net.parameters(), lr=a.lr); lf = nn.MSELoss()
print(f"[proxy s{a.seed}] {sum(p.numel() for p in net.parameters()):,} params, "
      f"SUPERVISED on z*, {len(tr)} train", flush=True)
for ep in range(a.epochs):
    opt.zero_grad(); loss = lf(net(Xn[tr]), Y); loss.backward(); opt.step()
    if ep % 50 == 0: print(f"  ep {ep:3d} mse {loss.item():.5f}", flush=True)
net.eval()
rows, t0 = [], time.time()
with torch.no_grad():
    ZP = net(Xn[te]).numpy()
for k, i in enumerate(te):
    t1 = time.time()
    zq = qz(D[i], np.clip(ZP[k], 0, S))
    el = time.time()-t1
    if zq is None: continue
    pr = R.solve(D[i], z_lo=zq, z_hi=zq)
    if pr is None: continue
    zt = np.array(REF[i]["z"]); pt = np.array(REF[i]["Peng"])
    rows.append(dict(gap=100*(pr["cost"]-REF[i]["cost"])/abs(REF[i]["cost"]),
                     disc=100*float(np.abs(zq-zt).sum()/max(zt.sum(), 1e-9)),
                     cont=100*float(np.abs(pr["Peng"]-pt).sum()/max(pt.sum(), 1e-9)),
                     secs=el))
v = np.array([[r["gap"], r["disc"], r["cont"], r["secs"]] for r in rows])
print(f"[TEST] NN proxy n={len(v)}  gap {v[:,0].mean():+.4f}%  disc {v[:,1].mean():.2f}%  "
      f"cont {v[:,2].mean():.2f}%  {v[:,3].mean():.3f}s/inst")
json.dump(dict(seed=a.seed, rows=rows, gap_mean=float(v[:,0].mean()),
               disc_mean=float(v[:,1].mean()), cont_mean=float(v[:,2].mean())),
          open(f"results/proxy_s{a.seed}.json", "w"), indent=1)

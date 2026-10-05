"""Critical 1: vehicle baselines the paper does not report, all on the same 48 test
instances and the same restoration as the method (qz_restore, copied from
vehicle/02_train.py: L1 projection to a feasible integer schedule, then re-solve).

  no-cut      uncut convex relaxation -> qz -> re-solve   (no learning at all)
  constant    train-mean reference z  -> qz -> re-solve
  ridge       ridge regression D -> z (lambda=10) -> qz -> re-solve
  paper proxy vehicle/03_nn_proxy.py's configuration (hidden 16, lr 1e-2, 200 steps), 2 seeds

Compare with the committed method rows (vehicle/results/train_s*.json) printed at the end.
Needs a Gurobi licence (the pip size-limited licence is enough). Runtime: ~2-5 min.
"""
import glob
import json
import _common
from _common import REPO
import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from src_hv import load_pool, splits, HVRelax

torch.set_num_threads(1)
VEH, D, REF = load_pool(f"{REPO}/vehicle/results/pool/pool_*.json")
T, S = VEH["T"], VEH["S_MODES"]
R = HVRelax(VEH, n_cuts=4)
te, va, tr = splits(len(REF))
ENV = gp.Env(empty=True); ENV.setParam("OutputFlag", 0); ENV.setParam("Threads", 1); ENV.start()


def qz(Dp, zf, tl=60.):
    """Same model as qz_restore in vehicle/02_train.py."""
    m = gp.Model(env=ENV); m.Params.TimeLimit = tl
    E = m.addVars(T + 1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"])
    Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER); w = m.addVars(T, lb=0)
    m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t + 1] - E[t] + VEH["tau"] * Pb[t] + VEH["gamma"] * Pb[t] * Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t] * (VEH["P_max"] / S))
        m.addConstr(Pb[t] + Pe[t] >= Dp[t])
        m.addConstr(w[t] >= z[t] - float(zf[t])); m.addConstr(w[t] >= float(zf[t]) - z[t])
    m.setObjective(gp.quicksum(w[t] for t in range(T)), GRB.MINIMIZE); m.optimize()
    return None if m.SolCount == 0 else np.array([round(z[t].X) for t in range(T)], float)


def evaluate(zpred):
    G, Dd, C = [], [], []
    for k, i in enumerate(te):
        zq = qz(D[i], np.clip(zpred[k], 0, S)); pr = R.solve(D[i], z_lo=zq, z_hi=zq)
        zt = np.array(REF[i]["z"]); pt = np.array(REF[i]["Peng"])
        G.append(100 * (pr["cost"] - REF[i]["cost"]) / abs(REF[i]["cost"]))
        Dd.append(100 * np.abs(zq - zt).sum() / zt.sum())
        C.append(100 * np.abs(pr["Peng"] - pt).sum() / pt.sum())
    return np.abs(G).mean(), np.mean(G), np.mean(Dd), np.mean(C)


def show(name, r):
    print(f"{name:28s} |gap| {r[0]:.4f}%  signed {r[1]:+.4f}%  disc {r[2]:.2f}%  cont {r[3]:.2f}%", flush=True)


out = {}
out["no-cut relaxation"] = evaluate([R.solve(D[i])["z"] for i in te]); show("no-cut relaxation", out["no-cut relaxation"])
X = torch.tensor(D / D.mean(), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu) / sg
Y = torch.tensor(np.array([REF[i]["z"] for i in range(len(REF))]), dtype=torch.float32)
out["constant train-mean z"] = evaluate(np.tile(Y[tr].mean(0).numpy(), (len(te), 1)))
show("constant train-mean z", out["constant train-mean z"])
Xa = np.c_[Xn[tr].numpy(), np.ones(len(tr))]
Wr = np.linalg.solve(Xa.T @ Xa + 10.0 * np.eye(Xa.shape[1]), Xa.T @ Y[tr].numpy())
out["ridge (lambda=10)"] = evaluate(np.c_[Xn[te].numpy(), np.ones(len(te))] @ Wr)
show("ridge (lambda=10)", out["ridge (lambda=10)"])
for seed in (0, 1):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(T, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, T))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad(); loss = nn.functional.mse_loss(net(Xn[tr]), Y[tr]); loss.backward(); opt.step()
    with torch.no_grad():
        out[f"paper proxy s{seed}"] = evaluate(net(Xn[te]).numpy())
    show(f"paper proxy s{seed}", out[f"paper proxy s{seed}"])
ours = [json.load(open(f)) for f in sorted(glob.glob(f"{REPO}/vehicle/results/train_s*.json"))]
g = np.array([r["gap"] if isinstance(r, dict) else r[0] for d in ours for r in d["rows"]])
print(f"{'method (committed, 4 seeds)':28s} |gap| {np.abs(g).mean():.4f}%  signed {g.mean():+.4f}%")
json.dump({k: list(map(float, v)) for k, v in out.items()},
          open(f"{_common.OUT}/c1_vehicle_baselines.json", "w"), indent=1)

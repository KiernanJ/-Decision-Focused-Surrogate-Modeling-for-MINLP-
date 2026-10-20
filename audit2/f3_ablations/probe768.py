# probe: raw MIQCQP commitment (before miqcqp's internal upward repair) for blow-up instances of A0 / A4
import os, sys, pickle
import os
REPO = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO + "/src")
import numpy as np, torch, torch.nn as nn
os.environ["DEPLOY_RHO"] = "3e5"
import combopt_ac as C, data2
from pipeline import grid
from cutnet_k import CutNetK, rows_torch
from vnetio import vnet_out
from gurobipy import GRB
import gurobipy as gp
from acuc import build
CASE = "case118"; HERE = REPO + "/acuc"
C.init(CASE, True)
g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
st = pickle.load(open("res_case118_fresh5.pkl", "rb")); dep = st["meta"]["scored"]
OUT, RHO, _ = vnet_out(HERE + "/results/vnet_layer_case118_env_nc.pt", Xn[dep], n)
UC = torch.load(HERE + "/results/unit_cuts_layer_case118_soft_lfc.pt", weights_only=False); c = UC["c"]
cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G)); cm.load_state_dict(UC["state"]); cm.eval()
with torch.no_grad(): Mc = torch.sigmoid(cm(Xn)).numpy()
for i in (768, 733):
    k = dep.index(i); V = (1.0 + OUT[k, :n], OUT[k, n:])
    on = [j for j in range(G) if Mc[i, j] > c]; off = [j for j in range(G) if Mc[i, j] < 1 - c]
    p, q = C._W["inst"][i]
    m, Xv, cost = build(g, nl, p, q, GRB.BINARY, exact=False, rho=RHO, Vr0=V[0], Vi0=V[1], threads=1)
    s = m.addVars(len(on) + len(off), lb=0)
    for r, j in enumerate(on): m.addConstr(-Xv["u"][j] <= -1 + s[r])
    for r, j in enumerate(off): m.addConstr(Xv["u"][j] <= 0 + s[len(on) + r])
    m.setObjective(m.getObjective() + 1e6*gp.quicksum(s.values()))
    m.Params.OutputFlag = 0; m.Params.Threads = 1; m.Params.TimeLimit = 1e6; m.Params.WorkLimit = 10; m.Params.MIPGap = 1e-4
    m.optimize()
    if m.SolCount == 0:
        print(f"#{i}: no incumbent in W=10 (miqcqp -> label-free fallback)"); continue
    u = np.array([round(Xv["u"][j].X) for j in range(G)]); viol = sum(s[r].X > 1e-6 for r in s)
    pr = C._price(i, u.astype(float))
    print(f"#{i}: raw MIQCQP units on {int(u.sum())}, fixings on/off {len(on)}/{len(off)}, violated fixings {viol}, MIP gap {m.MIPGap:.3g}, raw commitment prices: {'YES' if pr is not None else 'NO -> merit-order repair'}; ref units on {int(np.sum(st['meta']['SR'][i][5]))}")

# %% Vehicle: test metrics + per-instance timing for the method and the baselines,
#    every solver single-threaded (audit 2026-10-06).
#      ours       trained checkpoint: anchor solve (uncut relaxation) -> cut relaxation
#                 -> L1 projection (MIQCP) -> continuous re-solve
#      proxy      MLP on z*, hidden/lr/epochs chosen on VALIDATION MSE, same projection
#      ridge      ridge regression on z*, alpha on validation MSE, same projection
#      nocut      uncut relaxation -> same projection           (control)
#      constant   training-mean schedule -> same projection     (control)
#      exact      Gurobi on the MINLP, Threads=1 (reference time)
#    Usage (cwd = vehicle/):  python 05_baselines_timing.py
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import os, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from src_hv import load_pool, splits, HVRelax
from cutnet_hv import CutNetHV, rows_torch
torch.set_num_threads(1)

# %% data
VEH, D, REF = load_pool("results/pool/pool_*.json"); T, S = VEH["T"], VEH["S_MODES"]
R = HVRelax(VEH, n_cuts=4)
env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.setParam("Threads", 1); env.start()
te, va, tr = splits(len(REF))
X = torch.tensor(D/D.mean(), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
Z = {i: np.array(REF[i]["z"], float) for i in range(len(REF)) if REF[i]}
AL = np.array(VEH["alpha"]); BE = np.array(VEH["beta"])


def model(Dp, zf=None, tl=300.):
    """zf given -> L1 projection; zf None -> the exact MINLP."""
    m = gp.Model(env=env); m.Params.TimeLimit = tl
    E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"])
    Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER)
    m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t]+Pe[t] >= Dp[t])
    if zf is None:
        m.Params.MIPGap = 1e-4
        m.setObjective(gp.quicksum(AL[t]*Pe[t]*Pe[t] + BE[t]*z[t] for t in range(T))
                       + VEH["eta"]*(VEH["E_max"]-E[T]), GRB.MINIMIZE)
    else:
        w = m.addVars(T, lb=0)
        for t in range(T):
            m.addConstr(w[t] >= z[t]-float(zf[t])); m.addConstr(w[t] >= float(zf[t])-z[t])
        m.setObjective(gp.quicksum(w[t] for t in range(T)), GRB.MINIMIZE)
    m.optimize()
    return None if m.SolCount == 0 else np.array([round(z[t].X) for t in range(T)], float)


def finish(i, zf, t_pre):
    """projection + continuous re-solve; returns a row (failures counted)."""
    t0 = time.time()
    zq = model(D[i], np.clip(zf, 0, S))
    pr = R.solve(D[i], z_lo=zq, z_hi=zq) if zq is not None else None
    secs = t_pre + time.time() - t0
    if pr is None:
        return dict(idx=i, ok=False, secs=secs)
    zt = Z[i]; pt = np.array(REF[i]["Peng"])
    return dict(idx=i, ok=True, secs=secs,
                gap=100*(pr["cost"]-REF[i]["cost"])/abs(REF[i]["cost"]),
                disc=100*float(np.abs(zq-zt).sum()/max(zt.sum(), 1e-9)),
                cont=100*float(np.abs(pr["Peng"]-pt).sum()/max(pt.sum(), 1e-9)))


def summ(rows):
    ok = [r for r in rows if r["ok"]]
    return dict(n=len(rows), failed=len(rows)-len(ok),
                gap=float(np.mean([r["gap"] for r in ok])), disc=float(np.mean([r["disc"] for r in ok])),
                cont=float(np.mean([r["cont"] for r in ok])), median_secs=float(np.median([r["secs"] for r in rows])))


OUT = {}
# %% exact reference, single thread
t_ex = []
for i in te:
    t0 = time.time(); model(D[i]); t_ex.append(time.time()-t0)
OUT["exact"] = dict(median_secs=float(np.median(t_ex)), secs=t_ex)
print(f"[exact] median {np.median(t_ex):.3f}s (Threads=1)", flush=True)

# %% ours, 4 seeds
OUT["ours"] = {}
for seed in (0, 1, 2, 3):
    net = CutNetHV(T, k_rows=2, hidden=16, seed=seed, learn_thr=False)
    net.load_state_dict(torch.load(f"checkpoints/vehicle_s{seed}.pt", weights_only=False)["net"])
    net.eval(); rows = []
    for i in te:
        t0 = time.time()
        r0 = R.solve(D[i])                                   # anchor (uncut relaxation)
        w = torch.tensor(np.concatenate([r0["E"], r0["Peng"], r0["Pbatt"], r0["z"]])[None], dtype=torch.float32)
        with torch.no_grad():
            hh = net(Xn[[i]]); A, b = rows_torch(*hh[:7], w)
        r1 = R.solve(D[i], A=A[0].numpy().astype(float), b=b[0].numpy().astype(float), cut_cap=0.0)
        if r1 is None:
            rows.append(dict(idx=i, ok=False, secs=time.time()-t0)); continue
        rows.append(finish(i, r1["z"], time.time()-t0))
    OUT["ours"][seed] = dict(rows=rows, **summ(rows))
    print(f"[ours s{seed}] {summ(rows)}", flush=True)

# %% controls without learning
for name in ("nocut", "constant"):
    rows = []
    zbar = np.mean([Z[i] for i in tr], 0)
    for i in te:
        t0 = time.time()
        zf = R.solve(D[i])["z"] if name == "nocut" else zbar
        rows.append(finish(i, zf, time.time()-t0))
    OUT[name] = dict(rows=rows, **summ(rows)); print(f"[{name}] {summ(rows)}", flush=True)

# %% ridge, alpha on validation MSE
Xtr = np.c_[Xn[tr].numpy(), np.ones(len(tr))]; Ytr = np.stack([Z[i] for i in tr])
Xva = np.c_[Xn[va].numpy(), np.ones(len(va))]; Yva = np.stack([Z[i] for i in va])
best = None
for al in (1e-3, 1e-2, 1e-1, 1, 10, 100):
    Wt = np.linalg.solve(Xtr.T@Xtr + al*np.eye(Xtr.shape[1]), Xtr.T@Ytr)
    v = float(((Xva@Wt - Yva)**2).mean())
    if best is None or v < best[0]: best = (v, al, Wt)
rows = []
for i in te:
    t0 = time.time(); zf = np.r_[Xn[i].numpy(), 1.0] @ best[2]
    rows.append(finish(i, zf, time.time()-t0))
OUT["ridge"] = dict(alpha=best[1], rows=rows, **summ(rows)); print(f"[ridge a={best[1]}] {summ(rows)}", flush=True)

# %% MLP proxy, hyperparameters on validation MSE, 4 seeds
OUT["proxy"] = {}
Yt = torch.tensor(Ytr, dtype=torch.float32); Yv = torch.tensor(Yva, dtype=torch.float32)
for seed in (0, 1, 2, 3):
    best = None
    for h in (16, 64, 256):
        for lr in (1e-3, 1e-2):
            for ep in (500, 2000):
                torch.manual_seed(seed)
                m = nn.Sequential(nn.Linear(T, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, T))
                o = torch.optim.Adam(m.parameters(), lr=lr)
                for _ in range(ep):
                    o.zero_grad(); l = ((m(Xn[tr])-Yt)**2).mean(); l.backward(); o.step()
                with torch.no_grad():
                    v = float(((m(Xn[va])-Yv)**2).mean())
                if best is None or v < best[0]: best = (v, m, dict(hidden=h, lr=lr, epochs=ep))
    m = best[1].eval(); rows = []
    for i in te:
        t0 = time.time()
        with torch.no_grad():
            zf = m(Xn[[i]])[0].numpy()
        rows.append(finish(i, zf, time.time()-t0))
    OUT["proxy"][seed] = dict(cfg=best[2], rows=rows, **summ(rows))
    print(f"[proxy s{seed} {best[2]}] {summ(rows)}", flush=True)

json.dump(OUT, open("results/audit_vehicle.json", "w"), indent=1, default=float)
print("wrote results/audit_vehicle.json")

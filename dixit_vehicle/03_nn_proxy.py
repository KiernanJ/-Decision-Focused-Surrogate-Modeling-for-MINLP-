# %% Supervised NN proxy for the Dixit vehicle: D -> z* (MSE) on the same 100 training
#    profiles, hyperparameters on validation MSE, then the SAME L1 projection (no cut) and
#    continuous re-solve as the learned-cut method. Usage: python 03_nn_proxy.py --vid 0
import os, sys, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, torch, torch.nn as nn
from dixit_vehicle import Vehicle, T, SW
torch.set_num_threads(1)
ap = argparse.ArgumentParser(); ap.add_argument("--vid", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
V = Vehicle(a.vid); P = json.load(open(f"results/pool/vehicle_{a.vid}.json"))
te, va, tr = list(range(100)), list(range(100, 124)), list(range(124, 224))
D = np.stack([Vehicle.demand(p["u"]) for p in P]); Z = np.stack([p["z"] for p in P])
X = torch.tensor(D, dtype=torch.float32); mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
Y = torch.tensor(Z, dtype=torch.float32)
best = None
for h in (16, 64, 256):
    for lr in (1e-3, 1e-2):
        for ep in (500, 2000):
            torch.manual_seed(a.seed)
            m = nn.Sequential(nn.Linear(T, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, T))
            o = torch.optim.Adam(m.parameters(), lr=lr)
            for _ in range(ep):
                o.zero_grad(); l = ((m(Xn[tr])-Y[tr])**2).mean(); l.backward(); o.step()
            with torch.no_grad(): v = float(((m(Xn[va])-Y[va])**2).mean())
            if best is None or v < best[0]: best = (v, m, dict(hidden=h, lr=lr, epochs=ep))
m = best[1].eval(); rows = []
for i in te:
    t0 = time.time()
    with torch.no_grad(): zf = np.clip(m(Xn[[i]])[0].numpy(), 0, SW)
    pj = V.solve(D[i], target=zf); zq = pj["z"].round(); r = V.solve(D[i], zfix=zq)
    zt = np.array(P[i]["z"]); pt = np.array(P[i]["Pe"]); J = P[i]["cost"]
    rows.append(dict(idx=i, gap=100*(r["cost"]-J)/abs(J), disc=100*np.abs(zq-zt).sum()/max(zt.sum(), 1e-9),
                     cont=100*np.abs(r["Pe"]-pt).sum()/max(pt.sum(), 1e-9), secs=time.time()-t0))
g = np.array([r["gap"] for r in rows])
print(f"[PROXY] veh {a.vid} s{a.seed} {best[2]}: gap {g.mean():.4f}%  disc {np.mean([r['disc'] for r in rows]):.2f}%  "
      f"cont {np.mean([r['cont'] for r in rows]):.2f}%", flush=True)
json.dump(dict(rows=rows, cfg=best[2], args=vars(a)), open(f"results/proxy_v{a.vid}_s{a.seed}.json", "w"), indent=1)

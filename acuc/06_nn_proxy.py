# %% SUPERVISED NN proxy for AC-UC with the SAME restoration as our method.
#    Fixes two flaws in 42_proxy_sup.py: its restoration had no reserve top-up
#    and no downward pass (weaker than ours, unfair to the proxy), and its
#    timer measured an unrelated relaxation solve instead of the proxy's own
#    forward pass + restoration.
import os, sys, json, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--case", default="case118"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=400); ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--down", type=int, default=25)
ap.add_argument("--order", default="conf", choices=["conf", "merit"])
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case; os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(a.seed)
from pipeline import grid, set_cuts
from socp import Socp
from restore import restore
import poolload
QC = None; NT, NV, NTR = 48, 24, 144
set_cuts(4)
g, nl = grid(a.case); NG, NB = g.n_gen, g.n_bus
S = Socp(g, nl, n_cuts=4); poolload.set_grid(g)
POOL = poolload.load_pool(a.case, "wide", QC)
perm = np.random.default_rng(0).permutation(len(POOL))
te = [int(i) for i in perm[:NT]]; tr = [int(i) for i in perm[NT:NT+NTR+NV]][NV:]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
Y = torch.tensor(np.stack([POOL[i][2] for i in tr]), dtype=torch.float32)
net = nn.Sequential(nn.Linear(2*NB, a.hidden), nn.SiLU(), nn.Linear(a.hidden, a.hidden),
                    nn.SiLU(), nn.Linear(a.hidden, NG))
opt = torch.optim.Adam(net.parameters(), lr=a.lr); lf = nn.MSELoss()
t0 = time.time()
for ep in range(a.epochs):
    opt.zero_grad(); loss = lf(torch.sigmoid(net(Xn[tr])), Y); loss.backward(); opt.step()
train_s = time.time()-t0
MERIT = np.argsort(g.c1 + g.c2*g.pmax + nl/np.maximum(g.pmax, 1e-9))
net.eval(); rows = []
for i in te:
    rf = S.price(POOL[i][0], POOL[i][1], POOL[i][2].astype(float))
    if rf is None: continue
    t1 = time.time()
    with torch.no_grad():
        uf = torch.sigmoid(net(Xn[[i]]))[0].numpy().astype(float)
    order = MERIT if a.order == "merit" else None
    c, d = restore(S, g, POOL[i][0], POOL[i][1], uf, thr=0.5, order=order, ndown=a.down)
    el = time.time()-t1
    if d is None: continue
    rows.append(dict(gap=100*(c-POOL[i][3])/abs(POOL[i][3]),
                     disc=100*float((d["u"] != POOL[i][2]).mean()),
                     cont=100*float(np.abs(d["pg"]-rf["pg"]).sum()/rf["pg"].sum()), secs=el))
v = np.array([[r["gap"], r["disc"], r["cont"], r["secs"]] for r in rows])
print(f"[TEST] {a.case} s{a.seed} proxy(fair) n={len(v)}/{len(te)} |dev| {np.abs(v[:,0]).mean():.4f}%  "
      f"disc {v[:,1].mean():.2f}%  cont {v[:,2].mean():.2f}%  med {np.median(v[:,3]):.2f}s  train {train_s:.1f}s")
json.dump(dict(case=a.case, seed=a.seed, rows=rows, train_secs=train_s,
               gap_mean=float(np.abs(v[:,0]).mean()), disc_mean=float(v[:,1].mean()),
               cont_mean=float(v[:,2].mean())),
          open(f"results/proxyfair_{a.case}_s{a.seed}.json", "w"), indent=1)

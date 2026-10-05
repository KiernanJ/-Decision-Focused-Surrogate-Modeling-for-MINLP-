# %% Phase 2: take the threshold fitted on TRAINING deployed cost and evaluate
#    it on the held-out test split, against the shipped 0.5 rule. One seed per
#    process. Selection used only training instances; test is touched once.
import os, sys, json, glob, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--case", default="case118")
ap.add_argument("--pool", default="wide")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kcuts", type=int, default=2)
ap.add_argument("--down", type=int, default=6)
# case300 needs the cheapest-first MERIT order; case118 needs the
# network's confidence. Fitting under one and deploying under the
# other would tune the threshold against the wrong pipeline.
ap.add_argument("--order", default="conf", choices=["conf", "merit"])
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case; os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); QC = None   # wide pools live in results/wide
from pipeline import grid, RESERVE, set_cuts, set_down, deploy
from socp import Socp
from cutnet_k import CutNetK, rows_torch
import poolload
K = a.kcuts; ROWS = 2*K; RHO = 1e6; NT, NV, NTR = 48, 24, 144
set_cuts(ROWS); set_down(a.down)
g, nl = grid(a.case); NG, NB = g.n_gen, g.n_bus
S = Socp(g, nl, n_cuts=ROWS); V1, V0 = np.ones(NB), np.zeros(NB)
poolload.set_grid(g)
POOL = poolload.load_pool(a.case, a.pool, QC)
perm = np.random.default_rng(0).permutation(len(POOL))
te = [int(i) for i in perm[:NT]]
tr = [int(i) for i in perm[NT:NT+NTR+NV]][NV:]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
W = {}
for j in te:
    r = S.solve(POOL[j][0], POOL[j][1], V1, V0, rho=RHO)
    if r is not None:
        W[j] = np.concatenate([r["pg"], r["u"]])
ids = [j for j in te if j in W]
WR = torch.tensor(np.stack([W[j] for j in ids]), dtype=torch.float32)
net = CutNetK(NB, NG, k_rows=K, v_scale=0.5, seed=a.seed)
net.load_state_dict(torch.load(f"checkpoints/{a.case}_s{a.seed}.pt",
                               weights_only=False)["net"], strict=False)
net.eval()
with torch.no_grad():
    Vr_t, Vi_t, aa, al, a2, d, d2, _ = net(Xn[ids])
    A_t, b_t = rows_torch(aa, al, a2, d, d2, WR)
A = A_t.numpy().astype(float); b = b_t.numpy().astype(float)
Vr = Vr_t.numpy().astype(float); Vi = Vi_t.numpy().astype(float)
shards = [json.load(open(f)) for f in
          sorted(glob.glob(f"results/thr_lf/sh_{a.case}_s{a.seed}_*.json"))]
if not shards:
    raise SystemExit(f"FATAL: no shard results for seed {a.seed}")
win = min(shards, key=lambda d_: d_["best"])
c = np.array(win["c"])
ORD = (np.argsort(g.c1 + g.c2*g.pmax + nl/np.maximum(g.pmax, 1e-9))
       if a.order == 'merit' else None)
zz = lambda v: (v - v.mean())/(v.std() + 1e-12)
FEAT = np.stack([zz(g.pmax), zz(g.c1 + g.c2*g.pmax), zz(nl)])
th = np.clip(0.5 + c[0] + c[1]*FEAT[0] + c[2]*FEAT[1] + c[3]*FEAT[2], 0.05, 0.95)
print(f"[s{a.seed}] winning shard {win['shard']}  train {win['best']:.6f} "
      f"(shipped {win['base']:.6f})  c={np.round(c,4).tolist()}")
print(f"         thr range [{th.min():.3f}, {th.max():.3f}] mean {th.mean():.3f}", flush=True)
out = {}
for tag, t_ in (("shipped", None), ("fitted", th)):
    rows = []
    for k, j in enumerate(ids):
        cst, dd = deploy(S, g, POOL[j][0], POOL[j][1], Vr[k], Vi[k],
                         A=A[k], b=b[k], rho=RHO, thr=t_, order=ORD)
        if cst >= 1e8 or dd is None:
            continue
        pr = S.price(POOL[j][0], POOL[j][1], dd["u"].astype(float))
        rf = S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))
        if pr is None or rf is None:
            continue
        rows.append((100*(cst-POOL[j][3])/POOL[j][3],
                     100*float((dd["u"] != POOL[j][2]).mean()),
                     100*float(np.abs(pr["pg"]-rf["pg"]).sum()/rf["pg"].sum())))
    v = np.array(rows); out[tag] = rows
    print(f"  TEST {tag:8s} n={len(v):3d}  gap {v[:,0].mean():+.4f}%  "
          f"discrete {v[:,1].mean():.2f}%  continuous {v[:,2].mean():.2f}%", flush=True)
json.dump(dict(case=a.case, seed=a.seed, c=c.tolist(), out=out),
          open(f"results/thrlf_eval_{a.case}_s{a.seed}"
               + (f"_d{a.down}" if a.down != 6 else "") + ".json", "w"), indent=1)

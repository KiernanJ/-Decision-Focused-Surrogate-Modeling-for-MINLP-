# %% Per-instance deployment TIME at each case's FINAL configuration, for the
#    time-CDF panel. The earlier timing was taken at NDOWN=6; both cases now
#    ship a larger repair budget, which costs real seconds.
import os, sys, json, glob, pickle, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--case", default="case118")
ap.add_argument("--down", type=int, default=25)
ap.add_argument("--order", default="conf")
ap.add_argument("--thr", type=float, default=-1.0)   # <0 => use the FITTED vector
ap.add_argument("--ckpt", default="")
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case; os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); QC = None   # wide pools live in results/wide
from pipeline import grid, set_cuts, set_down, deploy
from socp import Socp
from cutnet_k import CutNetK, rows_torch
import poolload
K, ROWS, RHO, NT, NV, NTR = 2, 4, 1e6, 48, 24, 144
set_cuts(ROWS); set_down(a.down)
g, nl = grid(a.case); NG, NB = g.n_gen, g.n_bus
S = Socp(g, nl, n_cuts=ROWS); V1, V0 = np.ones(NB), np.zeros(NB)
poolload.set_grid(g)
POOL = poolload.load_pool(a.case, "wide", QC)
perm = np.random.default_rng(0).permutation(len(POOL))
te = [int(i) for i in perm[:NT]]
tr = [int(i) for i in perm[NT:NT+NTR+NV]][NV:]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
W = {}
for j in te:
    r = S.solve(POOL[j][0], POOL[j][1], V1, V0, rho=RHO)
    if r is not None: W[j] = np.concatenate([r["pg"], r["u"]])
ids = [j for j in te if j in W]
WR = torch.tensor(np.stack([W[j] for j in ids]), dtype=torch.float32)
MERIT = np.argsort(g.c1 + g.c2*g.pmax + nl/np.maximum(g.pmax, 1e-9))
ORD = MERIT if a.order == "merit" else None
zz = lambda v: (v - v.mean())/(v.std() + 1e-12)
F = np.stack([zz(g.pmax), zz(g.c1 + g.c2*g.pmax), zz(nl)])
T = []
for seed in (0, 1, 2, 3):
    ck = a.ckpt.format(seed=seed) if a.ckpt else f"checkpoints/{a.case}_s{seed}.pt"
    if not os.path.exists(ck): continue
    net = CutNetK(NB, NG, k_rows=K, v_scale=0.5, seed=seed)
    net.load_state_dict(torch.load(ck, weights_only=False)["net"], strict=False)
    net.eval()
    if a.thr >= 0:
        th = np.full(NG, a.thr)
    else:
        c = np.array(json.load(open(f"results/thrlf_eval_{a.case}_s{seed}_d25.json"))["c"])
        th = np.clip(0.5 + c[0] + c[1]*F[0] + c[2]*F[1] + c[3]*F[2], 0.05, 0.95)
    for k, j in enumerate(ids):
        t0 = time.time()
        with torch.no_grad():                      # network forward is IN the budget
            vr, vi, aa, al, a2, d, d2, _ = net(Xn[[j]])
            A_t, b_t = rows_torch(aa, al, a2, d, d2, WR[[k]])
        cst, dd = deploy(S, g, POOL[j][0], POOL[j][1],
                         vr[0].numpy().astype(float), vi[0].numpy().astype(float),
                         A=A_t[0].numpy().astype(float), b=b_t[0].numpy().astype(float),
                         rho=RHO, thr=th, order=ORD)
        el = time.time()-t0
        if cst < 1e8 and dd is not None: T.append(el)
REF = []
for f in sorted(glob.glob(f"results/wide/{a.case}_W_*.pkl")):
    d_ = pickle.load(open(f, "rb"))
    REF += [r["secs"] for r in d_["res"].values() if r is not None]
json.dump(dict(deploy=T, reference=REF, case=a.case, down=a.down, order=a.order),
          open(f"results/timingF_{a.case}.json", "w"))
print(f"[{a.case} NDOWN={a.down} {a.order}] deploy n={len(T)} median {np.median(T):.2f}s "
      f"mean {np.mean(T):.2f}s | reference median {np.median(REF):.1f}s "
      f"-> {np.median(REF)/np.median(T):.0f}x", flush=True)

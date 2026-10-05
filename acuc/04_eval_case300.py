# %% case300: the FULL pipeline, every stage swept together instead of one at a
#    time. Evaluation only -- the trained nets are untouched.
#
#    The question this answers: at NDOWN=40 our discrete error is 11.64% while a
#    CONSTANT commitment reaches 3.42%. Our STARTING point is worse than a fixed
#    guess, which is not "no headroom" -- something in the pipeline is wrong.
#    Measured leads:
#      - the net's uf ORDERING hurts on case300 (2x2 isolation: merit +0.613%
#        vs conf +0.798%) yet `deploy` still repairs in argsort(-uf) order
#      - the relaxation itself commits 50.9 against a 44.8 reference, so it
#        over-commits BEFORE any rounding
#      - case300 over-commits, so its threshold must go UP (case118's goes down)
#
#    Stages crossed here: repair ORDER x THRESHOLD x NDOWN.
import os, sys, json, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--case", default="case300")
ap.add_argument("--pool", default="wide")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--downs", default="6,25,40")
ap.add_argument("--kcuts", type=int, default=2)
# one COMBINATION per process: 48 instances x 30s at NDOWN=25 is
# 24 min per combo, so parallelising over seeds left each process
# running 6 combos back to back (~2.4h). One combo per process
# finishes the whole grid in one combo's time.
ap.add_argument("--order", default="")
# evaluate ANY checkpoint, not just ckptW -- stage D wrote its nets to
# ckptD and the trainer scored them with the DEFAULT (confidence)
# order, which is the losing one on case300.
ap.add_argument("--ckpt", default="")
ap.add_argument("--thr", default="")
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case; os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); QC = None   # wide pools live in results/wide
from pipeline import grid, RESERVE, set_cuts
from socp import Socp
from cutnet_k import CutNetK, rows_torch
import poolload
K = a.kcuts; ROWS = 2*K; RHO = 1e6; NT, NV, NTR = 48, 24, 144
set_cuts(ROWS)
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
CKPT = a.ckpt or f"checkpoints/{a.case}_s{a.seed}.pt"
net.load_state_dict(torch.load(CKPT, weights_only=False)["net"],
                    strict=False)
net.eval()
with torch.no_grad():
    Vr_t, Vi_t, aa, al, a2, d, d2, _ = net(Xn[ids])
    A_t, b_t = rows_torch(aa, al, a2, d, d2, WR)
A = A_t.numpy().astype(float); b = b_t.numpy().astype(float)
Vr = Vr_t.numpy().astype(float); Vi = Vi_t.numpy().astype(float)
MERIT = np.argsort(g.c1 + g.c2*g.pmax + nl/np.maximum(g.pmax, 1e-9))
zz = lambda v: (v - v.mean())/(v.std() + 1e-12)
FEAT = np.stack([zz(g.pmax), zz(g.c1 + g.c2*g.pmax), zz(nl)])
PG, UF = {}, {}
for k, j in enumerate(ids):
    pr = S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))
    r1 = S.solve(POOL[j][0], POOL[j][1], Vr[k], Vi[k], rho=RHO, A=A[k], b=b[k])
    if pr is not None and r1 is not None:
        PG[j] = pr["pg"]; UF[j] = r1["u"]
ok = [j for j in ids if j in UF]
print(f"[{a.case} s{a.seed}] {len(ok)} instances; relaxation commits "
      f"{np.mean([UF[j].sum() for j in ok]):.1f}, reference "
      f"{np.mean([POOL[j][2].sum() for j in ok]):.1f}", flush=True)


def restore(j, uf, thr, order, ndown):
    pd_, qd_ = POOL[j][0], POOL[j][1]
    z = (uf > thr).astype(float)
    need = (1.0+RESERVE)*float(pd_.sum())
    for t in order:
        if float(g.pmax @ z) >= need: break
        z[t] = 1.0
    best = None
    for _ in range(NG+1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"], pr["pg"]); break
        off = [t for t in order if z[t] < 0.5]
        if not off: return None
        z[off[0]] = 1.0
    if best is None: return None
    z, cost, pg = best
    # try removing the units the ORDER trusts least, cheapest-to-lose first
    for t in [t2 for t2 in order[::-1] if z[t2] > 0.5][:ndown]:
        w = z.copy(); w[t] = 0.0
        if float(g.pmax @ w) < need: continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w, pr["cost"], pr["pg"]
    return z, cost, pg


THRS = {"0.5": 0.5, "0.65": 0.65, "0.80": 0.80}
ORDS = {"conf": None, "merit": MERIT}
if a.thr:   THRS = {a.thr: float(a.thr)}
if a.order: ORDS = {a.order: (MERIT if a.order == "merit" else None)}
out = {}
print(f"\n{'order':>7s}{'thr':>7s}{'NDOWN':>7s}{'gap':>10s}{'disc':>9s}{'cont':>9s}{'n_on':>7s}{'s/inst':>9s}")
for oname, omer in ORDS.items():
    for tname, tv in THRS.items():
        for nd in [int(x) for x in a.downs.split(",")]:
            rows, t0 = [], time.time()
            for j in ok:
                uf = UF[j]
                order = MERIT if omer is not None else np.argsort(-uf)
                r = restore(j, uf, tv, order, nd)
                if r is None: continue
                z, c, pg = r
                rows.append((100*(c-POOL[j][3])/POOL[j][3],
                             100*float((z.astype(int) != POOL[j][2].astype(int)).mean()),
                             100*float(np.abs(pg-PG[j]).sum()/PG[j].sum()), float(z.sum())))
            if not rows: continue
            v = np.array(rows); el = (time.time()-t0)/len(rows)
            out[f"{oname}|{tname}|{nd}"] = rows
            print(f"{oname:>7s}{tname:>7s}{nd:>7d}{v[:,0].mean():>+9.3f}%{v[:,1].mean():>8.2f}%"
                  f"{v[:,2].mean():>8.2f}%{v[:,3].mean():>7.1f}{el:>8.2f}s", flush=True)
tag = f"_{a.order}{a.thr}" if (a.order or a.thr) else ""
tag += "_dt"   # paper nets were trained with --dthr 0.8
json.dump(out, open(f"results/c300pipe_s{a.seed}{tag}.json", "w"))

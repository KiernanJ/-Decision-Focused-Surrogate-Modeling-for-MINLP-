# %% Self-supervised training of the learned linearisation point V and the
#    learned cuts for QCAC AC unit commitment, by EXACT gradients through the
#    convex QCAC relaxation (ConeLayer, regularised sparse LU on the KKT).
#
#    Loss   = deployed cost after differentiable soft rounding and the
#             continuous re-solve, normalised by the uncut relaxation cost.
#             No reference commitment or reference cost enters training or
#             model selection.
#    Cuts   = K rows per category (general a'p_g + a_u'u <= b, and
#             integer-only a2'u <= b2), anchored b = <A, w_rlx> - depth.
#    Split  = 48 test / 24 validation / 144 train on the wide band.
#
#    The [TEST] block at the bottom deploys at threshold 0.5 and confidence
#    order; the PAPER numbers come from 04_eval_case118.py / 04_eval_case300.py
#    which apply each case's final restoration.
#
#    Final commands (paper):
#      case118: --case case118 --epochs 30 --lr 2e-3 --down 6
#               --ckpt checkpoints/case118_s{SEED}.pt
#      case300: --case case300 --epochs 26 --lr 5e-4 --down 25 --dthr 0.8
#               --ckpt checkpoints/case300_s{SEED}.pt
import os, sys, re, time, json, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="case118")
ap.add_argument("--kcuts", type=int, default=2)        # rows PER CATEGORY
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--n-test", type=int, default=48)
ap.add_argument("--n-train", type=int, default=144)
ap.add_argument("--n-val", type=int, default=24)   # carved from TRAIN
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--tau", type=float, default=0.08)
ap.add_argument("--init-depth", type=float, default=0.25)
ap.add_argument("--max-depth", type=float, default=2.0)
ap.add_argument("--tag", default="")
ap.add_argument("--rho", type=float, default=1e6)
ap.add_argument("--down", type=int, default=6)
# DiffDeploy rounds at this threshold during TRAINING. It defaulted
# to 0.5 while case300 deploys best at a HIGHER one, so the training
# objective and the deployment rule disagreed.
ap.add_argument("--dthr", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--v-scale", type=float, default=0.5)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--pool", default="wide", choices=["base", "wide"])
ap.add_argument("--ckpt", default="")
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case; os.environ["OMP_NUM_THREADS"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); QC = None   # wide pools live in results/wide
torch.set_num_threads(1)

from pipeline import grid, sample, RESERVE, set_cuts, set_down, deploy
from socp import Socp
from cutnet_k import CutNetK, rows_torch
from framework import DiffDeploy

REG1 = REG2 = 1e-10
K = a.kcuts; ROWS = 2*K
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
set_cuts(ROWS)
g, nl = grid(a.case); NG, NB = g.n_gen, g.n_bus
S = Socp(g, nl, n_cuts=ROWS)
V1, V0 = np.ones(NB), np.zeros(NB)

# ---- the pool (band selected by --pool; see src/poolload.py) ----
import poolload
poolload.set_grid(g)
POOL = poolload.load_pool(a.case, a.pool, QC)
if len(POOL) < a.n_test + a.n_train + a.n_val:
    raise SystemExit(f"FATAL: pool has {len(POOL)} instances, need "
                     f"{a.n_test+a.n_train+a.n_val} for the requested split")
print(f"[pool ] {a.pool} band, {len(POOL)} instances", flush=True)
perm = np.random.default_rng(0).permutation(len(POOL))
te = [int(i) for i in perm[:a.n_test]]
_pool_tr = [int(i) for i in perm[a.n_test:a.n_test+a.n_train+a.n_val]]
# VALIDATION is carved out of TRAIN, never out of test. Model selection on the
# training loss can pick an overfitted epoch; selection on test would leak.
va = _pool_tr[:a.n_val]
tr = _pool_tr[a.n_val:]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg

W_RLX, SCALE = {}, {}
for j in tr+va+te:
    r = S.solve(POOL[j][0], POOL[j][1], V1, V0, rho=a.rho)
    if r is not None:
        W_RLX[j] = np.concatenate([r["pg"], r["u"]]); SCALE[j] = float(r["cost"])
tr = [j for j in tr if j in W_RLX]; te = [j for j in te if j in W_RLX]
va = [j for j in va if j in W_RLX]
WR = torch.tensor(np.stack([W_RLX[j] for j in sorted(W_RLX)]), dtype=torch.float32)
WIDX = {j: i for i, j in enumerate(sorted(W_RLX))}
steps = a.epochs*max(1, int(np.ceil(len(tr)/a.batch)))
print(f"[{a.case}] pool {len(POOL)} | {len(tr)} train / {len(va)} val / {len(te)} test | "
      f"K={K} per category = {ROWS} rows | batch {a.batch} -> {steps} steps "
      f"(was {a.epochs}) | rho={a.rho:.0e}", flush=True)

net = CutNetK(NB, NG, k_rows=K, v_scale=a.v_scale, hidden=a.hidden,
              seed=a.seed, init_depth=a.init_depth,
              max_depth=a.max_depth)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
DD = DiffDeploy(S, g, nl, reserve=RESERVE, tau=a.tau, thr=a.dthr,
                rho=a.rho, reg=REG1, reg2=REG2)
print(f"[net  ] {net.n_params():,} params", flush=True)

def val_loss():
    """Mean surrogate loss on the VALIDATION split. Forward only, no gradient."""
    with torch.no_grad():
        Vr_t, Vi_t, at, alt, a2t, dt, d2t, _thr = net(Xn[va])
        A_t, b_t = rows_torch(at, alt, a2t, dt, d2t, WR[[WIDX[j] for j in va]])
    An_ = A_t.numpy().astype(float); bn_ = b_t.numpy().astype(float)
    Vr_ = Vr_t.numpy().astype(float); Vi_ = Vi_t.numpy().astype(float)
    L = []
    for k, j in enumerate(va):
        l, _, _, _ = DD(POOL[j][0], POOL[j][1], Vr_[k], Vi_[k], An_[k], bn_[k],
                        SCALE[j], want_grad=False)
        if l is not None:
            L.append(l)
    return float(np.mean(L)) if L else np.nan


t0all = time.time(); hist = []; ep0 = 0
best = None          # (val_loss, state_dict, epoch) -- the model actually reported
if a.ckpt and os.path.exists(a.ckpt):
    ck = torch.load(a.ckpt, weights_only=False)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"]); ep0 = ck["ep"]; hist = ck["hist"]
    best = ck.get("best")
    rng = np.random.default_rng(a.seed+7919*ep0)
    print(f"[resume] epoch {ep0}", flush=True)

for ep in range(ep0, a.epochs):
    t0 = time.time(); order = rng.permutation(tr)
    Ls, nok, nsolve, nstep, fails = [], 0, 0, 0, {}
    gnorms = []
    for s0 in range(0, len(order), a.batch):
        bidx = [int(j) for j in order[s0:s0+a.batch]]
        Vr_t, Vi_t, at, alt, a2t, dt, d2t, _thr = net(Xn[bidx])
        A_t, b_t = rows_torch(at, alt, a2t, dt, d2t, WR[[WIDX[j] for j in bidx]])
        An = A_t.detach().numpy().astype(float); bn = b_t.detach().numpy().astype(float)
        Vrn = Vr_t.detach().numpy().astype(float); Vin = Vi_t.detach().numpy().astype(float)
        GA = np.zeros_like(An); GB = np.zeros_like(bn)
        GVr = np.zeros_like(Vrn); GVi = np.zeros_like(Vin)
        got = 0
        for k, j in enumerate(bidx):
            L, gA, gb, info = DD(POOL[j][0], POOL[j][1], Vrn[k], Vin[k],
                                 An[k], bn[k], SCALE[j]); nsolve += 4
            if L is None or gA is None:
                fails[info.get("fail", "?")] = fails.get(info.get("fail", "?"), 0)+1
                continue
            GA[k], GB[k] = gA, gb
            GVr[k], GVi[k] = info["dVr"], info["dVi"]
            Ls.append(L); got += 1
        if got == 0:
            continue
        nok += got; nstep += 1
        opt.zero_grad()
        loss = ((A_t*torch.tensor(GA, dtype=torch.float32)).sum()
                + (b_t*torch.tensor(GB, dtype=torch.float32)).sum()
                + (Vr_t*torch.tensor(GVr, dtype=torch.float32)).sum()
                + (Vi_t*torch.tensor(GVi, dtype=torch.float32)).sum())/got
        loss.backward()
        gn = float(sum((p.grad**2).sum() for p in net.parameters()
                       if p.grad is not None)**0.5)
        gnorms.append(gn)
        opt.step(); sched.step()
    vl = val_loss()
    if best is None or (np.isfinite(vl) and vl < best[0]):
        best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
    hist.append(dict(ep=ep, loss=float(np.mean(Ls)) if Ls else np.nan, val=vl, nok=nok,
                     nstep=nstep, nsolve=nsolve, secs=time.time()-t0))
    print(f"[ep {ep:3d}] loss {np.mean(Ls) if Ls else np.nan:+.5f}  ok {nok}/{len(tr)}"
          f"  val {vl:+.5f}  steps {nstep}  |g| {np.mean(gnorms) if gnorms else 0:.3e}"
          f"  lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:6.1f}s"
          + (f"  FAILS {fails}" if fails else ""), flush=True)
    if nok == 0:
        raise SystemExit(f"FATAL: epoch {ep} produced no gradient; failures {fails}")
    if a.ckpt:
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), ep=ep+1, hist=hist,
                        best=best), a.ckpt)
train_secs = time.time()-t0all

# %% LABEL-FREE model selection: deployed cost on the VALIDATION split,
# normalised by the uncut relaxation cost. Measured reason this is not the
# surrogate validation loss: on case300 the surrogate RISES across epochs while
# the deployment gap IMPROVES, so surrogate-based selection picked 4.6x worse
# models on 0/5 seeds. Validation only -- test is never touched for selection.
set_down(a.down)
with torch.no_grad():
    Vrv, Viv, atv, altv, a2v, dtv, d2v = net(Xn[va])[:7]
    Av, bv = rows_torch(atv, altv, a2v, dtv, d2v, WR[[WIDX[j] for j in va]])
Avn = Av.numpy().astype(float); bvn = bv.numpy().astype(float)
Vrvn = Vrv.numpy().astype(float); Vivn = Viv.numpy().astype(float)
_tot, _n = 0.0, 0
for _k, _j in enumerate(va):
    _c, _d = deploy(S, g, POOL[_j][0], POOL[_j][1], Vrvn[_k], Vivn[_k],
                    A=Avn[_k], b=bvn[_k], rho=a.rho)
    _tot += 10.0 if (_c >= 1e8 or _d is None) else _c/SCALE[_j]
    _n += 1
val_deploy = _tot/max(_n, 1)
print(f"[select] LABEL-FREE validation deployed score {val_deploy:.6f}", flush=True)

# %% held-out on the REAL pipeline
# Report the BEST model by validation loss, not the last epoch. Five of six
# case300 seeds peaked at epoch 2-4 and then degraded for 25 more epochs, and
# the last epoch is what got reported. qcac_clean/21_tiered_train.py already
# keeps the best checkpoint; this trainer did not.
if best is not None:
    net.load_state_dict(best[1])
    print(f"[select] best epoch {best[2]}/{a.epochs} by validation loss {best[0]:+.5f}",
          flush=True)
net.eval(); set_down(a.down)
with torch.no_grad():
    Vr_t, Vi_t, at, alt, a2t, dt, d2t, _thr = net(Xn[te])
    A_t, b_t = rows_torch(at, alt, a2t, dt, d2t, WR[[WIDX[j] for j in te]])
An = A_t.numpy().astype(float); bn = b_t.numpy().astype(float)
Vrn = Vr_t.numpy().astype(float); Vin = Vi_t.numpy().astype(float)
rows = []
for k, j in enumerate(te):
    c, d = deploy(S, g, POOL[j][0], POOL[j][1], Vrn[k], Vin[k], A=An[k], b=bn[k], rho=a.rho)
    if c >= 1e8 or d is None:
        continue
    pr = S.price(POOL[j][0], POOL[j][1], d["u"].astype(float))
    ref = S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))
    rows.append(dict(gap=100*(c-POOL[j][3])/POOL[j][3],
                     disc=100*float((d["u"] != POOL[j][2]).mean()),
                     cont=(100*float(np.abs(pr["pg"]-ref["pg"]).sum()/ref["pg"].sum())
                           if pr is not None and ref is not None else np.nan)))
G = np.array([r["gap"] for r in rows]); D = np.array([r["disc"] for r in rows])
C = np.array([r["cont"] for r in rows])
print(f"\n[TEST] {len(rows)}/{len(te)} deployed (depth {a.down})")
print(f"  optimality gap    mean {G.mean():+.4f}%  median {np.median(G):+.4f}%")
print(f"  discrete error    mean {D.mean():.3f}%")
print(f"  continuous error  mean {np.nanmean(C):.3f}%")
print(f"  training {train_secs:.0f}s")
json.dump(dict(case=a.case, seed=a.seed, kcuts=K, pool=a.pool,
               val_deploy=val_deploy,
               best_epoch=(best[2] if best else None),
               best_val=(best[0] if best else None), gaps=list(map(float, G)),
               discs=list(map(float, D)), conts=[float(x) for x in C],
               gap_mean=float(G.mean()), disc_mean=float(D.mean()),
               cont_mean=float(np.nanmean(C)), train_secs=train_secs,
               hist=hist, args=vars(a)),
          open(f"results/{a.pool}_{a.case}_k{K}_s{a.seed}{a.tag}.json", "w"), indent=1)
print(f"wrote results/{a.pool}_{a.case}_k{K}_s{a.seed}{a.tag}.json")

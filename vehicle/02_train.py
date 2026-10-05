# %% Train the learned cut on the hybrid-vehicle convex MINLP by EXACT
#    gradients through two chained cone layers (relaxation with cut ->
#    soft-rounded integer schedule -> re-solve of the continuous dispatch).
#
#    Self-supervised: the loss is the deployed cost normalised by the uncut
#    relaxation cost. No reference solution is used for training or for model
#    selection (selection = label-free validation deployed cost). Test labels
#    are touched once, for scoring.
#
#    Final configuration (paper):  --kcuts 2 --hidden 16 --w-int 5 --epochs 25
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import os, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
torch.set_num_threads(1)
from src_hv import load_pool, splits, HVRelax, deploy
from cutnet_hv import CutNetHV, rows_torch
from diffhv import DiffHV
import gurobipy as gp
from gurobipy import GRB
_ENV = gp.Env(empty=True); _ENV.setParam("OutputFlag", 0); _ENV.start()


def qz_restore(VEH, T, S, Dp, z_frac, tl=60.):
    """L1 projection restoration: nearest FEASIBLE INTEGER schedule to z_frac.
    Fixed to the BEST restoration so the comparison isolates the LEARNED CUT --
    it is 8.4x better than round+repair on this problem (+0.088% vs +0.736%)."""
    m = gp.Model(env=_ENV); m.Params.TimeLimit = tl
    E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"])
    Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER); w = m.addVars(T, lb=0)
    m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S))
        m.addConstr(Pb[t] + Pe[t] >= Dp[t])
        m.addConstr(w[t] >= z[t] - float(z_frac[t]))
        m.addConstr(w[t] >= float(z_frac[t]) - z[t])
    m.setObjective(gp.quicksum(w[t] for t in range(T)), GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0: return None
    return np.array([round(z[t].X) for t in range(T)], float)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--kcuts", type=int, default=2)
ap.add_argument("--tau", type=float, default=0.08)
ap.add_argument("--init-depth", type=float, default=0.25)
ap.add_argument("--max-depth", type=float, default=2.0)
ap.add_argument("--ndown", type=int, default=6)
ap.add_argument("--hidden", type=int, default=16)
ap.add_argument("--pool", default="results/pool/pool_*.json")
ap.add_argument("--w-int", type=float, default=5.0)
ap.add_argument("--learn-thr", action="store_true")   # off in the paper
ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
VEH, D, REF = load_pool(a.pool); T, S = VEH["T"], VEH["S_MODES"]
R = HVRelax(VEH, n_cuts=2*a.kcuts)
te, va, tr = splits(len(REF))
te = [i for i in te if REF[i]]; va = [i for i in va if REF[i]]; tr = [i for i in tr if REF[i]]
X = torch.tensor(D/D.mean(), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
W, SCALE = {}, {}
for i in tr+va+te:
    r = R.solve(D[i])
    if r is not None:
        W[i] = np.concatenate([r["E"], r["Peng"], r["Pbatt"], r["z"]]); SCALE[i] = float(r["cost"])
tr = [i for i in tr if i in W]; va = [i for i in va if i in W]; te = [i for i in te if i in W]
WR = torch.tensor(np.stack([W[i] for i in sorted(W)]), dtype=torch.float32)
WI = {i: k for k, i in enumerate(sorted(W))}
net = CutNetHV(T, k_rows=a.kcuts, hidden=a.hidden, seed=a.seed,
               init_depth=a.init_depth, max_depth=a.max_depth,
               learn_thr=a.learn_thr)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
steps = a.epochs*max(1, int(np.ceil(len(tr)/a.batch)))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
DD = DiffHV(R, VEH, tau=a.tau, w_int=a.w_int)
print(f"[s{a.seed}] {len(tr)} train / {len(va)} val / {len(te)} test | "
      f"K={a.kcuts} -> {2*a.kcuts} rows | thr={'learned' if a.learn_thr else 'fixed 0.5'} | "
      f"{net.n_params():,} params | {steps} steps", flush=True)


def fwd(ids):
    aE, ape, apb, az, az2, d, d2, th = net(Xn[ids])
    A, b = rows_torch(aE, ape, apb, az, az2, d, d2,
                      WR[[WI[i] for i in ids]])
    return A, b, th


def val_deploy():
    with torch.no_grad():
        A, b, th = fwd(va)
    An, bn = A.numpy().astype(float), b.numpy().astype(float)
    thn = th.numpy().astype(float) if th is not None else None
    tot = 0.0
    for k, i in enumerate(va):
        r1 = R.solve(D[i], A=An[k], b=bn[k], cut_cap=0.0)
        zq = qz_restore(VEH, T, S, D[i], r1["z"]) if r1 is not None else None
        pr2 = R.solve(D[i], z_lo=zq, z_hi=zq) if zq is not None else None
        tot += 10.0 if pr2 is None else pr2["cost"]/SCALE[i]
    return tot/len(va)


best, hist, t0 = None, [], time.time()
for ep in range(a.epochs):
    te0 = time.time(); order = rng.permutation(tr); Ls, nok, fails = [], 0, {}
    for s0 in range(0, len(order), a.batch):
        bid = [int(i) for i in order[s0:s0+a.batch]]
        A, b, th = fwd(bid)
        An = A.detach().numpy().astype(float); bn = b.detach().numpy().astype(float)
        thn = th.detach().numpy().astype(float) if th is not None else None
        GA = np.zeros_like(An); GB = np.zeros_like(bn)
        GT = np.zeros_like(thn) if thn is not None else None
        got = 0
        for k, i in enumerate(bid):
            L, gA, gb, info = DD(D[i], An[k], bn[k], SCALE[i],
                                 thr=(thn[k] if thn is not None else None))
            if L is None or gA is None:
                f = info.get("fail", "?"); fails[f] = fails.get(f, 0)+1; continue
            GA[k], GB[k] = gA, gb
            if GT is not None: GT[k] = info["dthr"]
            Ls.append(L); got += 1
        if got == 0: continue
        nok += got
        obj = (A*torch.tensor(GA, dtype=torch.float32)).sum() \
            + (b*torch.tensor(GB, dtype=torch.float32)).sum()
        if th is not None:
            obj = obj + (th*torch.tensor(GT, dtype=torch.float32)).sum()
        opt.zero_grad(); (obj/got).backward(); opt.step(); sched.step()
    vd = val_deploy()
    if best is None or vd < best[0]:
        best = (vd, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
    hist.append(dict(ep=ep, loss=float(np.mean(Ls)) if Ls else np.nan, val=vd, nok=nok))
    print(f"[ep {ep:3d}] loss {np.mean(Ls) if Ls else np.nan:+.5f}  ok {nok}/{len(tr)}  "
          f"val {vd:.6f}  {time.time()-te0:5.1f}s" + (f"  FAILS {fails}" if fails else ""),
          flush=True)
    if nok == 0: raise SystemExit(f"FATAL: epoch {ep} produced no gradient {fails}")
if best: net.load_state_dict(best[1]); print(f"[select] best epoch {best[2]} val {best[0]:.6f}")
net.eval()
with torch.no_grad():
    A, b, th = fwd(te)
An, bn = A.numpy().astype(float), b.numpy().astype(float)
thn = th.numpy().astype(float) if th is not None else None
rows = []
for k, i in enumerate(te):
    r1 = R.solve(D[i], A=An[k], b=bn[k], cut_cap=0.0)
    if r1 is None: continue
    zq = qz_restore(VEH, T, S, D[i], r1["z"])
    if zq is None: continue
    pr2 = R.solve(D[i], z_lo=zq, z_hi=zq)
    if pr2 is None: continue
    zt = np.array(REF[i]["z"]); pt = np.array(REF[i]["Peng"])
    rows.append(dict(gap=100*(pr2["cost"]-REF[i]["cost"])/abs(REF[i]["cost"]),
                     disc=100*float(np.abs(zq-zt).sum()/max(zt.sum(), 1e-9)),
                     cont=100*float(np.abs(pr2["Peng"]-pt).sum()/max(pt.sum(), 1e-9))))
G = np.array([r["gap"] for r in rows])
print(f"\n[TEST] {len(rows)}/{len(te)}  gap {G.mean():+.4f}%  "
      f"discrete {np.mean([r['disc'] for r in rows]):.2f}%  "
      f"continuous {np.mean([r['cont'] for r in rows]):.2f}%  ({time.time()-t0:.0f}s)")
json.dump(dict(seed=a.seed, rows=rows, gap_mean=float(G.mean()),
               disc_mean=float(np.mean([r["disc"] for r in rows])),
               cont_mean=float(np.mean([r["cont"] for r in rows])),
               hist=hist, args=vars(a)),
          open(f"results/train_s{a.seed}{a.tag}.json", "w"), indent=1)
os.makedirs("checkpoints", exist_ok=True)
torch.save(dict(net=net.state_dict(), args=vars(a)), f"checkpoints/vehicle_s{a.seed}.pt")

# %% Deployment-aware training of the learned cut (vehicle), after the audit.
#    Forward = what is actually deployed: cut relaxation -> L1 projection (integer
#    schedule) -> continuous re-solve -> deployed cost.  Loss = deployed cost / J_R
#    (self-supervised, no labels).  Backward:
#      d cost / d z_int     exact, through the re-solve (cone layer 2, incl. E_T term)
#      d z_int / d z_relax  straight-through ("ste") or blackbox finite difference of
#                           the projection along the descent direction ("bb")
#      d z_relax / d (A,b)  exact, through the cut relaxation (cone layer 1)
#    Selection = deployed validation cost (label-free).  --check runs the descent test.
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import os, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import gurobipy as gp
from gurobipy import GRB
from src_hv import load_pool, splits, HVRelax
from cutnet_hv import CutNetHV, rows_torch
from diffhv import DiffHV
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--grad", default="ste", choices=["ste", "bb"]); ap.add_argument("--bb-lam", type=float, default=2.0)
ap.add_argument("--hidden", type=int, default=16); ap.add_argument("--check", action="store_true")
# d cost / d z_int: "cont" = cone layer 2 (exact for a continuous z, poor for unit moves);
# "int" = actual cost change of moving each z_t by +-1 (2T re-solves), CombOptNet-style
ap.add_argument("--zgrad", default="int", choices=["cont", "int"])
ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

# %% data and layers
VEH, D, REF = load_pool("results/pool/pool_*.json"); T, S = VEH["T"], VEH["S_MODES"]
R = HVRelax(VEH, n_cuts=4)
env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.setParam("Threads", 1); env.start()
te, va, tr = splits(len(REF))
X = torch.tensor(D/D.mean(), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
W, SC = {}, {}
for i in tr + va + te:
    r = R.solve(D[i]); W[i] = np.concatenate([r["E"], r["Peng"], r["Pbatt"], r["z"]]); SC[i] = float(r["cost"])
DD = DiffHV(R, VEH, tau=0.08)
DD._build(D[tr[0]], np.zeros((4, 4*T+1)), np.zeros(4))
AL, BE = np.array(VEH["alpha"]), np.array(VEH["beta"])


def project(i, zf, tl=60.):
    """nearest (L1) integer schedule feasible for the MINLP."""
    m = gp.Model(env=env); m.Params.TimeLimit = tl
    E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"])
    Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER); w = m.addVars(T, lb=0)
    m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t]+Pe[t] >= D[i][t])
        m.addConstr(w[t] >= z[t]-float(zf[t])); m.addConstr(w[t] >= float(zf[t])-z[t])
    m.setObjective(gp.quicksum(w[t] for t in range(T)), GRB.MINIMIZE); m.optimize()
    return None if m.SolCount == 0 else np.array([round(z[t].X) for t in range(T)], float)


def deploy(i, A, b):
    r1 = R.solve(D[i], A=A, b=b, cut_cap=0.0)
    if r1 is None: return None
    zq = project(i, np.clip(r1["z"], 0, S))
    if zq is None: return None
    r2 = R.solve(D[i], z_lo=zq, z_hi=zq)
    if r2 is None: return None
    return dict(zf=r1["z"], zq=zq, cost=float(r2["cost"]), r2=r2)


def grad(i, A, b, dep):
    """(loss, dL/dA, dL/db) of deployed cost / J_R."""
    zq, zf = dep["zq"], dep["zf"]
    R.solve(D[i], z_lo=zq, z_hi=zq)
    x2, st2 = DD.L2.forward([zq, zq], tol=1e-10)
    if x2 is None: return None
    pe = dep["r2"]["Peng"]
    dc = np.zeros(x2.size)
    dc[DD.sl2[R.v["Peng"].id]] = 2*AL*pe/SC[i]; dc[DD.sl2[R.v["z"].id]] = BE/SC[i]
    sE = DD.sl2[R.v["E"].id]; dc[sE.stop-1] = -VEH["eta"]/SC[i]
    g2 = DD.L2.vjp(st2, dc); dz = g2[0] + g2[1]
    if a.zgrad == "int":
        c0 = dep["cost"]; dz = np.zeros(T)
        for t in range(T):
            cu = cd = None
            if zq[t] < S:
                w = zq.copy(); w[t] += 1; r = R.solve(D[i], z_lo=w, z_hi=w); cu = r["cost"] if r else None
            if zq[t] > 0:
                w = zq.copy(); w[t] -= 1; r = R.solve(D[i], z_lo=w, z_hi=w); cd = r["cost"] if r else None
            if cu is not None and cd is not None: dz[t] = (cu - cd)/2
            elif cu is not None: dz[t] = cu - c0
            elif cd is not None: dz[t] = c0 - cd
            else: dz[t] = 0.0
            if cd is None and zq[t] > 0: dz[t] = min(dz[t], 0.0)      # going down is infeasible: never push down
        dz = dz/SC[i]
    if a.grad == "ste":
        dzf = dz
    else:                                            # blackbox: move the target downhill
        zl = project(i, np.clip(zf - a.bb_lam*dz/max(np.abs(dz).max(), 1e-12), 0, S))
        dzf = (zq - zl)/a.bb_lam * max(np.abs(dz).max(), 1e-12) if zl is not None else dz
    R.solve(D[i], A=A, b=b, cut_cap=0.0)
    x1, st1 = DD.L1.forward([A, b], tol=1e-10)
    if x1 is None: return None
    dl = np.zeros(x1.size); dl[DD.sl1[R.v["z"].id]] = dzf
    g1 = DD.L1.vjp(st1, dl)
    return dep["cost"]/SC[i], g1[0], g1[1]


net = CutNetHV(T, k_rows=2, hidden=a.hidden, seed=a.seed, learn_thr=False)
WR = {i: torch.tensor(W[i][None], dtype=torch.float32) for i in W}
def rows(ids):
    hh = net(Xn[ids]); return rows_torch(*hh[:7], torch.cat([WR[i] for i in ids]))

# %% descent test: does a small step along -g lower the DEPLOYED cost?
if a.check:
    out = []
    for i in va[:12]:
        with torch.no_grad(): A, b = rows([i])
        A, b = A[0].numpy().astype(float), b[0].numpy().astype(float)
        d0 = deploy(i, A, b); gr = grad(i, A, b, d0)
        if gr is None: continue
        L, gA, gb = gr; n = np.sqrt((gA**2).sum() + (gb**2).sum()) + 1e-12
        res = []
        for step in (0.05, 0.2, 0.5):
            dg = deploy(i, A - step*gA/n, b - step*gb/n)
            e = rng.normal(size=gA.shape); f_ = rng.normal(size=gb.shape); m_ = np.sqrt((e**2).sum()+(f_**2).sum())
            dr = deploy(i, A - step*e/m_, b - step*f_/m_)
            res.append((step, (dg["cost"]-d0["cost"])/SC[i] if dg else np.nan, (dr["cost"]-d0["cost"])/SC[i] if dr else np.nan))
        out.append(res); print(i, [(s, f"{g:+.2e}", f"{r:+.2e}") for s, g, r in res], flush=True)
    arr = np.array(out)
    for k, step in enumerate((0.05, 0.2, 0.5)):
        g_ = arr[:, k, 1]; r_ = arr[:, k, 2]
        print(f"step {step}: gradient lowers cost on {np.mean(g_ < -1e-9):.0%}, mean {np.nanmean(g_):+.2e} | "
              f"random lowers on {np.mean(r_ < -1e-9):.0%}, mean {np.nanmean(r_):+.2e}")
    raise SystemExit

# %% training
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
steps = a.epochs*int(np.ceil(len(tr)/a.batch)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
def val_score():
    with torch.no_grad(): A, b = rows(va)
    s = []
    for k, i in enumerate(va):
        d = deploy(i, A[k].numpy().astype(float), b[k].numpy().astype(float))
        s.append(10.0 if d is None else d["cost"]/SC[i])
    return float(np.mean(s))
best = (val_score(), {k: v.clone() for k, v in net.state_dict().items()}, -1); hist = []
print(f"[{a.grad} s{a.seed}] init val {best[0]:.6f}", flush=True)
t0 = time.time()
for ep in range(a.epochs):
    order = rng.permutation(tr); Ls, fails = [], 0
    for s0 in range(0, len(order), a.batch):
        ids = [int(i) for i in order[s0:s0+a.batch]]
        A_t, b_t = rows(ids)
        An, bn = A_t.detach().numpy().astype(float), b_t.detach().numpy().astype(float)
        GA, GB, got = np.zeros_like(An), np.zeros_like(bn), 0
        for k, i in enumerate(ids):
            d = deploy(i, An[k], bn[k]); gr = grad(i, An[k], bn[k], d) if d else None
            if gr is None: fails += 1; continue
            Ls.append(gr[0]); GA[k], GB[k] = gr[1], gr[2]; got += 1
        if not got: continue
        opt.zero_grad()
        ((A_t*torch.tensor(GA, dtype=torch.float32)).sum() + (b_t*torch.tensor(GB, dtype=torch.float32)).sum()).div(got).backward()
        opt.step(); sched.step()
    v = val_score()
    if v < best[0]: best = (v, {k: x.clone() for k, x in net.state_dict().items()}, ep)
    hist.append(dict(ep=ep, train=float(np.mean(Ls)), val=v, fails=fails))
    print(f"[ep {ep:2d}] train {np.mean(Ls):.6f}  val {v:.6f}  fails {fails}  {time.time()-t0:.0f}s", flush=True)
net.load_state_dict(best[1]); os.makedirs("checkpoints", exist_ok=True)
torch.save(dict(net=best[1], best_epoch=best[2], args=vars(a)), f"checkpoints/dep_{a.grad}_s{a.seed}{a.tag}.pt")
json.dump(dict(best_epoch=best[2], best_val=best[0], hist=hist, args=vars(a)),
          open(f"results/dep_{a.grad}_s{a.seed}{a.tag}.json", "w"), indent=1)
print(f"[done] best epoch {best[2]} val {best[0]:.6f}")

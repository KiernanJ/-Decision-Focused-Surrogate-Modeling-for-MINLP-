# %% Learned cuts INSIDE the integer restoration, CombOptNet-style (vehicle).
#    Forward:  uncut relaxation -> target z_hat -> L1 projection onto the MINLP's
#              integer-feasible set INTERSECTED with the learned cut rows A y <= b
#              (y = [E, P_eng, P_batt, z]) -> integer schedule -> re-solve -> cost.
#              A = 0 reproduces the no-cut baseline exactly.
#    Loss:     deployed cost / J_R  (self-supervised, no labels).
#    Backward: dL/dz from the true +-1 integer cost differences; decomposed into
#              downhill neighbours Delta_k (Paulus et al. 2021, eq. 3/9); for each
#              neighbour the CombOptNet mismatch (eq. 5) on the cut hyperplanes:
#                neighbour feasible w.r.t. the cuts but not chosen -> pull the
#                    nearest cut toward the current point (cut it off)
#                neighbour cut off by some rows -> push those rows past it
#              differentiated by autograd into the cut network.
#    Cuts start SLACK (margin) so training starts at the no-cut solution.
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import os, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import gurobipy as gp
from gurobipy import GRB
from src_hv import load_pool, splits, HVRelax
from cutnet_hv import CutNetHV
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--hidden", type=int, default=16); ap.add_argument("--kmoves", type=int, default=4)
ap.add_argument("--margin", type=float, default=1.0); ap.add_argument("--check", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

# %% data
VEH, D, REF = load_pool("results/pool/pool_*.json"); T, S = VEH["T"], VEH["S_MODES"]
R = HVRelax(VEH, n_cuts=4)
env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.setParam("Threads", 1); env.start()
te, va, tr = splits(len(REF))
X = torch.tensor(D/D.mean(), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
W, ZH, SC = {}, {}, {}
for i in tr + va + te:
    r = R.solve(D[i]); W[i] = np.concatenate([r["E"], r["Peng"], r["Pbatt"], r["z"]])
    ZH[i] = np.clip(r["z"], 0, S); SC[i] = float(r["cost"])
NY = 4*T + 1
net = CutNetHV(T, k_rows=2, hidden=a.hidden, seed=a.seed, learn_thr=False)


def rows_t(ids):
    """A (n,4,NY), b (n,4) in torch; b anchored SLACK: <A,w_rlx> + margin - depth."""
    aE, ape, apb, az, az2, d, d2, _ = net(Xn[ids])
    Z = torch.zeros_like(az)
    A = torch.cat([torch.cat([aE, ape, apb, az], 2), torch.cat([torch.zeros_like(aE), Z, Z, az2], 2)], 1)
    w = torch.tensor(np.stack([W[i] for i in ids]), dtype=torch.float32)
    b = torch.einsum("nkd,nd->nk", A, w) + a.margin - torch.cat([d, d2], 1)
    return A, b


def project(i, A=None, b=None, tl=60.):
    """L1 projection of z_hat onto {MINLP-feasible} cap {A y <= b + s}, s priced."""
    m = gp.Model(env=env); m.Params.TimeLimit = tl
    E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"])
    Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER); w = m.addVars(T, lb=0)
    m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t]+Pe[t] >= D[i][t])
        m.addConstr(w[t] >= z[t]-float(ZH[i][t])); m.addConstr(w[t] >= float(ZH[i][t])-z[t])
    obj = gp.quicksum(w[t] for t in range(T))
    if A is not None:
        yv = [E[t] for t in range(T+1)] + [Pe[t] for t in range(T)] + [Pb[t] for t in range(T)] + [z[t] for t in range(T)]
        s = m.addVars(len(b), lb=0)
        for k in range(len(b)):
            m.addConstr(gp.quicksum(float(A[k, j])*yv[j] for j in range(NY) if A[k, j] != 0) <= float(b[k]) + s[k])
        obj = obj + 1e3*gp.quicksum(s[k] for k in range(len(b)))
    m.setObjective(obj, GRB.MINIMIZE); m.optimize()
    if m.SolCount == 0: return None
    zq = np.array([round(z[t].X) for t in range(T)], float)
    y = np.array([E[t].X for t in range(T+1)] + [Pe[t].X for t in range(T)] + [Pb[t].X for t in range(T)] + list(zq))
    return zq, y


def resolve(i, zq):
    r = R.solve(D[i], z_lo=zq, z_hi=zq)
    if r is None: return None
    return float(r["cost"]), np.concatenate([r["E"], r["Peng"], r["Pbatt"], zq])


def deploy(i, A=None, b=None):
    p = project(i, A, b)
    if p is None: return None
    zq, y = p; r = resolve(i, zq)
    if r is None: return None
    return dict(zq=zq, y=y, cost=r[0])


def int_grad(i, zq, c0):
    """Feasible discrete steepest descent, per step t: +-1 move that is feasible AND
    cheaper; returns the signed descent vector (positive = increase z_t)."""
    desc = np.zeros(T)
    for t in range(T):
        best = 0.0
        for s_ in (+1, -1):
            if 0 <= zq[t] + s_ <= S:
                w = zq.copy(); w[t] += s_; r = resolve(i, w)
                if r is not None and c0 - r[0] > best:
                    best = c0 - r[0]; desc[t] = s_*best
    return desc/SC[i]


def mismatch(A_t, b_t, dep, i):
    """CombOptNet surrogate P = sum_k lambda_k P_Delta_k (torch, differentiable in A,b)."""
    zq, y = dep["zq"], dep["y"]
    desc = int_grad(i, zq, dep["cost"]); g = -desc     # feasible descent direction on z
    order = np.argsort(-np.abs(desc)); order = [t for t in order if abs(desc[t]) > 1e-12][:a.kmoves]
    if not order: return None, g
    mags = np.abs(desc[order]); lam = np.r_[mags[:-1] - mags[1:], mags[-1]]
    nrm = torch.linalg.norm(A_t, dim=1) + 1e-9
    dist = lambda pt: torch.abs(A_t @ torch.tensor(pt, dtype=torch.float32) - b_t)/nrm
    viol = lambda pt: (A_t @ torch.tensor(pt, dtype=torch.float32) - b_t) > 0
    P = torch.zeros(())
    Dk = np.zeros(T)
    for k, t in enumerate(order):
        Dk[t] = np.sign(desc[t])
        zn = np.clip(zq + Dk, 0, S)
        if np.allclose(zn, zq): continue
        r = resolve(i, zn)
        if r is None: continue                           # neighbour infeasible for the MINLP itself: P = 0
        yn = r[1]
        v = viol(yn)
        if v.any():                                      # cut off by learned rows -> push them past it
            P = P + float(lam[k]) * (dist(yn) * v.float()).sum()
        else:                                            # feasible but not chosen -> pull nearest cut to y
            P = P + float(lam[k]) * (-torch.logsumexp(-dist(y)/0.1, 0)*0.1)
    return P, g


# %% descent test
if a.check:
    res = []
    for i in va[:14]:
        A_t, b_t = rows_t([i]); A_t, b_t = A_t[0], b_t[0]
        An, bn = A_t.detach().numpy().astype(float), b_t.detach().numpy().astype(float)
        d0 = deploy(i, An, bn); dn = deploy(i)
        P, g = mismatch(A_t, b_t, d0, i)
        if P is None or not P.requires_grad:
            print(i, "no signal"); continue
        net.zero_grad(); P.backward()
        gp_ = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in net.parameters()]
        st = {k: v.clone() for k, v in net.state_dict().items()}
        out = []
        for lr in (0.02, 0.1, 0.5):
            with torch.no_grad():
                for p, gg in zip(net.parameters(), gp_): p -= lr*gg/(torch.sqrt(sum((x**2).sum() for x in gp_))+1e-12)
            A2, b2 = rows_t([i]); d1 = deploy(i, A2[0].detach().numpy().astype(float), b2[0].detach().numpy().astype(float))
            out.append((d1["cost"]-d0["cost"])/SC[i]*100 if d1 else np.nan)
            net.load_state_dict(st)
        res.append(out)
        print(i, f"no-cut {(dn['cost']-REF[i]['cost'])/abs(REF[i]['cost'])*100:+.3f}% | cut(init) {(d0['cost']-REF[i]['cost'])/abs(REF[i]['cost'])*100:+.3f}% | "
              f"after step (pp of J_R): " + " ".join(f"{x:+.3f}" for x in out), flush=True)
    r = np.array(res)
    for k, lr in enumerate((0.02, 0.1, 0.5)):
        print(f"step {lr}: cost lowered on {np.mean(r[:,k] < -1e-6):.0%}, raised on {np.mean(r[:,k] > 1e-6):.0%}, mean {np.nanmean(r[:,k]):+.4f} pp")
    raise SystemExit

# %% training (deployed cost; selection on deployed validation cost)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
def val_score(ids):
    with torch.no_grad(): A, b = rows_t(ids)
    s = []
    for k, i in enumerate(ids):
        d = deploy(i, A[k].numpy().astype(float), b[k].numpy().astype(float))
        s.append(10.0 if d is None else d["cost"]/SC[i])
    return float(np.mean(s))
v0 = val_score(va); best = (v0, {k: v.clone() for k, v in net.state_dict().items()}, -1); hist = []
print(f"[combopt s{a.seed}] init val {v0:.6f}  (no-cut val {np.mean([deploy(i)['cost']/SC[i] for i in va]):.6f})", flush=True)
t0 = time.time()
for ep in range(a.epochs):
    order = rng.permutation(tr); Ls = []
    for s0 in range(0, len(order), a.batch):
        ids = [int(i) for i in order[s0:s0+a.batch]]
        A_t, b_t = rows_t(ids); An, bn = A_t.detach().numpy().astype(float), b_t.detach().numpy().astype(float)
        Ptot, got = 0.0, 0
        for k, i in enumerate(ids):
            d = deploy(i, An[k], bn[k])
            if d is None: continue
            Ls.append(d["cost"]/SC[i])
            P, _ = mismatch(A_t[k], b_t[k], d, i)
            if P is not None and P.requires_grad: Ptot = Ptot + P; got += 1
        if got:
            opt.zero_grad(); (Ptot/got).backward(); opt.step()
    v = val_score(va)
    if v < best[0]: best = (v, {k: x.clone() for k, x in net.state_dict().items()}, ep)
    hist.append(dict(ep=ep, train=float(np.mean(Ls)), val=v))
    print(f"[ep {ep:2d}] train {np.mean(Ls):.6f}  val {v:.6f}  {time.time()-t0:.0f}s", flush=True)
net.load_state_dict(best[1])
# test: learned cut vs no cut, same projection
rows = []
with torch.no_grad(): A, b = rows_t(te)
for k, i in enumerate(te):
    d = deploy(i, A[k].numpy().astype(float), b[k].numpy().astype(float)); dn = deploy(i)
    g = lambda dd: 100*(dd["cost"]-REF[i]["cost"])/abs(REF[i]["cost"]) if dd else np.nan
    rows.append(dict(idx=i, gap=g(d), gap_nocut=g(dn),
                     disc=100*float(np.abs(d["zq"]-np.array(REF[i]["z"])).sum()/max(np.sum(REF[i]["z"]), 1e-9)) if d else np.nan))
G = np.array([[r["gap"], r["gap_nocut"]] for r in rows])
print(f"[TEST] learned cut {np.nanmean(G[:,0]):.4f}%  vs no-cut {np.nanmean(G[:,1]):.4f}%  | better on {int((G[:,0] < G[:,1]-1e-9).sum())}, "
      f"worse on {int((G[:,0] > G[:,1]+1e-9).sum())} of {len(G)} | best epoch {best[2]}")
os.makedirs("checkpoints", exist_ok=True)
torch.save(dict(net=best[1], args=vars(a)), f"checkpoints/combopt_s{a.seed}{a.tag}.pt")
json.dump(dict(rows=rows, hist=hist, best_epoch=best[2], args=vars(a)), open(f"results/combopt_s{a.seed}{a.tag}.json", "w"), indent=1)

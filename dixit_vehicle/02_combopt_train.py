# %% CombOptNet-style learned cuts on the Dixit/Gupta/Zhang hybrid-vehicle MILP.
#    Forward : LP relaxation -> z_hat -> L1 projection onto the MILP-feasible set
#              intersected with learned rows A y <= b (y = [E, P_eng, z], as in Dixit et al.)
#              -> z_int -> re-solve with z fixed -> deployed cost. A = 0 is the no-cut baseline.
#    Loss    : deployed cost (self-supervised; labels only score the test set).
#    Backward: feasible discrete steepest descent over +-1 moves of each z_t (true re-solves),
#              decomposed into neighbours, CombOptNet mismatch on the rows (Paulus et al. 2021).
#    Selection on deployed validation cost. Usage: python 02_combopt_train.py --vid 0
import os, sys, json, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from dixit_vehicle import Vehicle, T, SW, NY, EMAX, PMAX
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--vid", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rows", type=int, default=4); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--batch", type=int, default=20)
ap.add_argument("--margin", type=float, default=0.1); ap.add_argument("--kmoves", type=int, default=4)
ap.add_argument("--check", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--search", action="store_true")      # deploy = projection + dual-guided local search
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
V = Vehicle(a.vid)
P = json.load(open(f"results/pool/vehicle_{a.vid}.json"))
te, va, tr = list(range(100)), list(range(100, 124)), list(range(124, 224))
D = np.stack([Vehicle.demand(p["u"]) for p in P])
X = torch.tensor(D, dtype=torch.float32); mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
LPY = {i: np.array(P[i]["lp_y"]) for i in range(len(P))}; ZH = {i: np.array(P[i]["lp_z"]) for i in range(len(P))}
SC = {i: abs(P[i]["lp_cost"]) for i in range(len(P))}
# rows act on y scaled by its bounds, so every block lives in [0, 1]
SCL = np.r_[np.full(T+1, EMAX), np.full(T, PMAX), np.full(T, SW)]
# The projection prices only distance in z, so rows on the continuous block are absorbed by
# moving E / P_eng and never change the integer decision (measured). Rows act on z only.
ZMASK = np.r_[np.zeros(2*T+1), np.ones(T)]


class CutNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(T, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.ha = nn.Linear(h, a.rows*NY); self.hd = nn.Linear(h, a.rows)
        nn.init.zeros_(self.ha.weight); nn.init.normal_(self.ha.bias, std=0.5)
        nn.init.normal_(self.hd.weight, std=1e-3); nn.init.constant_(self.hd.bias, -2.0)
    def forward(self, x):
        h = self.trunk(x)
        return torch.tanh(self.ha(h)).view(-1, a.rows, NY)*torch.tensor(ZMASK, dtype=torch.float32), 0.2*torch.sigmoid(self.hd(h))


net = CutNet()
def rows(ids):
    A, d = net(Xn[ids]); w = torch.tensor(np.stack([LPY[i]/SCL for i in ids]), dtype=torch.float32)
    return A, torch.einsum("nrj,nj->nr", A, w) + a.margin - d


def deploy(i, A=None, b=None, search=False):
    pj = V.solve(D[i], target=ZH[i], A=None if A is None else A/SCL, b=b)
    if pj is None: return None
    zq = pj["z"].round(); r = V.solve(D[i], zfix=zq)
    if r is None: return None
    if search:
        zq, r = V.search(D[i], zq)
    return dict(zq=zq, y=pj["y"], cost=r["cost"], Pe=r["Pe"])


def descent(i, dep):
    c0 = dep["cost"]; desc = np.zeros(T); zq = dep["zq"]
    for t in range(T):
        best = 0.0
        for s_ in (1, -1):
            if 0 <= zq[t] + s_ <= SW:
                w = zq.copy(); w[t] += s_; r = V.solve(D[i], zfix=w)
                if r is not None and c0 - r["cost"] > best:
                    best = c0 - r["cost"]; desc[t] = s_*best
    return desc/SC[i]


def mismatch(A_t, b_t, i, dep):
    desc = descent(i, dep)
    order = [t for t in np.argsort(-np.abs(desc)) if abs(desc[t]) > 0][:a.kmoves]
    if not order: return None
    mags = np.abs(desc[order]); lam = np.r_[mags[:-1] - mags[1:], mags[-1]]
    nrm = torch.linalg.norm(A_t, dim=1) + 1e-9
    dist = lambda pt: torch.abs(A_t @ torch.tensor(pt/SCL, dtype=torch.float32) - b_t)/nrm
    P_ = torch.zeros(()); zn = dep["zq"].copy()
    for k, t in enumerate(order):
        zn = zn.copy(); zn[t] += np.sign(desc[t])
        r = V.solve(D[i], zfix=zn)
        if r is None: continue
        yn = r["y"]; v = (A_t @ torch.tensor(yn/SCL, dtype=torch.float32) - b_t) > 0
        if v.any(): P_ = P_ + float(lam[k])*(dist(yn)*v.float()).sum()
        else: P_ = P_ - float(lam[k])*0.1*torch.logsumexp(-dist(dep["y"])/0.1, 0)
    return P_


def score(ids, A=None, b=None, search=False):
    out = {}
    for k, i in enumerate(ids):
        d = deploy(i, None if A is None else A[k], None if b is None else b[k], search); out[i] = d
    return out


NC_va = score(va, search=a.search); vnc = float(np.mean([NC_va[i]["cost"]/SC[i] for i in va]))
if a.check:
    ids = va[:20]
    with torch.no_grad(): A, b = rows(ids)
    R0 = score(ids, A.numpy().astype(float), b.numpy().astype(float))
    for lr in (0.05, 0.2, 0.5):
        st = {k: v.clone() for k, v in net.state_dict().items()}
        A_t, b_t = rows(ids); Ps = [mismatch(A_t[k], b_t[k], i, R0[i]) for k, i in enumerate(ids) if R0[i]]
        Ps = [p for p in Ps if p is not None and p.requires_grad]
        if not Ps: print("no signal"); break
        net.zero_grad(); (sum(Ps)/len(Ps)).backward()
        gn = torch.sqrt(sum((p.grad**2).sum() for p in net.parameters() if p.grad is not None)) + 1e-12
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None: p -= lr*p.grad/gn
        with torch.no_grad(): A2, b2 = rows(ids)
        R1 = score(ids, A2.numpy().astype(float), b2.numpy().astype(float)); net.load_state_dict(st)
        dl = np.array([100*(R1[i]["cost"]-R0[i]["cost"])/SC[i] for i in ids if R0[i] and R1[i]])
        print(f"step {lr}: lowered {np.mean(dl < -1e-6):.0%} raised {np.mean(dl > 1e-6):.0%} mean {dl.mean():+.4f} pp (signal {len(Ps)})")
    print(f"init cut {np.mean([R0[i]['cost']/SC[i] for i in ids if R0[i]]):.6f} vs no-cut {np.mean([NC_va[i]['cost']/SC[i] for i in ids]):.6f}")
    raise SystemExit

opt = torch.optim.Adam(net.parameters(), lr=a.lr)
def val():
    with torch.no_grad(): A, b = rows(va)
    R_ = score(va, A.numpy().astype(float), b.numpy().astype(float), a.search)
    return float(np.mean([10.0 if R_[i] is None else R_[i]["cost"]/SC[i] for i in va]))
best = (val(), {k: v.clone() for k, v in net.state_dict().items()}, -1); hist = []; t0 = time.time()
print(f"[veh {a.vid}] init val {best[0]:.6f}  no-cut {vnc:.6f}", flush=True)
for ep in range(a.epochs):
    order = rng.permutation(tr)
    for s0 in range(0, len(order), a.batch):
        ids = [int(i) for i in order[s0:s0+a.batch]]
        A_t, b_t = rows(ids)
        R_ = score(ids, A_t.detach().numpy().astype(float), b_t.detach().numpy().astype(float))
        Ps = [mismatch(A_t[k], b_t[k], i, R_[i]) for k, i in enumerate(ids) if R_[i]]
        Ps = [p for p in Ps if p is not None and p.requires_grad]
        if Ps: opt.zero_grad(); (sum(Ps)/len(Ps)).backward(); opt.step()
    v = val(); hist.append(dict(ep=ep, val=v))
    if v < best[0]: best = (v, {k: x.clone() for k, x in net.state_dict().items()}, ep)
    print(f"[ep {ep:2d}] val {v:.6f} (no-cut {vnc:.6f})  {time.time()-t0:.0f}s", flush=True)
net.load_state_dict(best[1])
with torch.no_grad(): A, b = rows(te)
Rc = score(te, A.numpy().astype(float), b.numpy().astype(float), a.search); Rn = score(te, search=a.search)
rows_ = []
for i in te:
    zt = np.array(P[i]["z"]); pt = np.array(P[i]["Pe"]); J = P[i]["cost"]
    f = lambda d: dict(gap=100*(d["cost"]-J)/abs(J), disc=100*np.abs(d["zq"]-zt).sum()/max(zt.sum(), 1e-9),
                       cont=100*np.abs(d["Pe"]-pt).sum()/max(pt.sum(), 1e-9)) if d else None
    rows_.append(dict(idx=i, cut=f(Rc[i]), nocut=f(Rn[i])))
g = np.array([[r["cut"]["gap"], r["nocut"]["gap"]] for r in rows_ if r["cut"] and r["nocut"]])
print(f"[TEST] veh {a.vid}: learned cut gap {g[:,0].mean():.4f}%  no-cut {g[:,1].mean():.4f}% | better {int((g[:,0]<g[:,1]-1e-9).sum())} "
      f"worse {int((g[:,0]>g[:,1]+1e-9).sum())} of {len(g)} | best epoch {best[2]}", flush=True)
json.dump(dict(rows=rows_, hist=hist, best_epoch=best[2], args=vars(a)), open(f"results/combopt_v{a.vid}{a.tag}.json", "w"), indent=1)
torch.save(dict(net=best[1], args=vars(a)), f"results/combopt_v{a.vid}{a.tag}.pt")

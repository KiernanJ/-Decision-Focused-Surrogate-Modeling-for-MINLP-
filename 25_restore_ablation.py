"""How much of the result is the NETWORK, and how much is the restoration?

Deployment is: predict -> ONE solve -> round + restoration. The restoration has
four sub-steps (round, reserve top-up, upward repair, downward repair), which
invites a fair objection: is the restoration carrying the method?

Two cuts through it, on the same 48 held-out instances:

  ROWS  what the network supplies -- flat V and no cut, versus the learned V and
        learned cut. If restoration were doing the work, these rows would agree.
  COLS  how much restoration runs -- from bare rounding up to the full chain.
        If the network were doing nothing, the columns would agree.
"""
import os, sys, re
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts
from socp import Socp
exec(open(f"{HERE}/21_tiered_train.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

K, NTEST = 1, 48
set_cuts(2*K)
g, nl = grid(); NG = g.n_gen; S = Socp(g, nl, n_cuts=2*K)
POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
             for i in range(int(m.group(2))) if REF[i] is not None]
rng = np.random.default_rng(0); perm = rng.permutation(len(POOL))
te = [int(i) for i in perm[:NTEST]]; tr = [int(i) for i in perm[NTEST:]]
TIERS = [np.arange(NG)]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
net = TieredNet(g.n_bus, NG, [NG])
net.load_state_dict(torch.load(f"{HERE}/results/net_tier_k1best.pt")); net.eval()
with torch.no_grad():
    VR, VI, CNT = net(Xn[te])
VR, VI, CNT = [t.numpy().astype(float) for t in (VR, VI, CNT)]


def deploy(j, k, learned, level):
    pd_, qd_ = POOL[j][0], POOL[j][1]
    if learned:
        A, b = tier_rows(CNT[k], TIERS, NG, 2*K)
        r = S.solve(pd_, qd_, VR[k], VI[k], rho=1e6, A=A, b=b, cut_cap=0.0)
    else:
        r = S.solve(pd_, qd_, np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e6)
    if r is None:
        return None
    uf = r["u"]; z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0+RESERVE)*float(pd_.sum())
    if level >= 1:                                   # reserve top-up
        for t_ in o:
            if float(g.pmax @ z) >= need:
                break
            z[t_] = 1.0
    best = None
    tries = (NG+1) if level >= 2 else 1               # upward repair
    for _ in range(tries):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"]); break
        off = [t_ for t_ in o if z[t_] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, cost = best
    if level >= 3:                                   # downward repair
        for t_ in [t2 for t2 in np.argsort(uf) if z[t2] > 0.5][:6]:
            w = z.copy(); w[t_] = 0.0
            if float(g.pmax @ w) < need:
                continue
            pr = S.price(pd_, qd_, w)
            if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
                z, cost = w, pr["cost"]
    return z, cost


LEVELS = [(0, "round only"), (1, "+ reserve top-up"), (2, "+ upward repair"),
          (3, "+ downward repair (full)")]
print(f"\nOPTIMALITY GAP  --  48 held-out instances\n")
print(f"{'restoration':30s}{'flat V, no cut':>18s}{'learned V + cut':>18s}{'ratio':>9s}")
for lv, lname in LEVELS:
    out = []
    for learned in (False, True):
        G_ = []
        for k, j in enumerate(te):
            r = deploy(j, k, learned, lv)
            if r:
                G_.append(100*(r[1]-POOL[j][3])/POOL[j][3])
        out.append((np.mean(G_) if G_ else np.nan, len(G_)))
    a, b = out[0][0], out[1][0]
    print(f"{lname:30s}{a:>16.3f}% {b:>16.3f}% {a/b if b else float('nan'):>8.1f}x",
          flush=True)
print(f"\n  rows differ -> the NETWORK matters;  columns differ -> RESTORATION matters")

"""Remove the restoration asymmetry: let repair turn generators OFF too.

Restoration currently only ever switches generators ON. That makes
over-commitment unrecoverable while under-commitment is fixable, and the
network is forced to compensate -- the learned cardinality band converges to
[10.5, 12.9] against a true mean of 14.6, deliberately low, so the upward
repair can climb back. It works, but it is the network correcting for a
limitation of the restoration rather than predicting the right thing.

Downward repair closes the loop: after a feasible commitment is found, try
switching off the units the relaxation was least sure about, keep the change
when it stays feasible AND cheaper. Candidates are taken in ascending u_frac,
which is exactly the order the reserve top-up switched them on, and capped so
the cost stays bounded -- each trial is one price() solve.
"""
# %% setup
import os, sys, re, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample
from socp import Socp
exec(open(f"{HERE}/11_vcard.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

g, nl = grid(); NG = g.n_gen; S = Socp(g, nl, n_cuts=8)
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
te = [int(i) for i in perm[:24]]; tr = [int(i) for i in perm[24:]]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
order = np.argsort(-g.pmax); cum = np.cumsum(g.pmax[order])
SRt = torch.tensor([float(np.searchsorted(cum, (1.0+RESERVE)*float(p.sum())) + 1)
                    for p, _, _, _ in POOL], dtype=torch.float32)


def restore(pd_, qd_, uf, down=0):
    """Upward repair, then optionally `down` downward trials."""
    z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0 + RESERVE) * float(pd_.sum())
    for k in o:
        if float(g.pmax @ z) >= need:
            break
        z[k] = 1.0
    best = None
    for _ in range(NG + 1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"]); break
        off = [k for k in o if z[k] < 0.5]
        if not off:
            return None, 0
        z[off[0]] = 1.0
    if best is None:
        return None, 0
    z, cost = best; ntry = 0
    if down:
        # least-confident ON units first -- the ones top-up forced on
        cands = [k for k in np.argsort(uf) if z[k] > 0.5][:down]
        for k in cands:
            w = z.copy(); w[k] = 0.0
            if float(g.pmax @ w) < need:          # reserve would break
                continue
            ntry += 1
            pr = S.price(pd_, qd_, w)
            if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
                z, cost = w, pr["cost"]
    return (z.astype(int), cost), ntry


net = VCardNet(g.n_bus, NG)
net.load_state_dict(torch.load(f"{HERE}/results/net_fix_vcard.pt")); net.eval()
with torch.no_grad():
    vr, vi, lo, hi = net(Xn[te], SRt[te])
vr, vi, lo, hi = [t.numpy().astype(float) for t in (vr, vi, lo, hi)]

print(f"{'restoration':26s}{'gap':>10s}{'agree':>9s}{'units':>8s}"
      f"{'ref units':>11s}{'s/inst':>9s}")
for name, down in [("upward only (current)", 0), ("+ 3 downward trials", 3),
                   ("+ 6 downward trials", 6), ("+ 12 downward trials", 12)]:
    gaps, agrs, nu, nr, secs = [], [], [], [], []
    for k, j in enumerate(te):
        pd_, qd_, u_s, c_s = POOL[j]
        A, b = card_rows(lo[k], hi[k], NG)
        t0 = time.time()
        r = S.solve(pd_, qd_, vr[k], vi[k], rho=1e6, A=A, b=b)
        if r is None:
            continue
        res, _ = restore(pd_, qd_, r["u"], down)
        secs.append(time.time()-t0)
        if res is None:
            continue
        z, cost = res
        gaps.append(100*(cost-c_s)/c_s); agrs.append(100*float((z == u_s).mean()))
        nu.append(int(z.sum())); nr.append(int(u_s.sum()))
    print(f"{name:26s}{np.mean(gaps):>9.3f}%{np.mean(agrs):>8.1f}%"
          f"{np.mean(nu):>8.1f}{np.mean(nr):>11.1f}{np.mean(secs):>8.2f}s", flush=True)

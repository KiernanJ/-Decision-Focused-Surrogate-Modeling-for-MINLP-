"""Can CONFIDENCE-BASED VARIABLE FIXING cut the discrete error further?

Our cut constrains HOW MANY generators run. The UC learning literature's main
lever constrains WHICH: predict per-generator status and FIX the confident ones
as hard bounds, leaving only the marginal ones for the solver. Pure prediction
violates constraints often, which is why the solver stays in the loop -- so
fixing a confident SUBSET, rather than committing the whole vector, is the
hybrid the literature converges on.

The evidence it could work here: per-generator status is highly predictable
(the supervised proxy reaches 96.6% on case118, 98% on case300), so most
generators are easy and only a handful are genuinely marginal.

Measured BEFORE building a learner, as with every other cut family:

  ORACLE fixing   fix the k generators whose TRUE status we take as given, the
                  rest free. This is the ceiling -- how much is even available.
  PROXY fixing    fix where the existing self-supervised proxy is most
                  confident, which is what a real method could do TODAY with no
                  new training and no labels.

Both keep the learned V and the learned cardinality cut, so the only change is
the added fixing.
"""
import os, sys, re, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts
from socp import Socp
exec(open(f"{HERE}/21_tiered_train.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

K = 1; NDOWN = int(os.environ.get("NDOWN", "6"))
NTEST = int(os.environ.get("NTEST", "48"))
NET_TIER = os.environ.get("NET_TIER", "net_tier_k1best.pt")
NET_PS = os.environ.get("NET_PS", f"net_proxyself_{CASE}.pt")
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
net.load_state_dict(torch.load(f"{HERE}/results/{NET_TIER}")); net.eval()
with torch.no_grad():
    VR, VI, CNT = net(Xn[te])
VR, VI, CNT = [t.numpy().astype(float) for t in (VR, VI, CNT)]
PU = None
if os.path.exists(f"{HERE}/results/{NET_PS}"):
    pn = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.SiLU(),
                             torch.nn.Linear(256, 256), torch.nn.SiLU(),
                             torch.nn.Linear(256, NG))
    pn.load_state_dict(torch.load(f"{HERE}/results/{NET_PS}")); pn.eval()
    with torch.no_grad():
        PU = torch.sigmoid(pn(Xn[te])).numpy()
# A SUPERVISED per-generator predictor (3.09% error on case118 vs the
# self-supervised proxy's 9.61%). Fixing turns a prediction into a HARD
# constraint, so it needs far higher accuracy than a soft guess -- this row
# tells us whether fixing fails as an idea or merely fails with a weak
# predictor.
U = torch.tensor(np.stack([u.astype(float) for _, _, u, _ in POOL]), dtype=torch.float32)
sup = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.SiLU(),
                          torch.nn.Linear(256, 256), torch.nn.SiLU(),
                          torch.nn.Linear(256, NG))
_o = torch.optim.Adam(sup.parameters(), lr=2e-3)
_s = torch.optim.lr_scheduler.CosineAnnealingLR(_o, 8000)
_l = torch.nn.BCEWithLogitsLoss()
for _ in range(8000):
    _ll = _l(sup(Xn[tr]), U[tr]); _o.zero_grad(); _ll.backward(); _o.step(); _s.step()
with torch.no_grad():
    SUP = torch.sigmoid(sup(Xn[te])).numpy()
    acc = ((SUP > 0.5) == U[te].numpy().astype(bool)).mean()
print(f"  [supervised predictor] per-generator accuracy {100*acc:.1f}%")
PGS = {j: S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))["pg"] for j in te}


def run(name, fix_of):
    D_, C_, G_, NF = [], [], [], []
    for k, j in enumerate(te):
        pd_, qd_, u_s, c_s = POOL[j]
        lo, hi = np.zeros(NG), np.ones(NG)
        f = fix_of(k, j, u_s)
        if f is not None:
            idx, val = f
            lo[idx] = val; hi[idx] = val; NF.append(len(idx))
        A, b = tier_rows(CNT[k], TIERS, NG, 2*K)
        r = S.solve(pd_, qd_, VR[k], VI[k], rho=1e6, u_lo=lo, u_hi=hi,
                    A=A, b=b, cut_cap=0.0)
        if r is None:
            continue
        uf = r["u"]; z = (uf > 0.5).astype(float); o = np.argsort(-uf)
        need = (1.0+RESERVE)*float(pd_.sum())
        for t_ in o:
            if float(g.pmax @ z) >= need:
                break
            z[t_] = 1.0
        best = None
        for _ in range(NG+1):
            pr = S.price(pd_, qd_, z)
            if pr is not None and pr["slack"] < 1e-4:
                best = (z.copy(), pr["cost"], pr["pg"]); break
            off = [t_ for t_ in o if z[t_] < 0.5]
            if not off:
                break
            z[off[0]] = 1.0
        if best is None:
            continue
        z, cost, pg = best
        for t_ in [t2 for t2 in np.argsort(uf) if z[t2] > 0.5][:NDOWN]:
            w = z.copy(); w[t_] = 0.0
            if float(g.pmax @ w) < need:
                continue
            pr = S.price(pd_, qd_, w)
            if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
                z, cost, pg = w, pr["cost"], pr["pg"]
        D_.append(100*float((z.astype(int) != u_s).mean()))
        C_.append(100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()))
        G_.append(100*(cost-c_s)/c_s)
    nf = f"{np.mean(NF):.0f}" if NF else "-"
    print(f"{name:40s}{len(G_):>4d}{nf:>7s}{np.mean(D_):>10.2f}%"
          f"{np.mean(C_):>12.2f}%{np.mean(G_):>9.3f}%", flush=True)


print(f"\n{CASE}: {len(te)} held out, {NG} generators\n")
print(f"{'arm':40s}{'n':>4s}{'fixed':>7s}{'discrete':>11s}{'continuous':>13s}{'gap':>10s}")
run("ours (no fixing) -- current method", lambda k, j, u: None)
for frac in [0.5, 0.75, 0.9]:
    kk = int(frac*NG)
    run(f"ORACLE: fix {kk}/{NG} true statuses",
        lambda k, j, u, kk=kk: (np.arange(kk), u[:kk].astype(float)))
if PU is not None:
    for tau in [0.45, 0.4, 0.3, 0.2]:
        def pf(k, j, u, tau=tau):
            conf = np.abs(PU[k]-0.5)
            idx = np.where(conf > tau)[0]
            return None if len(idx) == 0 else (idx, (PU[k][idx] > 0.5).astype(float))
        run(f"PROXY fixing, confidence > {tau:.2f}", pf)
else:
    print("  NOTE: no self-supervised proxy checkpoint -- PROXY rows skipped")
for tau in [0.49, 0.45, 0.4, 0.3]:
    def sf(k, j, u, tau=tau):
        conf = np.abs(SUP[k]-0.5)
        idx = np.where(conf > tau)[0]
        return None if len(idx) == 0 else (idx, (SUP[k][idx] > 0.5).astype(float))
    run(f"SUPERVISED fixing (labels), conf > {tau:.2f}", sf)

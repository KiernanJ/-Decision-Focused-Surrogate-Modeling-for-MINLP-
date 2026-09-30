"""The honest headline table: everything against the NN proxy, one split.

The proxy is the baseline that matters -- it asks whether the optimiser is
needed at all. Comparing to "flat V, no cut" flatters us and settles nothing.

Every arm below is evaluated on the SAME 48 held-out instances with the SAME
restoration (round, reserve top-up, upward repair, 6 downward trials) and the
same AC pricing. Only the source of the commitment differs.

  proxy-sup    (pd,qd) -> net -> u       trained on reference commitments u*
                                         (LABELS we never use -- advantaged)
  proxy-self   (pd,qd) -> net -> u       same smoothed signal as ours, no labels
  ours-tiered  (pd,qd) -> net -> (V, K tier counts) -> HARD cuts -> relaxation
  ours-penalty (pd,qd) -> net -> V, plus a CONSTANT 1e4-per-generator penalty.
                                         This is what the old "learned
                                         cardinality cut" actually was once the
                                         dummy-band control exposed it: the band
                                         was decorative, the constant did the
                                         work. It is a legitimate method -- just
                                         not a learned cut -- so it belongs in
                                         the table under its real name.
"""
# %% setup
import os, sys, re, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp
exec(open(f"{HERE}/21_tiered_train.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())
exec(open(f"{HERE}/11_vcard.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

K, NTEST, NDOWN = 3, 48, 6
set_cuts(2*K); set_down(NDOWN)
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
avg = (g.c2*g.pmax**2 + g.c1*g.pmax + nl)/np.maximum(g.pmax, 1e-6)
MERIT = np.argsort(avg); TIERS = [MERIT[i::K] for i in range(K)]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
U = torch.tensor(np.stack([u.astype(float) for _, _, u, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
PGS = {j: S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))["pg"] for j in te}
print(f"[setup] {CASE}: {len(POOL)} inst, {len(tr)} train / {len(te)} test, "
      f"identical split & restoration for every arm\n", flush=True)


def restore(j, uf):
    pd_, qd_ = POOL[j][0], POOL[j][1]
    z = (uf > 0.5).astype(float); o = np.argsort(-uf)
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
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, cost, pg = best
    for t_ in [t2 for t2 in np.argsort(uf) if z[t2] > 0.5][:NDOWN]:
        w = z.copy(); w[t_] = 0.0
        if float(g.pmax @ w) < need:
            continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w, pr["cost"], pr["pg"]
    return z, cost, pg


def report(name, labels, uf_of):
    D, C_, G_, T = [], [], [], []
    for k, j in enumerate(te):
        t0 = time.time()
        uf = uf_of(k, j)
        if uf is None:
            continue
        out = restore(j, uf)
        if out is None:
            continue
        z, cost, pg = out
        T.append(time.time()-t0)
        D.append(100*float((z.astype(int) != POOL[j][2]).mean()))
        C_.append(100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()))
        G_.append(100*(cost-POOL[j][3])/POOL[j][3])
    print(f"{name:34s}{labels:>8s}{len(G_):>4d}{np.mean(D):>10.2f}%"
          f"{np.mean(C_):>12.2f}%{np.mean(G_):>9.3f}%{np.mean(T):>8.2f}s", flush=True)
    return np.mean(G_), np.mean(D)


print(f"{'arm':34s}{'labels':>8s}{'n':>4s}{'discrete':>11s}{'continuous':>13s}"
      f"{'gap':>10s}{'s/inst':>8s}")

# relaxation only
report("relaxation only (flat V)", "-",
       lambda k, j: (lambda r: r["u"] if r else None)(
           S.solve(POOL[j][0], POOL[j][1], np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e6)))

# supervised proxy
pnet = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.SiLU(),
                           torch.nn.Linear(256, 256), torch.nn.SiLU(),
                           torch.nn.Linear(256, NG))
opt = torch.optim.Adam(pnet.parameters(), lr=2e-3)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 8000)
lf = torch.nn.BCEWithLogitsLoss()
for _ in range(8000):
    l = lf(pnet(Xn[tr]), U[tr]); opt.zero_grad(); l.backward(); opt.step(); sch.step()
with torch.no_grad():
    PU = torch.sigmoid(pnet(Xn[te])).numpy()
report("NN proxy (SUPERVISED on u*)", "YES", lambda k, j: PU[k])

# self-supervised proxy
sp = f"{HERE}/results/net_proxyself_{CASE}.pt"
if os.path.exists(sp):
    snet = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.SiLU(),
                               torch.nn.Linear(256, 256), torch.nn.SiLU(),
                               torch.nn.Linear(256, NG))
    snet.load_state_dict(torch.load(sp)); snet.eval()
    with torch.no_grad():
        SU = torch.sigmoid(snet(Xn[te])).numpy()
    report("NN proxy (self-supervised)", "no", lambda k, j: SU[k])

# ours: constant-penalty variant (what the old soft "cardinality cut" really was)
cp = f"{HERE}/results/net_c118_big_vcard.pt"
if os.path.exists(cp):
    vnet = VCardNet(g.n_bus, NG); vnet.load_state_dict(torch.load(cp)); vnet.eval()
    order = np.argsort(-g.pmax); cum = np.cumsum(g.pmax[order])
    SRt = torch.tensor([float(np.searchsorted(cum, (1.0+RESERVE)*float(p.sum()))+1)
                        for p, _, _, _ in POOL], dtype=torch.float32)
    with torch.no_grad():
        vr, vi, lo, hi = vnet(Xn[te], SRt[te])
    vr, vi, lo, hi = [t.numpy().astype(float) for t in (vr, vi, lo, hi)]

    def uf_pen(k, j):
        A, b = card_rows(0.0, 0.0, NG, 2*K)   # always violated -> constant penalty
        r = S.solve(POOL[j][0], POOL[j][1], vr[k], vi[k], rho=1e6, A=A, b=b,
                    cut_cap=1e6)             # SOFT on purpose: that IS the method
        return r["u"] if r else None
    report("ours: V + constant penalty", "no", uf_pen)

# ours: learned tiered cuts, hard
tp = f"{HERE}/results/net_tier_k3full.pt"
tnet = TieredNet(g.n_bus, NG, [len(T) for T in TIERS])
tnet.load_state_dict(torch.load(tp)); tnet.eval()
with torch.no_grad():
    TVR, TVI, TCNT = tnet(Xn[te])
TVR, TVI, TCNT = [t.numpy().astype(float) for t in (TVR, TVI, TCNT)]


def uf_tier(k, j):
    A, b = tier_rows(TCNT[k], TIERS, NG, 2*K)
    r = S.solve(POOL[j][0], POOL[j][1], TVR[k], TVI[k], rho=1e6, A=A, b=b, cut_cap=0.0)
    return r["u"] if r else None


report("ours: V + LEARNED tiered cuts", "no", uf_tier)

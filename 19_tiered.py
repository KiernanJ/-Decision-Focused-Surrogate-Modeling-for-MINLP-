"""Is there headroom for a RICHER learned cut than global cardinality?

Global cardinality says only HOW MANY generators run. That is one number per
instance, which is exactly why a constant penalty could imitate it: with a soft
cut the right-hand side dropped out of the optimality conditions and a dummy
band [0,0] reproduced the learned band bit for bit.

Tiered cardinality says how many run IN EACH MERIT-ORDER TIER -- so it
constrains WHICH units, not just how many, while staying low-dimensional (2K
numbers for K tiers, against 864 for the free-form head that failed). A constant
cannot imitate it, because the tier counts move differently per instance.

This measures the CEILING with oracle tier counts, before anything is learned,
at realistic levels of linearisation error. All cuts are imposed HARD
(cut_cap=0); soft cuts are what made the previous family inert.
"""
# %% setup
import os, sys, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp

rng = np.random.default_rng(0)
set_cuts(24); set_down(6)
g, nl = grid(); NG = g.n_gen
S = Socp(g, nl, n_cuts=24)          # room for up to 12 tiers x 2 bounds

POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
             for i in range(int(m.group(2))) if REF[i] is not None]
perm = rng.permutation(len(POOL)); te = [int(i) for i in perm[:24]]

# merit order: average cost at full output, cheapest first
avg = (g.c2*g.pmax**2 + g.c1*g.pmax + nl) / np.maximum(g.pmax, 1e-6)
MERIT = np.argsort(avg)

VS = {}
for j in te:
    r = S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))
    VS[j] = np.r_[r["vr"], r["vi"]]
PGS = {j: S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))["pg"] for j in te}


def tiers(K):
    return [MERIT[i::K] for i in range(K)] if K > 1 else [MERIT]


def rows(K, u_star, w=0.0):
    """Oracle tier counts, imposed as hard bounds, +-w slack per tier."""
    A, b = np.zeros((24, 2*NG)), np.ones(24)*1e3
    k = 0
    for T in tiers(K):
        n_t = float(u_star[T].sum())
        A[k, NG + T] = -1.0; b[k] = -(max(0.0, n_t - w)); k += 1
        A[k, NG + T] = 1.0;  b[k] = min(len(T), n_t + w); k += 1
    return A, b


def deploy(j, V, A, b):
    pd_, qd_ = POOL[j][0], POOL[j][1]
    r = S.solve(pd_, qd_, V[:g.n_bus], V[g.n_bus:], rho=1e6, A=A, b=b, cut_cap=0.0)
    if r is None:
        return None
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
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, cost, pg = best
    for t_ in [t2 for t2 in np.argsort(uf) if z[t2] > 0.5][:6]:
        w_ = z.copy(); w_[t_] = 0.0
        if float(g.pmax @ w_) < need:
            continue
        pr = S.price(pd_, qd_, w_)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w_, pr["cost"], pr["pg"]
    return z.astype(int), cost, pg


def run(name, K, sd, w=0.0):
    D, C_, G_ = [], [], []
    for j in te:
        V = VS[j] + rng.normal(0, sd, 2*g.n_bus)
        A, b = (None, None) if K == 0 else rows(K, POOL[j][2].astype(float), w)
        out = deploy(j, V, A, b)
        if out is None:
            continue
        z, cost, pg = out
        D.append(100*float((z != POOL[j][2]).mean()))
        C_.append(100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()))
        G_.append(100*(cost-POOL[j][3])/POOL[j][3])
    print(f"{name:32s}{len(G_):>4d}{np.mean(D):>10.2f}%{np.mean(C_):>12.2f}%"
          f"{np.mean(G_):>9.3f}%", flush=True)


for sd, tag in [(0.02, "MODERATE V error (||dV||~0.31)"),
                (0.05, "LARGE V error (||dV||~0.76)")]:
    print(f"\n{tag}   -- all cuts HARD, oracle tier counts")
    print(f"{'cut family':32s}{'n':>4s}{'discrete':>11s}{'continuous':>13s}{'gap':>10s}")
    run("no cut", 0, sd)
    run("K=1  global cardinality", 1, sd)
    run("K=2  tiers", 2, sd)
    run("K=3  tiers", 3, sd)
    run("K=5  tiers", 5, sd)
    run("K=8  tiers", 8, sd)
    run("K=5  tiers, +-1 slack", 5, sd, 1.0)
    run("K=5  tiers, +-2 slack", 5, sd, 2.0)

"""Is there headroom in the ROUNDING THRESHOLD? (L2O-MINLP's learnable-threshold idea)

We round the relaxation's fractional commitment at a hard-coded 0.5, on a
solution where ~40 of 54 generators are fractional. L2O-MINLP (arXiv 2410.11061)
instead LEARNS a per-variable threshold h and rounds up when the fractional part
exceeds it, trained self-supervised with a smoothed-sigmoid surrogate.

That suits us better than variable fixing, which needed labels to help at all
and made instances infeasible when wrong: a threshold only moves the rounding
decision, and restoration still runs afterwards, so it cannot break feasibility.

Measured before building anything:
  GLOBAL   one threshold for every generator, swept. A free win if 0.5 is
           simply the wrong constant.
  ORACLE   per-instance, the threshold in the sweep that happens to score best
           -- the ceiling a perfect per-instance predictor could reach.
  ORACLE-G per-generator, rounded toward the TRUE status when the relaxation is
           fractional: the ceiling for a per-generator threshold.
"""
import os, sys, re, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts
from socp import Socp
exec(open(f"{HERE}/21_tiered_train.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

K = 1; NDOWN = int(os.environ.get("NDOWN", "6")); NTEST = int(os.environ.get("NTEST", "48"))
NET_TIER = os.environ.get("NET_TIER", "net_tier_k1best.pt")
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
PGS = {j: S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))["pg"] for j in te}

# solve the relaxation ONCE per instance; only the rounding varies after this
UF = {}
for k, j in enumerate(te):
    A, b = tier_rows(CNT[k], TIERS, NG, 2*K)
    r = S.solve(POOL[j][0], POOL[j][1], VR[k], VI[k], rho=1e6, A=A, b=b, cut_cap=0.0)
    if r is not None:
        UF[j] = r["u"]
fr = np.concatenate([v for v in UF.values()])
print(f"\n{CASE}: {len(UF)}/{len(te)} relaxations solved")
print(f"  fractional generators per instance: "
      f"{np.mean([((v>1e-4)&(v<1-1e-4)).sum() for v in UF.values()]):.1f} / {NG}")
print(f"  relaxed u distribution: <0.1 {100*(fr<0.1).mean():.0f}%  "
      f"0.1-0.9 {100*((fr>=0.1)&(fr<=0.9)).mean():.0f}%  >0.9 {100*(fr>0.9).mean():.0f}%\n")


def restore(j, z0, uf):
    pd_, qd_ = POOL[j][0], POOL[j][1]
    z = z0.copy(); o = np.argsort(-uf)
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


def score(rows):
    v = np.array(rows)
    return v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean()


PER = {}
THR = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print(f"{'rounding rule':34s}{'n':>4s}{'discrete':>11s}{'continuous':>13s}{'gap':>10s}")
for th in THR:
    rows = []
    for j in UF:
        out = restore(j, (UF[j] > th).astype(float), UF[j])
        if out is None:
            continue
        z, c, pg = out
        rec = (100*float((z.astype(int) != POOL[j][2]).mean()),
               100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()),
               100*(c-POOL[j][3])/POOL[j][3])
        rows.append(rec); PER.setdefault(j, {})[th] = rec
    d, c_, gp = score(rows)
    tag = f"GLOBAL threshold {th:.1f}" + ("   <- current" if th == 0.5 else "")
    print(f"{tag:34s}{len(rows):>4d}{d:>10.2f}%{c_:>12.2f}%{gp:>9.3f}%", flush=True)

best = [min(PER[j].values(), key=lambda r: r[2]) for j in PER if PER[j]]
d, c_, gp = score(best)
print(f"{'ORACLE per-instance threshold':34s}{len(best):>4d}{d:>10.2f}%{c_:>12.2f}%{gp:>9.3f}%")

rows = []
for j in UF:
    uf = UF[j]; z = (uf > 0.5).astype(float)
    frac = (uf > 1e-4) & (uf < 1-1e-4)
    z[frac] = POOL[j][2][frac].astype(float)      # oracle only where fractional
    out = restore(j, z, uf)
    if out is None:
        continue
    zz, c, pg = out
    rows.append((100*float((zz.astype(int) != POOL[j][2]).mean()),
                 100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()),
                 100*(c-POOL[j][3])/POOL[j][3]))
d, c_, gp = score(rows)
print(f"{'ORACLE per-generator rounding':34s}{len(rows):>4d}{d:>10.2f}%{c_:>12.2f}%{gp:>9.3f}%")

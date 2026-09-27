"""Is the cardinality cut robust to being predicted WRONG, and is it learnable?

09 found that a single integer -- how many generators run -- cuts the gap
5.625% -> 1.773% at large V error and lifts agreement 69.2% -> 88.0%, beating
even a cut that hands over 8 correct generators. That was the ORACLE n*.

A network will get n* wrong sometimes. Two things decide whether this is a
method or an artifact:

  A. ROBUSTNESS. Impose n_hat = n* + delta for delta in -2..+2, and a BAND
     [n*-w, n*+w]. If only delta=0 helps, the cut is too brittle to learn
     against. A band that tolerates error is the honest design.
  B. LEARNABILITY. n* is an aggregate of demand, so a linear fit on total
     demand should already be close. Its residual bounds how hard the head's
     job is -- and a trivial predictor that works is itself the baseline the
     network must beat.
"""
# %% setup
import os, sys, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, grid, sample, deploy
from socp import Socp

rng = np.random.default_rng(0)
g, nl = grid(); S = Socp(g, nl, n_cuts=8); G = g.n_gen

POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
             for i in range(int(m.group(2))) if REF[i] is not None]

# %% B. how predictable is n* from demand alone?
D = np.array([float(p.sum()) for p, _, _, _ in POOL])
NS = np.array([float(u.sum()) for _, _, u, _ in POOL])
c = np.corrcoef(D, NS)[0, 1]
A_ = np.c_[D, np.ones(len(D))]
coef, *_ = np.linalg.lstsq(A_, NS, rcond=None)
pred = A_ @ coef
print(f"B. n* from total demand: corr {c:+.3f}, "
      f"linear fit |error| mean {np.abs(pred-NS).mean():.2f}, "
      f"max {np.abs(pred-NS).max():.2f} units")
print(f"   n* ranges {NS.min():.0f}-{NS.max():.0f} over {len(POOL)} instances\n",
      flush=True)

# %% A. robustness of the cut to a wrong n_hat
sub = POOL[:24]
VS = []
for p, q, u, _ in sub:
    r = S.price(p, q, u.astype(float)); VS.append(np.r_[r["vr"], r["vi"]])
VS = np.stack(VS)


def band_cut(lo, hi):
    A, b = np.zeros((8, 2*G)), np.ones(8)
    A[0, G:] = -1.0; b[0] = -lo          # sum u >= lo
    A[1, G:] = 1.0;  b[1] = hi           # sum u <= hi
    return A, b


SPECS = [("no cut", None), ("n*-2", (-2, -2)), ("n*-1", (-1, -1)),
         ("n* exact", (0, 0)), ("n*+1", (1, 1)), ("n*+2", (2, 2)),
         ("band +-1", (-1, 1)), ("band +-2", (-2, 2)),
         ("[n*, n*+2]", (0, 2)), ("[n*-2, n*]", (-2, 0))]

print(f"A. robustness   (rows = imposed cardinality, cols = V error)")
print(f"{'cut':>12s}" + "".join(f"{s:>19s}" for s in ["dV=0.16", "dV=0.31", "dV=0.76"]))
for name, spec in SPECS:
    cells = []
    for sd in [0.01, 0.02, 0.05]:
        gaps, agrs = [], []
        for k, (pd_, qd_, u_star, c_ref) in enumerate(sub):
            v = VS[k] + rng.normal(0, sd, 2*g.n_bus)
            n_star = float(u_star.sum())
            if spec is None:
                A, b = None, None
            else:
                A, b = band_cut(n_star+spec[0], n_star+spec[1])
            cc, d = deploy(S, g, pd_, qd_, v[:g.n_bus], v[g.n_bus:], A, b)
            if cc < 1e8:
                gaps.append(100*(cc-c_ref)/c_ref)
                agrs.append(100*float((d["u"] == u_star).mean()))
        cells.append(f"{np.mean(gaps):>8.3f}% {np.mean(agrs):>6.1f}%"
                     if gaps else f"{'--':>16s}")
    print(f"{name:>12s}" + "".join(f"{c_:>19s}" for c_ in cells), flush=True)

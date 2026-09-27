"""Do cuts have headroom where it matters -- at an IMPERFECT linearisation point?

The earlier cut arms failed, but they were the wrong cuts aimed at the wrong
problem. At a good V the relaxation rounds to the exact optimum with 40 of 54
generators still FRACTIONAL, so removing fractionality buys nothing. And the
learned rows were free-form hyperplanes in (pg,u): nothing made them valid
inequalities, so they could cut off the optimum, and every one was active by
construction.

The real job for a cut is robustness. V will never be exact, and the tolerance
curve shows the cost of that: ||dV||=0.31 already costs 1.665% and 11 points of
agreement. If a cut restores that, cuts are the thing that makes ONE-SHOT
deployment work -- a stronger claim than fixing fractionality.

This measures the CEILING with oracle cuts, before building any learner. Every
family here is an aggregate statistic of the optimal commitment -- the kind of
thing a network can predict from (pd,qd) -- not the commitment itself:

  count      sum_i u_i >= n*                 how many units run
  count=     n* <= sum_i u_i <= n*           exactly how many
  capacity   sum_i pmax_i u_i >= P*          how much capacity runs
  count+cap  both of the above
  SET        u_i >= 1 for every i in S*      the answer itself -- upper bound,
                                             not a proposal
"""
# %% setup
import os, sys, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, RESERVE, grid, sample, deploy
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
POOL = POOL[:24]
VS = []
for p, q, u, _ in POOL:
    r = S.price(p, q, u.astype(float)); VS.append(np.r_[r["vr"], r["vi"]])
VS = np.stack(VS)
print(f"[setup] {len(POOL)} instances\n", flush=True)


# %% oracle cut families, all as A @ [pg; u] <= b
def cuts_for(kind, u_star):
    A, b = np.zeros((8, 2*G)), np.ones(8)          # unused rows are inactive
    n_star, P_star = float(u_star.sum()), float(g.pmax @ u_star)
    k = 0
    if kind in ("count", "count=", "count+cap"):
        A[k, G:] = -1.0; b[k] = -n_star; k += 1              # sum u >= n*
    if kind in ("count=", "count+cap"):
        A[k, G:] = 1.0; b[k] = n_star; k += 1                # sum u <= n*
    if kind in ("capacity", "count+cap"):
        A[k, G:] = -g.pmax; b[k] = -P_star; k += 1           # pmax.u >= P*
    if kind == "SET":
        on = np.where(u_star > 0.5)[0][:8]
        for j in on:
            A[k, G+j] = -1.0; b[k] = -1.0; k += 1
    return (A, b) if k else (None, None)


FAMILIES = ["none", "count", "count=", "capacity", "count+cap", "SET"]

# %% sweep V error x cut family
print(f"{'||dV||':>8s}  " + "".join(f"{f:>12s}" for f in FAMILIES))
for sd in [0.0, 0.01, 0.02, 0.05]:
    line_g, line_a, nrm = [], [], []
    for fam in FAMILIES:
        gaps, agrs = [], []
        for k, (pd_, qd_, u_star, c_ref) in enumerate(POOL):
            v = VS[k] + rng.normal(0, sd, 2*g.n_bus)
            if fam == "none":
                nrm.append(np.linalg.norm(v - VS[k]))
            A, b = cuts_for(fam, u_star.astype(float))
            cc, d = deploy(S, g, pd_, qd_, v[:g.n_bus], v[g.n_bus:], A, b)
            if cc < 1e8:
                gaps.append(100*(cc-c_ref)/c_ref)
                agrs.append(100*float((d["u"] == u_star).mean()))
        line_g.append(np.mean(gaps) if gaps else np.nan)
        line_a.append(np.mean(agrs) if agrs else np.nan)
    print(f"{np.mean(nrm):>8.4f}  " + "".join(f"{x:>11.3f}%" for x in line_g))
    print(f"{'agree':>8s}  " + "".join(f"{x:>11.1f}%" for x in line_a), flush=True)
    print()

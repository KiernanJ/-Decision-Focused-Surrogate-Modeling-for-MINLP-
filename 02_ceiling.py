"""How much of the remaining gap is V-prediction error, and how much is rounding?

The deployment gap mixes two things nobody has separated:

  (a) the network's V differs from QCAC's converged linearisation point
  (b) we solve the CONTINUOUS relaxation and round, while QCAC iterative solves
      a MIQCQP with 54 binaries

If (b) dominates, the V head is already at its ceiling and more training is
wasted -- and it would explain why the cuts failed: they were attacking (a)'s
symptom (a fractional relaxation) when the cost lives in (b).

ORACLE V: the linearisation point at the reference commitment's own solution,
i.e. exactly the fixed point QCAC's loop converges to. Deploying from it is the
best any V-predicting method can do. That is the floor.
"""
# %% setup
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, grid, sample, deploy, set_cuts
from socp import Socp

SEED, N, NTEST = 0, 32, 8

# %% data
g, nl = grid()
inst = sample(g, N, seed=SEED)
REF = list(np.load(f"{HERE}/results/ref_{CASE}_s{SEED}_n{N}.npz",
                   allow_pickle=True)["ref"])
te = [i for i in range(N-NTEST, N) if REF[i] is not None]
S = Socp(g, nl, n_cuts=8)

# %% three linearisation points, identical pipeline after that
rows = []
for i in te:
    pd_, qd_ = inst[i]
    ref_c, ref_u = REF[i]["cost"], REF[i]["u"]

    # flat V -- the untrained baseline
    c_flat, d_flat = deploy(S, g, pd_, qd_, np.ones(g.n_bus), np.zeros(g.n_bus))

    # ORACLE V -- QCAC's converged point, obtained by pricing the reference
    # commitment: price() re-linearises with u fixed until the slack vanishes,
    # so its V IS the fixed point of the map QCAC iterates.
    pr = S.price(pd_, qd_, ref_u.astype(float))
    c_orc, d_orc = deploy(S, g, pd_, qd_, pr["vr"], pr["vi"])

    # how fractional is the relaxation at the oracle point?
    r = S.solve(pd_, qd_, pr["vr"], pr["vi"], rho=1e6)
    nfrac = int(((r["u"] > 1e-4) & (r["u"] < 1-1e-4)).sum())

    rows.append(dict(
        i=i,
        flat=100*(c_flat-ref_c)/ref_c if c_flat < 1e8 else np.nan,
        orc=100*(c_orc-ref_c)/ref_c if c_orc < 1e8 else np.nan,
        agr_flat=100*float((d_flat["u"] == ref_u).mean()) if d_flat else np.nan,
        agr_orc=100*float((d_orc["u"] == ref_u).mean()) if d_orc else np.nan,
        nfrac=nfrac))

# %% report
f = lambda k: np.nanmean([x[k] for x in rows])
print(f"\ncase118, seed {SEED}, {len(te)} held-out instances")
print(f"{'linearisation point':28s}{'gap':>9s}{'agree':>9s}")
print(f"{'flat V (1,0) -- baseline':28s}{f('flat'):>8.3f}%{f('agr_flat'):>8.1f}%")
print(f"{'ORACLE V (QCAC fixed pt)':28s}{f('orc'):>8.3f}%{f('agr_orc'):>8.1f}%")
print(f"\n  learned V (measured earlier)      +1.388%    86.1%")
print(f"\n  mean fractional generators at the ORACLE point: "
      f"{np.mean([x['nfrac'] for x in rows]):.1f} / {g.n_gen}")
print(f"""
  The ORACLE row is the FLOOR: perfect V, same rounding. Whatever gap remains
  there is pure rounding cost and no amount of V training can remove it.""")

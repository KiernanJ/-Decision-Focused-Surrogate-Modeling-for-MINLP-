"""Instances, split and references after the 2026-10-06 audit.

Changes from poolload.py:
  * the split is defined over the DRAWN instances, not over the instances whose
    reference happened to succeed -- so failed references can no longer reshape
    the instance distribution or the split, and the label-free method trains on
    exactly the drawn training instances;
  * a reference is usable only if the QCAC iteration CONVERGED (slack <= 1e-3);
  * scoring prices the reference commitment with the SAME evaluator as the
    methods (J_ref = price(u_ref)), see eval code.
"""
from __future__ import annotations
import glob, os, pickle
import numpy as np

N_DRAW, LO, SEED = 240, 0.55, 200
# case300's true capacity (326.8 pu with the corrected slack-bus unit) is below
# the 10% reserve requirement at s = 1.30 (337.8 pu): those draws are infeasible,
# not hard. Its band therefore stops at 1.20.
HI = {"case118": 1.30, "case300": 1.20}
# PGLib "api" is ALREADY at its stressed operating point (lines near their ratings) and its
# reserve becomes infeasible above s ~ 1.16, so its band sits around nominal instead.
BAND = {"pglib_opf_case118_ieee__api": (0.85, 1.05), "pglib_opf_case300_ieee__api": (0.70, 0.90)}   # 0.85-1.05: 11/22 infeasible
NT, NV, NTR = 48, 24, 144


def instances(g, case):
    """[(pd, qd)] for all N_DRAW draws -- identical to the wide pool's sampler."""
    import hardgate
    lo, hi = BAND.get(case, (LO, HI.get(case)))
    return [(p, q) for p, q, _ in hardgate.sample_hard(g, N_DRAW, seed=SEED, lo=lo,
                                                       hi=hi, p_out=0.0)]


def split():
    perm = np.random.default_rng(0).permutation(N_DRAW)
    te = [int(i) for i in perm[:NT]]
    va = [int(i) for i in perm[NT:NT+NV]]
    tr = [int(i) for i in perm[NT+NV:NT+NV+NTR]]
    return te, va, tr


def refs(case, d="results/pool2"):
    """{instance index: ref dict} for every solved index (converged or not)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, f"{case}_*.pkl"))):
        out.update(pickle.load(open(f, "rb"))["res"])
    return out


def converged(r):
    return r is not None and bool(r.get("converged", False))


# FRESH TEST SET (2026-10-08, after the second audit). Every earlier test number was
# looked at while the method was being designed (W, the safeguard), so the frozen
# protocol is scored ONCE on draws nothing has seen: same sampler, same band, new seed.
# Indices N_DRAW .. N_DRAW+N_FRESH-1, so no existing index changes.
N_FRESH, FRESH_SEED = 48, 777


def all_instances(g, case):
    import hardgate
    lo, hi = BAND.get(case, (LO, HI.get(case)))
    fresh = [(p, q) for p, q, _ in hardgate.sample_hard(g, N_FRESH, seed=FRESH_SEED, lo=lo,
                                                        hi=hi, p_out=0.0)]
    fresh2 = [(p, q) for p, q, _ in hardgate.sample_hard(g, N_FRESH, seed=FRESH2_SEED, lo=lo,
                                                         hi=hi, p_out=0.0)]
    fresh3 = [(p, q) for p, q, _ in hardgate.sample_hard(g, N_FRESH, seed=FRESH3_SEED, lo=lo,
                                                         hi=hi, p_out=0.0)]
    extra = [(p, q) for p, q, _ in hardgate.sample_hard(g, N_EXTRA, seed=EXTRA_SEED, lo=lo,
                                                        hi=hi, p_out=0.0)]
    fresh4 = [(p, q) for p, q, _ in hardgate.sample_hard(g, N_FRESH, seed=FRESH4_SEED, lo=lo,
                                                         hi=hi, p_out=0.0)]
    fresh5 = [(p, q) for p, q, _ in hardgate.sample_hard(g, N_FRESH, seed=FRESH5_SEED, lo=lo,
                                                         hi=hi, p_out=0.0)]
    band = [(p, q) for p, q, _ in hardgate.sample_hard(g, N_BAND, seed=BAND_SEED, lo=0.78, hi=0.88, p_out=0.0)]
    fresh6 = [(p, q) for p, q, _ in hardgate.sample_hard(g, N_FRESH, seed=FRESH6_SEED, lo=lo,
                                                         hi=hi, p_out=0.0)]
    return instances(g, case) + fresh + fresh2 + fresh3 + extra + fresh4 + fresh5 + band + fresh6


def fresh_ids():
    return list(range(N_DRAW, N_DRAW + N_FRESH))


def strict_refs(case, root=None):
    """CACHED strict prices of the converged reference commitments (2026-10-09): {i: (J_ref, pg, vr, vi,
    n_repaired, u)} from combopt_ac.price_strict -- computed once by acuc/32_cache_refs.py, loaded by
    every evaluation instead of re-pricing the same commitments in each script."""
    import pickle
    root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "acuc", "results")
    return pickle.load(open(os.path.join(root, f"strict_refs_{case}.pkl"), "rb"))


# UNTOUCHED TEST SET (2026-10-09, after re-checking Kiernan's audit): the first fresh set of case118 was
# used to DESIGN the label-free V (scripts 26-39 ran on fresh_ids()[:22]), so it is no longer a clean test.
# A second fresh draw, same sampler/band, new seed, indices after the first fresh set. Evaluated ONCE.
FRESH2_SEED = 4242


def fresh2_ids():
    return list(range(N_DRAW + N_FRESH, N_DRAW + 2*N_FRESH))


# FINAL TEST SET (2026-10-11, after the third audit): case118 fresh2 was scored during development (41/42, 10-09/10) and
# the case300/PGLib fresh2 results were seen before the audit's bug fixes, so the fixed method is scored ONCE on a third
# draw: same sampler/band, new seed, indices after fresh2 (no existing index changes).
FRESH3_SEED = 9191


def fresh3_ids():
    return list(range(N_DRAW + 2*N_FRESH, N_DRAW + 3*N_FRESH))


# EXTRA UNLABELED TRAINING DRAWS (2026-10-11, label-free cuts): same sampler/band, new seed, indices after fresh3 (no existing
# index changes). Never referenced/solved by a reference solver: their only targets are our own cheap best-of solves.
N_EXTRA, EXTRA_SEED = 300, 5151


def extra_ids():
    return list(range(N_DRAW + 3*N_FRESH, N_DRAW + 3*N_FRESH + N_EXTRA))


# FINAL TEST SET for the label-free method (2026-10-11 night): every earlier test draw was scored with an earlier method
# version, so the label-free method is scored ONCE on a fourth draw: same sampler/band, new seed, indices after extra.
FRESH4_SEED = 6464


def fresh4_ids():
    return list(range(N_DRAW + 3*N_FRESH + N_EXTRA, N_DRAW + 4*N_FRESH + N_EXTRA))


# SECOND FINAL TEST SET (2026-10-13): fresh4 was scored with the _lfx method and its result seen, so a re-designed case118 method
# (clean V-net, cuts/rows trained at the deployed V) is scored ONCE on a fifth draw: same sampler/band, new seed, indices after fresh4.
FRESH5_SEED = 7373


def fresh5_ids():
    return list(range(N_DRAW + 4*N_FRESH + N_EXTRA, N_DRAW + 5*N_FRESH + N_EXTRA))


# SPARSE-REGIME DIAGNOSTIC SET (2026-10-13, case118 only): 60 unlabeled draws with load scale 0.78-0.88 (the regime where the reference commits ~13 units and both methods fail), NOT a test set and
# disjoint from every other block (indices after fresh5). Used to study how the deployment penalty rho matters in that regime (priced cost vs the best rho; reference gap).
N_BAND, BAND_SEED = 60, 8484


def band_ids():
    return list(range(N_DRAW + 5*N_FRESH + N_EXTRA, N_DRAW + 5*N_FRESH + N_EXTRA + N_BAND))


# THIRD FINAL TEST SET (2026-10-13): fresh5 was scored with the clean method and its failures seen, so the retry safeguard (and the clean method without it) are scored ONCE on a sixth draw:
# same sampler/band as fresh..fresh5 (load scale 0.55-1.30), new seed, indices after the band diagnostic block.
FRESH6_SEED = 9393


def fresh6_ids():
    return list(range(N_DRAW + 5*N_FRESH + N_EXTRA + N_BAND, N_DRAW + 6*N_FRESH + N_EXTRA + N_BAND))

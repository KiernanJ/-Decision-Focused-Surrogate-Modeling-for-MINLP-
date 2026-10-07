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
NT, NV, NTR = 48, 24, 144


def instances(g, case):
    """[(pd, qd)] for all N_DRAW draws -- identical to the wide pool's sampler."""
    import hardgate
    return [(p, q) for p, q, _ in hardgate.sample_hard(g, N_DRAW, seed=SEED, lo=LO,
                                                       hi=HI[case], p_out=0.0)]


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

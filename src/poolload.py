"""One place that knows how to build an instance pool, for both bands.

  base  the original sampler, s ~ U[0.80,1.15], references in
        qcac_clean/results/ref_<case>_s<seed>_n<n>.npz
  wide  s ~ U[0.55,1.30], references in difflayer_work/results/wide/*.pkl

Measured reason the wide band exists: on `base` a majority-vote CONSTANT
commitment lands +0.347% from optimal on case118 and +0.215% on case300, so no
learned map has more than that to win -- the tuning wall every case300 variant
hit. Widening the band moves the constant to +1.713% and is a sampler-only
change, unlike outages, which also need availability plumbed into the network
input and into u_hi through deploy/DiffDeploy.
"""
from __future__ import annotations

import glob
import os
import pickle
import re

import numpy as np

WIDE = dict(lo=0.55, hi=1.30, p_out=0.0, seed=200)
BASE = dict(lo=0.80, hi=1.15, p_out=0.0)


def load_pool(case, kind, qc_dir, wide_dir="results/wide"):
    """-> [(pd, qd, u_ref, cost_ref)], newest-consistent ordering."""
    if kind == "base":
        from pipeline import sample
        from acopf_data import load as _load          # noqa: F401  (grid built by caller)
        pool = []
        for f in sorted(os.listdir(f"{qc_dir}/results")):
            m = re.fullmatch(rf"ref_{case}_s(\d+)_n(\d+)\.npz", f)
            if not m:
                continue
            ref = list(np.load(f"{qc_dir}/results/{f}", allow_pickle=True)["ref"])
            n = int(m.group(2))
            inst = sample(_G[0], n, seed=int(m.group(1)))
            pool += [(inst[i][0], inst[i][1], ref[i]["u"], ref[i]["cost"])
                     for i in range(n) if ref[i] is not None]
        return pool

    if kind != "wide":
        raise ValueError(f"unknown pool kind {kind!r}")
    # The shards each re-derive the SAME instance list from (seed, N, lo, hi),
    # so an instance is identified by its global index and never re-sampled here.
    merged = {}
    for f in sorted(glob.glob(f"{wide_dir}/{case}_W_*.pkl")):
        d = pickle.load(open(f, "rb"))
        for i, r in d["res"].items():
            if r is not None:
                merged[i] = (d["inst"][i], r)
    return [(merged[i][0][0], merged[i][0][1],
             np.asarray(merged[i][1]["u"]), float(merged[i][1]["cost"]))
            for i in sorted(merged)]


_G = [None]


def set_grid(g):
    """`base` needs the grid to replay pipeline.sample; `wide` does not."""
    _G[0] = g

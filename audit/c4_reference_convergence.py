"""Critical 4: non-converged QCAC references in the pools, and what they do to the
published case300 numbers.

Part 1: count references whose stored slack exceeds the QCAC stopping tolerance
        (1e-3, src/acuc.py) and how many land in the test split.
Part 2: re-score the COMMITTED case300 result rows (ours and proxy) on converged
        test instances only. Rows are aligned to test positions with test
        position 39 (pool index 238) removed; that instance fails restoration for
        every seed and both arms (verified by parity alignment of the rows).
Runtime: seconds.
"""
import json
import _common
from _common import REPO, ref_slack, splits
import numpy as np

TOL = 1e-3
for case in ("case118", "case300"):
    sl = ref_slack(case)
    te, _, tr = splits(len(sl))
    bad_te = [p for p, j in enumerate(te) if sl[j] > TOL]
    print(f"{case}: {len(sl)} references, {int((sl > TOL).sum())} with slack > {TOL:g} "
          f"(max {np.nanmax(sl):.3g}); test positions affected: {bad_te}; "
          f"train labels affected: {sum(sl[j] > TOL for j in tr)}")

sl = ref_slack("case300")
te, _, _ = splits(len(sl))
DROP = 39
conv = np.array([sl[j] <= TOL for p, j in enumerate(te) if p != DROP])
A = f"{REPO}/acuc/results"
out = {}
for arm, pat, get in (("ours", "c300pipe_s{s}_merit0.80_dt.json", lambda d: [r[0] for r in d["merit|0.80|40"]]),
                      ("proxy", "proxyfair_case300_s{s}.json", lambda d: [r["gap"] for r in d["rows"]])):
    G = np.array([get(json.load(open(f"{A}/{pat.format(s=s)}"))) for s in range(4)])
    assert G.shape == (4, 47), G.shape
    out[arm] = dict(all_abs=float(np.abs(G).mean()), all_signed=float(G.mean()),
                    conv_abs=float(np.abs(G[:, conv]).mean()), conv_signed=float(G[:, conv].mean()),
                    nonconv_abs=float(np.abs(G[:, ~conv]).mean()), nonconv_signed=float(G[:, ~conv].mean()))
print(f"\ncase300 test: {conv.sum()} converged / {(~conv).sum()} non-converged references "
      f"among the 47 scored instances")
print(f"{'arm':6s} {'|gap| all':>10s} {'signed all':>11s} {'|gap| conv':>11s} {'signed conv':>12s} {'|gap| non-conv':>15s}")
for arm, o in out.items():
    print(f"{arm:6s} {o['all_abs']:10.4f} {o['all_signed']:+11.4f} {o['conv_abs']:11.4f} "
          f"{o['conv_signed']:+12.4f} {o['nonconv_abs']:15.4f}")
json.dump(out, open(f"{_common.OUT}/c4_reference_convergence.json", "w"), indent=1)

"""Major (statistics): per-seed spread and paired instance-level comparison of the
method vs the NN proxy, from the COMMITTED result files.

All seeds share the same test instances, so seeds are averaged per instance and the
comparison is paired over instances (n = 48, or 47 on case300 where test position 39
fails for both arms). Bootstrap CI over instances; Wilcoxon signed-rank test.
Runtime: seconds.
"""
import glob
import json
import _common
from _common import REPO
import numpy as np
from scipy import stats

A = f"{REPO}/acuc/results"; V = f"{REPO}/vehicle/results"
L = lambda p: [json.load(open(f)) for f in sorted(glob.glob(p))]
row = lambda r: [r["gap"], r["disc"], r["cont"]] if isinstance(r, dict) else list(r[:3])
CASES = {
    "case118": ([np.array(d["out"]["fitted"], float)[:, :3] for d in L(f"{A}/thrlf_eval_case118_s*_d25.json")],
                [np.array([row(r) for r in d["rows"]]) for d in L(f"{A}/proxyfair_case118_s*.json")]),
    "case300": ([np.array(d["merit|0.80|40"], float)[:, :3] for d in L(f"{A}/c300pipe_s*_merit0.80_dt.json")],
                [np.array([row(r) for r in d["rows"]]) for d in L(f"{A}/proxyfair_case300_s*.json")]),
    "vehicle": ([np.array([row(r) for r in d["rows"]]) for d in L(f"{V}/train_s?.json")],
                [np.array([row(r) for r in d["rows"]]) for d in L(f"{V}/proxy_s*.json")]),
}
rng = np.random.default_rng(0)
out = {}
for case, (O, P) in CASES.items():
    O, P = np.stack(O), np.stack(P)                       # seeds x instances x (gap, disc, cont)
    print(f"===== {case}  ours {O.shape}  proxy {P.shape}")
    for name, M in (("ours", O), ("proxy", P)):
        ps = np.abs(M[:, :, 0]).mean(1)
        print(f"  {name:5s} per-seed |gap| {np.round(ps, 3)}  (sd {ps.std(ddof=1):.3f})")
    o, p = np.abs(O[:, :, 0]).mean(0), np.abs(P[:, :, 0]).mean(0)
    d = o - p
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    ci = np.percentile(bs, [2.5, 97.5])
    pv = stats.wilcoxon(o, p).pvalue
    print(f"  paired |gap| ours - proxy: {d.mean():+.4f} pp, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], "
          f"Wilcoxon p = {pv:.3g}; ours better on {(d < -1e-9).sum()}, worse on {(d > 1e-9).sum()}, "
          f"tied on {(np.abs(d) <= 1e-9).sum()} of {len(d)}")
    out[case] = dict(diff=float(d.mean()), ci=list(map(float, ci)), p=float(pv))
json.dump(out, open(f"{_common.OUT}/stats_paired.json", "w"), indent=1)

"""Summarise the case118 ablation. Paired, instance-clustered: average each arm over
seeds per test instance, then bootstrap over the 48 instances."""
import json, glob, os
import numpy as np
from scipy.stats import wilcoxon
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ARMS = ["flat_nocut", "flat_cut", "pred_nocut", "pred_cut", "proxy"]
LABEL = {"flat_nocut": "flat V, no cuts", "flat_cut": "flat V + learned cuts",
         "pred_nocut": "learned V, no cuts", "pred_cut": "learned V + cuts (paper)",
         "proxy": "supervised proxy"}

R = {}   # (arm, thr) -> {seed: {j: row}}
for arm in ARMS:
    for f in sorted(glob.glob(f"{OUT}/eval_{arm}_s*.json")):
        s = int(f.rsplit("_s", 1)[1].split(".")[0])
        for thr, rows in json.load(open(f)).items():
            R.setdefault((arm, thr), {})[s] = {r["j"]: r for r in rows}

print("=== per-arm test results (48 instances x 4 seeds; mean over seeds) ===")
print(f"{'arm':28s} {'thr':7s} {'|gap| %':>9s} {'per-seed |gap|':>32s} {'signed %':>9s} {'disc %':>7s} {'fails':>6s}")
for arm in ARMS:
    for thr in ("0.5", "fitted"):
        if (arm, thr) not in R: continue
        per = R[(arm, thr)]
        ps = [np.mean([abs(r["gap"]) for r in per[s].values() if "gap" in r]) for s in sorted(per)]
        allr = [r for s in per for r in per[s].values()]
        ok = [r for r in allr if "gap" in r]
        print(f"{LABEL[arm]:28s} {thr:7s} {np.mean([abs(r['gap']) for r in ok]):9.4f} "
              f"{' '.join(f'{x:.3f}' for x in ps):>32s} {np.mean([r['gap'] for r in ok]):+9.4f} "
              f"{np.mean([r['disc'] for r in ok]):7.2f} {len(allr)-len(ok):6d}")


def inst_mean(arm, thr):
    per = R[(arm, thr)]
    js = sorted(set.intersection(*[set(j for j, r in per[s].items() if "gap" in r) for s in per]))
    return {j: np.mean([abs(per[s][j]["gap"]) for s in per]) for j in js}


def paired(a, b, label):
    A, B = inst_mean(*a), inst_mean(*b)
    js = sorted(set(A) & set(B)); d = np.array([A[j] - B[j] for j in js])
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = wilcoxon(d)[1] if np.any(d != 0) else 1.0
    print(f"{label:62s} {d.mean():+8.4f} pp  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3g}  "
          f"(better on {(d < 0).sum()}/{len(d)})")


print("\n=== paired differences in |gap| (first minus second; negative = first is better) ===")
for a, b, lab in [
    (("pred_nocut", "0.5"), ("flat_nocut", "0.5"), "learned V vs flat V (no cuts, thr 0.5)"),
    (("pred_nocut", "fitted"), ("flat_nocut", "fitted"), "learned V vs flat V (no cuts, fitted thr)"),
    (("pred_cut", "0.5"), ("pred_nocut", "0.5"), "cuts on vs off (learned V, thr 0.5)"),
    (("pred_cut", "fitted"), ("pred_nocut", "fitted"), "cuts on vs off (learned V, fitted thr)"),
    (("flat_cut", "0.5"), ("flat_nocut", "0.5"), "cuts on vs off (flat V, thr 0.5)"),
    (("flat_cut", "fitted"), ("flat_nocut", "fitted"), "cuts on vs off (flat V, fitted thr)"),
    (("pred_cut", "fitted"), ("pred_cut", "0.5"), "fitted vs 0.5 threshold (paper arm)"),
    (("flat_nocut", "fitted"), ("flat_nocut", "0.5"), "fitted vs 0.5 threshold (no learning)"),
    (("pred_cut", "fitted"), ("proxy", "0.5"), "paper method vs proxy as published (thr 0.5)"),
    (("pred_cut", "fitted"), ("proxy", "fitted"), "paper method vs proxy, both fitted thr"),
    (("pred_nocut", "fitted"), ("proxy", "fitted"), "learned V only vs proxy, both fitted thr"),
]:
    if a in R and b in R:
        paired(a, b, lab)

print("\n=== fidelity checks against committed results ===")
for s in range(4):
    mine = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/fit_pred_cut_s{s}_*.json"))]
    com = [json.load(open(f)) for f in sorted(glob.glob(f"{REPO}/acuc/results/thr_lf/sh_case118_s{s}_*.json"))]
    if mine and com:
        wm = min(mine, key=lambda d: d["best"]); wc = min(com, key=lambda d: d["best"])
        print(f"s{s} threshold fit: mine c={np.round(wm['c'], 4).tolist()}  committed c={np.round(wc['c'], 4).tolist()}")
    ce = json.load(open(f"{REPO}/acuc/results/thrlf_eval_case118_s{s}_d25.json"))["out"]
    if ("pred_cut", "fitted") in R and s in R[("pred_cut", "fitted")]:
        m = np.mean([abs(r["gap"]) for r in R[("pred_cut", "fitted")][s].values() if "gap" in r])
        c = np.mean([abs(r[0]) for r in ce["fitted"]])
        m5 = np.mean([abs(r["gap"]) for r in R[("pred_cut", "0.5")][s].values() if "gap" in r])
        c5 = np.mean([abs(r[0]) for r in ce["shipped"]])
        print(f"s{s} paper arm |gap|: mine fitted {m:.4f} vs committed {c:.4f};  mine 0.5 {m5:.4f} vs committed {c5:.4f}")
    cp = json.load(open(f"{REPO}/acuc/results/proxyfair_case118_s{s}.json"))
    if ("proxy", "0.5") in R and s in R[("proxy", "0.5")]:
        m = np.mean([abs(r["gap"]) for r in R[("proxy", "0.5")][s].values() if "gap" in r])
        print(f"s{s} proxy thr 0.5 |gap|: mine {m:.4f} vs committed {cp['gap_mean']:.4f}")

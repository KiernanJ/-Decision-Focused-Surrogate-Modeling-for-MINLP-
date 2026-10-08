"""Summarise the case300 ablation. Paired and instance-clustered: average each arm
over seeds per test instance, then bootstrap over instances.
Two scorings:
  stored     gap vs the stored reference cost (as in the paper), all test instances
  converged  gap vs the reference commitment re-priced with Socp.price, on test
             instances whose stored reference converged (slack <= 1e-3)
Two threshold rules: the paper's 0.8, and the LABEL-FREE validation choice per arm/seed.
Usage: python analyse300.py  (writes nothing; redirect to out/analysis.txt)"""
import json, glob, os
import numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ARMS = ["flat_nocut", "flat_cut", "pred_nocut", "pred_cut", "proxy"]
LABEL = {"flat_nocut": "flat V, no cuts", "flat_cut": "flat V + learned cuts",
         "pred_nocut": "learned V, no cuts", "pred_cut": "learned V + cuts (paper)",
         "proxy": "supervised proxy"}
GRID = ("0.5", "0.65", "0.8")

refs = {int(k): v for k, v in json.load(open(f"{OUT}/refs.json")).items()}
CONV = {j for j, v in refs.items() if v["stored_slack"] <= 1e-3 and v["priced_cost"] is not None
        and v["priced_slack"] is not None and v["priced_slack"] < 1e-4}

T = {}     # (arm, seed, thr) -> {j: row}
for f in glob.glob(f"{OUT}/test_*_s*_t*.json"):
    b = os.path.basename(f)[5:-5]
    arm, rest = b.rsplit("_s", 1)
    seed, thr = rest.split("_t")
    T[(arm, int(seed), thr)] = {r["j"]: r for r in json.load(open(f))}
SEL = {}   # (arm, seed) -> selected threshold (label-free, validation)
VAL = {}
for f in glob.glob(f"{OUT}/val_*_s*.json"):
    d = json.load(open(f)); SEL[(d["arm"], d["seed"])] = str(d["selected"]); VAL[(d["arm"], d["seed"])] = d["score"]


def seeds(arm):
    return sorted({s for (a, s, _) in T if a == arm})


def rows_for(arm, rule):
    """{seed: {j: row}} under a threshold rule ('0.8', '0.5', '0.65', or 'val')."""
    out = {}
    for s in seeds(arm):
        thr = SEL.get((arm, s)) if rule == "val" else rule
        thr = {"0.5": "0.5", "0.65": "0.65", "0.8": "0.8"}.get(thr, thr)
        if thr is not None and (arm, s, thr) in T:
            out[s] = T[(arm, s, thr)]
    return out


def gap(r, scoring):
    if "cost" not in r:
        return None
    if scoring == "stored":
        return r["gap"]
    if r["j"] not in CONV:
        return None
    pc = refs[r["j"]]["priced_cost"]
    return 100*(r["cost"] - pc)/pc


def summary(arm, rule, scoring):
    per = rows_for(arm, rule)
    if not per:
        return None
    G, D, ON, F, ps = [], [], [], 0, []
    for s, rows in per.items():
        gs = []
        for r in rows.values():
            if scoring == "converged" and r["j"] not in CONV:
                continue
            x = gap(r, scoring)
            if x is None:
                F += 1; continue
            G.append(x); D.append(r["disc"]); ON.append(r["n_on"]); gs.append(abs(x))
        ps.append(np.mean(gs))
    G = np.array(G)
    return dict(abs=np.abs(G).mean(), signed=G.mean(), disc=np.mean(D), on=np.mean(ON), fails=F, per=ps)


print(f"test instances {len(refs)}; converged references {len(CONV)}")
print("label-free threshold selected on validation (arm: seed0..3):")
for arm in ARMS:
    print(f"  {LABEL[arm]:28s} " + "  ".join(f"s{s}:{SEL.get((arm, s), '-')}" for s in seeds(arm)))

for scoring in ("stored", "converged"):
    print(f"\n=== per-arm test |gap| % ({scoring} reference; mean over instances x seeds) ===")
    print(f"{'arm':28s} {'rule':6s} {'|gap|':>8s} {'per-seed |gap|':>30s} {'signed':>8s} {'disc%':>6s} {'on':>5s} {'fails':>5s}")
    for arm in ARMS:
        for rule in ("0.8", "val", "0.5", "0.65"):
            s_ = summary(arm, rule, scoring)
            if s_ is None:
                continue
            print(f"{LABEL[arm]:28s} {rule:6s} {s_['abs']:8.4f} {' '.join(f'{x:.3f}' for x in s_['per']):>30s} "
                  f"{s_['signed']:+8.4f} {s_['disc']:6.2f} {s_['on']:5.1f} {s_['fails']:5d}")


def inst_mean(arm, rule, scoring):
    per = rows_for(arm, rule)
    js = None
    for rows in per.values():
        ok = {j for j, r in rows.items() if gap(r, scoring) is not None}
        js = ok if js is None else js & ok
    return {j: np.mean([abs(gap(per[s][j], scoring)) for s in per]) for j in (js or [])}


def paired(a, b, rule, scoring, label):
    A, B = inst_mean(a, rule, scoring), inst_mean(b, rule, scoring)
    js = sorted(set(A) & set(B))
    if len(js) < 5:
        return
    d = np.array([A[j] - B[j] for j in js])
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = wilcoxon(d)[1] if np.any(d != 0) else 1.0
    print(f"{label:52s} {d.mean():+8.4f} pp  95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3g}  "
          f"(first better on {(d < 0).sum()}/{len(d)}, worse on {(d > 0).sum()})")


for scoring in ("stored", "converged"):
    for rule in ("val", "0.8"):
        print(f"\n=== paired |gap| differences ({scoring} reference, threshold rule {rule}); negative = first better ===")
        paired("pred_nocut", "flat_nocut", rule, scoring, "learned V vs flat V (no cuts)")
        paired("pred_cut", "pred_nocut", rule, scoring, "cuts on vs off (learned V)")
        paired("flat_cut", "flat_nocut", rule, scoring, "cuts on vs off (flat V)")
        paired("pred_cut", "flat_nocut", rule, scoring, "paper method vs no learning")
        paired("pred_cut", "proxy", rule, scoring, "paper method vs proxy")
        paired("pred_nocut", "proxy", rule, scoring, "learned V only vs proxy")

print("\n=== fidelity: paper arm, selected-epoch weights, thr 0.8 vs audit/out/precomputed/case300_best_s*.json ===")
for s in range(4):
    f = f"{REPO}/audit/out/precomputed/case300_best_s{s}.json"
    if not os.path.exists(f) or ("pred_cut", s, "0.8") not in T:
        continue
    pre = json.load(open(f))
    mine = list(T[("pred_cut", s, "0.8")].values())   # rows are stored in test order
    diffs = [abs(mine[int(pos)]["gap"] - v[0]) for pos, v in pre.items()
             if int(pos) < len(mine) and "gap" in mine[int(pos)]]
    print(f"s{s}: {len(diffs)} instances compared, max |difference| in gap {max(diffs):.2e} pp")

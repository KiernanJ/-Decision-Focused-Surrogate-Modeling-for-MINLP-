# %% Aggregate the audit evaluation: per-method means, per-seed spread, failures,
#    paired per-INSTANCE statistics (seed-averaged, so n = scored test instances),
#    bootstrap 95% CI and Wilcoxon signed-rank. Writes results/eval2/summary.json.
import os, sys, json, glob
import numpy as np
from scipy.stats import wilcoxon
OUT = "results/eval2"
S = {}
rng = np.random.default_rng(0)
for case in ("case118", "case300"):
    S[case] = {}
    per_inst = {}
    for meth in ("ours", "proxy", "nocut", "untrained"):
        fs = sorted(glob.glob(f"{OUT}/{case}_s?_{meth}.json"))
        if not fs:
            continue
        D = [json.load(open(f)) for f in fs]
        ok = [r for d in D for r in d["rows"] if r["ok"]]
        n_all = sum(len(d["rows"]) for d in D)
        seeds = [np.mean([abs(r["gap"]) for r in d["rows"] if r["ok"]]) for d in D]
        S[case][meth] = dict(
            n_seeds=len(D), n_scored=n_all, n_failed=n_all-len(ok),
            abs_gap=float(np.mean([abs(r["gap"]) for r in ok])),
            signed_gap=float(np.mean([r["gap"] for r in ok])),
            median_abs_gap=float(np.median([abs(r["gap"]) for r in ok])),
            disc=float(np.mean([r["disc"] for r in ok])), cont=float(np.mean([r["cont"] for r in ok])),
            median_secs=float(np.median([r["secs"] for d in D for r in d["rows"]])),
            per_seed_abs_gap=[float(x) for x in seeds], choices=[d["choice"] for d in D],
            excluded=D[0]["excluded"], best_epochs=[d.get("best_epoch") for d in D],
            proxy_cfg=[d.get("proxy_cfg") for d in D])
        acc = {}
        for d in D:
            for r in d["rows"]:
                acc.setdefault(r["idx"], []).append(abs(r["gap"]) if r["ok"] else np.nan)
        per_inst[meth] = {k: float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan for k, v in acc.items()}
        ref = [r["ref_secs"] for d in D[:1] for r in d["rows"] if "ref_secs" in r]
        S[case]["reference_median_secs"] = float(np.median(ref)) if ref else None
    for other in ("proxy", "nocut", "untrained"):
        if "ours" not in per_inst or other not in per_inst:
            continue
        ks = [k for k in per_inst["ours"] if k in per_inst[other]
              and np.isfinite(per_inst["ours"][k]) and np.isfinite(per_inst[other][k])]
        dlt = np.array([per_inst["ours"][k] - per_inst[other][k] for k in ks])
        boot = [rng.choice(dlt, len(dlt)).mean() for _ in range(10000)]
        try:
            p = float(wilcoxon(dlt).pvalue) if np.any(dlt != 0) else 1.0
        except ValueError:
            p = 1.0
        S[case][f"ours_minus_{other}"] = dict(
            n=len(ks), mean=float(dlt.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            wilcoxon_p=p, ours_better=int((dlt < -1e-9).sum()), ours_worse=int((dlt > 1e-9).sum()),
            tied=int((np.abs(dlt) <= 1e-9).sum()))
json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(S, indent=1))

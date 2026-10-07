"""Summaries of the case300 diagnostics from out/a_state_s*.json (no solves):
cut-row activity by category, and reserve-margin escalation vs loss.
Usage: python summarise.py"""
import json, os
import numpy as np
from scipy.stats import spearmanr
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

print("=== cut-row activity at the predicted V (24 val x 4 seeds) ===")
for which in ("init", "best", "net"):
    act, sl = [], []
    for s in range(4):
        for r in json.load(open(f"{OUT}/a_state_s{s}.json"))[which]["rows"]:
            x = np.array(r["rowslack_pred"]); act.append(x < 1e-4); sl.append(x)
    act, sl = np.array(act), np.array(sl)
    print(f"{which:4s} active fraction by row [general0 general1 integer0 integer1]: {np.round(act.mean(0), 2)}  "
          f"overall {act.mean():.2f}  median slack {np.round(np.median(sl, 0), 3)}")

print("\n=== reserve-margin escalation in the training loss (DiffDeploy) ===")
for which in ("init", "best", "net"):
    M, L = [], []
    for s in range(4):
        for r in json.load(open(f"{OUT}/a_state_s{s}.json"))[which]["rows"]:
            if "L" in r:
                M.append(r["margin"]); L.append(r["L"])
    M, L = np.array(M), np.array(L)
    vals, c = np.unique(np.round(M, 2), return_counts=True)
    print(f"{which:4s} margins used {dict(zip(vals.tolist(), c.tolist()))}  needs escalation {np.mean(M > 0.02):.0%}  "
          f"spearman(loss, margin) {spearmanr(L, M)[0]:+.2f}")

print("\n=== per-checkpoint state summary (mean over 4 seeds) ===")
keys = ["vmin", "vmax", "sat_r", "sat_i", "xi_flat", "xi_pred", "relax_pred", "rows_active_pred",
        "rows_active_flat", "u_diff_cut", "L", "margin"]
for which in ("init", "best", "net"):
    S = [json.load(open(f"{OUT}/a_state_s{s}.json"))[which]["summary"] for s in range(4)]
    print(f"{which:4s} " + "  ".join(f"{k}={np.mean([x[k] for x in S]):.3g}" for k in keys))
    H = {h: np.mean([x["head_grad"].get(h, 0) for x in S]) for h in S[0]["head_grad"]}
    print("      head gradient norms: " + "  ".join(f"{h}={v:.2g}" for h, v in H.items()))

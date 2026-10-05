"""Critical 2: which epoch's weights do the eval scripts actually score?

acuc/02_train.py saves ck["net"] (last epoch) and ck["best"] = (val, state, epoch).
04_eval_case*.py, 03_threshold_fit.py and 05_timing.py all load ck["net"].
Runtime: seconds.
"""
import _common  # noqa: F401  (sets paths)
from _common import REPO
import torch

print(f"{'checkpoint':16s} {'last ep':>8s} {'selected ep':>12s} {'max|net-best|':>14s}")
for case in ("case118", "case300"):
    for s in range(4):
        ck = torch.load(f"{REPO}/acuc/checkpoints/{case}_s{s}.pt", weights_only=False,
                        map_location="cpu")
        diff = max(float((ck["net"][k] - ck["best"][1][k]).abs().max()) for k in ck["net"])
        print(f"{case}_s{s:<9d} {ck['ep']-1:8d} {ck['best'][2]:12d} {diff:14.4f}")
print("\nIf 'selected ep' != 'last ep', the published numbers do not use the selected model.")

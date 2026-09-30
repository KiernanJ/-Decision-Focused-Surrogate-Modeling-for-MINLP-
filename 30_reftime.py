"""A defensible reference timing: enough samples that the median is stable.

The speed-up is the paper's headline efficiency claim and it was resting on
FOUR re-timed reference solves. QCAC iterative varies enormously per instance
(case300: median 201s, mean 231s, max 961s under contention), so a 4-sample
median is noise: the same quantity measured twice gave 120.4s and 345.1s, i.e.
"25x" and "71x" for the same method.

This re-times a proper sample SEQUENTIALLY on an idle machine, with every Gurobi
thread available, and reports the median with an interquartile range so the
spread is visible rather than hidden in a single number.
"""
import os, sys, re, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, RESERVE, grid, sample
from acuc import solve_qcac_iterative

N = int(os.environ.get("NREF", "15"))
_seeds = os.environ.get("POOL_SEEDS")
_keep = set(_seeds.split(",")) if _seeds else None
g, nl = grid()
POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m or (_keep is not None and m.group(1) not in _keep):
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1]) for i in range(int(m.group(2)))
             if REF[i] is not None]
rng = np.random.default_rng(0); perm = rng.permutation(len(POOL))
NTEST = int(os.environ.get("NTEST", "40"))
te = [int(i) for i in perm[:NTEST]][:N]
print(f"{CASE}: re-timing {len(te)} reference solves, sequential, 12 threads\n",
      flush=True)
T = []
for c, j in enumerate(te):
    t0 = time.perf_counter()
    solve_qcac_iterative(g, nl, POOL[j][0], POOL[j][1], reserve=RESERVE, threads=12)
    T.append(time.perf_counter()-t0)
    print(f"  {c+1:2d}/{len(te)}  {T[-1]:7.1f}s   running median {np.median(T):7.1f}s",
          flush=True)
T = np.array(T)
print(f"\n  median {np.median(T):.1f}s   IQR {np.percentile(T,25):.1f}-"
      f"{np.percentile(T,75):.1f}s   mean {T.mean():.1f}s   n={len(T)}")
json.dump(list(map(float, T)), open(f"{HERE}/results/reftime_{CASE}.json", "w"))
print(f"  wrote results/reftime_{CASE}.json")

# %% Reference-solver timing on an IDLE machine (audit 2026-10-06): re-solve the
#    QCAC iterative reference for the converged TEST instances, single thread,
#    one shard per core, after all other jobs have finished.
#    Usage: python 09_timing_ref.py --case case300 --shard 0 --nshard 8 ; then --merge
import os, sys, json, time, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case
OUT = "results/eval2"
if a.merge:
    secs = []
    for f in sorted(glob.glob(f"{OUT}/tref_{a.case}_*.json")):
        secs += json.load(open(f))
    json.dump(dict(case=a.case, secs=secs), open(f"{OUT}/timing_ref_{a.case}.json", "w"))
    print(a.case, len(secs), "median", np.median(secs)); raise SystemExit
import hardgate, data2
hardgate.init(a.case)
INST = data2.instances(hardgate._W["g"], a.case)
te, va, tr = data2.split(); REF = data2.refs(a.case)
ids = [j for j in te if data2.converged(REF.get(j))][a.shard::a.nshard]
secs = []
for j in ids:
    t0 = time.time()
    hardgate.work((j, INST[j][0], INST[j][1], np.ones(hardgate._W["g"].n_gen)))
    secs.append(time.time()-t0); print(j, f"{secs[-1]:.1f}s", flush=True)
json.dump(secs, open(f"{OUT}/tref_{a.case}_{a.shard}.json", "w"))

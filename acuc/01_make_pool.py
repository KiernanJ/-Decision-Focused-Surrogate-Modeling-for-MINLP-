# %% QCAC iterative references for one shard of instance indices (audit version).
#    Solves only the indices used by the split (test first, then train, then
#    validation), single-threaded Gurobi, and stores convergence status.
#    Usage:  QCAC_CASE=case300 python 01_make_pool.py --case case300 --shard 0 --nshard 8
import os, sys, time, pickle, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--nshard", type=int, required=True)
ap.add_argument("--tag", default="")              # top-up wave: output {case}_{tag}{shard}.pkl
ap.add_argument("--skip-done", action="store_true")
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case
import hardgate, data2
hardgate.init(a.case)
inst = data2.instances(hardgate._W["g"], a.case)
te, va, tr = data2.split()
order = te + tr + va                              # test first, then train, then validation
if a.skip_done:                                    # re-split only what is not solved yet
    done = set(data2.refs(a.case).keys())
    order = [i for i in order if i not in done]
todo = order[a.shard::a.nshard]
os.makedirs("results/pool2", exist_ok=True)
OUT = f"results/pool2/{a.case}_{a.tag}{a.shard}.pkl"
res, t0 = {}, time.time()
for i in todo:
    pd_, qd_ = inst[i]
    _, r, err = hardgate.work((i, pd_, qd_, np.ones(hardgate._W["g"].n_gen)))
    res[i] = r
    print(f"[{a.case} {i}] {'FAIL '+str(err) if r is None else r['status']} "
          f"{'' if r is None else f'''slack {r['slack']:.1e} n_on {r['u'].sum():.0f} {r['secs']:.0f}s'''}"
          f"  [{time.time()-t0:.0f}s]", flush=True)
    pickle.dump(dict(res=res, case=a.case), open(OUT, "wb"))
print(f"[{a.case} shard {a.shard}] done in {time.time()-t0:.0f}s", flush=True)

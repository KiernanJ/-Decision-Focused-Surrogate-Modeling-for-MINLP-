# %% One SHARD of the headroom gate: solve references for a contiguous slice of
#    one arm, serially, in its own OS process. Independent processes instead of
#    multiprocessing.Pool because macOS spawn re-imports __main__ and this
#    project has already lost time to that twice.
import os, sys, time, pickle
SC = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SC)
sys.path.insert(0, os.path.join(SC, "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import hardgate

CASE = os.environ["QCAC_CASE"]; ARM = os.environ["GATE_ARM"]
LO, HI = float(os.environ["GATE_LO"]), float(os.environ["GATE_HI"])
POUT = float(os.environ["GATE_POUT"]); SEED = int(os.environ["GATE_SEED"])
N = int(os.environ["GATE_N"]); I0 = int(os.environ["GATE_I0"]); I1 = int(os.environ["GATE_I1"])
OUT = os.environ["GATE_OUT"]
# A top-up batch uses a different seed, so its instance 0 is NOT the first
# batch's instance 0. Offsetting the stored key keeps the two disjoint.
OFF = int(os.environ.get("GATE_OFFSET", "0"))

hardgate.init(CASE)
inst = hardgate.sample_hard(hardgate._W["g"], N, seed=SEED, lo=LO, hi=HI, p_out=POUT)
res, t0 = {}, time.time()
for i in range(I0, min(I1, N)):
    pd_, qd_, av = inst[i]
    key, r, err = hardgate.work((i, pd_, qd_, av))
    res[i+OFF] = r
    print(f"[{ARM} {i}] {'OK' if r else 'FAIL '+str(err)} "
          f"{'' if not r else f'''n_on {r['u'].sum():.0f} cost {r['cost']:.2f} {r['secs']:.0f}s'''}"
          f"  out={int((av<0.5).sum())}  [{time.time()-t0:.0f}s]", flush=True)
    pickle.dump(dict(res=res, inst={k: inst[k-OFF] for k in res}, arm=ARM, case=CASE,
                     lo=LO, hi=HI, p_out=POUT, seed=SEED, N=N),
                open(OUT, "wb"))
print(f"[{ARM} {I0}:{I1}] done in {time.time()-t0:.0f}s", flush=True)

# %% LABEL-FREE TEACHER for the hybrid vehicle (2026-10-10): OUR OWN longer solves (WorkLimit W_LONG, no cuts) on the
#    TRAINING profiles only (never the reference pool's z). Both versions: MILP (Dixit, Dixit split: train = 124..223)
#    and MINLP (convex MIQCQP, src_hv.splits train set). Saves z, P_eng, cost per instance.
#    Usage: python dixit_vehicle/11_teacher.py 0,1,2,4,6,7,8,9
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np
from importlib import import_module
H = import_module("10_headroom")
from dixit_vehicle import Vehicle
W_LONG = 1.0
if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "results", "teacher"), exist_ok=True); t0 = time.time()
    for vid in [int(v) for v in sys.argv[1].split(",")]:
        P = json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json"))); V = Vehicle(vid, env=H.ENV); out = {}
        for i in range(124, 224):
            r = H.milp_short(V, Vehicle.demand(P[i]["u"]), W_LONG)
            out[i] = None if r is None else dict(cost=r[0], z=r[1].tolist(), Pe=r[2].tolist(), secs=r[3])
        json.dump(out, open(os.path.join(HERE, "results", "teacher", f"milp_v{vid}_w{W_LONG:g}.json"), "w"))
        print(f"[MILP veh {vid}] {sum(v is not None for v in out.values())}/100 teacher solves  [{time.time()-t0:.0f}s]", flush=True)
    from src_hv import load_pool, splits
    VEH, D, REF = load_pool(os.path.join(HERE, "..", "vehicle", "results", "pool", "pool_*.json"))
    te, va, tr = splits(len(REF)); out = {}
    for i in tr:
        r = H.minlp_short(VEH, D[i], W_LONG)
        out[i] = None if r is None else dict(cost=r[0], z=r[1].tolist(), Pe=r[2].tolist(), secs=r[3])
    json.dump(out, open(os.path.join(HERE, "results", "teacher", f"minlp_w{W_LONG:g}.json"), "w"))
    print(f"[MINLP] {sum(v is not None for v in out.values())}/{len(tr)} teacher solves  [{time.time()-t0:.0f}s]", flush=True)

# %% FRESH-2 VEHICLE TEST POOL (2026-10-11 night) for the LABEL-FREE method: pool_fresh/ was scored with the earlier method (11:55),
#    so the label-free method gets new profiles, scored ONCE. Same samplers and reference settings as 17_fresh_test_pool.py
#    (MILP: 100 unused u per vehicle; MINLP: 48 new D ~ U(0.5, 2.5); exact solves Threads=1, TL 120 s, MIPGap 1e-4), new seeds,
#    disjoint from the pool, pool_fresh AND the extra training profiles (results/extra/).   Usage: python dixit_vehicle/19_fresh2_test_pool.py
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, multiprocessing as mp
from importlib import import_module
F17 = import_module("17_fresh_test_pool")
OUT = os.path.join(HERE, "results", "pool_fresh2"); os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    t0 = time.time(); jobs = []
    for vid in F17.VIDS:
        used = {int(round(p["u"]*1000)) for p in json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json")))}
        used |= {int(round(p["u"]*1000)) for p in json.load(open(os.path.join(HERE, "results", "pool_fresh", f"vehicle_{vid}.json")))}
        EX = os.path.join(HERE, "results", "extra", f"milp_v{vid}.json")
        if os.path.exists(EX): used |= {int(round(u*1000)) for u in json.load(open(EX))["u"]}
        avail = np.array(sorted(set(range(100, 2001)) - used))
        jobs += [(vid, u) for u in (np.random.default_rng(95000 + vid).choice(avail, F17.N_MILP, replace=False)/1000).tolist()]
        print(f"[MILP v{vid}] {len(avail)} unused u values (pool, fresh and extra excluded)", flush=True)
    from src_hv import load_pool
    VEH, Dm, RL = load_pool(os.path.join(HERE, "..", "vehicle", "results", "pool", "pool_*.json"))
    Dn = np.random.default_rng(20261012).uniform(0.5, 2.5, (F17.N_MINLP, VEH["T"]))
    with mp.get_context("spawn").Pool(6) as pool:   # (audit 2026-10-12) = 17 (TL is wall-clock; same contention as the original pools)
        for vid in F17.VIDS:
            R = [r for _, r in pool.imap(F17.milp_job, [j for j in jobs if j[0] == vid])]
            json.dump(R, open(os.path.join(OUT, f"vehicle_{vid}.json"), "w"))
            g = np.array([r["mipgap"] for r in R]); print(f"[MILP v{vid}] {len(R)} refs, MIPGap>1e-4: {(g > 1e-4 + 1e-12).sum()}  [{time.time()-t0:.0f}s]", flush=True)
        M = dict(pool.imap(F17.minlp_job, [(k, Dn[k], VEH) for k in range(F17.N_MINLP)]))
    json.dump(dict(D=Dn.tolist(), ref={str(k): v for k, v in M.items()}), open(os.path.join(OUT, "minlp.json"), "w"))
    print(f"[MINLP] {sum(v is not None for v in M.values())}/{F17.N_MINLP} solved  [{time.time()-t0:.0f}s]", flush=True)

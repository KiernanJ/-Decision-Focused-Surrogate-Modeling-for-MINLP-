# %% LABEL-FREE TEACHER for the hybrid vehicle (2026-10-11 night), same rule as the AC case (acuc/57, 58, 60):
#    the old teacher (11_teacher.py) = our own W=1 solves = 100x the deployment budget (W=0.01) -> near-reference labels.
#    New teacher, per instance, nothing longer than deployment:
#      candidates (a) own short solve W=0.01, no cuts;  (b) the relaxation's rounded z;
#      then a local search from the cheaper one: single-step z_t +-1 moves, accepted only if the fixed-z re-solve cost drops;
#      target = the result (cost = fixed-z re-solve, the same evaluator deployment is scored with).
#    Also draws EXTRA unlabeled training profiles (MILP: 300 unused u per vehicle; MINLP: 300 new D), disjoint from pool and fresh test.
#    PRE-REGISTERED check on the ORIGINAL training profiles vs the W=1 teacher (both re-priced by the fixed-z re-solve):
#      mean (new - W1)/|W1| <= 0.05% AND median <= 0.01%.  Four metrics vs the exact pool reference = [diag only] report.
#    Usage: python dixit_vehicle/18_lf_teacher.py   -> results/teacher/<case>_lf.json, results/extra/<case>.json, logs
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, multiprocessing as mp
from importlib import import_module
W, N_EXTRA, ROUNDS = 0.01, 300, 10
CASES = [f"milp_v{v}" for v in (0, 1, 2, 4, 6, 7, 8, 9)] + ["minlp"]


def extra_profiles(name, C):
    """unlabeled extra training profiles, disjoint from the pool and the fresh test profiles."""
    from dixit_vehicle import Vehicle
    if name.startswith("milp_v"):
        vid = int(name[6:]); P = json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json")))
        used = {int(round(p["u"]*1000)) for p in P}
        FP = os.path.join(HERE, "results", "pool_fresh", f"vehicle_{vid}.json")
        if os.path.exists(FP): used |= {int(round(p["u"]*1000)) for p in json.load(open(FP))}
        avail = np.array(sorted(set(range(100, 2001)) - used)); u = (np.random.default_rng(70000 + vid).choice(avail, N_EXTRA, replace=False)/1000).tolist()
        return dict(u=u), np.stack([Vehicle.demand(x) for x in u])
    T = C["D"].shape[1]; D = np.random.default_rng(31337).uniform(0.5, 2.5, (N_EXTRA, T))
    return dict(D=D.tolist()), D


def price(C, D, z):
    p = C["solve"](D, 1e6, relax=True, bnd=(z, z))
    return None if p is None else (p[0], np.asarray(p[2], float))


def lf_target(C, D):
    """-> dict(cost, z, Pe, secs, picked) : own short solve / rounded relaxation, then +-1 local search on the fixed-z cost."""
    t0 = time.time(); SW = C["SW"]; cand = {}
    r = C["solve"](D, W)
    if r is not None: cand["short W=0.01"] = np.asarray(r[1], float)
    rr = C["solve"](D, 1e6, relax=True)
    if rr is not None: cand["rounded relaxation"] = np.clip(np.asarray(rr[1], float), 0, SW)
    best = None
    for nm, z in cand.items():
        p = price(C, D, z)
        if p is not None and (best is None or p[0] < best[1]): best = (z.copy(), p[0], p[1], nm)
    if best is None: return None
    z, c, Pe, picked = best
    for _ in range(ROUNDS):
        moved = False
        for t in range(len(z)):
            for dz in (-1, 1):
                if not 0 <= z[t] + dz <= SW: continue
                w = z.copy(); w[t] += dz; p = price(C, D, w)
                if p is not None and p[0] < c - 1e-9: z, c, Pe, moved = w, p[0], p[1], True
        if not moved: break
    return dict(cost=float(c), z=z.tolist(), Pe=Pe.tolist(), secs=time.time() - t0, picked=picked)


def case_job(name):
    os.environ["OMP_NUM_THREADS"] = "1"; t0 = time.time()
    F = import_module("16_vehicle_final"); C = F.load_case(name); tr = C["tr"]
    ex_meta, Dex = extra_profiles(name, C)
    T1 = {i: lf_target(C, C["D"][i]) for i in tr}
    TX = {f"x{k}": lf_target(C, Dex[k]) for k in range(len(Dex))}
    # ---- pre-registered check vs the W=1 teacher (both re-priced by the fixed-z re-solve), original training profiles
    rel, rows_new, rows_w1 = [], [], []
    for i in tr:
        if T1.get(i) is None or C["TG"].get(i) is None: continue
        p1 = price(C, C["D"][i], np.array(C["TG"][i]["z"], float))
        if p1 is None: continue
        rel.append(100*(T1[i]["cost"] - p1[0])/abs(p1[0]))
        if i in C["REF"] and C["REF"][i]["ok"]:
            ref = C["REF"][i]
            for rows, cost, z, Pe in ((rows_new, T1[i]["cost"], np.array(T1[i]["z"]), np.array(T1[i]["Pe"])), (rows_w1, p1[0], np.array(C["TG"][i]["z"]), p1[1])):
                rows.append((100*abs(cost - ref["cost"])/abs(ref["cost"]), 100*np.abs(z - ref["z"]).sum()/max(ref["z"].sum(), 1e-9), 100*np.abs(Pe - ref["Pe"]).sum()/max(ref["Pe"].sum(), 1e-9)))
    rel = np.array(rel); passed = bool(rel.mean() <= 0.05 and np.median(rel) <= 0.01)
    four = lambda v: (f"{np.mean(v[:,0]):.4f}% | {np.median(v[:,0]):.4f}% | {np.mean(v[:,1]):.2f}% | {np.mean(v[:,2]):.2f}%") if len(v) else "n/a"
    msg = (f"[{name}] train n={len(rel)}: new vs W=1 teacher mean {rel.mean():+.4f}% median {np.median(rel):+.4f}% max {rel.max():+.4f}% -> "
           f"{'PASSED' if passed else 'FAILED'} | [diag vs ref, n={len(rows_new)}] W=1 teacher {four(np.array(rows_w1))} || label-free {four(np.array(rows_new))} | "
           f"extra {sum(v is not None for v in TX.values())}/{len(TX)} | median {np.median([v['secs'] for v in T1.values() if v]):.2f}s/inst  [{time.time()-t0:.0f}s]")
    os.makedirs(os.path.join(HERE, "results", "extra"), exist_ok=True)
    json.dump(ex_meta, open(os.path.join(HERE, "results", "extra", f"{name}.json"), "w"))
    T = {str(i): v for i, v in T1.items() if v is not None}; T.update({k: v for k, v in TX.items() if v is not None})
    json.dump(T, open(os.path.join(HERE, "results", "teacher", f"{name}_lf.json"), "w"))
    return name, passed, msg


if __name__ == "__main__":
    import runlog
    RUN = runlog.start("vehicle_lf_teacher"); t0 = time.time()
    with mp.get_context("spawn").Pool(9) as pool:
        for name, passed, msg in pool.imap_unordered(case_job, CASES):
            print(msg, flush=True); runlog.add_result(RUN, f"VEHICLE label-free teacher check (TRAIN profiles) {msg}")
    print(f"[done] {time.time()-t0:.0f}s", flush=True)

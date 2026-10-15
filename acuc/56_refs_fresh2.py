# %% References for the UNTOUCHED test draw (data2.fresh2_ids) of a case: sequential QCAC to convergence + final-iteration
#    polish (combopt_ac.ref_polish), saving every iteration. Resumable. Usage: python 56_refs_fresh2.py <case>
import os, sys, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import multiprocessing as mp
CASE = sys.argv[1]; DRAW = sys.argv[2] if len(sys.argv) > 2 else "fresh2"
if __name__ == "__main__":
    import combopt_ac, data2
    RP = os.path.join(HERE, "results", f"{DRAW}_refs_{CASE}.pkl")
    REF = pickle.load(open(RP, "rb")) if os.path.exists(RP) else {}
    todo = [i for i in getattr(data2, f"{DRAW}_ids")() if i not in REF]; t0 = time.time()
    with mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True)) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(combopt_ac.ref_polish, [(i, 1e-4, 600.0) for i in todo])):
            REF[i] = r; pickle.dump(REF, open(RP, "wb"))
            print(f"[{k+1}/{len(todo)}] {i} {r.get('status', r.get('err'))} {r.get('secs', 0):.0f}s  [{time.time()-t0:.0f}s]", flush=True)
    print(f"[done] {sum(r.get('converged', False) for r in REF.values())}/{len(REF)} converged", flush=True)

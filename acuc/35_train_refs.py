# %% Sequential-QCAC solutions of the TRAINING instances (V-net teacher), saved in the pool2 format so
#    data2.refs() finds them; every iteration's V and commitment are kept (acuc hist). Standard settings.
#    Usage: python 35_train_refs.py <case>
import os, sys, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import multiprocessing as mp
CASE = sys.argv[1]

if __name__ == "__main__":
    import combopt_ac, data2
    te, va, tr = data2.split()
    OUT = os.path.join(HERE, "results", "pool2", f"{CASE}_train.pkl")
    res = pickle.load(open(OUT, "rb"))["res"] if os.path.exists(OUT) else {}
    todo = [i for i in tr if i not in res]; t0 = time.time()
    with mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True)) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(combopt_ac.ref_tight, [(i, 1e-3, 120.0) for i in todo])):
            res[i] = r; pickle.dump(dict(res=res, case=CASE), open(OUT, "wb"))
            print(f"[{k+1}/{len(todo)}] {i} {r.get('status', r.get('err'))} iters {r.get('iters')} {r.get('secs', 0):.0f}s  [{time.time()-t0:.0f}s]", flush=True)
    print(f"[done] {sum(r.get('converged', False) for r in res.values())}/{len(res)} converged", flush=True)

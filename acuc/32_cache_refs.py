# %% Price every converged reference commitment ONCE with the strict evaluator and cache it
#    (results/strict_refs_{case}.pkl; loaded by data2.strict_refs). Fresh/split test refs + training refs.
import os, sys, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import multiprocessing as mp
SRC = {"case118": ["fresh_refs_case118.pkl", "pool2"], "case300": ["fresh_refs_case300.pkl", "pool2"],
       "pglib_opf_case118_ieee__api": ["split_refs_pglib_opf_case118_ieee__api.pkl", "pool2"]}

if __name__ == "__main__":
    import combopt_ac, data2
    for case, srcs in [(c, s) for c, s in SRC.items() if len(sys.argv) < 2 or c == sys.argv[1]]:
        U = {}
        for s in srcs:
            R = data2.refs(case, os.path.join(HERE, "results", "pool2")) if s == "pool2" else pickle.load(open(os.path.join(HERE, "results", s), "rb"))
            U.update({i: r["u"] for i, r in R.items() if r is not None and r.get("converged")})
        with mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(case, True)) as pool:
            out = dict(pool.imap_unordered(combopt_ac.price_strict, list(U.items())))
        pickle.dump(out, open(os.path.join(HERE, "results", f"strict_refs_{case}.pkl"), "wb"))
        print(f"[{case}] cached {sum(v is not None for v in out.values())}/{len(out)} strict reference prices", flush=True)

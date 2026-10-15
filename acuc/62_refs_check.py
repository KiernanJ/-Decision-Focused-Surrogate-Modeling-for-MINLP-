# %% fresh-draw REFERENCE CHECK before the final test (2026-10-12): per case list converged / not converged / crashed references;
#    re-solve ONLY entries that crashed with an exception (transient; references only -- the method is untouched).
#    Genuine non-convergence stays excluded (pre-registered).   Usage: python 62_refs_check.py <draw, e.g. fresh4>
import os, sys, pickle, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import data2
DRAW = sys.argv[1] if len(sys.argv) > 1 else "fresh4"
for C in ("case118", "case300", "pglib_opf_case118_ieee__api"):
    p = os.path.join(HERE, "results", f"{DRAW}_refs_{C}.pkl"); ids = getattr(data2, f"{DRAW}_ids")()
    if not os.path.exists(p):
        print(f"{C}: NO reference file {p}", flush=True); continue
    R = pickle.load(open(p, "rb"))
    crash = [i for i in ids if i in R and "err" in R[i] and R[i]["err"] != "no solution"]
    nonconv = [i for i in ids if i in R and not R[i].get("converged")]
    conv = sum(bool(R[i].get("converged")) for i in ids if i in R)
    print(f"{C}: {len(R)}/{len(ids)} refs, converged {conv}, not converged {len(nonconv)} {nonconv}, crashed {crash}", flush=True)
    if crash:
        for i in crash:
            del R[i]
        pickle.dump(R, open(p, "wb")); print(f"   re-solving {len(crash)} crashed references", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "56_refs_fresh2.py"), C, DRAW], check=False)
        R = pickle.load(open(p, "rb")); print(f"   after retry: converged {sum(bool(R[i].get('converged')) for i in ids if i in R)}/{len(ids)}", flush=True)

# F2 task 4b: reference noise floor -- re-solve a few references with a 10x tighter MIP gap (ref_tight, mip_gap 1e-4) and price strictly
import sys, os, pickle, glob, time, numpy as np, warnings; warnings.filterwarnings("ignore")
R, case, draw = sys.argv[1], sys.argv[2], sys.argv[3]; ids = [int(x) for x in sys.argv[4].split(",")]
sys.path.insert(0, os.path.join(R, "src"))
import combopt_ac as C
C.init(case, True)
REF = pickle.load(open(os.path.join(R, f"acuc/results/{draw}_refs_{case}.pkl"), "rb"))
out = {}
for i in ids:
    t0 = time.time()
    _, J0 = C.price_strict((i, REF[i]["u"]))
    _, r = C.ref_tight((i, 1e-4, 600.0))
    if not r.get("converged"):
        print(f"{case} {i}: tight re-solve not converged ({r.get('status', r.get('err'))}) [{time.time()-t0:.0f}s]", flush=True); out[i] = (J0, None, r); continue
    _, J1 = C.price_strict((i, r["u"]))
    du = int(np.sum(np.asarray(r["u"]) != np.asarray(REF[i]["u"])))
    print(f"{case} {i}: stored ref strict {J0[0]:.2f} (repairs {J0[4]}) | tight re-solve: {du} units differ, strict {J1[0] if J1 else float('nan'):.2f} -> {100*(J1[0]-J0[0])/J0[0]:+.4f}%; tight MIP gaps {np.round(r['mip_gaps'],5).tolist()} [{time.time()-t0:.0f}s]", flush=True)
    out[i] = (J0, J1, r)
pickle.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"t4b_{case}_{draw}.pkl"), "wb"))

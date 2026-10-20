# F2 task 4c: why is the PGLib reference's QCAC cost ~3.6% above the strict price of its own commitment?
# Hypothesis: the reference stops at slack <= 1e-3 with NO physics check, at an iterate whose dispatch is far from converged
# (binding lines). Test: re-price u_ref with u fixed using the reference's own stopping rule (rho0 1e3, x4, stop at slack <= 1e-3).
import sys, os, pickle, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src"))
import combopt_ac as C
for case, pat, draw in (("pglib_opf_case118_ieee__api", "*FINAL_pglib*fresh5*", "fresh5"), ("case118", "*FINAL_case118_fresh6*", "fresh6")):
    C._W.clear(); C.init(case, True); S = C._W["S"]
    F = pickle.load(open(glob.glob(os.path.join(R, "audit2", "repro", "runs", f"{pat}", "final.pkl"))[0], "rb")); REF = pickle.load(open(os.path.join(R, f"acuc/results/{draw}_refs_{case}.pkl"), "rb"))
    rows = []
    for i in F["tid"][:12]:
        p, q = C._W["inst"][i]; u = np.asarray(F["SR"][i][5], float)
        r = S.price(p, q, u, eps=1e-3, max_iter=30)          # the reference's stopping rule, u fixed
        rows.append((100*(REF[i]["cost"] - F["SR"][i][0])/F["SR"][i][0], 100*(r["cost"] - F["SR"][i][0])/F["SR"][i][0], r["iters"], r["slack"]))
    a = np.array(rows)
    print(f"{case}: REF QCAC cost vs strict: mean {a[:,0].mean():+.3f}% | same-u price with the reference stopping rule (slack<=1e-3, no physics check) vs strict: mean {a[:,1].mean():+.3f}% "
          f"[min {a[:,1].min():+.3f}, max {a[:,1].max():+.3f}], iters {np.round(a[:,2]).astype(int).tolist()}", flush=True)

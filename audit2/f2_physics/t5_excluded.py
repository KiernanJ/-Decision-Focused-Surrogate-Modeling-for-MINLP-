# F2 task 5b: are the excluded (non-converged reference) test instances AC-infeasible, or reference-solver failures?
import sys, os, pickle, time, numpy as np, warnings; warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combopt_ac as C, data2
from evalac import opf_cost
for case, draw in (("case118","fresh5"),("case118","fresh6"),("pglib_opf_case118_ieee__api","fresh5"),("case300","fresh5")):
    C._W.clear(); C.init(case, True); g, nl = C._W["g"], C._W["nl"]
    F = pickle.load(open(os.path.join(R, f"acuc/results/{draw}_refs_{case}.pkl"), "rb")); ids = getattr(data2, f"{draw}_ids")()
    bad = [i for i in ids if not F.get(i, {}).get("converged")]
    res = []
    for i in bad:
        p, q = C._W["inst"][i]; u = np.ones(g.n_gen)
        t0 = time.time(); _, r = C.price_strict((i, u))
        ok, Jo, _ = opf_cost(g, nl, p, q, u)
        res.append((i, round(float(p.sum()/g.pd.sum()), 3), r is not None, ok))
        print(f"  {case} {draw} {i}: load {p.sum()/g.pd.sum():.3f} | all-units-on strict pricing {'FEASIBLE' if r else 'fails'} | PYPOWER OPF all-on {'success' if ok else 'FAIL'} [{time.time()-t0:.0f}s]", flush=True)
    print(f"== {case} {draw}: excluded {len(bad)}; AC-feasible with all units on (strict / PYPOWER): {sum(x[2] for x in res)} / {sum(x[3] for x in res)}", flush=True)

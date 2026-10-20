# F2 task 2b: case300 voltage band -- source data say 0.94-1.06; the code uses 0.90-1.10 ("infeasible at 0.94/1.06"). Check feasibility + cost effect.
import sys, os, pickle, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src"))
import combopt_ac as C
from socp import Socp
C.init("case300", True); g, nl = C._W["g"], C._W["nl"]
F = pickle.load(open(glob.glob(os.path.join(R, "audit2", "repro", "runs", "*FINAL_case300_fresh5*", "final.pkl"))[0], "rb"))
import copy
g2 = copy.deepcopy(g); g2.vmin = np.full(g.n_bus, 0.94); g2.vmax = np.full(g.n_bus, 1.06)
S2 = Socp(g2, nl, n_cuts=4)
C._W["g"] = g2; C._W["S"] = S2          # price_strict / _phys read the grid from _W
ok = 0; rows = []
for i in F["tid"][:10]:
    u = np.asarray(F["SR"][i][5], float); p, q = C._W["inst"][i]
    _, r = C.price_strict((i, u))
    _, r1 = C.price_strict((i, np.ones(g.n_gen)))
    rows.append((i, p.sum()/g.pd.sum(), r is not None, (r[4] if r else None), (100*(r[0]-F["SR"][i][0])/F["SR"][i][0] if r else np.nan), r1 is not None))
    print(f"case300 {i} load {p.sum()/g.pd.sum():.3f}: u_ref at 0.94-1.06 -> {'feasible' if r else 'INFEASIBLE'}" + (f" (repairs {r[4]}, cost {100*(r[0]-F['SR'][i][0])/F['SR'][i][0]:+.3f}% vs 0.90-1.10)" if r else "") + f"; all units on -> {'feasible' if r1 else 'INFEASIBLE'}", flush=True)

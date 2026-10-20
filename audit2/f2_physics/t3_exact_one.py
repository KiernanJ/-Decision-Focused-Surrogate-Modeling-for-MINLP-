# F2 task 3a: exact nonconvex AC-OPF (Gurobi global, u fixed) for one commitment vs the strict price
import sys, os, pickle, time, numpy as np, warnings; warnings.filterwarnings("ignore")
R, case, fin, i, which, tl = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], float(sys.argv[6])
sys.path.insert(0, os.path.join(R, "src"))
import combopt_ac as C
C.init(case, True)
from acuc import build
from gurobipy import GRB
F = pickle.load(open(fin, "rb")); SR = F["SR"]
u = np.asarray(SR[i][5] if which == "ref" else (F["rows"][0]["OP"][i]["u"] if which == "ours" else F["rows"][0]["PX"][i]["u"]), float)
J = SR[i][0] if which == "ref" else (F["rows"][0]["OP"][i]["cost"] if which == "ours" else F["rows"][0]["PX"][i]["cost"])
g, nl = C._W["g"], C._W["nl"]; p, q = C._W["inst"][i]
m, V, cost = build(g, nl, p, q, GRB.CONTINUOUS, exact=True, threads=2)
for x in range(g.n_gen): V["u"][x].lb = V["u"][x].ub = float(u[x])
m.Params.TimeLimit = tl; m.Params.MIPGap = 1e-5
t0 = time.time(); m.optimize()
print(f"{case} inst {i} {which}: strict price J={J:.4f}; Gurobi exact: status {m.Status}, sols {m.SolCount}, obj {m.ObjVal if m.SolCount else float('nan'):.4f}, bound {m.ObjBound:.4f}, gap {m.MIPGap if m.SolCount else float('nan'):.2e}, {time.time()-t0:.0f}s; "
      f"J vs bound {100*(J-m.ObjBound)/m.ObjBound:+.4f}%" + (f", J vs exact incumbent {100*(J-m.ObjVal)/m.ObjVal:+.4f}%" if m.SolCount else ""), flush=True)

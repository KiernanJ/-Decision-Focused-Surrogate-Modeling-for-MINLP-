# C2: did V-net training reduce its OWN training objective (penalised relaxed QCAC value J(V) = cost + rho*slack, uncut)? at the training rho
# (stored, 1e4) and at the deployment rho; validation + fresh5 draws. Also the linearisation slack itself.
import sys, numpy as np, torch, os
from common import *
case = sys.argv[1]; RD = float(sys.argv[2])
D = setup(case); n = D["n"]; data2 = D["data2"]
from socp import Socp
S = Socp(D["g"], D["nl"], n_cuts=4)
OUT, RHO, SEL = vnet(D, case)
ids = list(D["va"]) + data2.fresh5_ids()
for rho in sorted({RHO, RD}):
    R = []
    for i in ids:
        p, q = D["INST"][i]
        rf = S.solve(p, q, np.ones(n), np.zeros(n), rho=rho); rv = S.solve(p, q, 1.0 + OUT[i, :n], OUT[i, n:], rho=rho)
        if rf is None or rv is None: continue
        R.append((rv["cost"] + rho*rv["slack"], rf["cost"] + rho*rf["slack"], rv["slack"], rf["slack"], rv["cost"], rf["cost"]))
    R = np.array(R)
    print(f"[{case}] rho {rho:.0e} (train rho {RHO:.0e}) n={len(R)}: J(V-net)/J(flat) mean {np.mean(R[:,0]/R[:,1]):.4f} (min {np.min(R[:,0]/R[:,1]):.4f}, max {np.max(R[:,0]/R[:,1]):.4f}); "
          f"lin. slack V-net {np.median(R[:,2]):.3f} vs flat {np.median(R[:,3]):.3f} (median); relaxation cost part V-net/flat {np.mean(R[:,4]/R[:,5]):.4f}", flush=True)

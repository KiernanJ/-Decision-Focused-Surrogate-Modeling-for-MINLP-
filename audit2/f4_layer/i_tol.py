# I: is the FD self-inconsistency at small h (trained V) a kink, or just the default Clarabel tolerance against a ~30x smaller gradient?
# Re-do the D1 FD at the deployed V-net point with TIGHT solver tolerances.
import sys, numpy as np, os
from common import *
import cvxpy as cp
case = sys.argv[1]; insts = [int(x) for x in sys.argv[2].split(",")]
D = setup(case); n = D["n"]
import combopt_ac
combopt_ac.init(case, True); S = combopt_ac._W["S"]; INST = combopt_ac._W["inst"]
OUT, RHO, SEL = vnet(D, case)
def loss(i, Vr, Vi, rho, tight):
    p, q = INST[i]; fb, tb = D["g"].f_bus.astype(int), D["g"].t_bus.astype(int)
    S.p_pd.value, S.p_qd.value = p, q; S.p_Vr0.value, S.p_Vi0.value = Vr, Vi; S.p_V0sq.value = Vr**2 + Vi**2
    S.p_VVc.value = Vr[fb]*Vr[tb] + Vi[fb]*Vi[tb]; S.p_VVs.value = Vr[fb]*Vi[tb] - Vr[tb]*Vi[fb]; S.p_rho.value = rho
    S.p_ulo.value = np.zeros(D["G"]); S.p_uhi.value = np.ones(D["G"]); S.p_A.value = np.zeros(S.p_A.shape); S.p_b.value = np.ones(S.p_b.shape); S.p_cutcap.value = 1e6
    kw = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500) if tight else {}
    S.prob.solve(solver=cp.CLARABEL, warm_start=False, **kw)
    return float(S.cost.value + rho*S.v["xi"].value.sum()), S.prob.status
rng = np.random.default_rng(1)
for i in insts:
    Vr, Vi = 1.0 + OUT[i, :n], OUT[i, n:]
    _, e = combopt_ac.vgrad_env((i, Vr, Vi, RHO, None)); ge = np.r_[e[1], e[2]]
    Ds = []
    for _ in range(8):
        d = rng.standard_normal(2*n); d[n + int(D["g"].ref)] = 0; d /= np.linalg.norm(d); Ds.append(d)
    for tight in (False, True):
        L0, st0 = loss(i, Vr, Vi, RHO, tight)
        for h in (1e-3, 1e-4, 1e-5):
            Rw, sts = [], set()
            for d in Ds:
                Lp, s1 = loss(i, Vr + h*d[:n], Vi + h*d[n:], RHO, tight); Lm, s2 = loss(i, Vr - h*d[:n], Vi - h*d[n:], RHO, tight); sts |= {s1, s2}
                Rw.append(((Lp - Lm)/(2*h), (Lp - L0)/h, (L0 - Lm)/h, ge @ d))
            R = np.array(Rw); fd, an = R[:, 0], R[:, 3]
            print(f"[{case}] inst {i} V-net point, {'TIGHT tol 1e-12' if tight else 'default tol    '} h {h:.0e} | FD self-cons {np.corrcoef(R[:,1], R[:,2])[0,1]:+.4f} | cos {fd @ an/(np.linalg.norm(fd)*np.linalg.norm(an)):+.4f} rel.err {np.linalg.norm(an-fd)/np.linalg.norm(fd):.4f} | statuses {sorted(sts)}", flush=True)

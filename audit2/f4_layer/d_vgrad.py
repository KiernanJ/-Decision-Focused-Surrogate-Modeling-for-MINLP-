# D1: FD check of the V-net training gradient (combopt_ac.vgrad_env) at the DEPLOYED clean V-nets (_env_nc), exactly as trained:
# uncut relaxation (cuts=([],[])), rho = stored training rho; flat V and trained V; h in {1e-3,1e-4,1e-5}; FD self-consistency;
# metrics: Pearson corr (as the repo), UNCENTRED cosine and relative error. Also records layer forward statuses (AlmostSolved acceptance).
import sys, numpy as np, torch, os
from common import *
case = sys.argv[1]; NI = int(sys.argv[2]) if len(sys.argv) > 2 else 3
D = setup(case); n = D["n"]; G = D["G"]
import combopt_ac, conelayer
STAT = []
_fw = conelayer.ConeLayer.forward
def fw(self, vals, tol=None):
    x, st = _fw(self, vals, tol); STAT.append(st["status"]); return x, st
conelayer.ConeLayer.forward = fw
combopt_ac.init(case, True); S = combopt_ac._W["S"]; INST = combopt_ac._W["inst"]
OUT, RHO, SEL = vnet(D, case)
def loss(i, Vr, Vi, rho):
    r = S.solve(*INST[i], Vr, Vi, rho=rho); return r["cost"] + rho*r["slack"]
rng = np.random.default_rng(1)
inst = D["tr"][:NI]
for i in inst:
    for where, Vr, Vi in (("flat V", np.ones(n), np.zeros(n)), ("V-net(dep)", 1.0 + OUT[i, :n], OUT[i, n:])):
        _, e = combopt_ac.vgrad_env((i, Vr, Vi, RHO, None)); ge = np.r_[e[1], e[2]]
        L0 = loss(i, Vr, Vi, RHO); Ds = []
        for _ in range(8):
            d = rng.standard_normal(2*n); d[n + int(D["g"].ref)] = 0; d /= np.linalg.norm(d); Ds.append(d)
        for h in (1e-3, 1e-4, 1e-5):
            R = []
            for d in Ds:
                Lp = loss(i, Vr + h*d[:n], Vi + h*d[n:], RHO); Lm = loss(i, Vr - h*d[:n], Vi - h*d[n:], RHO)
                R.append(((Lp - Lm)/(2*h), (Lp - L0)/h, (L0 - Lm)/h, ge @ d))
            R = np.array(R); fd, an = R[:, 0], R[:, 3]
            cfd = np.corrcoef(R[:, 1], R[:, 2])[0, 1]; corr = np.corrcoef(fd, an)[0, 1]; cos = fd @ an/(np.linalg.norm(fd)*np.linalg.norm(an)); rel = np.linalg.norm(an - fd)/np.linalg.norm(fd)
            print(f"[{case}] inst {i:3d} {where:10s} rho {RHO:.0e} h {h:.0e} | FD self-cons {cfd:+.4f} | corr {corr:+.4f} cos {cos:+.4f} rel.err {rel:.4f} signs {(np.sign(fd)==np.sign(an)).sum()}/8 | |grad| {np.linalg.norm(ge):.3e}", flush=True)
from collections import Counter
print(f"[{case}] layer forward statuses: {dict(Counter(STAT))}")

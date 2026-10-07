# %% LICQ / KKT conditioning of the QCAC relaxation at its solution (audit 2026-10-06).
#    case118, first test instance, flat V, rho = 1e6, no cut. Active rows of the
#    canonical cone program: all zero-cone rows, nonnegative rows with s ~ 0, and
#    the normal of every SOC block on its boundary. Reports rank deficiency and the
#    condition number of the KKT matrix used by the layer.
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ["QCAC_CASE"] = "case118"; warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from pipeline import grid
from socp import Socp
from conelayer import ConeLayer
from conedesc import cone_blocks
import data2
g, nl = grid("case118"); S = Socp(g, nl, n_cuts=4)
INST = data2.instances(g, "case118"); te, va, tr = data2.split(); pd_, qd_ = INST[te[0]]
V1, V0 = np.ones(g.n_bus), np.zeros(g.n_bus)
S.solve(pd_, qd_, V1, V0, rho=1e6)
lay = ConeLayer(S.prob, [S.p_b], [S.v["u"]]); lay.build_structure([S.p_b.value])
x, st = lay.forward([S.p_b.value], tol=1e-10)
A, s, y, dims = st["A"].tocsr(), st["s"], st["y"], st["dims"]
rows, tol = [], 1e-7
for kind, i, k in cone_blocks(dims):
    if kind == "zero":
        rows += [A[i+r].toarray().ravel() for r in range(k)]
    elif kind == "nonneg":
        rows += [A[i+r].toarray().ravel() for r in range(k) if s[i+r] < tol]
    elif kind == "soc":
        t, v = s[i], s[i+1:i+k]
        if abs(t - np.linalg.norm(v)) < tol*max(1.0, t) and t > tol:       # boundary, not apex
            nrm = np.r_[t, -v]                                               # normal of the cone surface
            rows.append(nrm @ A[i:i+k].toarray())
Aact = np.array(rows)
sv = np.linalg.svd(Aact, compute_uv=False)
rank = int((sv > sv[0]*1e-9).sum())
K, _ = lay._kkt(st); Kd = K.toarray(); skk = np.linalg.svd(Kd, compute_uv=False)
out = dict(n_vars=int(A.shape[1]), n_active=len(rows), rank=rank, deficiency=len(rows)-rank,
           sv_at_rank=float(sv[rank-1]), sv_next=float(sv[rank]) if rank < len(sv) else None,
           kkt_size=int(Kd.shape[0]), kkt_cond=float(skk[0]/max(skk[-1], 1e-300)))
print(out)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "degeneracy.json"), "w"), indent=1)

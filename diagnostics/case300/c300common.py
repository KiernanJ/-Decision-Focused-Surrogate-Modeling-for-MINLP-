"""case300 diagnostics: shared setup. Read-only on the repo; reuses audit/_common.
Training-time objects mirror acuc/02_train.py (REG1=REG2=1e-10, rho 1e6, tau .08,
dthr 0.8, K=2, anchor at flat V)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "audit"))
import _common
from _common import setup, load_net, RHO
import numpy as np, torch
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

E = setup("case300")
g, S, POOL, Xn = E["g"], E["S"], E["POOL"], E["Xn"]
from pipeline import RESERVE
from framework import DiffDeploy, soft_round
from cutnet_k import rows_torch
NB, NG = g.n_bus, g.n_gen
V1, V0 = np.ones(NB), np.zeros(NB)
DD = DiffDeploy(S, g, E["nl"], reserve=RESERVE, tau=0.08, thr=0.8, rho=RHO, reg=1e-10, reg2=1e-10)

_W, _SC = {}, {}


def anchor(j):
    """Uncut relaxation at FLAT V -- the trainer's anchor and loss scale."""
    if j not in _W:
        r = S.solve(POOL[j][0], POOL[j][1], V1, V0, rho=RHO)
        _W[j] = (np.r_[r["pg"], r["u"]], r)
        _SC[j] = float(r["cost"])
    return _W[j]


def forward(net, ids, anchor_fn=None):
    """Network outputs for ids, with b anchored at w (default: flat-V uncut relaxation)."""
    W = np.stack([(anchor_fn or (lambda j: anchor(j)[0]))(j) for j in ids])
    out = net(Xn[ids])
    Vr, Vi, a, al, a2, d, d2, _ = out
    A, b = rows_torch(a, al, a2, d, d2, torch.tensor(W, dtype=torch.float32))
    return Vr, Vi, A, b


def npf(t):
    return t.detach().numpy().astype(float)


def surrogate(j, Vr, Vi, A, b, want_grad=True):
    anchor(j)
    return DD(POOL[j][0], POOL[j][1], Vr, Vi, A, b, _SC[j], want_grad=want_grad)

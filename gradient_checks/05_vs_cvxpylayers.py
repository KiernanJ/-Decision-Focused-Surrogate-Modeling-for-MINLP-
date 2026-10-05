# %% Head-to-head: MY generic layer vs cvxpylayers/diffcp vs finite differences.
#    Point A: a small WELL-POSED SOCP   -> do the two methods agree?
#    Point B: the real case118 QCAC     -> do they fail together?
#    If A agrees and B fails for both, the method is not the problem; the
#    PROBLEM is the problem.
import os, sys, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("QCAC_CASE", "case118"); os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np, cvxpy as cp, torch
from cvxpylayers.torch import CvxpyLayer
from conelayer import ConeLayer

rng = np.random.default_rng(0)
torch.set_default_dtype(torch.float64)


def cl_grad(prob, params, var, vals, gl):
    """Gradient of <gl, var> w.r.t. params, via cvxpylayers."""
    lay = CvxpyLayer(prob, parameters=params, variables=[var])
    ts = [torch.tensor(np.asarray(v, float), requires_grad=True) for v in vals]
    out, = lay(*ts, solver_args=dict(eps=1e-10, max_iters=200_000))
    (out * torch.tensor(gl)).sum().backward()
    return [t.grad.detach().numpy().copy() for t in ts], out.detach().numpy()


# %% ---------- POINT A: small well-posed SOCP ----------
print("="*70); print("POINT A: small well-posed SOCP")
n = 6
xv = cp.Variable(n); pq = cp.Parameter(n); pr = cp.Parameter(2)
M = rng.normal(size=(3, n))
prob = cp.Problem(cp.Minimize(cp.sum_squares(xv - pq)),
                  [cp.norm(M@xv) <= pr[0], cp.sum(xv) == pr[1], xv >= -5])
vals = [rng.normal(size=n), np.array([2.0, 1.0])]

gl_u = rng.normal(size=n)
# -- cvxpylayers
g_cl, x_cl = cl_grad(prob, [pq, pr], xv, vals, gl_u)
# -- mine: gradient on the USER variable, so restrict the canonical loss to xv
lay = ConeLayer(prob, [pq, pr], [xv]); lay.build_structure(vals)
x, st = lay.forward(vals, tol=1e-12)
# locate xv inside canonical x by matching values
pq.value, pr.value = vals[0], vals[1]; prob.solve(solver=cp.CLARABEL)
idx = [int(np.argmin(np.abs(x - v))) for v in xv.value]
gl_can = np.zeros(x.size); gl_can[idx] = gl_u
g_me = lay.vjp(st, gl_can)
print(f"  [mine] KKT solve relative residual: {lay.last_residual:.2e}")
# -- finite differences on the same scalar loss
def fd_loss(vals, pi, j, h):
    o = []
    for s_ in (+1, -1):
        v = [np.array(a, float) for a in vals]; v[pi].reshape(-1)[j] += s_*h
        pq.value, pr.value = v[0], v[1]; prob.solve(solver=cp.CLARABEL)
        o.append(float(gl_u @ xv.value))
    return (o[0]-o[1])/(2*h)
fd = np.array([fd_loss(vals, 0, j, 1e-6) for j in range(n)])
print(f"  solution match (mine vs cvxpylayers): {np.abs(x_cl - xv.value).max():.2e}")
print(f"  d loss / d pq   finite differences : {np.array2string(fd, precision=4)}")
print(f"                  MINE               : {np.array2string(g_me[0], precision=4)}")
print(f"                  cvxpylayers        : {np.array2string(g_cl[0], precision=4)}")
print(f"  rel err  MINE vs fd        : {np.abs(g_me[0]-fd).max()/np.abs(fd).max():.2e}")
print(f"  rel err  cvxpylayers vs fd : {np.abs(g_cl[0]-fd).max()/np.abs(fd).max():.2e}")

# %% ---------- POINT B: real case118 QCAC ----------
print(); print("="*70); print("POINT B: case118 QCAC, cuts binding")
from pipeline import grid, sample
from socp import Socp
g, nl = grid("case118"); pd_, qd_ = sample(g, 1, seed=0)[0]; K = 8
S = Socp(g, nl, n_cuts=K)
A0 = 0.05*rng.normal(size=(K, 2*g.n_gen))
r0 = S.solve(pd_, qd_, np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e4,
             A=A0, b=np.full(K, 1e3), cut_cap=0.0)
b0 = A0 @ np.concatenate([r0["pg"], r0["u"]]) - 0.05
S.solve(pd_, qd_, np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e4, A=A0, b=b0, cut_cap=0.0)
u_ref = S.v["u"].value.copy()
gl_u = rng.normal(size=g.n_gen)

# finite differences on the scalar loss <gl, u>, per entry of b
def fd_b(j, h):
    o = []
    for s_ in (+1, -1):
        bb = np.array(b0); bb[j] += s_*h
        r = S.solve(pd_, qd_, np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e4,
                    A=A0, b=bb, cut_cap=0.0)
        o.append(float(gl_u @ r["u"]))
    return (o[0]-o[1])/(2*h)
print("  finite differences of <gl,u> w.r.t. b, at three step sizes")
FD = {h: np.array([fd_b(j, h) for j in range(K)]) for h in (1e-1, 1e-2, 1e-3)}
for h, v in FD.items():
    print(f"    h={h:<6.0e} {np.array2string(v, precision=4)}")
ref = FD[1e-2]

t0 = time.time()
try:
    pars = [S.p_pd, S.p_qd, S.p_Vr0, S.p_Vi0, S.p_V0sq, S.p_VVc, S.p_VVs,
            S.p_rho, S.p_ulo, S.p_uhi, S.p_A, S.p_b, S.p_cutcap]
    vals = [p.value for p in pars]
    g_cl, x_cl = cl_grad(S.prob, pars, S.v["u"], vals, gl_u)
    print(f"\n  cvxpylayers  u match: {np.abs(x_cl-u_ref).max():.2e}   ({time.time()-t0:.0f}s)")
    print(f"  cvxpylayers  d loss/db : {np.array2string(g_cl[-2], precision=4)}")
    print(f"  cvxpylayers  rel err vs fd(h=1e-2): "
          f"{np.abs(g_cl[-2]-ref).max()/max(1e-12,np.abs(ref).max()):.2e}")
except Exception as e:
    print(f"\n  cvxpylayers FAILED: {type(e).__name__}: {str(e)[:200]}")

# -- MY layer on the same point, for the record
lay2 = ConeLayer(S.prob, [S.p_b], [S.v["u"]]); lay2.build_structure([b0])
x2, st2 = lay2.forward([b0], tol=1e-12)
uidx = [int(np.argmin(np.abs(x2 - v))) for v in u_ref]
glc = np.zeros(x2.size); glc[uidx] = gl_u
g2 = lay2.vjp(st2, glc)
print(f"\n  MINE         d loss/db : {np.array2string(g2[0], precision=4)}")
print(f"  MINE         KKT relative residual: {lay2.last_residual:.2e}")
print(f"  MINE         rel err vs fd(h=1e-2): "
      f"{np.abs(g2[0]-ref).max()/max(1e-12,np.abs(ref).max()):.2e}")
print(f"\n  reference (finite differences, h=1e-2): {np.array2string(ref, precision=4)}")

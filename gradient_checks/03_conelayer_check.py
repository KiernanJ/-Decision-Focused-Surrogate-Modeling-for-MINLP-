# %% gradient check for the GENERIC cone layer, against finite differences
#    two problem shapes, neither of them QCAC, to prove nothing is special-cased:
#      (a) a QP   -> zero + nonneg cones only   (the hybrid-vehicle shape)
#      (b) an SOCP-> zero + nonneg + soc cones  (the QCAC shape)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, cvxpy as cp
from conelayer import ConeLayer

rng = np.random.default_rng(0)


def fd_jvp(layer, vals, dvals, h):
    """(x(theta + h d) - x(theta - h d)) / 2h  -- central differences."""
    up = [np.asarray(v, float) + h*np.asarray(d, float) for v, d in zip(vals, dvals)]
    dn = [np.asarray(v, float) - h*np.asarray(d, float) for v, d in zip(vals, dvals)]
    xu, _ = layer.forward(up)
    xd, _ = layer.forward(dn)
    return (xu - xd) / (2*h)


def check(name, prob, params, vars_, vals, n_dir=6):
    lay = ConeLayer(prob, params, vars_)
    lay.build_structure(vals)
    x, st = lay.forward(vals)
    if x is None:
        print(f"{name}: forward FAILED {st}"); return
    dims = st["dims"]
    nsoc = len(getattr(dims, "q", []) or [])
    print(f"\n{name}:  canonical n={x.size} m={st['A'].shape[0]}  "
          f"zero={dims.zero} nonneg={dims.nonneg} soc_blocks={nsoc}")

    # ---- forward mode vs central differences, random directions ----
    worst = 0.0
    for _ in range(n_dir):
        d = [rng.normal(size=np.asarray(v).shape) for v in vals]
        an = lay.jvp(st, d)
        best = np.inf
        for h in (1e-4, 1e-5, 1e-6):
            fd = fd_jvp(lay, vals, d, h)
            err = np.abs(an - fd).max() / max(1.0, np.abs(fd).max())
            best = min(best, err)
        worst = max(worst, best)
    print(f"  jvp  vs finite differences : rel err {worst:.2e}")

    # ---- reverse mode: check <dL_dx, jvp(d)> == <vjp(dL_dx), d> ----
    wr = 0.0
    for _ in range(n_dir):
        gl = rng.normal(size=x.size)
        d = [rng.normal(size=np.asarray(v).shape) for v in vals]
        lhs = float(gl @ lay.jvp(st, d))
        gr = lay.vjp(st, gl)
        rhs = float(sum(np.sum(a*b) for a, b in zip(gr, d)))
        wr = max(wr, abs(lhs - rhs)/max(1.0, abs(lhs)))
    print(f"  vjp  adjoint identity      : rel err {wr:.2e}")

    # ---- reverse mode end-to-end on a real scalar loss, vs finite differences ----
    gl = rng.normal(size=x.size)
    gr = lay.vjp(st, gl)
    p0 = np.asarray(vals[0], float)
    idx = np.ndindex(p0.shape)
    errs = []
    for k, ij in enumerate(idx):
        if k >= 5: break
        h = 1e-5
        up = [np.array(v, float) for v in vals]; up[0][ij] += h
        dn = [np.array(v, float) for v in vals]; dn[0][ij] -= h
        xu, _ = lay.forward(up); xd, _ = lay.forward(dn)
        fd = float(gl @ (xu - xd)) / (2*h)
        errs.append(abs(gr[0][ij] - fd)/max(1.0, abs(fd)))
    print(f"  vjp  vs finite differences : rel err {max(errs):.2e}  (5 entries)")
    lay.forward(vals)


# %% (a) QP -- the hybrid-vehicle cone shape: zero + nonneg only
n, m = 8, 5
Q = rng.normal(size=(n, n)); Q = Q.T@Q + 0.5*np.eye(n)
Am = rng.normal(size=(m, n))
xv = cp.Variable(n)
pb = cp.Parameter(m)
pc = cp.Parameter(n)
prob = cp.Problem(cp.Minimize(0.5*cp.quad_form(xv, cp.psd_wrap(Q)) + pc@xv),
                  [Am@xv <= pb, xv >= -3, xv <= 3])
check("(a) QP  zero+nonneg", prob, [pb, pc], [xv],
      [rng.normal(size=m)+2.0, rng.normal(size=n)])

# %% (b) SOCP -- the QCAC cone shape: a second-order cone in the constraints
n2 = 6
xv2 = cp.Variable(n2)
pq = cp.Parameter(n2)
pr = cp.Parameter(2)
M = rng.normal(size=(3, n2))
prob2 = cp.Problem(cp.Minimize(cp.sum_squares(xv2 - pq)),
                   [cp.norm(M@xv2) <= pr[0], cp.sum(xv2) == pr[1], xv2 >= -5])
check("(b) SOCP zero+nonneg+soc", prob2, [pq, pr], [xv2],
      [rng.normal(size=n2), np.array([2.0, 1.0])])

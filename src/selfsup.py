"""Label-free self-improvement worker (2026-10-11).

Replaces the long-solve teacher (W=120/180 MIQCQP) by the method's OWN deployment followed by an
exact-layer-guided local search: deploy (one short MIQCQP at V, optional cut rows) -> combopt_ac.improve
(ranks single flips / swaps by the conic layer's d cost/d u, prices the top candidates, accepts a move only
if the priced cost strictly decreases). No reference solution and no solve longer than deployment.
Pool initializer = combopt_ac.init (this module only reads its worker state).
Lives in a module because macOS spawn pickles workers by module path.
"""
import time
import numpy as np
import combopt_ac as C


def deploy_improve(args):
    """-> (i, dict(u0, c0 = own deployment; u, c = after improvement; n_moves, fallback, secs)) or (i, None)."""
    i, V, A, b, work, rho, k, rounds, swaps = args
    t0 = time.time()
    _, R = C.miqcqp((i, V, A, b, work, rho))
    if R is None:
        return i, None
    u0, c0 = np.asarray(R["u"], float), float(R["cost"])
    u, c, _ = C.improve(i, u0, c0, R["pg"], R["vr"], R["vi"], k=k, rounds=rounds, swaps=swaps)
    return i, dict(u0=u0, c0=c0, u=np.asarray(u, float), c=float(c), n_moves=int(np.sum(u != u0)),
                   fallback=bool(R.get("fallback")), secs=time.time() - t0)


_S = {}


def check_rows(args):
    """LABEL-FREE validity check of learned cut rows at deployment (2026-10-11): solve the convex relaxation at V with the
    rows as soft constraints (violation priced 1e6, above the linearisation penalty rho). Rows the relaxation still has to
    violate (slack > tol) are incompatible with the relaxation's feasible set on this instance -> released (A=0, b=1).
    Repeated until no row is violated (max `passes`). -> (i, (A, b, released rows, secs, slack per pass))."""
    i, V, rho, A, b, tol, passes = args
    t0 = time.time()
    if "S" not in _S:
        from socp import Socp
        _S["S"] = Socp(C._W["g"], C._W["nl"], n_cuts=A.shape[0], cut_price=1e6)
    S = _S["S"]; A, b = np.array(A, float), np.array(b, float); rel, hist = [], []
    for _ in range(passes):
        r = S.solve(*C._W["inst"][i], V[0], V[1], rho=rho, A=A, b=b, cut_cap=1e6, soft_fallback=False)
        if r is None:
            break
        sc = np.asarray(S.v["sc"].value, float).ravel(); hist.append(float(sc.sum()))
        new = [k for k in range(len(b)) if sc[k] > tol and k not in rel]
        if not new:
            break
        rel += new; A[new] = 0.0; b[new] = 1.0
    return i, (A, b, rel, time.time() - t0, hist)


def restore_candidates(args):
    """the two cheap label-free restorations of the relaxation at V (round 0.8 + merit order / round 0.5 + confidence
    order, reserve top-up, upward repair, downward pass) -> (i, {name: u}). Same heuristics miqcqp uses as its fallback."""
    i, V = args
    t0 = time.time()
    _, r = C.relax((i, V))
    out = {}
    if r is not None:
        for name, thr, o in (("restore merit0.8", 0.8, "merit"), ("restore conf0.5", 0.5, "conf")):
            z = C.old_restore(i, r[0], thr, o)
            if z is not None:
                out[name] = np.asarray(z[0], float)
    return i, dict(cands=out, secs=time.time() - t0)

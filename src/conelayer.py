"""Generic differentiable layer for ANY DPP cvxpy problem.

Forward: cvxpy canonicalises, CLARABEL solves.
Backward: implicit differentiation of the conic optimality conditions.

Nothing in this file knows what problem it differentiates.  The same object
handles AC unit commitment (SOC cones from the rank-1 and line-limit rows) and
the hybrid vehicle (a QP: zero and nonneg cones only).

In standard conic form

    min  1/2 x'Px + q'x    s.t.   Ax + s = b,   s in K,   y in K*

write w = s - y, so that s = Proj_K(w) and y = Proj_K(w) - w.  The optimality
conditions become two equations in (x, w),

    F1:  P x + q + A'(Proj(w) - w) = 0
    F2:  A x + Proj(w) - b         = 0

and differentiating with respect to a parameter gives

    [ P   A'(DProj - I) ] [ dx ]   [ -dq - dP x - dA' y ]
    [ A   DProj         ] [ dw ] = [  db - dA x         ]   ...  K z = r

Every constraint here is LINEAR; the nonlinearity lives entirely in the cone
membership, so there is no per-problem constraint Hessian to hand-derive.  That
is what makes this transfer across problems, unlike differentiating the smooth
form of each constraint.

REVERSE MODE.  For a scalar loss L(x),

    dL/dtheta_i = dL_dx' [I 0] K^-1 r_i = g' r_i,    g = K^-T [dL_dx; 0]

so ONE transposed solve yields g and each parameter costs a single inner
product.  Because the problem is DPP the canonical data are AFFINE in theta, so
the per-parameter directions (dP, dq, dA, db) are CONSTANT -- extracted once at
build time and reused for every instance and every training step.

Why not cvxpylayers: diffcp's SCS branch is 61% off at eps=1e-9 on the QCAC cone
program, and its CLARABEL branch returns an all-zero vector.  Both measured, see
README.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import cvxpy as cp
import clarabel

from conedesc import proj_soc, proj_nonneg, cone_blocks, dproj_cone


# %% cone projection over a full canonical vector
def proj_cone(v, dims):
    out = np.zeros_like(v)
    for kind, i, k in cone_blocks(dims):
        seg = v[i:i + k]
        if kind == "zero":
            out[i:i + k] = 0.0
        elif kind == "nonneg":
            out[i:i + k] = proj_nonneg(seg)
        elif kind == "soc":
            out[i:i + k] = proj_soc(seg)
        else:
            raise NotImplementedError(kind)
    return out


# %% canonical solve
def solve_canonical(P, q, A, b, dims, tol=None):
    """Solve  min 1/2 x'Px + q'x  s.t.  Ax + s = b, s in K  with CLARABEL."""
    cones = []
    for kind, _, k in cone_blocks(dims):
        if kind == "zero":
            cones.append(clarabel.ZeroConeT(k))
        elif kind == "nonneg":
            cones.append(clarabel.NonnegativeConeT(k))
        elif kind == "soc":
            cones.append(clarabel.SecondOrderConeT(k))
        else:
            raise NotImplementedError(kind)
    st = clarabel.DefaultSettings()
    st.verbose = False
    if tol is not None:
        st.tol_gap_abs = st.tol_gap_rel = tol
        st.tol_feas = tol
    sol = clarabel.DefaultSolver(sp.triu(sp.csc_matrix(P)).tocsc(),
                                 np.asarray(q, float),
                                 sp.csc_matrix(A), np.asarray(b, float),
                                 cones, st).solve()
    return (np.asarray(sol.x, float), np.asarray(sol.z, float),
            np.asarray(sol.s, float), str(sol.status))


# %% the layer
class ConeLayer:
    """Differentiable solve of a parameterised DPP cvxpy problem.

    problem     : cvxpy Problem, DPP in `parameters`
    parameters  : list of cp.Parameter, the differentiable inputs
    variables   : list of cp.Variable whose values the caller wants back
    """

    def __init__(self, problem, parameters, variables, reg=None):
        assert problem.is_dpp(), "canonicalisation must be affine in the parameters"
        # Per-INSTANCE regularisation. It was a class attribute, which meant the
        # two layers of a chained objective silently shared one value -- and the
        # right value differs between them: layer 1 needs 1e-8 at rho=1e6 while
        # layer 2 needs 1e-10, and running layer 2 at 1e-8 drove its fuel-cost
        # gradient to cosine 0.42 instead of 0.9999.
        self.reg = reg
        self.problem = problem
        self.parameters = list(parameters)
        self.variables = list(variables)
        self._struct = None          # affine canonical structure, built lazily
        self._slices = None          # where each variable sits inside canonical x

    # ---- canonical data at the current parameter values ----
    def _data(self):
        d, _, _ = self.problem.get_problem_data(cp.CLARABEL)
        A = sp.csc_matrix(d["A"])
        n = A.shape[1]
        P = d.get("P", None)
        P = sp.csc_matrix((n, n)) if P is None else sp.csc_matrix(P)
        return P, np.asarray(d["c"], float), A, np.asarray(d["b"], float), d["dims"]

    def _set(self, vals):
        for p, v in zip(self.parameters, vals):
            p.value = np.asarray(v, float).reshape(p.shape)

    # ---- affine structure: canonical data are affine in theta, so extract the
    #      constant per-parameter directions ONCE and reuse them forever ----
    def build_structure(self, base_vals):
        base = [np.asarray(v, float).reshape(p.shape).copy()
                for v, p in zip(base_vals, self.parameters)]
        self._set(base)
        P0, q0, A0, b0, dims = self._data()
        dirs = []
        for pi, bv in enumerate(base):
            flat_shape = bv.shape
            nflat = int(np.prod(flat_shape)) if flat_shape else 1
            per = []
            for j in range(nflat):
                pert = [x.copy() for x in base]
                f = pert[pi].reshape(-1)
                h = 1.0                                  # affine => exact at any h
                f[j] += h
                self._set(pert)
                P1, q1, A1, b1, _ = self._data()
                per.append(dict(dP=(sp.csc_matrix(P1) - P0) / h,
                                dq=(q1 - q0) / h,
                                dA=(sp.csc_matrix(A1) - A0) / h,
                                db=(b1 - b0) / h))
            dirs.append(per)
        self._set(base)
        self._struct = dict(dims=dims, dirs=dirs)
        return self._struct

    # ---- where each user variable lives inside the canonical x ----
    #
    # cvxpy's InverseData carries var_offsets (variable id -> start index), so
    # the mapping is exact and generic.  Matching canonical entries to variable
    # VALUES instead -- which earlier throwaway scripts did -- silently breaks
    # whenever two entries happen to be equal.
    def var_slices(self):
        if self._slices is not None:
            return self._slices
        _, _, inv = self.problem.get_problem_data(cp.CLARABEL)
        off = None
        for el in inv:
            o = getattr(el, "var_offsets", None)
            if o:
                off = o
                break
        if off is None:
            raise RuntimeError("cvxpy did not expose var_offsets for this problem")
        out = {}
        for v in self.variables:
            if v.id not in off:
                raise RuntimeError(f"variable {v.id} absent from the canonical map")
            n = int(np.prod(v.shape)) if v.shape else 1
            out[v.id] = slice(int(off[v.id]), int(off[v.id]) + n)
        self._slices = out
        return out

    # ---- forward ----
    def forward(self, vals, tol=None):
        self._set(vals)
        P, q, A, b, dims = self._data()
        x, y, s, status = solve_canonical(P, q, A, b, dims, tol=tol)
        if "Solved" not in status:
            return None, dict(status=status)
        state = dict(P=P, q=q, A=A, b=b, dims=dims, x=x, y=y, s=s,
                     status=status, vals=[np.asarray(v, float).copy() for v in vals])
        return x, state

    # ---- KKT matrix at the solution ----
    def _kkt(self, state):
        A, P, dims = state["A"], state["P"], state["dims"]
        w = state["s"] - state["y"]
        DP = dproj_cone(w, dims)
        m = A.shape[0]
        I = sp.eye(m, format="csc")
        return sp.bmat([[P, A.T @ (sp.csc_matrix(DP) - I)],
                        [A, sp.csc_matrix(DP)]], format="csc"), w

    # ---- linear solve for the KKT system ----
    #
    # On a NONDEGENERATE problem the KKT matrix is invertible and a plain sparse
    # LU is exact.  On a DEGENERATE one it is not, and the choice made here is
    # the difference between a usable gradient and noise.
    #
    # Measured on case118 QCAC, where LICQ fails (2154 active rows, rank 1809,
    # deficiency 345) and the KKT condition number reaches ~1e21:
    #
    #   plain spsolve          -> SuperLU raises, or returns nan
    #   lsqr / lsmr            -> cosine 0.18 against finite differences, and
    #                             IDENTICAL for every damping value, i.e. the
    #                             Krylov iteration stalls rather than solving.
    #                             This is what diffcp/cvxpylayers use, and it is
    #                             why cvxpylayers returns a 97%-wrong gradient
    #                             on this problem.
    #   truncated-SVD pinv     -> cosine 0.984, but an 81 s dense 7951^2 SVD
    #   DIRECT LU on K + eps I -> cosine 0.987 in 0.01 s          <-- used here
    #
    # eps was chosen on a 5-instance validation split and reported on 5 held-out
    # instances: mean cosine 0.990, min 0.967.  The plateau is broad (1e-6 to
    # 1e-9 all above 0.993), so this is not a knife-edge constant.
    #
    # The regularisation is legitimate rather than a fudge because the null space
    # of K lies almost entirely in the DUAL block: measured |x-component|^2 ~ 0
    # over the smallest singular vectors.  dx is unique even where dy is not, so
    # damping the dual directions leaves the primal derivative intact.
    REG = 1e-8

    # SuperLU's SUPERNODAL kernels call cblas_dtrsv with a leading dimension
    # smaller than the block size, which Apple's Accelerate BLAS rejects and
    # then ABORTS the process on ("lda must be >= MAX(N,1)"). It is not
    # catchable from Python: it killed two training runs at epochs 14 and 5,
    # and only the arm that calls this solver was ever affected. relax=1 and
    # panel_size=1 disable supernodes, which removes the bad call. Measured on
    # the same system: identical residual (8.06e-12), marginally faster.
    # ---- factorisation in a PERSISTENT helper process ----
    # See src/lu_server.py for why: SuperLU aborts (not raises) after enough
    # factorisations, fork deadlocks, spawn re-runs __main__. A helper spoken
    # to over pipes can simply be restarted, so an abort costs one solve.
    _SRV = None
    _SRV_N = 0

    @staticmethod
    def _lu_inproc(M):
        # DROP EXPLICIT ZEROS FIRST. (DProj - I) is exactly zero wherever the
        # projection derivative is 1, and those entries are STORED: 1724 of
        # them in a case118 KKT. SuperLU treats a stored entry as a pivot
        # candidate, finds a hard zero, and on this machine's Accelerate BLAS
        # that surfaced as "cblas_dtrsv/dgemv invalid value" or a SIGSEGV --
        # an abort Python cannot catch, which killed three training runs.
        M = sp.csc_matrix(M)
        M.eliminate_zeros()
        return spla.splu(M)

    @classmethod
    def _server(cls):
        if cls._SRV is None or cls._SRV.poll() is not None:
            import subprocess
            here = os.path.dirname(os.path.abspath(__file__))
            cls._SRV = subprocess.Popen(
                [sys.executable, os.path.join(here, "lu_server.py")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, bufsize=0)
            cls._SRV_N += 1
        return cls._SRV

    @staticmethod
    def _lu_solve(M, r):
        if os.environ.get("QCAC_LU_INPROC") == "1":
            return ConeLayer._lu_inproc(M).solve(r)
        import struct
        M = sp.csc_matrix(M); M.eliminate_zeros()
        M.sort_indices()
        data = np.ascontiguousarray(M.data, np.float64)
        ind = np.ascontiguousarray(M.indices, np.int32)
        ptr = np.ascontiguousarray(M.indptr, np.int32)
        rr = np.ascontiguousarray(np.asarray(r, float), np.float64)
        # On repeated helper death, JITTER the diagonal. The abort is
        # matrix-specific -- a checkpoint-resume died at the same epoch every
        # time -- so retrying the identical matrix is futile, while a slightly
        # different one factors fine. The gradient is insensitive to this: its
        # cosine is flat across eps 1e-9..1e-11.
        JITTER = (0.0, 1e-9, 1e-8, 1e-7, 1e-6)
        for attempt, jit in enumerate(JITTER):
            if jit:
                Mj = (M + jit*sp.eye(M.shape[0], format="csc")).tocsc()
                Mj.sort_indices()
                data = np.ascontiguousarray(Mj.data, np.float64)
                ind = np.ascontiguousarray(Mj.indices, np.int32)
                ptr = np.ascontiguousarray(Mj.indptr, np.int32)
                nnz = Mj.nnz
            else:
                nnz = M.nnz
            req = (struct.pack("<4q", M.shape[0], M.shape[1], nnz, rr.size)
                   + data.tobytes() + ind.tobytes() + ptr.tobytes() + rr.tobytes())
            srv = ConeLayer._server()
            try:
                srv.stdin.write(req); srv.stdin.flush()
                tag = srv.stdout.read(2)
                if tag == b"OK":
                    buf = b""
                    need = 8*rr.size
                    while len(buf) < need:
                        c = srv.stdout.read(need-len(buf))
                        if not c:
                            raise RuntimeError("helper closed mid-reply")
                        buf += c
                    return np.frombuffer(buf, dtype=np.float64).copy()
                if tag == b"ER":
                    n = struct.unpack("<q", srv.stdout.read(8))[0]
                    raise RuntimeError(srv.stdout.read(n).decode())
                raise RuntimeError("helper died")        # abort: tag is empty
            except RuntimeError:
                try:
                    srv.kill()
                except Exception:
                    pass
                ConeLayer._SRV = None
        # Never fall back in-process: that abort would kill the TRAINING run,
        # and a resumed checkpoint dies at the same epoch forever. Raising lets
        # the caller skip this instance and keep going.
        raise RuntimeError("LU helper failed on every jitter level")

    @staticmethod
    def _solve(M, r, self_or_reg=None):
        info = {}
        nrm = max(1e-30, np.linalg.norm(r))
        z = None
        try:
            z = ConeLayer._lu_solve(M, r)
        except Exception:
            z = None
        if z is not None and np.all(np.isfinite(z)) and \
           np.linalg.norm(M @ z - r)/nrm <= 1e-8:
            info.update(mode="lu", residual=float(np.linalg.norm(M @ z - r)/nrm))
            return z, info
        # singular: regularise and factor directly.  NOT lsqr -- see above.
        reg = ConeLayer.REG if getattr(self_or_reg, "reg", None) is None \
            else self_or_reg.reg
        Mr = (M + reg*sp.eye(M.shape[0], format="csc")).tocsc()
        z = ConeLayer._lu_solve(Mr, r)
        info.update(mode="lu+reg", reg=reg,
                    residual=float(np.linalg.norm(M @ z - r)/nrm))
        if not np.all(np.isfinite(z)):
            raise RuntimeError("regularised KKT solve produced non-finite values")
        return z, info

    # ---- forward-mode: derivative of canonical x in ONE parameter direction ----
    def jvp(self, state, dvals):
        K, _ = self._kkt(state)
        x, y = state["x"], state["y"]
        n, m = x.size, state["A"].shape[0]
        r1 = np.zeros(n)
        r2 = np.zeros(m)
        for pi, dv in enumerate(dvals):
            flat = np.asarray(dv, float).reshape(-1)
            for j, c in enumerate(flat):
                if c == 0.0:
                    continue
                D = self._struct["dirs"][pi][j]
                r1 += c * (-D["dq"] - D["dP"] @ x - D["dA"].T @ y)
                r2 += c * (D["db"] - D["dA"] @ x)
        z, info = self._solve(K, np.concatenate([r1, r2]), self)
        self.last_residual, self.last_mode = info["residual"], info["mode"]
        return z[:n]

    # ---- reverse-mode: gradient of a scalar loss w.r.t. every parameter ----
    def vjp(self, state, dL_dx):
        K, _ = self._kkt(state)
        x, y = state["x"], state["y"]
        n, m = x.size, state["A"].shape[0]
        g, info = self._solve(K.T.tocsc(),
                              np.concatenate([np.asarray(dL_dx, float), np.zeros(m)]),
                              self)
        self.last_residual, self.last_mode = info["residual"], info["mode"]
        g1, g2 = g[:n], g[n:]
        out = []
        for pi, p in enumerate(self.parameters):
            per = self._struct["dirs"][pi]
            gr = np.zeros(len(per))
            for j, D in enumerate(per):
                gr[j] = (g1 @ (-D["dq"] - D["dP"] @ x - D["dA"].T @ y)
                         + g2 @ (D["db"] - D["dA"] @ x))
            out.append(gr.reshape(p.shape))
        return out




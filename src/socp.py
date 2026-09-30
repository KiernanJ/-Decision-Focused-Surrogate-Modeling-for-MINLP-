"""
The QCAC relaxation as an SOCP, compiled once and re-solved as a parameter sweep.

WHY THIS FILE EXISTS. `acuc.build` hands the linearised model to Gurobi, and on
case118 the CONTINUOUS relaxation -- no integer variables at all -- takes 120 s
and still hits its time limit. Gurobi reports "Continuous model is non-convex --
solving as a MIP" and runs spatial branch-and-bound on it, because it will not
certify `cij^2 + sij^2 <= cii_i cii_j` as a rotated cone (8 of case118's 186
stay bilinear, and rewriting it as a plain SOC did not help). The model IS an
SOCP. Given to a conic solver it takes 0.06-0.10 s -- a 1200-2000x difference --
and returns the same cost Gurobi does, to the digit, at every rho tested.

That gap is the difference between a trainable method and an untrainable one:
training needs thousands of these solves.

Gurobi keeps the two jobs that genuinely need a MINLP solver: `solve_global`
(exact nonconvex AC) and `solve_qcac_iterative` (54 binaries). Only the
relaxation moves here.

The problem is built ONCE with cvxpy Parameters and re-solved by assigning to
them, so per-instance cost is solve time alone. That also keeps the formulation
DPP-compliant, which is what a differentiable layer needs later.

  u_lo/u_hi   select the two uses of the same model: (0,1) is the relaxation,
              (z,z) fixes the commitment and dispatches.
  A, b        the learned cuts, always K rows; A=0, b=1 disables them without
              changing the compiled structure.
  V0sq, VV0   Vr0^2+Vi0^2 and the branch cross terms, passed as their own
              parameters because a parameter SQUARED is not DPP.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

# %% the compiled problem


class Socp:
    def __init__(self, g, nl, n_cuts=8, cut_price=1e4):
        n, L, G = g.n_bus, g.n_branch, g.n_gen
        fb, tb = g.f_bus.astype(int), g.t_bus.astype(int)
        self.g, self.nl, self.n_cuts = g, nl, n_cuts
        self.n, self.L, self.G = n, L, G

        self.p_pd, self.p_qd = cp.Parameter(n), cp.Parameter(n)
        self.p_Vr0, self.p_Vi0 = cp.Parameter(n), cp.Parameter(n)
        self.p_V0sq = cp.Parameter(n)                  # Vr0^2 + Vi0^2
        self.p_VVc = cp.Parameter(L)                   # Vr0_i Vr0_j + Vi0_i Vi0_j
        self.p_VVs = cp.Parameter(L)                   # Vr0_i Vi0_j - Vr0_j Vi0_i
        self.p_rho = cp.Parameter(nonneg=True)
        self.p_ulo, self.p_uhi = cp.Parameter(G), cp.Parameter(G)
        self.p_A, self.p_b = cp.Parameter((n_cuts, 2*G)), cp.Parameter(n_cuts)
        # Cap on how far a cut may be violated. 0 makes the cut HARD.
        #
        # Soft cuts alone are a trap. With a linear price on the violation, if
        # a cut is violated at the optimum then the penalty's GRADIENT in the
        # cut variables is the price, a constant, and the cut's RIGHT-HAND SIDE
        # drops out of the optimality conditions entirely. Measured: the learned
        # band was violated on 8/8 instances, and replacing it with [0,0], or
        # with [0,0] shifted by -20, gave bit-identical results. The band was
        # decorative; the gain came from a constant 1e4-per-generator penalty.
        self.p_cutcap = cp.Parameter(nonneg=True)

        vr, vi = cp.Variable(n), cp.Variable(n)
        pg, qg, u = cp.Variable(G), cp.Variable(G), cp.Variable(G)
        cii, cij, sij = cp.Variable(n), cp.Variable(L), cp.Variable(L)
        xi = cp.Variable(n + 2*L, nonneg=True)
        sc = cp.Variable(n_cuts, nonneg=True)
        self.v = dict(vr=vr, vi=vi, pg=pg, qg=qg, u=u, cii=cii, cij=cij,
                      sij=sij, xi=xi, sc=sc)

        ph = g.vmax[fb]*g.vmax[tb]
        C = [cii >= g.vmin**2, cii <= g.vmax**2,
             cij >= -ph, cij <= ph, sij >= -ph, sij <= ph,
             vr >= -g.vmax, vr <= g.vmax, vi >= -g.vmax, vi <= g.vmax,
             pg >= 0, pg <= g.pmax, qg >= g.qmin, qg <= g.qmax,
             u >= self.p_ulo, u <= self.p_uhi,
             vi[int(g.ref)] == 0, vr[int(g.ref)] >= 0]

        # linearisation of the voltage products, residual absorbed by xi
        lin_ii = 2*cp.multiply(self.p_Vr0, vr) + 2*cp.multiply(self.p_Vi0, vi) - self.p_V0sq
        lc = (cp.multiply(self.p_Vr0[tb], vr[fb]) + cp.multiply(self.p_Vr0[fb], vr[tb])
              + cp.multiply(self.p_Vi0[tb], vi[fb]) + cp.multiply(self.p_Vi0[fb], vi[tb])
              - self.p_VVc)
        ls = (cp.multiply(self.p_Vi0[tb], vr[fb]) - cp.multiply(self.p_Vi0[fb], vr[tb])
              - cp.multiply(self.p_Vr0[tb], vi[fb]) + cp.multiply(self.p_Vr0[fb], vi[tb])
              - self.p_VVs)
        C += [cp.abs(cii - lin_ii) <= xi[:n],
              cp.abs(cij - lc) <= xi[n:n+L],
              cp.abs(sij - ls) <= xi[n+L:]]

        # SOC strengthening. Load-bearing, not cosmetic: without it the slack
        # still goes to 1e-10 but (cii,cij,sij) leaves the rank-1 manifold and
        # case118 inst0 returns 161,352.6 -- BELOW the proven optimum 164,101.4.
        C += [cp.SOC(cii[fb] + cii[tb],
                     cp.vstack([2*cij, 2*sij, cii[fb] - cii[tb]]), axis=0)]

        gg, bb, bsh, T, tm2 = g.g, g.b, g.b_fr, g.tap, g.tm2
        p_fr = cp.multiply(gg/tm2, cii[fb]) - cp.multiply(gg/T, cij) + cp.multiply(bb/T, sij)
        p_to = cp.multiply(gg, cii[tb]) - cp.multiply(gg/T, cij) - cp.multiply(bb/T, sij)
        q_fr = -cp.multiply((bb+bsh)/tm2, cii[fb]) + cp.multiply(gg/T, sij) + cp.multiply(bb/T, cij)
        q_to = -cp.multiply(bb+bsh, cii[tb]) - cp.multiply(gg/T, sij) + cp.multiply(bb/T, cij)
        Ain, Aout, Ag = np.zeros((n, L)), np.zeros((n, L)), np.zeros((n, G))
        for k in range(L):
            Ain[fb[k], k] = 1.0; Aout[tb[k], k] = 1.0
        for gi, b_ in enumerate(g.gen_bus):
            Ag[int(b_), gi] = 1.0
        C += [Ag@pg - self.p_pd - cp.multiply(g.gs, cii) == Ain@p_fr + Aout@p_to,
              Ag@qg - self.p_qd + cp.multiply(g.bs, cii) == Ain@q_fr + Aout@q_to,
              pg <= cp.multiply(g.pmax, u), pg >= cp.multiply(g.pmin, u),
              qg <= cp.multiply(g.qmax, u), qg >= cp.multiply(g.qmin, u)]
        idx = np.where(np.isfinite(g.rate))[0]
        if len(idx):
            C += [cp.SOC(g.rate[idx], cp.vstack([p_fr[idx], q_fr[idx]]), axis=0),
                  cp.SOC(g.rate[idx], cp.vstack([p_to[idx], q_to[idx]]), axis=0)]

        # reserve, as a parameter of the demand so it re-solves with it
        C += [g.pmax @ u >= 1.10 * cp.sum(self.p_pd)]
        # learned cuts, priced rather than hard: a bad cut costs, never kills
        C += [self.p_A @ cp.hstack([pg, u]) <= self.p_b + sc, sc <= self.p_cutcap]

        self.cost = cp.sum(cp.multiply(g.c2, cp.square(pg))
                           + cp.multiply(g.c1, pg) + cp.multiply(nl, u))
        self.prob = cp.Problem(
            cp.Minimize(self.cost + self.p_rho*cp.sum(xi) + cut_price*cp.sum(sc)), C)

# %% one solve

    def solve(self, pd_, qd_, Vr0, Vi0, rho=1e6, u_lo=None, u_hi=None,
              A=None, b=None, cut_cap=0.0, soft_fallback=True):
        n, G, K = self.n, self.G, self.n_cuts
        fb, tb = self.g.f_bus.astype(int), self.g.t_bus.astype(int)
        self.p_pd.value, self.p_qd.value = np.asarray(pd_), np.asarray(qd_)
        Vr0, Vi0 = np.asarray(Vr0, float), np.asarray(Vi0, float)
        self.p_Vr0.value, self.p_Vi0.value = Vr0, Vi0
        self.p_V0sq.value = Vr0**2 + Vi0**2
        self.p_VVc.value = Vr0[fb]*Vr0[tb] + Vi0[fb]*Vi0[tb]
        self.p_VVs.value = Vr0[fb]*Vi0[tb] - Vr0[tb]*Vi0[fb]
        self.p_rho.value = float(rho)
        self.p_ulo.value = np.zeros(G) if u_lo is None else np.asarray(u_lo, float)
        self.p_uhi.value = np.ones(G) if u_hi is None else np.asarray(u_hi, float)
        if A is None:
            self.p_A.value = np.zeros((K, 2*G)); self.p_b.value = np.ones(K)
            self.p_cutcap.value = 1e6          # no cut: cap is irrelevant
        else:
            self.p_A.value = np.asarray(A, float); self.p_b.value = np.asarray(b, float)
            self.p_cutcap.value = float(cut_cap)
        try:
            self.prob.solve(solver=cp.CLARABEL)
        except Exception:
            self.v["u"].value = None
        if self.v["u"].value is None and A is not None and soft_fallback:
            # A hard cut can make the instance infeasible. Rather than lose the
            # instance -- a constant failure penalty carries no gradient -- fall
            # back to a softly priced cut so the step still costs rather than
            # kills.
            self.p_cutcap.value = 1e6
            try:
                self.prob.solve(solver=cp.CLARABEL)
            except Exception:
                return None
        if self.v["u"].value is None:
            return None
        return dict(u=self.v["u"].value.copy(), pg=self.v["pg"].value.copy(),
                    vr=self.v["vr"].value.copy(), vi=self.v["vi"].value.copy(),
                    cost=float(self.cost.value), slack=float(self.v["xi"].value.sum()),
                    cut_slack=float(self.v["sc"].value.sum()), status=self.prob.status)

# %% price a commitment: QCAC's own re-linearisation loop with u FIXED

    def price(self, pd_, qd_, u_fixed, rho0=1e3, mu=4.0, rho_max=1e10,
              eps=1e-6, max_iter=25):
        """Cost of a given commitment, re-linearising until the slack vanishes.

        This is how a commitment gets a physically meaningful price without the
        exact nonconvex solve. At convergence the slack is ~0, so the linearised
        products equal the true ones at that V and the point satisfies AC.
        """
        z = np.asarray(u_fixed, float)
        Vr, Vi = np.ones(self.n), np.zeros(self.n)
        rho, last = rho0, None
        for it in range(max_iter):
            r = self.solve(pd_, qd_, Vr, Vi, rho=rho, u_lo=z, u_hi=z)
            if r is None:
                # A re-linearisation step failed. Return the last good iterate,
                # WITH iters set -- callers index it, and omitting it turned a
                # rare solver hiccup into a crash that killed two 60-epoch runs.
                return dict(**last, iters=it) if last is not None else None
            last, Vr, Vi = r, r["vr"], r["vi"]
            if r["slack"] <= eps:
                return dict(**r, iters=it+1)
            rho = min(rho*mu, rho_max)
        return dict(**last, iters=max_iter) if last else None

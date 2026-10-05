"""The FULLY differentiable deployment objective.

    (pd,qd) -> net -> V, (A,b)
                 |
                 +--[ LAYER 1 : SOCP relaxation with the learned cut ]--> u_frac
                 |
                 +--[ soft round : sigmoid((u_frac - thr)/tau)       ]--> z
                 |
                 +--[ LAYER 2 : SOCP priced at u = z                 ]--> AC cost
                 |
                 +--  L = cost/scale + mu * relu(need - pmax'z)^2

Every stage above is differentiable, so dL/dtheta is exact -- no smoothing.

WHAT REPLACED WHAT, relative to `pipeline.deploy`:

  deploy                        here
  --------------------------    ------------------------------------------
  z = (uf > 0.5)                z = zc + (1-2 zc) sigmoid((uf - thr)/tau)
  reserve top-up LOOP           thr is LOWERED until pmax'z meets reserve,
                                which turns units on in order of uf exactly as
                                the loop does, and is differentiated through by
                                the implicit function theorem
  price() re-linearisation      ONE solve at the predicted V -- which is the
    loop (up to 25 solves)      method's own claim: predicting V removes the loop
  upward / downward repair      dropped from TRAINING, kept at inference

The repair loops are a safety net that can only lower cost or restore
feasibility, so training without them optimises a slight UPPER bound on what
deployment achieves. That is the one approximation in the chain, and it is an
approximation in the safe direction.

Why `deploy`'s version cannot be trained on directly: the rounding and the
top-up loop make the cost a STAIRCASE in u_frac -- measured flat to the cent
over a perturbation of 0.1 in u, then jumping. Its true gradient is exactly
zero almost everywhere, which is why the previous trainer had to smooth.
"""
from __future__ import annotations

import numpy as np

from conelayer import ConeLayer


def _sig(uf, thr, tau):
    return 1.0/(1.0 + np.exp(-np.clip((uf - thr)/tau, -60, 60)))


def soft_round(uf, thr, tau, zc=0.02):
    """Squashed into [zc, 1-zc] AFFINELY rather than clipped.

    np.clip would zero the derivative for every entry it touches, and at
    tau=0.08 almost every unit is saturated -- the gradient would vanish for
    all but the handful near the threshold. The affine squash keeps z strictly
    interior (layer 2 needs that) while preserving the derivative everywhere.
    """
    return zc + (1.0 - 2.0*zc)*_sig(uf, thr, tau)


def dsoft_round(uf, thr, tau, zc=0.02):
    s = _sig(uf, thr, tau)
    return (1.0 - 2.0*zc)*s*(1.0 - s)/tau


def reserve_threshold(uf, pmax, target, tau, thr0=0.5, zc=0.02):
    """Largest thr <= thr0 with pmax @ soft_round(uf,thr) >= target.

    pmax @ z is strictly decreasing in thr, so bisection is exact. Returns
    thr0 unchanged when the default threshold already meets reserve -- the
    deployment pipeline only ever ADDS units, never removes them, at this step.
    """
    cap = lambda t: float(pmax @ soft_round(uf, t, tau, zc))
    if cap(thr0) >= target:
        return thr0, False
    lo, hi = thr0 - 1.0, thr0
    for _ in range(80):
        if cap(lo) >= target:
            break
        lo -= 1.0
    else:
        return lo, True
    for _ in range(80):
        mid = 0.5*(lo + hi)
        if cap(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo, True


class DiffDeploy:
    """Two chained cone layers over one compiled Socp."""

    def __init__(self, S, g, nl, reserve=0.10, tau=0.08, thr=0.5, mu=10.0,
                 z_clip=0.02, rho=1e6, reg=1e-8, reg2=1e-10, margin=0.02,
                 margins=(0.02, 0.08, 0.16, 0.28, 0.45, 0.70)):
        self.S, self.g, self.nl = S, g, nl
        self.reserve, self.tau, self.thr, self.mu = reserve, tau, thr, mu
        self.z_clip, self.rho, self.margin = z_clip, rho, margin
        self.margins = tuple(margins)
        # The two layers need DIFFERENT eps -- validated separately, each
        # against its own loss direction. See the note in ConeLayer.__init__.
        #
        # V is a differentiable input of BOTH layers: layer 1 linearises at it,
        # and layer 2 prices at it. Its five Socp parameters are carried in both
        # and folded back to (Vr, Vi) by v_chain.
        self.fb = g.f_bus.astype(int); self.tb = g.t_bus.astype(int)
        VP = [S.p_Vr0, S.p_Vi0, S.p_V0sq, S.p_VVc, S.p_VVs]
        self.L1 = ConeLayer(S.prob, [S.p_A, S.p_b] + VP,
                            [S.v["pg"], S.v["u"]], reg=reg)
        self.L2 = ConeLayer(S.prob, [S.p_ulo, S.p_uhi] + VP,
                            [S.v["pg"], S.v["u"]], reg=reg2)
        self._built = False

    def _build(self, pd_, qd_, Vr, Vi, A, b):
        if self._built:
            return
        VP = v_params(Vr, Vi, self.fb, self.tb)
        self.S.solve(pd_, qd_, Vr, Vi, rho=self.rho, A=A, b=b, cut_cap=0.0)
        self.L1.build_structure([np.asarray(A, float), np.asarray(b, float)] + VP)
        z = np.full(self.g.n_gen, 0.5)
        self.S.solve(pd_, qd_, Vr, Vi, rho=self.rho, u_lo=z, u_hi=z)
        self.L2.build_structure([z, z] + VP)
        self.sl1, self.sl2 = self.L1.var_slices(), self.L2.var_slices()
        self._built = True

    def __call__(self, pd_, qd_, Vr, Vi, A, b, scale, want_grad=True,
                 thr=None):
        """Return (loss, dL/dA, dL/db, info); info carries dL/dVr and dL/dVi.

        V flows through BOTH layers -- layer 1 linearises at it and layer 2
        prices at it -- so its gradient is the SUM of the two paths.
        """
        S, g = self.S, self.g
        self._build(pd_, qd_, Vr, Vi, A, b)
        pid, uid = S.v["pg"].id, S.v["u"].id

        # ---- LAYER 1: the relaxation with the learned cut ----
        r1 = S.solve(pd_, qd_, Vr, Vi, rho=self.rho, A=A, b=b, cut_cap=0.0)
        if r1 is None:
            return None, None, None, dict(fail="layer1")
        uf = r1["u"]
        VP = v_params(Vr, Vi, self.fb, self.tb)
        x1, st1 = self.L1.forward([np.asarray(A, float), np.asarray(b, float)] + VP,
                                  tol=1e-10) if want_grad else (None, None)
        if want_grad and x1 is None:
            return None, None, None, dict(fail="layer1_canon")

        # ---- soft rounding, with the threshold lowered to meet reserve ----
        #
        # Meeting the RESERVE is necessary but not sufficient: the priced AC
        # problem can still be infeasible with a commitment that clears reserve
        # by a thin margin (reactive capability scales with z too). The real
        # pipeline handles this with its upward REPAIR loop -- turn more units
        # on and retry. The differentiable analogue is to escalate the reserve
        # margin, which lowers thr and turns more units on, and retry. Without
        # it 5 of 6 case300 instances failed at layer 2 and the arm trained on
        # one instance.
        need = (1.0 + self.reserve)*float(pd_.sum())
        r2 = None
        for mg in self.margins:
            # thr0 may be a per-generator VECTOR. reserve_threshold needs no
            # change for that: lo = thr0 - 1.0 keeps the bisection offset
            # UNIFORM across components, so it bisects a scalar shift of the
            # whole vector -- exactly the reserve top-up's behaviour.
            thr0 = self.thr if thr is None else np.asarray(thr, float)
            thr_e, binding = reserve_threshold(uf, g.pmax, need*(1.0 + mg),
                                               self.tau, thr0, self.z_clip)
            z = soft_round(uf, thr_e, self.tau, self.z_clip)
            r2 = S.solve(pd_, qd_, Vr, Vi, rho=self.rho, u_lo=z, u_hi=z)
            if r2 is not None:
                break
        if r2 is None:
            return None, None, None, dict(fail="layer2")
        pg = r2["pg"]
        cost = float((g.c2*pg**2 + g.c1*pg + self.nl*z).sum())

        short = max(0.0, need - float(g.pmax @ z))
        loss = cost/scale + self.mu*(short/scale)**2
        info = dict(cost=cost, short=short, thr=thr_e, binding=binding, margin=mg,
                    nfrac=int(((uf > 1e-6) & (uf < 1-1e-6)).sum()))
        if not want_grad:
            return loss, None, None, info

        # ---- backward: dL/dz from layer 2 + the reserve penalty ----
        x2, st2 = self.L2.forward([z, z] + VP, tol=1e-10)
        if x2 is None:
            return loss, None, None, dict(**info, fail="layer2_canon")
        dcost = np.zeros(x2.size)
        dcost[self.sl2[pid]] = (2*g.c2*pg + g.c1)/scale
        dcost[self.sl2[uid]] = self.nl/scale            # nl'z enters through u = z
        try:
            g2 = self.L2.vjp(st2, dcost)
        except RuntimeError:
            return loss, None, None, dict(**info, fail="lu_layer2")
        glo, ghi = g2[0], g2[1]
        dz = glo + ghi                                   # z pins BOTH bounds
        dVr2, dVi2 = v_chain(g2[2:], Vr, Vi, self.fb, self.tb)   # V via layer 2
        if short > 0:
            dz += -2.0*self.mu*short/scale**2*g.pmax

        # ---- through the soft rounding, then into layer 1 ----
        # When the reserve bisection moved the threshold, thr is a FUNCTION of
        # uf, so dz/duf is the diagonal PLUS a rank-one term from dthr/duf:
        #   F(thr,uf) = pmax'z(thr,uf) - target = 0
        #   dthr/duf_j = pmax_j d_j / sum_k pmax_k d_k,   d = dz/dsigmoid-arg
        # Dropping that term is the obvious chain-rule slip here, so it is
        # written out explicitly.
        d = dsoft_round(uf, thr_e, self.tau, self.z_clip)
        duf = dz*d
        if binding:
            den = float(g.pmax @ d)
            if abs(den) > 1e-300:
                duf -= (float(dz @ d)/den)*(g.pmax*d)
        # dL/dthr = -dL/duf, in BOTH the binding and non-binding cases:
        #   z depends on (uf - thr), so the diagonal terms are exact negatives,
        #   and the rank-one reserve term flips sign with them.
        info["dthr"] = -duf
        dL1 = np.zeros(x1.size)
        dL1[self.sl1[uid]] = duf
        try:
            g1 = self.L1.vjp(st1, dL1)
        except RuntimeError:
            return loss, None, None, dict(**info, fail="lu_layer1")
        gA, gb = g1[0], g1[1]
        dVr1, dVi1 = v_chain(g1[2:], Vr, Vi, self.fb, self.tb)   # V via layer 1
        info["dVr"] = dVr1 + dVr2
        info["dVi"] = dVi1 + dVi2
        return loss, gA, gb, info


# %% chain rule for the LINEARISATION POINT
#
# Socp takes V through FIVE parameters, three of which are nonlinear in V:
#     p_Vr0 = Vr                 p_Vi0 = Vi
#     p_V0sq = Vr^2 + Vi^2
#     p_VVc  = Vr[f]Vr[t] + Vi[f]Vi[t]
#     p_VVs  = Vr[f]Vi[t] - Vr[t]Vi[f]
# A cone layer differentiates w.r.t. each PARAMETER; turning that into dL/dVr,
# dL/dVi needs this chain rule. Omitting the three derived terms would leave a
# gradient that looks plausible and is wrong, so it is gated against finite
# differences like everything else.
V_PARAM_NAMES = ("p_Vr0", "p_Vi0", "p_V0sq", "p_VVc", "p_VVs")


def v_params(Vr, Vi, fb, tb):
    Vr = np.asarray(Vr, float); Vi = np.asarray(Vi, float)
    return [Vr, Vi, Vr**2 + Vi**2,
            Vr[fb]*Vr[tb] + Vi[fb]*Vi[tb],
            Vr[fb]*Vi[tb] - Vr[tb]*Vi[fb]]


def v_chain(grads, Vr, Vi, fb, tb):
    """(dL/dp_Vr0, dL/dp_Vi0, dL/dp_V0sq, dL/dp_VVc, dL/dp_VVs) -> (dL/dVr, dL/dVi)."""
    gVr0, gVi0, gV0sq, gVVc, gVVs = grads
    Vr = np.asarray(Vr, float); Vi = np.asarray(Vi, float)
    dVr = np.array(gVr0, float) + 2.0*np.asarray(gV0sq, float)*Vr
    dVi = np.array(gVi0, float) + 2.0*np.asarray(gV0sq, float)*Vi
    # p_VVc
    np.add.at(dVr, fb, gVVc*Vr[tb]); np.add.at(dVr, tb, gVVc*Vr[fb])
    np.add.at(dVi, fb, gVVc*Vi[tb]); np.add.at(dVi, tb, gVVc*Vi[fb])
    # p_VVs
    np.add.at(dVr, fb,  gVVs*Vi[tb]); np.add.at(dVr, tb, -gVVs*Vi[fb])
    np.add.at(dVi, tb,  gVVs*Vr[fb]); np.add.at(dVi, fb, -gVVs*Vr[tb])
    return dVr, dVi

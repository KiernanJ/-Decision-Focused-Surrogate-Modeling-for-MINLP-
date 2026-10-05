"""Fully differentiable deployment for the hybrid-vehicle convex MINLP.

    D -> net -> (A, b)
           |
           +--[ LAYER 1 : convex relaxation with the learned cut ]--> z_frac
           |
           +--[ soft round over S level boundaries ]---------------> z
           |
           +--[ LAYER 2 : priced at z ]----------------------------> cost
           |
           +--  L = cost / scale

Exact gradients through both solves via the same ConeLayer used for AC-UC.
No V: the problem is convex, so there is no linearisation point.

SOFT ROUNDING for an INTEGER variable. The binary case used one sigmoid; with
z in {0..S} the hard rule is a sum of S step functions,
    z = sum_{k=1..S} 1{ zf > (k-1) + thr },
so the smooth version is a sum of S sigmoids. It reduces to the binary case at
S=1, and it keeps the identity dz/dthr = -dz/dzf that made the AC-UC threshold
gradient cheap.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ))
from conelayer import ConeLayer


def _sig(x):
    return 1.0/(1.0 + np.exp(-np.clip(x, -60, 60)))


def soft_round_int(zf, thr, tau, S, zc=0.02):
    """Sum of S sigmoids, squashed AFFINELY into [zc, S-zc].

    The affine squash (not np.clip) matters: clip zeroes the derivative wherever
    it bites, and at tau=0.08 almost every step is saturated, so the gradient
    would vanish for all but the few steps near a boundary.
    """
    s = np.stack([_sig((zf - k - np.asarray(thr, float))/tau) for k in range(S)])
    z = s.sum(0)
    return zc + (1.0 - 2.0*zc/max(S, 1e-9))*z, s


def dsoft_round_int(s, tau, S, zc=0.02):
    """d z / d(zf) ; d z / d(thr) is the NEGATIVE of this."""
    return (1.0 - 2.0*zc/max(S, 1e-9))*(s*(1.0 - s)).sum(0)/tau


class DiffHV:
    """Two chained cone layers over one compiled relaxation."""

    def __init__(self, R, veh, tau=0.08, thr=0.5, reg=1e-10, reg2=1e-10,
                 zc=0.02, drops=(0.0, 0.10, 0.20, 0.32, 0.45), w_int=0.0):
        # INTEGRALITY PENALTY, as in run_hybrid_vehicle.py's DF surrogate:
        #     loss = obj + 20 * sum( sin(pi z)^2 ) + 1e4 * sum(s)
        # sin^2(pi z) is zero at every integer and maximal at half-integers, so
        # it pushes the RELAXATION toward an integral z. That is what the cut is
        # really for here: not to lower the relaxed cost but to make rounding
        # harmless. Our loss was deployed cost ALONE, which reaches integrality
        # only indirectly through two layers and a soft-round -- too weak a
        # signal, which is why 34.8 of 40 steps stayed fractional.
        self.w_int = float(w_int)
        # THRESHOLD ESCALATION. A rounded z is often infeasible -- that is why
        # `deploy` has a repair loop at all. The differentiable analogue is to
        # LOWER the threshold, which rounds more steps UP and buys engine power,
        # and retry. Without it layer 2 fails on most instances and the arm
        # trains on a biased subset; the AC-UC version needed the same thing
        # (there it escalated the reserve margin).
        self.drops = tuple(drops)
        self.R, self.veh, self.tau, self.thr, self.zc = R, veh, tau, thr, zc
        self.S, self.T = veh["S_MODES"], veh["T"]
        self.L1 = ConeLayer(R.prob, [R.p_A, R.p_b],
                            [R.v["Peng"], R.v["z"]], reg=reg)
        self.L2 = ConeLayer(R.prob, [R.p_zlo, R.p_zhi],
                            [R.v["Peng"], R.v["z"]], reg=reg2)
        self._built = False

    def _build(self, Dp, A, b):
        if self._built:
            return
        self.R.solve(Dp, A=A, b=b, cut_cap=0.0)
        self.L1.build_structure([np.asarray(A, float), np.asarray(b, float)])
        z0 = np.full(self.T, 0.5*self.S)
        self.R.solve(Dp, z_lo=z0, z_hi=z0)
        self.L2.build_structure([z0, z0])
        self.sl1, self.sl2 = self.L1.var_slices(), self.L2.var_slices()
        self._built = True

    def __call__(self, Dp, A, b, scale, thr=None, want_grad=True):
        R, veh = self.R, self.veh
        self._build(Dp, A, b)
        pid, zid = R.v["Peng"].id, R.v["z"].id
        r1 = R.solve(Dp, A=A, b=b, cut_cap=0.0)
        if r1 is None:
            return None, None, None, dict(fail="layer1")
        zf = r1["z"]
        th0 = self.thr if thr is None else np.asarray(thr, float)
        r2 = None
        for drop in self.drops:
            th = np.clip(th0 - drop, 0.02, 0.98)
            z, s = soft_round_int(zf, th, self.tau, self.S, self.zc)
            r2 = R.solve(Dp, z_lo=z, z_hi=z)
            if r2 is not None:
                break
        if r2 is None:
            return None, None, None, dict(fail="layer2")
        pe = r2["Peng"]
        al = np.asarray(veh["alpha"]); be = np.asarray(veh["beta"])
        cost = float((al*pe**2 + be*z).sum() + veh["eta"]*(veh["E_max"]-r2["E"][-1]))
        loss = cost/scale
        # the penalty is on the RELAXATION's z (zf), which is what the cut moves
        if self.w_int > 0.0:
            loss = loss + self.w_int*float(np.sin(np.pi*zf)**2).__float__() \
                if np.isscalar(zf) else loss + self.w_int*float((np.sin(np.pi*zf)**2).sum())/scale
        info = dict(cost=cost, drop=float(drop),
                    int_pen=float((np.sin(np.pi*zf)**2).sum()),
            nfrac=int((np.abs(zf-np.round(zf)) > 1e-4).sum()))
        if not want_grad:
            return loss, None, None, info
        # The layer-2 solve above PINNED p_zlo/p_zhi to z. ConeLayer.forward
        # only rewrites the parameters it owns, so layer 1 would otherwise be
        # canonicalised against those pinned bounds -- a different (usually
        # infeasible) problem. Restore the free range first.
        R.solve(Dp, A=A, b=b, cut_cap=0.0)
        x1, st1 = self.L1.forward([np.asarray(A, float), np.asarray(b, float)],
                                  tol=1e-10)
        if x1 is None:
            return loss, None, None, dict(**info, fail="layer1_canon")
        R.solve(Dp, z_lo=z, z_hi=z)          # likewise, restore layer 2's own
        x2, st2 = self.L2.forward([z, z], tol=1e-10)
        if x2 is None:
            return loss, None, None, dict(**info, fail="layer2_canon")
        dcost = np.zeros(x2.size)
        dcost[self.sl2[pid]] = (2*al*pe)/scale
        dcost[self.sl2[zid]] = be/scale          # be'z enters through z
        try:
            g2 = self.L2.vjp(st2, dcost)
        except RuntimeError:
            return loss, None, None, dict(**info, fail="lu2")
        dz = g2[0] + g2[1]                        # z pins BOTH bounds
        duf = dz*dsoft_round_int(s, self.tau, self.S, self.zc)
        info["dthr"] = -duf                       # exact negative, as in AC-UC
        if self.w_int > 0.0:
            duf = duf + self.w_int*(2*np.pi*np.sin(np.pi*zf)*np.cos(np.pi*zf))/scale
        dL1 = np.zeros(x1.size)
        dL1[self.sl1[zid]] = duf
        try:
            g1 = self.L1.vjp(st1, dL1)
        except RuntimeError:
            return loss, None, None, dict(**info, fail="lu1")
        return loss, g1[0], g1[1], info

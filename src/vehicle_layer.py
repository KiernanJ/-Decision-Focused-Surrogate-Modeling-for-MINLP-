"""Exact differentiable layer for the HYBRID VEHICLE (2026-10-10), both versions:
  kind="milp"  : Dixit et al.'s vehicle (src/dixit_vehicle.py), LP relaxation (z continuous in [0, 3]);
  kind="minlp" : our convex MIQCQP vehicle (vehicle/01_make_pool.py), QCQP relaxation.
Per-step INTEGER cut rows (soft, violation priced) with learned right-hand sides:
  rows 0..T-1 :  -z_t <= -l_t      rows T..2T-1 :  z_t <= h_t        (b = [-l ; h] is the layer parameter)
inactive rows use b_lower = +1, b_upper = SW + 1. A small strongly convex term gamma*sum (z - SW/2)^2 makes the relaxed
solution unique and smooth in b (the AC case showed the unsmoothed derivative depends on the KKT regularisation).
Loss helper: tracking ||z_relax - z_own||^2 / T and its exact VJP w.r.t. (l, h)."""
import numpy as np
import cvxpy as cp
from conelayer import ConeLayer


class VehLayer:
    def __init__(self, kind, veh, gamma=1.0, cut_price=1e3, reg=1e-10, k_gen=0):
        self.kind, self.k_gen = kind, k_gen
        if kind == "milp":
            from dixit_vehicle import T, SW, TAU, E0, EMAX, PMAX
            self.T, self.SW = T, SW
            E = cp.Variable(T+1); Pe = cp.Variable(T); Pb = cp.Variable(T); z = cp.Variable(T); s = cp.Variable(2*T, nonneg=True)
            self.D = cp.Parameter(T); self.b = cp.Parameter(2*T)
            C = [E[0] == E0, E >= 0, E <= EMAX, Pe >= 0, Pe <= PMAX, z >= 0, z <= SW,
                 E[1:] == E[:-1] - TAU*Pb, Pb + Pe >= self.D, Pe <= z*PMAX/SW]
            cost = (-veh.cT*(EMAX - E[T]) + veh.cE @ E[:T] + veh.cP @ Pe + veh.cZ @ z)
        else:
            T, S = veh["T"], veh["S_MODES"]; self.T, self.SW = T, S
            AL = np.array(veh["alpha"]); BE = np.array(veh["beta"])
            E = cp.Variable(T+1); Pe = cp.Variable(T); Pb = cp.Variable(T); z = cp.Variable(T); s = cp.Variable(2*T, nonneg=True)
            self.D = cp.Parameter(T); self.b = cp.Parameter(2*T)
            C = [E[0] == veh["E_init"], E >= 0, E <= veh["E_max"], Pe >= 0, Pe <= veh["P_max"],
                 Pb >= -veh["P_batt_max"], Pb <= veh["P_batt_max"], z >= 0, z <= S,
                 E[1:] - E[:-1] + veh["tau"]*Pb + veh["gamma"]*cp.square(Pb) <= 0,
                 Pe <= z*(veh["P_max"]/S), Pb + Pe >= self.D]
            cost = cp.sum(cp.multiply(AL, cp.square(Pe))) + BE @ z + veh["eta"]*(veh["E_max"] - E[T])
        A = np.vstack([-np.eye(self.T), np.eye(self.T)])
        C += [A @ z <= self.b + s]
        pen = cut_price*cp.sum(s)
        if k_gen:                                   # GENERAL rows  Ag [Pe; z] <= bg  (soft, same price)
            self.Ag = cp.Parameter((k_gen, 2*self.T)); self.bg = cp.Parameter(k_gen); sg_ = cp.Variable(k_gen, nonneg=True)
            C += [self.Ag @ cp.hstack([Pe, z]) <= self.bg + sg_]; pen = pen + cut_price*cp.sum(sg_)
        self.cost = cost
        self.prob = cp.Problem(cp.Minimize(cost + pen + gamma*cp.sum_squares(z - self.SW/2)), C)
        self.z, self.Pe = z, Pe
        self.L = ConeLayer(self.prob, [self.b] + ([self.Ag, self.bg] if k_gen else []), [self.Pe, self.z], reg=reg); self.built = False

    def rhs(self, lo, hi):
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        return np.r_[np.where(lo > 0, -lo, 1.0), np.where(hi < self.SW, hi, self.SW + 1.0)]

    def _gen(self, Ag=None, bg=None):
        if not self.k_gen: return []
        Ag = np.zeros((self.k_gen, 2*self.T)) if Ag is None else np.asarray(Ag, float); bg = np.ones(self.k_gen) if bg is None else np.asarray(bg, float)
        self.Ag.value, self.bg.value = Ag, bg; return [Ag, bg]

    def anchor(self, D, lo, hi):
        """relaxation with the integer cut rows only (general rows disabled) -> [Pe; z], the general rows' anchor."""
        self.D.value = np.asarray(D, float); self.b.value = self.rhs(lo, hi); self._gen()
        try: self.prob.solve(solver=cp.CLARABEL, warm_start=False)
        except Exception: return None
        return None if self.z.value is None else np.r_[self.Pe.value, self.z.value]

    def track_gen(self, D, lo, hi, Ag, bg, z_own, want_grad=True):
        """tracking loss with integer cuts (frozen) + general rows; returns (loss, dL/dAg, dL/dbg)."""
        self.D.value = np.asarray(D, float); b = self.rhs(lo, hi); self.b.value = b; vals = [b] + self._gen(Ag, bg)
        try: self.prob.solve(solver=cp.CLARABEL, warm_start=False)
        except Exception: return None
        if self.z.value is None: return None
        if not want_grad: return (float(((self.z.value - z_own)**2).sum()/self.T),)
        if not self.built: self.L.build_structure(vals); self.sl = self.L.var_slices(); self.built = True
        x, st = self.L.forward(vals, tol=1e-10)
        if x is None: return None
        zz = x[self.sl[self.z.id]]; dx = np.zeros(x.size); dx[self.sl[self.z.id]] = 2.0*(zz - z_own)/self.T
        try: gr = self.L.vjp(st, dx)
        except RuntimeError: return None
        return (float(((zz - z_own)**2).sum()/self.T), np.asarray(gr[1]).reshape(self.Ag.shape), np.asarray(gr[2]))

    def track(self, D, lo, hi, z_own, want_grad=True):
        self.D.value = np.asarray(D, float); b = self.rhs(lo, hi); self.b.value = b; self._gen()
        try:
            self.prob.solve(solver=cp.CLARABEL, warm_start=False)
        except Exception:
            return None
        if self.z.value is None:
            return None
        loss = float(((self.z.value - z_own)**2).sum()/self.T)
        if not want_grad:
            return (loss,)
        vals = [b] + self._gen()
        if not self.built:
            self.L.build_structure(vals); self.sl = self.L.var_slices(); self.built = True
        x, st = self.L.forward(vals, tol=1e-10)
        if x is None:
            return None
        zz = x[self.sl[self.z.id]]; dx = np.zeros(x.size); dx[self.sl[self.z.id]] = 2.0*(zz - z_own)/self.T
        try:
            gb = self.L.vjp(st, dx)[0]
        except RuntimeError:
            return None
        T = self.T
        return (float(((zz - z_own)**2).sum()/T), np.where(np.asarray(lo) > 0, -gb[:T], 0.0), np.where(np.asarray(hi) < self.SW, gb[T:], 0.0))

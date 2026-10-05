"""Shared pieces for the hybrid-vehicle case study: pool loading, the exact
model, the convex relaxation with a learned cut, and deployment.

CONVEX MINLP -- every nonlinearity (alpha*P_eng^2 in the objective,
+gamma*P_batt^2 on the <= side of the dynamics) is convex, so the relaxation is
convex and there is NO linearisation point to learn, unlike QCAC. The network
predicts the CUT only.
"""
from __future__ import annotations
import glob, json
import numpy as np


def load_pool(pat="results/pool/pool_*.json"):
    """-> (veh, D, REF) with REF[i] = dict(cost, z, ...) or None."""
    veh, Ds, refs = None, [], {}
    for f in sorted(glob.glob(pat), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0])):
        d = json.load(open(f))
        veh = veh or d["veh"]
        i0 = d["i0"]
        for k, v in d["ref"].items():
            refs[int(k)] = v
        Ds.append((i0, np.array(d["D"])))
    n = max(refs) + 1
    T = veh["T"]
    D = np.zeros((n, T))
    for i0, arr in Ds:
        D[i0:i0+len(arr)] = arr
    return veh, D, [refs.get(i) for i in range(n)]


def splits(n, n_test=48, n_val=24, n_train=144, seed=0):
    """Same convention as the AC-UC study: test first, validation carved from
    train, and the test split is never used for any selection."""
    perm = np.random.default_rng(seed).permutation(n)
    te = [int(i) for i in perm[:n_test]]
    rest = [int(i) for i in perm[n_test:n_test+n_train+n_val]]
    return te, rest[:n_val], rest[n_val:]


# ---------------------------------------------------------------------------
# The convex relaxation with a learned cut, as a cvxpy problem.
# ---------------------------------------------------------------------------
class HVRelax:
    """Convex relaxation of the hybrid-vehicle MINLP with K learned cut rows.

    z is relaxed to [0, S]. Everything else is the exact model, so the only
    approximation is integrality -- which the cut is there to tighten.

    The cut is PRICED, not hard: `A @ [Peng; z] <= b + sc` with `sc` penalised
    and capped. A bad cut then costs money and can never make the problem
    infeasible, which is what let the AC-UC version train without the layer
    failing. Same choice here.

    DPP-compliant: every parameter enters affinely, so the problem compiles once
    and the cone layer can extract per-parameter directions a single time.
    """

    def __init__(self, veh, n_cuts=4, cut_price=1e3):
        import cvxpy as cp
        T, S = veh["T"], veh["S_MODES"]
        self.T, self.S, self.K = T, S, n_cuts
        al = np.asarray(veh["alpha"]); be = np.asarray(veh["beta"])
        E = cp.Variable(T+1); Pe = cp.Variable(T); Pb = cp.Variable(T)
        z = cp.Variable(T); sc = cp.Variable(n_cuts, nonneg=True)
        self.v = dict(E=E, Peng=Pe, Pbatt=Pb, z=z, sc=sc)
        self.p_D = cp.Parameter(T)
        # width 4T+1 = [E (T+1) | P_eng (T) | P_batt (T) | z (T)]. The earlier
        # version spanned only [P_eng | z] and could not express ANY
        # constraint on the energy trajectory -- which is where this
        # problem's whole-horizon coupling lives. It made the cut harmful.
        self.p_A = cp.Parameter((n_cuts, 4*T+1))
        self.p_b = cp.Parameter(n_cuts)
        self.p_zlo = cp.Parameter(T); self.p_zhi = cp.Parameter(T)
        self.p_cutcap = cp.Parameter(nonneg=True)
        C = [E[0] == veh["E_init"], E >= 0, E <= veh["E_max"],
             Pe >= 0, Pe <= veh["P_max"],
             Pb >= -veh["P_batt_max"], Pb <= veh["P_batt_max"],
             z >= self.p_zlo, z <= self.p_zhi,
             # battery dynamics; cp.square keeps it a convex (SOC) constraint
             E[1:] - E[:-1] + veh["tau"]*Pb + veh["gamma"]*cp.square(Pb) <= 0,
             Pe <= (veh["P_max"]/S)*z,
             Pb + Pe >= self.p_D,
             self.p_A @ cp.hstack([E, Pe, Pb, z]) <= self.p_b + sc,
             sc <= self.p_cutcap]
        self.cost = cp.sum(cp.multiply(al, cp.square(Pe)) + cp.multiply(be, z)) \
            + veh["eta"]*(veh["E_max"] - E[T])
        self.prob = cp.Problem(cp.Minimize(self.cost + cut_price*cp.sum(sc)), C)
        assert self.prob.is_dpp(), "relaxation must be DPP for the cone layer"
        self.veh = veh

    def solve(self, Dp, A=None, b=None, z_lo=None, z_hi=None, cut_cap=0.0,
              solver="CLARABEL", **kw):
        T, K, S = self.T, self.K, self.S
        self.p_D.value = np.asarray(Dp, float)
        self.p_A.value = np.zeros((K, 4*T+1)) if A is None else np.asarray(A, float)
        self.p_b.value = np.zeros(K) if b is None else np.asarray(b, float)
        self.p_zlo.value = np.zeros(T) if z_lo is None else np.asarray(z_lo, float)
        self.p_zhi.value = np.full(T, S) if z_hi is None else np.asarray(z_hi, float)
        self.p_cutcap.value = float(cut_cap)
        try:
            self.prob.solve(solver=solver, **kw)
        except Exception:
            return None
        if self.v["z"].value is None:
            return None
        return dict(z=np.asarray(self.v["z"].value).ravel(),
                    Peng=np.asarray(self.v["Peng"].value).ravel(),
                    Pbatt=np.asarray(self.v["Pbatt"].value).ravel(),
                    E=np.asarray(self.v["E"].value).ravel(),
                    cost=float(self.cost.value))


# ---------------------------------------------------------------------------
# Rounding + feasibility restoration.
# ---------------------------------------------------------------------------
def round_z(zf, thr=0.5, S=3):
    """Integer generalisation of the binary threshold.

        z = floor(zf) + 1{ frac(zf) > thr }

    thr=0.5 is round-to-nearest; thr<0.5 rounds UP more often (more engine),
    thr>0.5 rounds DOWN. `thr` may be a scalar or a per-step vector, which is
    the analogue of the per-generator threshold that was worth 2.6x on case118.
    One boundary per step rather than S separate ones: the fractional part
    carries all the information, so S thresholds would be over-parameterised.
    """
    zf = np.asarray(zf, float)
    fl = np.floor(zf); fr = zf - fl
    return np.clip(fl + (fr > np.asarray(thr, float)).astype(float), 0, S)


def deploy(R, veh, Dp, A=None, b=None, thr=0.5, order=None, ndown=6,
           cut_cap=0.0, want_relax=True, zf=None):
    """One convex solve -> round -> repair UP to feasibility -> repair DOWN.

    Mirrors pipeline.deploy from the AC-UC study, including the two findings
    that mattered most there:
      - the repair ORDER is an argument, not hard-coded (worth 4.4x on case300)
      - the DOWNWARD pass walks that same order REVERSED, so the two mirror each
        other (worth a further 3.0x; mixing them cost 3x)
    Returns (cost, info) or (inf, None).
    """
    S, T = veh["S_MODES"], veh["T"]
    if zf is None:
        if not want_relax:
            return np.inf, None
        r = R.solve(Dp, A=A, b=b, cut_cap=cut_cap)
        if r is None:
            return np.inf, None
        zf = r["z"]
    z = round_z(zf, thr, S)
    # order: which steps to raise first. Default is the relaxation's own
    # confidence -- the steps whose fractional part was closest to rounding up.
    if order is None:
        order = np.argsort(-(zf - np.floor(zf)))
    order = np.asarray(order, int)
    best = None
    for _ in range(T*S + 1):
        c = _price(R, veh, Dp, z)
        if c is not None:
            best = (z.copy(), c[0], c[1]); break
        up = [t for t in order if z[t] < S]
        if not up:
            return np.inf, None
        z[up[0]] += 1.0
    if best is None:
        return np.inf, None
    z, cost, pe = best
    for t in order[::-1][:ndown]:          # mirror the up-order
        if z[t] <= 0:
            continue
        w = z.copy(); w[t] -= 1.0
        c = _price(R, veh, Dp, w)
        if c is not None and c[0] < cost:
            z, cost, pe = w, c[0], c[1]
    return cost, dict(z=z.astype(int), z_frac=zf, Peng=pe)


_PRICE = {}


def _price(R, veh, Dp, z):
    """Exact continuous cost of a GIVEN integer schedule; None if infeasible.
    z is pinned through the relaxation's own z bounds, so one compiled problem
    serves both roles and nothing is rebuilt per call."""
    r = R.solve(Dp, A=None, b=None, z_lo=z, z_hi=z)
    if r is None:
        return None
    return r["cost"], r["Peng"]

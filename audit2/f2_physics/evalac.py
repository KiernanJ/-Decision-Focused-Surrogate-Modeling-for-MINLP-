"""F2 independent AC evaluator: PYPOWER (MATPOWER formulation) AC-OPF at a fixed commitment, and a
raw-voltage feasibility check of a priced point with a complex Ybus built by PYPOWER's makeYbus."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from pypower.api import ppoption, runopf
from pypower.makeYbus import makeYbus
from pypower.ext2int import ext2int


def build_ppc(g, p, q, u, base=100.0):
    n, L, G = g.n_bus, g.n_branch, g.n_gen
    bus = np.zeros((n, 13))
    bus[:, 0] = np.arange(1, n + 1); bus[:, 1] = 1
    for k in range(G):
        if u[k] > 0.5: bus[g.gen_bus[k], 1] = 2
    bus[int(g.ref), 1] = 3
    bus[:, 2] = p * base; bus[:, 3] = q * base
    bus[:, 4] = g.gs * base; bus[:, 5] = g.bs * base
    bus[:, 6] = 1; bus[:, 7] = 1.0; bus[:, 8] = 0.0; bus[:, 9] = 100.0; bus[:, 10] = 1
    bus[:, 11] = g.vmax; bus[:, 12] = g.vmin
    ys = g.g + 1j * g.b; z = 1.0 / ys
    br = np.zeros((L, 13))
    br[:, 0] = g.f_bus + 1; br[:, 1] = g.t_bus + 1; br[:, 2] = z.real; br[:, 3] = z.imag
    br[:, 4] = 2 * g.b_fr
    rate = np.where(np.isfinite(g.rate), g.rate * base, 0.0)
    br[:, 5] = rate; br[:, 6] = rate; br[:, 7] = rate
    br[:, 8] = np.where(np.abs(g.tap - 1.0) < 1e-12, 0.0, g.tap); br[:, 9] = 0.0; br[:, 10] = 1
    br[:, 11] = -360; br[:, 12] = 360
    gen = np.zeros((G, 21))
    gen[:, 0] = g.gen_bus + 1; gen[:, 3] = g.qmax * base; gen[:, 4] = g.qmin * base; gen[:, 5] = 1.0
    gen[:, 6] = base; gen[:, 7] = (np.asarray(u) > 0.5).astype(float); gen[:, 8] = g.pmax * base; gen[:, 9] = g.pmin * base
    gc = np.zeros((G, 7)); gc[:, 0] = 2; gc[:, 3] = 3; gc[:, 4] = g.c2 / base**2; gc[:, 5] = g.c1 / base; gc[:, 6] = 0.0
    return dict(version="2", baseMVA=base, bus=bus, branch=br, gen=gen, gencost=gc)


def opf_cost(g, nl, p, q, u):
    ppc = build_ppc(g, p, q, u)
    r = runopf(ppc, ppoption(VERBOSE=0, OUT_ALL=0, PDIPM_MAX_IT=300))
    pg = r["gen"][:, 1] / 100.0
    return bool(r["success"]), float(r["f"] + nl @ (np.asarray(u) > 0.5)), r


def check_point(g, p, q, u, vr, vi, pg, base=100.0):
    """raw-voltage feasibility of (V, pg, u): P residual, Q feasibility per bus vs committed Q ranges, |V| limits,
    branch |S| at both ends vs rating, all in pu."""
    ppc = build_ppc(g, p, q, u, base)
    bi, bri = ppc["bus"].copy(), ppc["branch"].copy(); bi[:, 0] -= 1; bri[:, 0:2] -= 1   # internal 0-based numbering
    Ybus, Yf, Yt = makeYbus(base, bi, bri)
    V = np.asarray(vr) + 1j * np.asarray(vi)
    S = V * np.conj(Ybus @ V)                     # injections (pu), shunts included
    n, G = g.n_bus, g.n_gen
    Cg = np.zeros((n, G)); Cg[g.gen_bus, np.arange(G)] = 1.0
    uu = (np.asarray(u) > 0.5).astype(float)
    pres = np.abs(Cg @ (pg * uu) - p - S.real).max()
    qreq = S.imag + q                              # Q the generators at each bus must supply
    qlo, qhi = Cg @ (g.qmin * uu), Cg @ (g.qmax * uu)
    qviol = np.maximum(np.maximum(qlo - qreq, qreq - qhi), 0).max()
    vm = np.abs(V); vviol = max(float(np.max(g.vmin - vm)), float(np.max(vm - g.vmax)), 0.0)
    Sf = V[g.f_bus] * np.conj(Yf @ V); St = V[g.t_bus] * np.conj(Yt @ V)
    fin = np.isfinite(g.rate)
    load = float(np.max(np.maximum(np.abs(Sf[fin]), np.abs(St[fin])) / g.rate[fin])) if fin.any() else 0.0
    pgv = max(float(np.max(pg - g.pmax * uu)), float(np.max(g.pmin * uu - pg)), 0.0)
    return dict(p_res=float(pres), q_viol=float(qviol), v_viol=vviol, max_load=load, pg_viol=pgv)

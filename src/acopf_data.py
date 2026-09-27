
"""
Network data for the rung-D QCAC case study, in the form the formulation needs.

Pulled from pandapower's internal MATPOWER-format case (`net._ppc`) rather than
from its own element tables, because the QCAC formulation in Constante-Flores &
Li is written in MATPOWER conventions -- series admittance g+jb, line charging
split g_fr/b_fr and g_to/b_to, transformer tap ratio and phase shift -- and
converting once at the source avoids a second set of sign conventions to get
wrong.

Everything is per-unit on the system base.

The branch model (MATPOWER §3.2), which the QCAC power-flow equations (1d)-(1g)
assume:

    y_series = 1 / (r + jx)          tap = T e^{j.shift}
    p_fr = (g+g_fr)/|T|^2 . c_ii  +  (-g.tr + b.ti)/|T|^2 . c_ij  + ...

`validate()` checks the extracted data by reconstructing bus power injections
from a converged pandapower AC power flow and comparing against what pandapower
itself reports. If the admittances or tap conventions are wrong the residual
blows up, which is the same style of cross-check used on the vehicle MIQP
(exact solver vs brute-force enumeration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    name: str
    n_bus: int
    ref: int                      # slack bus index
    # branches
    f_bus: np.ndarray             # (L,) int
    t_bus: np.ndarray             # (L,) int
    g: np.ndarray                 # series conductance
    b: np.ndarray                 # series susceptance
    g_fr: np.ndarray; b_fr: np.ndarray
    g_to: np.ndarray; b_to: np.ndarray
    tap: np.ndarray                  # tap magnitude (QCAC assumes real T)
    tm2: np.ndarray                  # T^2
    rate: np.ndarray                 # MVA limit, per-unit (inf where unset)
    # buses
    vmin: np.ndarray; vmax: np.ndarray
    gs: np.ndarray; bs: np.ndarray   # shunt
    pd: np.ndarray; qd: np.ndarray   # nominal demand
    # generators
    gen_bus: np.ndarray           # (G,) int
    pmin: np.ndarray; pmax: np.ndarray
    qmin: np.ndarray; qmax: np.ndarray
    c2: np.ndarray; c1: np.ndarray; c0: np.ndarray   # quadratic cost
    # Real per-unit no-load cost, when the case file actually carries one
    # (only case14_uctest.m does). None => qcac.no_load_cost synthesises it
    # from nl_frac, which is what every other case relies on.
    nl_override: object = None
    # True for generators that are NOT a commitment decision. pandapower's
    # ext_grid is a slack BUS, not a dispatchable unit: its capacity arrives as
    # a sentinel and is replaced below by 1.5*load_tot, which by construction
    # lets it serve the whole system and satisfy the reserve constraint alone.
    # Left committable, the MIQCQP exploits exactly that -- on case300 the
    # optimum was "commit only the phantom unit" in 7 of 32 instances, all with
    # a ~6.75% base gap against 0.33% for the rest. The slack bus is always
    # energised in practice, so u is pinned to 1 for these.
    always_on: object = None

    @property
    def n_branch(self): return len(self.f_bus)

    @property
    def n_gen(self): return len(self.gen_bus)

    def summary(self):
        return (f"{self.name}: {self.n_bus} buses, {self.n_branch} branches, "
                f"{self.n_gen} gens, load {self.pd.sum():.3f}+j{self.qd.sum():.3f} pu, "
                f"gen cap {self.pmax.sum():.3f} pu")


def load(case="case14") -> Grid:
    import pandapower as pp
    import pandapower.networks as pn

    net = getattr(pn, case)()
    pp.runpp(net, calculate_voltage_angles=True, init="flat")
    ppc = net._ppc
    base = ppc["baseMVA"]
    bus, branch, gen = ppc["bus"], ppc["branch"], ppc["gen"]

    # MATPOWER bus columns
    BUS_I, BUS_TYPE, PD, QD, GS, BS = 0, 1, 2, 3, 4, 5
    VMAX, VMIN = 11, 12
    idx = {int(r[BUS_I]): i for i, r in enumerate(bus)}
    n = len(bus)

    ref_rows = np.flatnonzero(bus[:, BUS_TYPE] == 3)
    ref = int(ref_rows[0]) if len(ref_rows) else 0

    # MATPOWER branch columns
    F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, TAP, SHIFT = 0, 1, 2, 3, 4, 5, 8, 9
    r, x, bc = branch[:, BR_R], branch[:, BR_X], branch[:, BR_B]
    ys = 1.0 / (r + 1j * x)
    tap = np.where(branch[:, TAP] == 0, 1.0, branch[:, TAP])
    shift = np.deg2rad(branch[:, SHIFT])
    if np.abs(shift).max() > 1e-9:
        print(f'  ! {int((np.abs(shift)>1e-9).sum())} phase-shifting branches; '
              'the QCAC model assumes a real tap ratio')

    rate = branch[:, RATE_A] / base
    rate = np.where(rate <= 0, np.inf, rate)

    g_fr = np.zeros_like(bc); b_fr = bc / 2.0
    g_to = np.zeros_like(bc); b_to = bc / 2.0

    # generator columns
    GEN_BUS, QMAX, QMIN, PMAX, PMIN = 0, 3, 4, 8, 9
    gb = np.array([idx[int(v)] for v in gen[:, GEN_BUS]])
    pmax, pmin = gen[:, PMAX] / base, gen[:, PMIN] / base
    load_tot = float(np.sum(bus[:, PD]) / base)
    huge = pmax > 50 * max(load_tot, 1e-6)      # pandapower ext_grid sentinel
    if huge.any():
        # The sentinel used to become 1.5*load_tot, which by construction lets
        # this ONE unit serve the entire system and satisfy the reserve
        # constraint alone. The unit commitment then had a trivial optimum:
        # on case300, 7 of 32 instances optimally committed nothing but this
        # phantom unit (base gap ~6.75% against 0.33% for the rest), and no
        # demand band removed it because it is not a demand effect.
        #
        # The ext_grid is an interconnection, not a local generator, so it is
        # capped like a large tie-line: twice the biggest real unit, or a
        # quarter of nominal load, whichever is larger. Measured, this keeps
        # both cases feasible with margin and stops any single unit covering
        # the system:
        #            slack cap   sum(pmax)   worst-case need   covers alone?
        #   case300      58.8       361.6             306.2    no
        #   case118      14.1       105.8              52.7    no
        real = pmax[~huge]
        cap = max(2.0 * float(real.max()) if real.size else load_tot,
                  0.25 * load_tot)
        pmax = np.where(huge, cap, pmax)
        pmin = np.where(huge, 0.0, pmin)
    always_on = huge.copy()                     # see Grid.always_on
    qmax, qmin = gen[:, QMAX] / base, gen[:, QMIN] / base

    # Quadratic cost, converted to per-unit input.
    #
    # NOT from ppc["gencost"]: pandapower leaves that None after runpp (it is
    # populated only by runopp), so reading it silently produced c2=0, c1=1,
    # c0=0 for EVERY generator via the fallback below -- identical costs, a
    # degenerate economic dispatch, and a unit commitment with nothing to
    # choose between units. Every rung D number before 2026-09-16 was computed
    # on that degenerate problem.
    #
    # The real coefficients live in net.poly_cost, in MW units, keyed by
    # (et, element). The ppc generator order is [ext_grid rows, gen rows] --
    # the same convention ac_metric.py relies on -- so the lookup follows it.
    G = len(gb)
    c2 = np.zeros(G); c1 = np.zeros(G); c0 = np.zeros(G)
    pc = getattr(net, "poly_cost", None)
    order = ([("ext_grid", int(i)) for i in net.ext_grid.index]
             + [("gen", int(i)) for i in net.gen.index])
    found = 0
    if pc is not None and len(pc) and len(order) == G:
        key = {(str(r.et), int(r.element)): r for r in pc.itertuples()}
        for i, k in enumerate(order):
            r = key.get(k)
            if r is None:
                continue
            # cost(p_pu) = cp2*(base*p_pu)^2 + cp1*(base*p_pu) + cp0  [dollars]
            c2[i] = float(r.cp2_eur_per_mw2) * base ** 2
            c1[i] = float(r.cp1_eur_per_mw) * base
            c0[i] = float(r.cp0_eur)
            found += 1
    if found < G:
        print(f"  ! {case}: cost data for only {found}/{G} generators; "
              "the rest default to linear unit cost")
    if np.allclose(c1, 0) and np.allclose(c2, 0):
        print(f"  ! {case}: NO generator cost data found -- using c1=1 for all. "
              "The economic problem is degenerate; do not report gaps.")
        c1 = np.ones(G)

    return Grid(
        name=case, n_bus=n, ref=ref,
        f_bus=np.array([idx[int(v)] for v in branch[:, F_BUS]]),
        t_bus=np.array([idx[int(v)] for v in branch[:, T_BUS]]),
        g=ys.real, b=ys.imag, g_fr=g_fr, b_fr=b_fr, g_to=g_to, b_to=b_to,
        tap=tap, tm2=tap ** 2, rate=rate,
        vmin=bus[:, VMIN].copy(), vmax=bus[:, VMAX].copy(),
        gs=bus[:, GS] / base, bs=bus[:, BS] / base,
        pd=bus[:, PD] / base, qd=bus[:, QD] / base,
        gen_bus=gb, pmin=pmin, pmax=pmax, qmin=qmin, qmax=qmax,
        c2=c2, c1=c1, c0=c0, always_on=always_on,
    )


def injections(grid: Grid, vr, vi):
    """
    Bus active/reactive injections implied by a voltage profile, using the same
    lifted quantities the QCAC constraints use:
        c_ii = vr_i^2 + vi_i^2,  c_ij = vr_i vr_j + vi_i vi_j,
        s_ij = vr_i vi_j - vr_j vi_i
    Returns (p_inj, q_inj) per bus.
    """
    i, j = grid.f_bus, grid.t_bus
    cii = vr**2 + vi**2
    cij = vr[i] * vr[j] + vi[i] * vi[j]
    sij = vr[i] * vi[j] - vr[j] * vi[i]          # = -v_i v_j sin(th_i - th_j)
    g, b, bsh, T, tm2 = grid.g, grid.b, grid.b_fr, grid.tap, grid.tm2

    # Constante-Flores & Li, equations (1d)-(1g), in their own c/s convention.
    p_fr = g / tm2 * cii[i] - g / T * cij + b / T * sij
    p_to = g * cii[j] - g / T * cij - b / T * sij
    q_fr = -(b + bsh) / tm2 * cii[i] + g / T * sij + b / T * cij
    q_to = -(b + bsh) * cii[j] - g / T * sij + b / T * cij

    p = np.zeros(grid.n_bus); q = np.zeros(grid.n_bus)
    np.add.at(p, i, p_fr); np.add.at(p, j, p_to)
    np.add.at(q, i, q_fr); np.add.at(q, j, q_to)
    p += grid.gs * cii
    q -= grid.bs * cii
    return p, q


def validate(case="case14", tol=1e-6):
    """
    Reconstruct injections from a converged pandapower AC power flow and compare
    against pandapower's own reported bus powers. A mismatch means the
    admittance, shunt or tap conventions were extracted wrongly.
    """
    import pandapower as pp
    import pandapower.networks as pn

    net = getattr(pn, case)()
    pp.runpp(net, calculate_voltage_angles=True, init="flat")
    grid = load(case)
    base = net._ppc["baseMVA"]

    vm = net._ppc["bus"][:, 7]
    va = np.deg2rad(net._ppc["bus"][:, 8])
    vr, vi = vm * np.cos(va), vm * np.sin(va)

    p, q = injections(grid, vr, vi)

    # pandapower's own net injection per bus = generation - load
    gen_p = np.zeros(grid.n_bus); gen_q = np.zeros(grid.n_bus)
    gcols = net._ppc["gen"]
    for row in gcols:
        bi = int(row[0])
        k = {int(r[0]): m for m, r in enumerate(net._ppc["bus"])}[bi]
        gen_p[k] += row[1] / base
        gen_q[k] += row[2] / base
    ref_p = gen_p - grid.pd
    ref_q = gen_q - grid.qd

    dp, dq = np.abs(p - ref_p).max(), np.abs(q - ref_q).max()
    print(f"{grid.summary()}")
    print(f"  max |P injection residual| = {dp:.3e}")
    print(f"  max |Q injection residual| = {dq:.3e}")
    print(f"  -> {'OK' if max(dp, dq) < 1e-4 else 'MISMATCH — check conventions'}")
    return grid, max(dp, dq)


if __name__ == "__main__":
    import sys
    for c in (sys.argv[1:] or ["case5", "case14", "case30"]):
        validate(c)
        print()

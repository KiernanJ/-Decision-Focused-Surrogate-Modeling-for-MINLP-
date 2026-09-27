"""
Deployment: one continuous solve + feasibility restoration -> discrete decision.

This is the surrogate path the proposal describes -- "learn problem-specific
constraints that guide the continuous relaxation, followed by a feasibility
restoration step to quickly recover valid discrete decisions". No binaries are
solved anywhere here; the integer decision comes out of restoration.

    predicted V, cuts (A,b)
        -> CONTINUOUS QCAC NLP, linearised at V, with A w <= b
        -> fractional u
        -> round, top up to the reserve, dispatch with u fixed
        -> repair upward while infeasible
        -> cost

Two things this shares with the validated code rather than reimplementing:
`acuc.build` constructs the model (so the surrogate and the references cannot
drift apart), and the cost returned EXCLUDES any penalty term, so it is
comparable to the global and iterative solutions directly.

The repair only ever switches generators ON. That is a real asymmetry: a
commitment that is too large cannot be walked back, so over-commitment is
unrecoverable while under-commitment is fixable. It is why the uncut relaxation
-- which under-commits badly -- still reaches a feasible point, and why the
cuts' job is to move the relaxed solution toward the right integer answer
BEFORE rounding.
"""
from __future__ import annotations

import time

import numpy as np
from gurobipy import GRB

from acuc import build


def solve_nlp_with_cuts(g, nl, pd_, qd_, Vr, Vi, A=None, b=None, reserve=0.10,
                        threads=4, time_limit=60.0, cut_slack_price=1e4):
    """One CONTINUOUS solve: QCAC linearised at (Vr, Vi), plus learned cuts.

    The cuts enter softly, priced at `cut_slack_price`: a cut that would make
    the instance infeasible costs rather than kills. Without that a single bad
    predicted cut returns no solution at all and the training signal for that
    instance is a constant penalty, which carries no gradient.
    """
    import gurobipy as gp
    n, L, G = g.n_bus, g.n_branch, g.n_gen
    m, V, cost = build(g, nl, pd_, qd_, u_type=GRB.CONTINUOUS, exact=False,
                       rho=1e6, Vr0=Vr, Vi0=Vi, reserve=reserve, threads=threads)
    if A is not None and b is not None:
        K = A.shape[0]
        sc = m.addVars(K, lb=0.0)
        w = [V["pg"][x] for x in range(G)] + [V["u"][x] for x in range(G)]
        for k in range(K):
            m.addConstr(gp.quicksum(float(A[k, d]) * w[d] for d in range(2*G))
                        <= float(b[k]) + sc[k])
        m.setObjective(m.getObjective() + cut_slack_price*gp.quicksum(sc[k] for k in range(K)))
    m.Params.TimeLimit = time_limit
    m.optimize()
    if m.SolCount == 0:
        return None
    return dict(u=np.array([V["u"][x].X for x in range(G)]),
                pg=np.array([V["pg"][x].X for x in range(G)]),
                vr=np.array([V["vr"][i].X for i in range(n)]),
                vi=np.array([V["vi"][i].X for i in range(n)]))


def dispatch_convex(g, nl, pd_, qd_, u_fixed, Vr, Vi, reserve=0.10, threads=4,
                    time_limit=60.0):
    """Dispatch with u FIXED, on the CONVEX QCAC model at (Vr, Vi).

    Training needs thousands of these. The exact nonconvex dispatch costs
    ~60-120 s each; this is the convex one the surrogate actually controls, and
    it is what the network can influence. The exact AC cost is used at
    EVALUATION, where comparability to the global optimum is what matters.
    """
    m, V, cost = build(g, nl, pd_, qd_, u_type=GRB.CONTINUOUS, exact=False,
                       rho=1e6, Vr0=Vr, Vi0=Vi, reserve=reserve, threads=threads)
    for x in range(g.n_gen):
        V["u"][x].lb = V["u"][x].ub = float(u_fixed[x])
    m.Params.TimeLimit = time_limit
    m.optimize()
    if m.SolCount == 0:
        return None
    return dict(cost=float(cost.getValue()),
                pg=np.array([V["pg"][x].X for x in range(g.n_gen)]))


def restore(g, nl, pd_, qd_, u_frac, reserve=0.10, threads=4, time_limit=60.0,
            exact=True, Vr=None, Vi=None):
    """Feasibility restoration: round -> reserve top-up -> dispatch -> repair up.

    Returns the TRUE AC cost of the recovered commitment, so it is directly
    comparable to the global optimum and to the iterative method.
    """
    from acuc import cost_commitment
    z = (np.asarray(u_frac, float) > 0.5).astype(float)
    order = np.argsort(-np.asarray(u_frac, float))
    need = (1.0 + reserve) * float(pd_.sum())
    for k in order:                              # top up to the reserve margin
        if float(g.pmax @ z) >= need:
            break
        z[k] = 1.0
    for _ in range(g.n_gen + 1):                 # repair upward until feasible
        if exact:
            r = cost_commitment(g, nl, pd_, qd_, z, reserve=reserve,
                                time_limit=time_limit, threads=threads)
        else:
            r = dispatch_convex(g, nl, pd_, qd_, z, Vr, Vi, reserve=reserve,
                                threads=threads, time_limit=time_limit)
        if r is not None:
            return dict(u=z.astype(int), cost=r["cost"], pg=r["pg"],
                        vr=r.get("vr"), vi=r.get("vi"))
        off = [k for k in order if z[k] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    return None


def deployment_cost(g, nl, pd_, qd_, Vr, Vi, A=None, b=None, reserve=0.10,
                    threads=4, fail=1e9, want_detail=False, exact=True):
    """The objective the network is trained on: one solve + restoration.

    A failure returns `fail` rather than raising, so one bad instance cannot
    kill a training step -- but note a constant penalty carries no gradient, so
    a high failure rate silently starves learning. `want_detail` reports it.
    """
    t0 = time.time()
    s = solve_nlp_with_cuts(g, nl, pd_, qd_, Vr, Vi, A, b, reserve, threads)
    if s is None:
        return (fail, None) if want_detail else fail
    r = restore(g, nl, pd_, qd_, s["u"], reserve, threads, exact=exact,
                Vr=Vr, Vi=Vi)
    if r is None:
        return (fail, None) if want_detail else fail
    if want_detail:
        r["secs"] = time.time() - t0
        r["u_frac"] = s["u"]
        return r["cost"], r
    return r["cost"]

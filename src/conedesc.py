"""Cone projections and their derivatives, in standard conic form.

Everything here is PROBLEM-AGNOSTIC. Any DPP cvxpy problem canonicalises to

    min  1/2 x'Px + q'x    s.t.  Ax + s = b,   s in K

where ALL constraints are LINEAR and the only nonlinearity is membership in the
cone K. So there is no per-problem constraint Hessian to hand-derive -- the
earlier smooth-constraint version needed grad^2 g for each constraint and would
not have transferred from AC unit commitment to the hybrid vehicle. The cone
enters instead through complementarity, i.e. through the derivative of the
projection onto K.

Cones covered, which is everything these case studies produce:

    zero      equalities                      projection 0, derivative 0
    nonneg    bounds and inequalities         max(x,0), derivative 1[x>0]
    SOC       rank-1 and line limits          closed form below

PSD is deliberately absent -- neither case study generates one, and it is the
only cone whose derivative is genuinely awkward.

SOC projection, for (t, z) with r = ||z||:
    r <= t          already inside           -> identity
    r <= -t         polar cone               -> 0
    otherwise       boundary                 -> ((t + r)/2) * (1, z/r)

and its Jacobian in the boundary case, with u = z/r:

    1/2 * [ 1      u'                      ]
          [ u   (1 + t/r) I - (t/r) u u'   ]
"""
from __future__ import annotations

import numpy as np


def proj_nonneg(x):
    return np.maximum(x, 0.0)


def dproj_nonneg(x):
    return (x > 0).astype(float)


def proj_soc(v):
    t, z = v[0], v[1:]
    r = np.linalg.norm(z)
    if r <= t:
        return v.copy()
    if r <= -t:
        return np.zeros_like(v)
    out = np.empty_like(v)
    sc = (t + r)/2.0
    out[0] = sc
    out[1:] = sc*(z/r)
    return out


def dproj_soc(v):
    """Jacobian of the SOC projection at v. Dense but small: one block per cone."""
    n = v.size
    t, z = v[0], v[1:]
    r = np.linalg.norm(z)
    if r <= t:
        return np.eye(n)
    if r <= -t:
        return np.zeros((n, n))
    u = z/r
    J = np.zeros((n, n))
    J[0, 0] = 1.0
    J[0, 1:] = u
    J[1:, 0] = u
    J[1:, 1:] = (1.0 + t/r)*np.eye(n-1) - (t/r)*np.outer(u, u)
    return 0.5*J


def cone_blocks(dims):
    """Yield (kind, start, length) for each cone block, in canonical order."""
    i = 0
    if getattr(dims, "zero", 0):
        yield ("zero", i, int(dims.zero)); i += int(dims.zero)
    if getattr(dims, "nonneg", 0):
        yield ("nonneg", i, int(dims.nonneg)); i += int(dims.nonneg)
    for q in getattr(dims, "soc", []):
        yield ("soc", i, int(q)); i += int(q)
    for e in range(getattr(dims, "exp", 0) or 0):
        yield ("exp", i, 3); i += 3
    if getattr(dims, "psd", None):
        for p in dims.psd:
            k = int(p*(p+1)//2)
            yield ("psd", i, k); i += k


def dproj_cone(v, dims):
    """Block-diagonal Jacobian of the projection onto the product cone."""
    import scipy.sparse as sp
    blocks = []
    for kind, i, k in cone_blocks(dims):
        seg = v[i:i+k]
        if kind == "zero":
            blocks.append(sp.csc_matrix((k, k)))
        elif kind == "nonneg":
            blocks.append(sp.diags(dproj_nonneg(seg)).tocsc())
        elif kind == "soc":
            blocks.append(sp.csc_matrix(dproj_soc(seg)))
        else:
            raise NotImplementedError(
                f"cone '{kind}' has no derivative here; neither case study "
                f"produces one, so this is a guard, not a gap")
    return sp.block_diag(blocks, format="csc") if blocks else sp.csc_matrix((0, 0))

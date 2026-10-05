"""CutNet with K rows per cut CATEGORY, instead of one of each.

    rows 0 .. K-1     alpha_i'pg + a_i'u <= b_i     the abstract's general cut
    rows K .. 2K-1    a2_i'u <= b2_i                the integer/binary category

Same guarantees as the K=1 version: delta and alpha start at (near) zero so the
cut family begins at the validated cardinality cut, and head_v is zero so V
starts at the flat point.

TWO things differ and both matter.

1. The network emits a DEPTH per row, not b itself, and the caller sets
       b_i = <A_i, w_rlx> - depth_i
   where w_rlx is the UNCUT relaxation's own [pg; u]. Every row is then violated
   by exactly depth_i at that point, i.e. ACTIVE. This is the anchoring
   qcac_clean/src/netw.py already validated. Emitting b directly left rows with
   slack 0.23 and 0.54 at init -- a row that starts inactive has zero derivative
   with respect to its own coefficients and never trains.

2. With K rows the obvious initialisation -- every row identical -- creates K-1
   REDUNDANT active constraints, exactly the LICQ failure that makes the KKT
   singular. The rows get a random spread in `a` to keep them distinct; `spread`
   trades that against starting at the validated cardinality cut.
"""
from __future__ import annotations

import numpy as np
import torch


class CutNetK(torch.nn.Module):
    def __init__(self, n_bus, n_gen, k_rows=2, hidden=256, v_scale=0.5,
                 a_scale=2.0, alpha_scale=2.0, init_depth=0.25, max_depth=2.0,
                 spread=0.25, seed=0, learn_thr=True, thr_span=0.4):
        super().__init__()
        self.n_bus, self.n_gen, self.K = n_bus, n_gen, k_rows
        self.v_scale, self.a_scale, self.alpha_scale = v_scale, a_scale, alpha_scale
        self.max_depth = max_depth
        self.learn_thr, self.thr_span = learn_thr, thr_span
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(2*n_bus, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU())
        self.head_v = torch.nn.Linear(hidden, 2*n_bus)
        self.head_a = torch.nn.Linear(hidden, k_rows*n_gen)      # general: a
        self.head_al = torch.nn.Linear(hidden, k_rows*n_gen)     # general: alpha
        self.head_d = torch.nn.Linear(hidden, k_rows)       # DEPTH, not b
        self.head_a2 = torch.nn.Linear(hidden, k_rows*n_gen)     # binary-only: a2
        self.head_d2 = torch.nn.Linear(hidden, k_rows)
        # Per-generator ROUNDING THRESHOLD. The relaxation leaves 41 of 54
        # units fractional on case118 and we collapse all of it at a
        # hard-coded 0.5; an ORACLE per-generator threshold reaches 1.39%
        # discrete / 2.87% continuous there against 5.71% / 15.93% shipped.
        # Label-free: a threshold only moves the rounding decision, and
        # restoration still runs after it, so it cannot cause infeasibility.
        self.head_thr = torch.nn.Linear(hidden, n_gen) if learn_thr else None
        torch.nn.init.zeros_(self.head_v.weight); torch.nn.init.zeros_(self.head_v.bias)
        for h in (self.head_a, self.head_al, self.head_a2):
            torch.nn.init.zeros_(h.weight); torch.nn.init.zeros_(h.bias)
        # break the tie between rows -- identical rows are redundant constraints
        gen = torch.Generator().manual_seed(seed)
        for h in (self.head_a, self.head_a2):
            with torch.no_grad():
                h.bias.copy_(spread*torch.randn(k_rows*n_gen, generator=gen))
        if self.head_thr is not None:
            # start EXACTLY at the shipped rule: thr = 0.5 for every unit
            torch.nn.init.zeros_(self.head_thr.weight)
            torch.nn.init.zeros_(self.head_thr.bias)
        f = float(np.clip(init_depth/max_depth, 1e-3, 1-1e-3))
        for h in (self.head_d, self.head_d2):
            torch.nn.init.normal_(h.weight, std=1e-3)
            torch.nn.init.constant_(h.bias, float(np.log(f/(1-f))))

    def forward(self, x):
        h = self.trunk(x)
        n, K, G = x.shape[0], self.K, self.n_gen
        dv = self.v_scale*torch.tanh(self.head_v(h))
        Vr = 1.0 + dv[:, :self.n_bus]
        Vi = 0.0 + dv[:, self.n_bus:]
        a = 1.0 + self.a_scale*torch.tanh(self.head_a(h)).view(n, K, G)
        al = self.alpha_scale*torch.tanh(self.head_al(h)).view(n, K, G)
        a2 = 1.0 + self.a_scale*torch.tanh(self.head_a2(h)).view(n, K, G)
        d = self.max_depth*torch.sigmoid(self.head_d(h))
        d2 = self.max_depth*torch.sigmoid(self.head_d2(h))
        # thr in 0.5 +/- thr_span, centred so tanh(0) reproduces 0.5 exactly.
        # Measured on case118: thresholds ABOVE 0.5 are inert because the
        # reserve top-up turns those units straight back on, so the useful
        # range is downward and the span is kept modest.
        thr = (0.5 + self.thr_span*torch.tanh(self.head_thr(h))
               if self.head_thr is not None else None)
        return Vr, Vi, a, al, a2, d, d2, thr

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def rows_torch(a, al, a2, d, d2, w_rlx):
    """Assemble (A, b) IN TORCH so autograd routes b's gradient to A and depth.

    b = <A, w_rlx> - depth, so every row is active by `depth` at the uncut
    relaxation point. Building b outside torch would silently drop the dL/dA
    term that flows through the anchor.
    """
    n, K, G = a.shape
    Agen = torch.cat([al, a], dim=2)                       # (n,K,2G)
    Abin = torch.cat([torch.zeros_like(a2), a2], dim=2)    # (n,K,2G)
    A = torch.cat([Agen, Abin], dim=1)                     # (n,2K,2G)
    depth = torch.cat([d, d2], dim=1)                      # (n,2K)
    b = torch.einsum("nkd,nd->nk", A, w_rlx) - depth
    return A, b




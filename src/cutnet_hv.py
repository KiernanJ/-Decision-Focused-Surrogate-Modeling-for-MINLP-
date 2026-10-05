"""Cut network for the hybrid-vehicle convex MINLP.

Predicts the CUT ONLY -- the problem is convex, so unlike QCAC there is no
linearisation point to learn.

    rows 0..K-1     alpha'Peng + a'z <= b     general category
    rows K..2K-1    a2'z <= b2                integer-only category

Same two PI categories and the same DEPTH ANCHORING as the AC-UC study:
    b = <A, w_rlx> - depth
so every row is violated by exactly `depth` at the UNCUT relaxation point and
therefore starts ACTIVE. Emitting b directly left rows slack at init, and a
slack row has zero derivative w.r.t. its own coefficients and never trains.
"""
from __future__ import annotations
import numpy as np
import torch


class CutNetHV(torch.nn.Module):
    def __init__(self, T, k_rows=2, hidden=256, a_scale=2.0, alpha_scale=2.0,
                 init_depth=0.25, max_depth=2.0, spread=0.25, seed=0,
                 learn_thr=True, thr_span=0.25):
        super().__init__()
        self.T, self.K, self.max_depth = T, k_rows, max_depth
        self.learn_thr, self.thr_span = learn_thr, thr_span
        self.a_scale, self.alpha_scale = a_scale, alpha_scale
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(T, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU())
        # one head per variable block, matching the original layer's
        # A_e / A_peng / A_pbatt / A_z
        self.head_E = torch.nn.Linear(hidden, k_rows*(T+1))
        self.head_pe = torch.nn.Linear(hidden, k_rows*T)
        self.head_pb = torch.nn.Linear(hidden, k_rows*T)
        self.head_z = torch.nn.Linear(hidden, k_rows*T)
        self.head_z2 = torch.nn.Linear(hidden, k_rows*T)   # integer-only category
        self.head_d = torch.nn.Linear(hidden, k_rows)
        self.head_d2 = torch.nn.Linear(hidden, k_rows)
        self.head_thr = torch.nn.Linear(hidden, T) if learn_thr else None
        for h in (self.head_E, self.head_pe, self.head_pb, self.head_z, self.head_z2):
            torch.nn.init.zeros_(h.weight); torch.nn.init.zeros_(h.bias)
        gen = torch.Generator().manual_seed(seed)
        for h in (self.head_z, self.head_z2):      # identical rows are redundant
            with torch.no_grad():                   # constraints -> LICQ failure
                h.bias.copy_(spread*torch.randn(k_rows*T, generator=gen))
        if self.head_thr is not None:               # start EXACTLY at 0.5
            torch.nn.init.zeros_(self.head_thr.weight)
            torch.nn.init.zeros_(self.head_thr.bias)
        f = float(np.clip(init_depth/max_depth, 1e-3, 1-1e-3))
        for h in (self.head_d, self.head_d2):
            torch.nn.init.normal_(h.weight, std=1e-3)
            torch.nn.init.constant_(h.bias, float(np.log(f/(1-f))))

    def forward(self, x):
        h = self.trunk(x); n, K, T = x.shape[0], self.K, self.T
        aE = self.alpha_scale*torch.tanh(self.head_E(h)).view(n, K, T+1)
        ape = self.alpha_scale*torch.tanh(self.head_pe(h)).view(n, K, T)
        apb = self.alpha_scale*torch.tanh(self.head_pb(h)).view(n, K, T)
        az = 1.0 + self.a_scale*torch.tanh(self.head_z(h)).view(n, K, T)
        az2 = 1.0 + self.a_scale*torch.tanh(self.head_z2(h)).view(n, K, T)
        d = self.max_depth*torch.sigmoid(self.head_d(h))
        d2 = self.max_depth*torch.sigmoid(self.head_d2(h))
        thr = (0.5 + self.thr_span*torch.tanh(self.head_thr(h))
               if self.head_thr is not None else None)
        return aE, ape, apb, az, az2, d, d2, thr

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def rows_torch(aE, ape, apb, az, az2, d, d2, w_rlx):
    """(A, b) assembled IN TORCH so autograd routes b's gradient to A and depth.
    Building b outside torch silently drops the dL/dA term through the anchor."""
    n, K, T = az.shape
    Z = torch.zeros_like(az)
    gen = torch.cat([aE, ape, apb, az], dim=2)                     # general
    binr = torch.cat([torch.zeros_like(aE), Z, Z, az2], dim=2)     # integer-only
    A = torch.cat([gen, binr], dim=1)
    depth = torch.cat([d, d2], dim=1)
    return A, torch.einsum("nkd,nd->nk", A, w_rlx) - depth

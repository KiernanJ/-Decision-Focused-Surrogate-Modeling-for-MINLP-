"""
The surrogate network: instance parameters -> cuts AND linearisation point.

This is the architecture the proposal describes, with BOTH heads live. The
previous codebase had the same two heads but ran every experiment with
`--freeze-v`, so the linearisation head never trained. The stated reason was an
ablation finding that learning V "cheats by shrinking slack without improving
the commitment" -- but that was measured on an objective that was 97% slack, so
anything reducing slack looked like cheating. On a correct objective the
argument evaporates, and predicting V is the half of the method that removes
the iteration.

    (pd, qd) --> trunk --+--> head_v    --> (Vre, Vim)   the linearisation point
                         |
                         +--> head_dir  --> A            K cut directions
                         +--> head_rhs  --> depth        K cut depths

What each head is for, and why neither replaces the other:

  head_v     Can Li's iterative method finds the linearisation point by solving
             3-4 times, re-linearising at each solution. Predict it and ONE
             solve lands where the loop converges.

  head_cuts  Even at a perfect linearisation point the relaxation returns a
             FRACTIONAL commitment. The cuts push it toward the integer
             optimum so the restoration step rounds to the right units.

Design details that carry over from the validated implementation:

  * head_v is initialised to ZERO, so training starts at V = V0, the AC
    power-flow point. That is already ahead of the iterative baseline, which
    starts flat at (1, 0) and still converges. The network learns a bounded
    correction, |dV| <= v_scale, not V from scratch.
  * A's rows are normalised to unit length, so the network chooses a DIRECTION
    per cut and never a magnitude.
  * b is anchored per instance: b = <A, w_rlx> - depth, where w_rlx is the
    uncut relaxation's own solution. Every cut is therefore violated by exactly
    `depth` at that point, i.e. active. An inactive cut has zero derivative
    with respect to its own coefficients and would never train.

No supervised targets. Training is on the deployment objective, as the proposal
specifies -- labels would require the exact solve the surrogate exists to avoid.
"""
from __future__ import annotations

import numpy as np
import torch


class SurrogateNet(torch.nn.Module):
    def __init__(self, n_bus, n_gen, n_cuts=8, hidden=128, v_scale=0.15,
                 init_depth=0.20, max_depth=0.80):
        super().__init__()
        self.n_bus, self.n_gen, self.n_cuts = n_bus, n_gen, n_cuts
        self.dim = 2 * n_gen                  # w = [pg; u]
        self.v_scale, self.max_depth = v_scale, max_depth

        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(2 * n_bus, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh())
        self.head_v = torch.nn.Linear(hidden, 2 * n_bus)
        self.head_dir = torch.nn.Linear(hidden, n_cuts * self.dim)
        self.head_rhs = torch.nn.Linear(hidden, n_cuts)

        # V starts exactly at the power-flow point: zero correction.
        torch.nn.init.zeros_(self.head_v.weight)
        torch.nn.init.zeros_(self.head_v.bias)
        # Directions start essentially instance-independent (small weight, real
        # bias) so the map from instance to direction has to be learned.
        torch.nn.init.normal_(self.head_dir.weight, std=1e-2)
        torch.nn.init.normal_(self.head_dir.bias, std=0.5)
        torch.nn.init.normal_(self.head_rhs.weight, std=1e-2)
        r = init_depth / max_depth
        torch.nn.init.constant_(self.head_rhs.bias, float(np.log(r / (1 - r))))

    def forward(self, x, Vr0, Vi0, w_rlx):
        """x = [pd; qd]; Vr0/Vi0 = power-flow point; w_rlx = uncut [pg; u]."""
        h = self.trunk(x)
        dv = self.v_scale * torch.tanh(self.head_v(h))
        Vr = Vr0 + dv[:, :self.n_bus]
        Vi = Vi0 + dv[:, self.n_bus:]
        A = self.head_dir(h).view(-1, self.n_cuts, self.dim)
        A = A / (A.norm(dim=-1, keepdim=True) + 1e-8)
        depth = self.max_depth * torch.sigmoid(self.head_rhs(h))
        b = torch.einsum("bkd,bd->bk", A, w_rlx) - depth
        return Vr, Vi, A, b

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

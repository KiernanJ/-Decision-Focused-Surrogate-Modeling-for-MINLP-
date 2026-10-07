"""Cut network for the CombOptNet-style AC-UC method (integer-only rows)."""
import torch, torch.nn as nn


class CutNet(nn.Module):
    """integer-only rows a_r'u <= b_r; a = 1 + 2 tanh(.), depth in (0, 2)."""
    def __init__(self, n_in, G, R, h=256, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.G, self.R = G, R
        self.trunk = nn.Sequential(nn.Linear(n_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.ha = nn.Linear(h, R*G); self.hd = nn.Linear(h, R)
        nn.init.zeros_(self.ha.weight); nn.init.normal_(self.ha.bias, std=0.25)
        nn.init.normal_(self.hd.weight, std=1e-3); nn.init.constant_(self.hd.bias, -2.0)
    def forward(self, x):
        h = self.trunk(x)
        return 1.0 + 2.0*torch.tanh(self.ha(h)).view(-1, self.R, self.G), 2.0*torch.sigmoid(self.hd(h))

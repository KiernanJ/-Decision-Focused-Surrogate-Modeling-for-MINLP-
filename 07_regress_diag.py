"""Why did the learning curve go the wrong way? Train error vs test error.

06 produced a non-monotone curve (||dV|| 0.33 -> 0.59 -> 1.53 -> 1.49 -> 0.69).
More data cannot genuinely hurt, so the regression itself is at fault. The
distinguishing measurement is TRAIN error:

  train error also high  -> optimisation failure (too few steps, bad scaling)
  train low, test high   -> genuine generalisation limit, i.e. really data-bound

Also reported: ||V* - V_flat||, the distance the network has to cover at all. A
predictor that does nothing scores that, so any ||dV|| near it has learned
nothing -- which is what 0.33 vs 1.53 was really saying.
"""
# %% setup
import os, sys, time, re
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, grid, sample, deploy
from socp import Socp

torch.manual_seed(0); rng = np.random.default_rng(0)
g, nl = grid(); S = Socp(g, nl, n_cuts=8)

POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
             for i in range(int(m.group(2))) if REF[i] is not None]
VS = np.stack([S.price(p, q, u.astype(float)) for p, q, u, _ in POOL]
              if False else
              [np.r_[(lambda r: (r["vr"], r["vi"]))(S.price(p, q, u.astype(float)))[0],
                     (lambda r: (r["vr"], r["vi"]))(S.price(p, q, u.astype(float)))[1]]
               for p, q, u, _ in POOL])
print(f"[pool] {len(POOL)} labelled instances", flush=True)

base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
Y = torch.tensor(VS, dtype=torch.float32)
V0 = torch.tensor(np.r_[np.ones(g.n_bus), np.zeros(g.n_bus)], dtype=torch.float32)
flat_err = float(torch.norm(Y - V0, dim=1).mean())
print(f"[scale] ||V* - V_flat|| = {flat_err:.4f}   "
      f"(a do-nothing predictor scores this)")
print(f"[scale] target for ~exact commitments: 0.08\n", flush=True)

perm = rng.permutation(len(POOL)); te = perm[:24]; pool_tr = perm[24:]


def fit(ntr, steps=30000, lr=3e-3):
    tr = pool_tr[:ntr]
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.Tanh(),
                              torch.nn.Linear(256, 256), torch.nn.Tanh(),
                              torch.nn.Linear(256, 2*g.n_bus))
    torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for _ in range(steps):
        p = V0 + 0.5*torch.tanh(net((X[tr]-mu)/sg))
        loss = ((p - Y[tr])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        ptr = V0 + 0.5*torch.tanh(net((X[tr]-mu)/sg))
        pte = V0 + 0.5*torch.tanh(net((X[te]-mu)/sg))
    return (pte.numpy(),
            float(torch.norm(ptr - Y[tr], dim=1).mean()),
            float(torch.norm(pte - Y[te], dim=1).mean()), time.time()-t0)


print(f"{'n_train':>9s}{'train':>9s}{'test':>9s}{'gap':>10s}{'agree':>9s}{'fit':>8s}")
for ntr in [16, 32, 48, 64, len(pool_tr)]:
    if ntr > len(pool_tr):
        continue
    pte, e_tr, e_te, ft = fit(ntr)
    gaps, agrs = [], []
    for k, j in enumerate(te):
        pd_, qd_, u, c = POOL[j]
        cc, d = deploy(S, g, pd_, qd_, pte[k][:g.n_bus], pte[k][g.n_bus:])
        if cc < 1e8:
            gaps.append(100*(cc-c)/c); agrs.append(100*float((d["u"] == u).mean()))
    print(f"{ntr:>9d}{e_tr:>9.4f}{e_te:>9.4f}{np.mean(gaps):>9.3f}%"
          f"{np.mean(agrs):>8.1f}%{ft:>7.1f}s", flush=True)

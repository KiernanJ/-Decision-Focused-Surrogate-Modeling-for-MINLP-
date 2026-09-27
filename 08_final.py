"""Was v_scale=0.15 clamping the target out of reach, and how far does data go?

07 showed regression onto V* reaching +0.174% gap / 96.4% agreement at n=16-32,
while n=48/64 DIVERGED (train error 4.0 against a do-nothing baseline of 1.07)
-- the tanh output head saturating, where gradients vanish. That is an
optimisation artifact, not a data limit.

Two things to settle:

  A. Every smoothed run used v_scale=0.15, i.e. |dV_i| <= 0.15 per component.
     If V* needs more than that anywhere, the network could not represent the
     target however long it trained, and the 1.388% was an architectural
     ceiling rather than a result.

  B. Refit with a LINEAR output head on standardised targets -- no saturation,
     nothing to diverge into -- and read the learning curve honestly.
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
VS = []
for p, q, u, _ in POOL:
    r = S.price(p, q, u.astype(float)); VS.append(np.r_[r["vr"], r["vi"]])
VS = np.stack(VS)

# %% A. is v_scale=0.15 binding?
V0n = np.r_[np.ones(g.n_bus), np.zeros(g.n_bus)]
D = np.abs(VS - V0n)
print(f"\nA. per-component |V* - V_flat| over {len(POOL)} instances")
print(f"     mean {D.mean():.4f}   median {np.median(D):.4f}   "
      f"p95 {np.percentile(D,95):.4f}   MAX {D.max():.4f}")
frac = 100*float((D > 0.15).mean())
print(f"     components needing MORE than v_scale=0.15: {frac:.1f}%")
print(f"     -> {'CLAMP WAS BINDING: the target was unreachable' if D.max()>0.15 else 'clamp was not binding'}")

# %% B. honest learning curve, linear head on standardised targets
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
Yr = torch.tensor(VS - V0n, dtype=torch.float32)          # predict the CORRECTION
V0t = torch.tensor(V0n, dtype=torch.float32)
perm = rng.permutation(len(POOL)); te = perm[:24]; pool_tr = perm[24:]


def fit(ntr, steps=20000, lr=1e-3):
    tr = pool_tr[:ntr]
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
    ym, ys = Yr[tr].mean(0), Yr[tr].std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.SiLU(),
                              torch.nn.Linear(256, 256), torch.nn.SiLU(),
                              torch.nn.Linear(256, 2*g.n_bus))   # LINEAR head
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        loss = ((net((X[tr]-mu)/sg) - (Yr[tr]-ym)/ys)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        f_ = lambda ix: (V0t + ym + ys*net((X[ix]-mu)/sg))
        ptr, pte = f_(tr), f_(te)
    return (pte.numpy(),
            float(torch.norm(ptr-(V0t+Yr[tr]), dim=1).mean()),
            float(torch.norm(pte-(V0t+Yr[te]), dim=1).mean()))


print(f"\nB. regression onto V*, linear head   "
      f"(do-nothing ||dV|| = {float(torch.norm(Yr[te],dim=1).mean()):.4f}, "
      f"target 0.08)")
print(f"{'n_train':>9s}{'train':>9s}{'test':>9s}{'gap':>10s}{'agree':>9s}")
for ntr in [16, 32, 48, 64, len(pool_tr)]:
    if ntr > len(pool_tr):
        continue
    pte, e_tr, e_te = fit(ntr)
    gaps, agrs = [], []
    for k, j in enumerate(te):
        pd_, qd_, u, c = POOL[j]
        cc, d = deploy(S, g, pd_, qd_, pte[k][:g.n_bus], pte[k][g.n_bus:])
        if cc < 1e8:
            gaps.append(100*(cc-c)/c); agrs.append(100*float((d["u"] == u).mean()))
    print(f"{ntr:>9d}{e_tr:>9.4f}{e_te:>9.4f}{np.mean(gaps):>9.3f}%{np.mean(agrs):>8.1f}%",
          flush=True)

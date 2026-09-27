"""Is the remaining gap data-limited? Learning curve on pooled references.

05 established: the target V* is worth -0.002% / 100% agreement; regression onto
it trains in 2.4 s with NO solver in the loop; and the pipeline needs
||dV|| <~ 0.08 to commit correctly. With 24 instances the regression reaches
only 0.3733, which the tolerance curve maps to ~2.4% -- exactly what it scored.

So the question is not the objective or the architecture. It is whether ||dV||
falls with more labels. If it does, the remaining work is data generation, which
is cheap and parallel, rather than more training.
"""
# %% setup
import os, sys, time, itertools
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, grid, sample, deploy
from socp import Socp

torch.manual_seed(0); rng = np.random.default_rng(0)
g, nl = grid()
S = Socp(g, nl, n_cuts=8)

# %% pool every reference set on disk
POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    import re
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:          # older tag-named caches (run1, smoke) have no seed/n
        continue
    seed, n = int(m.group(1)), int(m.group(2))
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, n, seed=seed)
    for i in range(n):
        if REF[i] is not None:
            POOL.append((inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"]))
    print(f"[pool] {f}: +{sum(x is not None for x in REF)}", flush=True)
print(f"[pool] {len(POOL)} labelled instances total", flush=True)

# %% V* labels
t0 = time.time(); VS = []
for pd_, qd_, u, c in POOL:
    pr = S.price(pd_, qd_, u.astype(float))
    VS.append(np.r_[pr["vr"], pr["vi"]])
VS = np.stack(VS)
print(f"[V*   ] {len(VS)} labels in {time.time()-t0:.0f}s", flush=True)

# %% learning curve
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
Y = torch.tensor(VS, dtype=torch.float32)
perm = rng.permutation(len(POOL))
te = perm[:24]; pool_tr = perm[24:]
V0 = torch.tensor(np.r_[np.ones(g.n_bus), np.zeros(g.n_bus)], dtype=torch.float32)


def fit(ntr, epochs=6000):
    tr = pool_tr[:ntr]
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.Tanh(),
                              torch.nn.Linear(256, 256), torch.nn.Tanh(),
                              torch.nn.Linear(256, 2*g.n_bus))
    torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    t0 = time.time()
    for _ in range(epochs):
        p = V0 + 0.15*torch.tanh(net((X[tr]-mu)/sg))
        loss = ((p - Y[tr])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pte = (V0 + 0.15*torch.tanh(net((X[te]-mu)/sg))).numpy()
    return pte, np.linalg.norm(pte - Y[te].numpy(), axis=1).mean(), time.time()-t0


print(f"\n{'n_train':>9s}{'||dV||':>10s}{'gap':>10s}{'agree':>9s}{'fit':>8s}")
for ntr in [16, 32, 48, 64, len(pool_tr)]:
    if ntr > len(pool_tr):
        continue
    pte, dv, ft = fit(ntr)
    gaps, agrs = [], []
    for k, j in enumerate(te):
        pd_, qd_, u, c = POOL[j]
        cc, d = deploy(S, g, pd_, qd_, pte[k][:g.n_bus], pte[k][g.n_bus:])
        if cc < 1e8:
            gaps.append(100*(cc-c)/c); agrs.append(100*float((d["u"] == u).mean()))
    print(f"{ntr:>9d}{dv:>10.4f}{np.mean(gaps):>9.3f}%{np.mean(agrs):>8.1f}%{ft:>7.1f}s",
          flush=True)
print(f"\n  target for ~exact commitments: ||dV|| <= 0.08   (from 05's tolerance curve)")

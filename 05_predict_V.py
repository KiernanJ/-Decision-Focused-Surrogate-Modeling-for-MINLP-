"""Predict the linearisation point directly. No solver in the training loop.

What the diagnostics established:
  02: the oracle linearisation point V* deploys at -0.002% / 100% agreement,
      from ONE continuous solve, even with 40 of 54 generators fractional.
      Rounding costs nothing; the whole remaining gap is V error.
  03: the label-free fixed point (u free) caps at 1.039% and never converges.
  04: bootstrapping on the network's own commitment collapses to 7.599% --
      it re-linearises at a bad commitment and locks it in.

So V* is the right target and no label-free surrogate reaches it. But V* is
already in hand: one price() call on references we have paid for. That makes
this a regression, with NO solver in the training loop -- seconds instead of the
smoothed estimator's five solves per instance per step.

Two questions this answers:
  A. how accurately must V be predicted before the commitment degrades?
  B. what does a plain regression onto V* actually deliver?
"""
# %% setup
import os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, grid, sample, deploy
from socp import Socp

SEED, N, NTEST = 0, 32, 8
rng = np.random.default_rng(0); torch.manual_seed(0)
g, nl = grid()
inst = sample(g, N, seed=SEED)
REF = list(np.load(f"{HERE}/results/ref_{CASE}_s{SEED}_n{N}.npz",
                   allow_pickle=True)["ref"])
ok = [i for i in range(N) if REF[i] is not None]
tr = [i for i in ok if i < N-NTEST]; te = [i for i in ok if i >= N-NTEST]
S = Socp(g, nl, n_cuts=8)

# %% labels: V* = linearisation point at the reference commitment
t0 = time.time()
VS = {}
for i in ok:
    pr = S.price(*inst[i], REF[i]["u"].astype(float))
    VS[i] = np.r_[pr["vr"], pr["vi"]]
print(f"[labels] V* for {len(ok)} instances in {time.time()-t0:.1f}s "
      f"(reuses references already paid for)", flush=True)

# %% A. how much V error can the pipeline absorb?
print(f"\n{'A. tolerance to V error  (oracle V + gaussian noise)':52s}")
print(f"{'  ||dV||':>12s}{'gap':>10s}{'agree':>9s}")
for sd in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]:
    gaps, agrs, nrm = [], [], []
    for i in te:
        v = VS[i] + rng.normal(0, sd, 2*g.n_bus)
        nrm.append(np.linalg.norm(v - VS[i]))
        c, d = deploy(S, g, *inst[i], v[:g.n_bus], v[g.n_bus:])
        if c < 1e8:
            gaps.append(100*(c-REF[i]["cost"])/REF[i]["cost"])
            agrs.append(100*float((d["u"] == REF[i]["u"]).mean()))
    print(f"{np.mean(nrm):>12.4f}{np.mean(gaps):>9.3f}%{np.mean(agrs):>8.1f}%", flush=True)

# %% B. plain regression onto V*, no solver in the loop
X = torch.tensor(np.stack([np.concatenate(inst[i])/float(g.pd.sum()) for i in ok]),
                 dtype=torch.float32)
Y = torch.tensor(np.stack([VS[i] for i in ok]), dtype=torch.float32)
V0 = torch.tensor(np.tile(np.r_[np.ones(g.n_bus), np.zeros(g.n_bus)], (len(ok), 1)),
                  dtype=torch.float32)
idx = {v: k for k, v in enumerate(ok)}
itr = [idx[i] for i in tr]; ite = [idx[i] for i in te]
mu, sg = X[itr].mean(0), X[itr].std(0) + 1e-6

net = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.Tanh(),
                          torch.nn.Linear(256, 256), torch.nn.Tanh(),
                          torch.nn.Linear(256, 2*g.n_bus))
torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
t0 = time.time()
for ep in range(4000):
    p = V0[itr] + 0.15*torch.tanh(net((X[itr]-mu)/sg))
    loss = ((p - Y[itr])**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
tt = time.time()-t0
with torch.no_grad():
    pte = (V0[ite] + 0.15*torch.tanh(net((X[ite]-mu)/sg))).numpy()
err = np.linalg.norm(pte - Y[ite].numpy(), axis=1)
print(f"\nB. regression onto V*: trained in {tt:.1f}s "
      f"(NO solver in the loop), test ||dV|| = {err.mean():.4f}")

gaps, agrs, secs = [], [], []
for k, i in enumerate(te):
    t0 = time.time()
    c, d = deploy(S, g, *inst[i], pte[k][:g.n_bus], pte[k][g.n_bus:])
    secs.append(time.time()-t0)
    if c < 1e8:
        gaps.append(100*(c-REF[i]["cost"])/REF[i]["cost"])
        agrs.append(100*float((d["u"] == REF[i]["u"]).mean()))
print(f"\n{'method':36s}{'gap':>9s}{'agree':>9s}{'solves':>8s}")
print(f"{'flat V (baseline)':36s}{10.885:>8.3f}%{56.9:>8.1f}%{1:>8}")
print(f"{'learned V (smoothed, 20 min)':36s}{1.388:>8.3f}%{86.1:>8.1f}%{1:>8}")
print(f"{'label-free fixed point':36s}{1.039:>8.3f}%{88.4:>8.1f}%{12:>8}")
print(f"{'REGRESSION onto V*':36s}{np.mean(gaps):>8.3f}%{np.mean(agrs):>8.1f}%{1:>8}")
print(f"{'ORACLE V*':36s}{-0.002:>8.3f}%{100.0:>8.1f}%{1:>8}")
print(f"\n  deployment {np.mean(secs):.2f}s/instance   "
      f"QCAC iterative reference: 152s/instance")

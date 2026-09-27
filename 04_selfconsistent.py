"""A LABEL-FREE target that reaches the oracle: self-consistent re-linearisation.

02_ceiling: the oracle linearisation point deploys at -0.002% / 100% agreement.
It is the point reached by re-linearising with u FIXED at the optimal integer
commitment -- so it needs the answer, and cannot train anything.

03_fixedpoint: the obvious label-free substitute, the fixed point of the
relaxation with u FREE, caps at 1.039% and never converges (residual 3e-2 after
12 iterations). Its distance to the oracle is 1.05. Integrality is what the
oracle has and it lacks.

The network produces its OWN integer commitment at every step, so close the loop
on that instead:

    V -> solve relaxation -> round + reserve top-up -> z
      -> re-linearise with z FIXED -> V'   -> repeat

No labels. If z is correct, V' IS the oracle point. As z improves the target
sharpens itself. This measures whether the loop reaches the oracle, and how many
solves it needs -- which is the number the one-shot network must beat.
"""
# %% setup
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, RESERVE, grid, sample, deploy
from socp import Socp

SEED, N, NTEST = 0, 32, 8
g, nl = grid()
inst = sample(g, N, seed=SEED)
REF = list(np.load(f"{HERE}/results/ref_{CASE}_s{SEED}_n{N}.npz",
                   allow_pickle=True)["ref"])
te = [i for i in range(N-NTEST, N) if REF[i] is not None]
S = Socp(g, nl, n_cuts=8)


def round_up(uf, pd_):
    z = (uf > 0.5).astype(float)
    order = np.argsort(-uf)
    need = (1.0 + RESERVE) * float(pd_.sum())
    for k in order:
        if float(g.pmax @ z) >= need:
            break
        z[k] = 1.0
    return z


# %% the self-consistent loop
def selfconsistent(pd_, qd_, iters=8, rho=1e6):
    Vr, Vi = np.ones(g.n_bus), np.zeros(g.n_bus)
    seen, nsolve = [], 0
    for it in range(iters):
        r = S.solve(pd_, qd_, Vr, Vi, rho=rho); nsolve += 1
        if r is None:
            break
        z = round_up(r["u"], pd_)
        pr = S.price(pd_, qd_, z)                 # re-linearise with z FIXED
        if pr is None:
            break
        nsolve += pr.get("iters", 1)
        seen.append((tuple(z.astype(int)), pr["cost"], it+1, nsolve))
        if np.allclose(pr["vr"], Vr, atol=1e-6) and np.allclose(pr["vi"], Vi, atol=1e-6):
            break
        Vr, Vi = pr["vr"], pr["vi"]
    return Vr, Vi, seen, nsolve


# %% run
rows = []
for i in te:
    pd_, qd_ = inst[i]
    ref_c, ref_u = REF[i]["cost"], REF[i]["u"]
    t0 = time.time()
    Vr, Vi, seen, nsolve = selfconsistent(pd_, qd_)
    dt = time.time() - t0
    if not seen:
        continue
    best = min(seen, key=lambda s: s[1])           # best commitment it visited
    last = seen[-1]
    pr = S.price(pd_, qd_, ref_u.astype(float))
    rows.append(dict(
        gap_last=100*(last[1]-ref_c)/ref_c,
        gap_best=100*(best[1]-ref_c)/ref_c,
        agr_last=100*np.mean(np.array(last[0]) == ref_u),
        agr_best=100*np.mean(np.array(best[0]) == ref_u),
        its=len(seen), nsolve=nsolve, secs=dt,
        dist=float(np.linalg.norm(np.r_[Vr-pr["vr"], Vi-pr["vi"]]))))

# %% report
f = lambda k: np.nanmean([x[k] for x in rows])
print(f"\ncase118, seed {SEED}, {len(rows)} held-out instances")
print(f"{'method':34s}{'gap':>9s}{'agree':>9s}{'solves':>8s}")
print(f"{'flat V, one solve (baseline)':34s}{10.885:>8.3f}%{56.9:>8.1f}%{1:>8}")
print(f"{'learned V, one solve (smoothed)':34s}{1.388:>8.3f}%{86.1:>8.1f}%{1:>8}")
print(f"{'label-free fixed point (u free)':34s}{1.039:>8.3f}%{88.4:>8.1f}%{12:>8}")
print(f"{'SELF-CONSISTENT (last iterate)':34s}{f('gap_last'):>8.3f}%{f('agr_last'):>8.1f}%"
      f"{f('nsolve'):>8.1f}")
print(f"{'SELF-CONSISTENT (best visited)':34s}{f('gap_best'):>8.3f}%{f('agr_best'):>8.1f}%"
      f"{f('nsolve'):>8.1f}")
print(f"{'ORACLE V (needs the answer)':34s}{-0.002:>8.3f}%{100.0:>8.1f}%{1:>8}")
print(f"\n  outer iterations {f('its'):.1f}, wall {f('secs'):.2f}s, "
      f"distance to oracle point {f('dist'):.4f}")

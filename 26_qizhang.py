"""Qi-Zhang L1 projection restoration vs our continuous repair heuristic.

The hybrid-vehicle case study in the MINLP project folder restores feasibility
by PROJECTION (`qi_zhang_miqcqp_restoration`): solve a MIQCQP whose objective is
the L1 distance to the relaxation's fractional solution, subject to every
original constraint, with the decision variables restored to integer. Nearest
feasible integer point, guaranteed feasible, principled.

For that case study it is cheap -- 30 integer variables. For AC-UC it means a
MIQCQP with 54 binaries AND the AC physics, i.e. exactly the solve this method
exists to avoid. Our restoration stays continuous: round, top up to the reserve,
then repair up and down, pricing each candidate with the QCAC re-linearisation
loop.

The question a reviewer will ask is what the heuristic costs in solution
quality. Both start from the SAME learned relaxation solution, so only the
restoration differs.
"""
import os, sys, re, time
import numpy as np, torch
import gurobipy as gp
from gurobipy import GRB
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts
from socp import Socp
from acuc import build
exec(open(f"{HERE}/21_tiered_train.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

K, NTEST, NDOWN = 1, 48, 6
set_cuts(2*K)
g, nl = grid(); NG = g.n_gen; S = Socp(g, nl, n_cuts=2*K)
POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
             for i in range(int(m.group(2))) if REF[i] is not None]
rng = np.random.default_rng(0); perm = rng.permutation(len(POOL))
te = [int(i) for i in perm[:NTEST]]; tr = [int(i) for i in perm[NTEST:]]
TIERS = [np.arange(NG)]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
net = TieredNet(g.n_bus, NG, [NG])
net.load_state_dict(torch.load(f"{HERE}/results/net_tier_k1best.pt")); net.eval()
with torch.no_grad():
    VR, VI, CNT = net(Xn[te])
VR, VI, CNT = [t.numpy().astype(float) for t in (VR, VI, CNT)]


def qi_zhang(pd_, qd_, u_frac, tl=60.0, threads=3):
    """Nearest FEASIBLE INTEGER commitment to u_frac, in L1 -- the vehicle
    study's restoration, transplanted. Exact AC model, u binary, objective is
    distance not cost."""
    m, V, cost = build(g, nl, pd_, qd_, u_type=GRB.BINARY, exact=True,
                       reserve=RESERVE, threads=threads)
    w = m.addVars(NG, lb=0.0)
    for x in range(NG):
        m.addConstr(w[x] >= V["u"][x] - float(u_frac[x]))
        m.addConstr(w[x] >= float(u_frac[x]) - V["u"][x])
    m.setObjective(gp.quicksum(w[x] for x in range(NG)), GRB.MINIMIZE)
    m.Params.TimeLimit = tl
    m.optimize()
    if m.SolCount == 0:
        return np.clip(np.round(u_frac), 0, 1)       # the study's own fallback
    return np.array([round(V["u"][x].X) for x in range(NG)], dtype=float)


def ours(pd_, qd_, uf):
    z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0+RESERVE)*float(pd_.sum())
    for t_ in o:
        if float(g.pmax @ z) >= need:
            break
        z[t_] = 1.0
    best = None
    for _ in range(NG+1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"]); break
        off = [t_ for t_ in o if z[t_] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, c = best
    for t_ in [t2 for t2 in np.argsort(uf) if z[t2] > 0.5][:NDOWN]:
        wv = z.copy(); wv[t_] = 0.0
        if float(g.pmax @ wv) < need:
            continue
        pr = S.price(pd_, qd_, wv)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < c:
            z, c = wv, pr["cost"]
    return z


# PAIRED: keep per-instance results and compare only where BOTH succeeded.
# The unpaired run averaged ours over 48 instances and Qi-Zhang over 24 -- the
# projection's failures are exactly the hard instances, so dropping them
# flatters whichever method survives them.
REC = {}
for name, fn in [("ours (continuous repair)", "ours"),
                 ("Qi-Zhang L1 projection (MIQCQP)", "qz")]:
    REC[name] = {}
    for k, j in enumerate(te):
        pd_, qd_, u_s, c_s = POOL[j]
        A, b = tier_rows(CNT[k], TIERS, NG, 2*K)
        r = S.solve(pd_, qd_, VR[k], VI[k], rho=1e6, A=A, b=b, cut_cap=0.0)
        if r is None:
            continue
        t0 = time.time()
        z = ours(pd_, qd_, r["u"]) if fn == "ours" else qi_zhang(pd_, qd_, r["u"])
        if z is None:
            continue
        pr = S.price(pd_, qd_, z)
        if pr is None or pr["slack"] > 1e-4:
            continue
        REC[name][j] = (100*float((z.astype(int) != u_s).mean()),
                        100*(pr["cost"]-c_s)/c_s, time.time()-t0)
    print(f"  {name}: solved {len(REC[name])}/{len(te)}", flush=True)

common = sorted(set(REC["ours (continuous repair)"]) &
                set(REC["Qi-Zhang L1 projection (MIQCQP)"]))
print(f"\nPAIRED on the {len(common)} instances BOTH restorations solved")
print(f"{'restoration':34s}{'n':>4s}{'discrete':>11s}{'gap':>10s}{'s/inst':>9s}")
for name in REC:
    v = np.array([REC[name][j] for j in common])
    print(f"{name:34s}{len(common):>4d}{v[:,0].mean():>10.2f}%{v[:,1].mean():>9.3f}%"
          f"{v[:,2].mean():>8.2f}s", flush=True)
qz_only = sorted(set(REC["ours (continuous repair)"]) -
                 set(REC["Qi-Zhang L1 projection (MIQCQP)"]))
if qz_only:
    v = np.array([REC["ours (continuous repair)"][j] for j in qz_only])
    print(f"\n  on the {len(qz_only)} instances Qi-Zhang FAILED, ours scored "
          f"discrete {v[:,0].mean():.2f}%  gap {v[:,1].mean():+.3f}%")

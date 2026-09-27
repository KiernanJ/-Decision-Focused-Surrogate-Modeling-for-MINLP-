"""Is 2 rows the right cut, and is the advantage a tuning artifact?

Two questions a reviewer will ask about the cardinality cut:

  A. WHY TWO ROWS? A band is structurally 2 (a lower and an upper bound), so
     the count is not tuned -- unlike the old free-form head where K was a
     hyperparameter AND was selected on the test set. But "structural" is only
     a defence if more rows do not help. Tested here: lower-only, upper-only,
     the band, and the band plus a capacity row.

  B. IS THE GAIN A TUNING ARTIFACT? init_lo/init_hi were set by hand so the
     band would bind on both cases, not by a validation search. If the result
     only survives at one init, that is a finding the paper has to state.

Both are measured on the TRAINED net, varying only how its band is used, so
nothing here is refit and the comparison is clean.
"""
# %% setup
import os, sys, re, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp
exec(open(f"{HERE}/11_vcard.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

NTEST = int(os.environ.get("NTEST", "48"))
NDOWN = int(os.environ.get("NDOWN", "6"))
CKPT = os.environ.get("NET_VC", "net_c118_big_vcard.pt")
set_cuts(8); set_down(NDOWN)
g, nl = grid(); NG = g.n_gen; S = Socp(g, nl, n_cuts=8)

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
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
order = np.argsort(-g.pmax); cum = np.cumsum(g.pmax[order])
SRt = torch.tensor([float(np.searchsorted(cum, (1.0+RESERVE)*float(p.sum())) + 1)
                    for p, _, _, _ in POOL], dtype=torch.float32)
PGS = {j: (lambda r: r["pg"] if r else None)(
    S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))) for j in te}
net = VCardNet(g.n_bus, NG); net.load_state_dict(torch.load(f"{HERE}/results/{CKPT}"))
net.eval()
with torch.no_grad():
    vr, vi, LO, HI = net(Xn[te], SRt[te])
vr, vi, LO, HI = [t.numpy().astype(float) for t in (vr, vi, LO, HI)]
print(f"[setup] {CASE}: {len(POOL)} instances, {len(te)} held out, net {CKPT}")
print(f"[band ] learned [{LO.mean():.1f}, {HI.mean():.1f}]   "
      f"true n* mean {np.mean([u.sum() for _,_,u,_ in [POOL[j] for j in te]]):.1f}\n",
      flush=True)


def deploy(pd_, qd_, Vr, Vi, A, b):
    r = S.solve(pd_, qd_, Vr, Vi, rho=1e6, A=A, b=b)
    if r is None:
        return None
    uf = r["u"]; z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0+RESERVE)*float(pd_.sum())
    for k in o:
        if float(g.pmax @ z) >= need:
            break
        z[k] = 1.0
    best = None
    for _ in range(NG+1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"], pr["pg"]); break
        off = [k for k in o if z[k] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, cost, pg = best
    for k in [t for t in np.argsort(uf) if z[t] > 0.5][:NDOWN]:
        w = z.copy(); w[k] = 0.0
        if float(g.pmax @ w) < need:
            continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w, pr["cost"], pr["pg"]
    return z.astype(int), cost, pg


def rows_for(kind, k, shift=0.0):
    A, b = np.zeros((8, 2*NG)), np.ones(8)
    lo, hi = LO[k] + shift, HI[k] + shift
    if kind == "none":
        return None, None
    if kind in ("lower only", "band", "band + capacity"):
        A[0, NG:] = -1.0; b[0] = -lo
    if kind in ("upper only", "band", "band + capacity"):
        A[1, NG:] = 1.0; b[1] = hi
    if kind == "band + capacity":
        need = None
    return A, b


def run(name, kind, shift=0.0):
    D, C_, G_ = [], [], []
    for k, j in enumerate(te):
        pd_, qd_, u_s, c_s = POOL[j]
        A, b = rows_for(kind, k, shift)
        if kind == "band + capacity":
            A[2, NG:] = -g.pmax; b[2] = -(1.0+RESERVE)*float(pd_.sum())*1.05
        out = deploy(pd_, qd_, vr[k], vi[k], A, b)
        if out is None or PGS[j] is None:
            continue
        z, cost, pg = out
        D.append(100*float((z != u_s).mean()))
        C_.append(100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()))
        G_.append(100*(cost-c_s)/c_s)
    print(f"{name:30s}{len(G_):>4d}{np.mean(D):>10.2f}%{np.mean(C_):>12.2f}%"
          f"{np.mean(G_):>9.3f}%", flush=True)


print("A. which rows are doing the work")
print(f"{'cut rows':30s}{'n':>4s}{'discrete':>11s}{'continuous':>13s}{'gap':>10s}")
run("0  none", "none")
run("1  lower only  (sum u >= lo)", "lower only")
run("1  upper only  (sum u <= hi)", "upper only")
run("2  BAND  (the method)", "band")
run("3  band + capacity row", "band + capacity")

print(f"\nB. sensitivity to the band placement (shift BOTH bounds)")
print(f"{'shift [units]':30s}{'n':>4s}{'discrete':>11s}{'continuous':>13s}{'gap':>10s}")
for s_ in [-3, -2, -1, 0, 1, 2, 3]:
    run(f"   {s_:+d}", "band", float(s_))

"""NN proxy baseline: predict the commitment DIRECTLY, skipping the optimiser.

The question this answers is whether the solver is needed at all. If a network
maps (pd, qd) -> u well enough on its own, then predicting a linearisation point
and a cut to steer a relaxation is wasted machinery.

Everything downstream is held identical -- rounding, reserve top-up, upward and
downward repair, and AC pricing -- so the ONLY difference is where the
fractional commitment comes from:

    ours    (pd,qd) -> net -> (V, cut) -> SOCP relaxation -> u_frac -> restore
    proxy   (pd,qd) -> net --------------------------------> u_frac -> restore

Two proxies, because the training signal has to be controlled:

  proxy (supervised)   trained on the REFERENCE commitments u*, i.e. given
                       labels our method never sees. Deliberately advantaged:
                       if our self-supervised method still wins, the optimiser
                       is doing real work rather than compensating for a weak
                       baseline.
  proxy (self-sup.)    same smoothed gradient on deployment cost that our
                       method uses. The like-for-like comparison.
"""
# %% setup
import os, sys, re, time, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp

NTEST = int(os.environ.get("NTEST", "24"))
NDOWN = int(os.environ.get("NDOWN", "6"))
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
print(f"[setup] {CASE}: {len(POOL)} instances, {len(tr)} train / {len(te)} test, "
      f"downward repair {NDOWN}", flush=True)

base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
U = torch.tensor(np.stack([u.astype(float) for _, _, u, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
Xn = (X - mu)/sg


# %% shared restoration -- identical for every arm
def restore_price(pd_, qd_, uf):
    z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0 + RESERVE)*float(pd_.sum())
    for k in o:
        if float(g.pmax @ z) >= need:
            break
        z[k] = 1.0
    best = None
    for _ in range(NG + 1):
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


# %% supervised proxy: (pd,qd) -> u,  trained on the reference commitments
def train_proxy_supervised(steps=8000, lr=2e-3, hidden=256):
    net = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, NG))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    for _ in range(steps):
        loss = lossf(net(Xn[tr]), U[tr])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        acc_tr = ((net(Xn[tr]) > 0).float() == U[tr]).float().mean().item()
        acc_te = ((net(Xn[te]) > 0).float() == U[te]).float().mean().item()
    print(f"[proxy-sup] trained {time.time()-t0:.1f}s | raw commitment accuracy "
          f"train {100*acc_tr:.1f}%  test {100*acc_te:.1f}%", flush=True)
    return net


# %% evaluate
PGS = {}
for j in te:
    r = S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))
    PGS[j] = r["pg"] if r else None


def evaluate(name, uf_of):
    rows = []
    for k, j in enumerate(te):
        pd_, qd_, u_s, c_s = POOL[j]
        t0 = time.time()
        uf = uf_of(k, j, pd_, qd_)
        if uf is None:
            continue
        out = restore_price(pd_, qd_, uf)
        if out is None or PGS[j] is None:
            continue
        z, cost, pg = out
        rows.append(dict(disc=100*float((z != u_s).mean()),
                         cont=100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()),
                         gap=100*(cost-c_s)/c_s, secs=time.time()-t0))
    d = lambda k: np.mean([x[k] for x in rows]) if rows else np.nan
    print(f"{name:34s}{len(rows):>4d}{d('disc'):>10.2f}%{d('cont'):>12.2f}%"
          f"{d('gap'):>9.3f}%{d('secs'):>8.2f}s", flush=True)
    return rows


print(f"\n{'arm':34s}{'n':>4s}{'discrete':>11s}{'continuous':>13s}{'gap':>10s}{'s/inst':>9s}")

# relaxation-based arms, for context
exec(open(f"{HERE}/11_vcard.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())
order = np.argsort(-g.pmax); cum = np.cumsum(g.pmax[order])
SRt = torch.tensor([float(np.searchsorted(cum, (1.0+RESERVE)*float(p.sum())) + 1)
                    for p, _, _, _ in POOL], dtype=torch.float32)

evaluate("relaxation only (flat V, no cut)",
         lambda k, j, p, q: (lambda r: r["u"] if r else None)(
             S.solve(p, q, np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e6)))

proxy = train_proxy_supervised()
with torch.no_grad():
    PU = torch.sigmoid(proxy(Xn[te])).numpy()
evaluate("NN proxy (SUPERVISED on u*)", lambda k, j, p, q: PU[k])

for tag, ckpt, cut in [("ours: V only", os.environ.get("NET_V", "net_d6_v.pt"), False),
                       ("ours: V + cardinality cut",
                        os.environ.get("NET_VC", "net_d6_vcard.pt"), True)]:
    pth = f"{HERE}/results/{ckpt}"
    if not os.path.exists(pth):
        print(f"{tag:34s}   -- {ckpt} missing --"); continue
    net = VCardNet(g.n_bus, NG); net.load_state_dict(torch.load(pth)); net.eval()
    with torch.no_grad():
        vr, vi, lo, hi = net(Xn[te], SRt[te])
    vr, vi, lo, hi = [t.numpy().astype(float) for t in (vr, vi, lo, hi)]

    def uf_of(k, j, p, q, vr=vr, vi=vi, lo=lo, hi=hi, cut=cut):
        A, b = (card_rows(lo[k], hi[k], NG) if cut else (None, None))
        r = S.solve(p, q, vr[k], vi[k], rho=1e6, A=A, b=b)
        return r["u"] if r else None
    evaluate(tag, uf_of)

"""case118 component ablation. Read-only on the repo; outputs to ./out.

Every arm uses the paper's case118 deployment (confidence order, NDOWN=25 at test)
and differs only in how the fractional commitment uf is produced:
  flat_nocut   uncut relaxation at flat V (no learning)
  pred_nocut   relaxation at the network's V, cuts disabled
  pred_cut     relaxation at the network's V with its cuts (the paper's method)
  flat_cut     relaxation at flat V with the network's cuts (anchor point, rows active)
  proxy        supervised NN proxy (exact copy of acuc/06_nn_proxy.py), restore()
Thresholds: 0.5 ('shipped') or a per-generator threshold fitted LABEL-FREE on
training deployed cost with the procedure of acuc/03_threshold_fit.py
(32 train instances, NDOWN=6, 2 shards x 24 configs, same RNG seeds).

  python abl.py fit  ARM SEED SHARD
  python abl.py eval ARM SEED
"""
import os, sys, json, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "audit"))
import _common
from _common import setup, load_net, RHO
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

mode, arm, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
E = setup("case118")
g, S, POOL, Xn, nl = E["g"], E["S"], E["POOL"], E["Xn"], E["nl"]
te, tr = E["te"], E["tr"]
from pipeline import set_down, deploy
from restore import restore
from cutnet_k import rows_torch
NB, NG = g.n_bus, g.n_gen
V1, V0 = np.ones(NB), np.zeros(NB)
zz = lambda v: (v - v.mean())/(v.std() + 1e-12)
FEAT = np.stack([zz(g.pmax), zz(g.c1 + g.c2*g.pmax), zz(nl)])
thr_of = lambda c: np.clip(0.5 + c[0] + c[1]*FEAT[0] + c[2]*FEAT[1] + c[3]*FEAT[2], 0.05, 0.95)

_RLX = {}
def rlx(j):
    if j not in _RLX:
        _RLX[j] = S.solve(POOL[j][0], POOL[j][1], V1, V0, rho=RHO)
    return _RLX[j]


def proxy_net(seed):
    """Exact replica of acuc/06_nn_proxy.py training (seed, 400 full-batch epochs)."""
    torch.manual_seed(seed)
    Y = torch.tensor(np.stack([POOL[i][2] for i in tr]), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(2*NB, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, NG))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lf = nn.MSELoss()
    for _ in range(400):
        opt.zero_grad(); loss = lf(torch.sigmoid(net(Xn[tr])), Y); loss.backward(); opt.step()
    net.eval()
    return net


def make_deployer(arm, seed, ids):
    """Returns f(k, j, thr, ndown) -> (cost, u) for instance ids[k]."""
    if arm == "proxy":
        pn = proxy_net(seed)
        with torch.no_grad():
            UF = torch.sigmoid(pn(Xn[ids])).numpy().astype(float)
        def f(k, j, thr, ndown):
            c, d = restore(S, g, POOL[j][0], POOL[j][1], UF[k], thr=0.5 if thr is None else thr,
                           order=None, ndown=ndown)
            return (c, d["u"]) if d is not None else (np.inf, None)
        return f
    if arm == "flat_nocut":
        def f(k, j, thr, ndown):
            set_down(ndown)
            c, d = deploy(S, g, POOL[j][0], POOL[j][1], V1, V0, rho=RHO, thr=thr)
            return (c, d["u"]) if d is not None and c < 1e8 else (np.inf, None)
        return f
    net = load_net("case118", seed, "net", g)       # case118: selected epoch == last epoch
    W = torch.tensor(np.stack([np.r_[rlx(j)["pg"], rlx(j)["u"]] for j in ids]), dtype=torch.float32)
    with torch.no_grad():
        Vr_t, Vi_t, a, al, a2, d_, d2, _ = net(Xn[ids])
        A_t, b_t = rows_torch(a, al, a2, d_, d2, W)
    Vr, Vi = Vr_t.numpy().astype(float), Vi_t.numpy().astype(float)
    A, b = A_t.numpy().astype(float), b_t.numpy().astype(float)
    def f(k, j, thr, ndown):
        set_down(ndown)
        vr, vi = (Vr[k], Vi[k]) if arm.startswith("pred") else (V1, V0)
        kw = dict(A=A[k], b=b[k]) if arm.endswith("_cut") else {}
        c, d = deploy(S, g, POOL[j][0], POOL[j][1], vr, vi, rho=RHO, thr=thr, **kw)
        return (c, d["u"]) if d is not None and c < 1e8 else (np.inf, None)
    return f


if mode == "fit":
    shard = int(sys.argv[4])
    ids = tr[:32]
    f = make_deployer(arm, seed, ids)
    SCALE = np.array([rlx(j)["cost"] for j in ids])          # label-free normaliser
    def score(c):
        th = None if c is None else thr_of(c)
        tot = 0.0
        for k, j in enumerate(ids):
            cst, u = f(k, j, th, 6)
            tot += 10.0 if u is None else cst/SCALE[k]
        return tot/len(ids)
    t0 = time.time(); base = score(None)
    rng = np.random.default_rng(7000 + 97*seed + shard)
    best = (base, np.zeros(4))
    for t in range(24):
        c = np.array([0.5*(-0.30+0.05), 0, 0, 0]) if (t == 0 and shard == 0) else \
            np.r_[rng.uniform(-0.30, 0.05), rng.normal(0, 0.12, 3)]
        sc = score(c)
        if sc < best[0]:
            best = (sc, c)
    json.dump(dict(arm=arm, seed=seed, shard=shard, base=base, best=best[0], c=best[1].tolist(),
                   secs=time.time()-t0), open(f"{OUT}/fit_{arm}_s{seed}_{shard}.json", "w"))
    print(f"fit {arm} s{seed} sh{shard}: base {base:.6f} best {best[0]:.6f} {time.time()-t0:.0f}s", flush=True)

elif mode == "eval":
    ids = te
    f = make_deployer(arm, seed, ids)
    fits = [json.load(open(p)) for p in sorted(glob.glob(f"{OUT}/fit_{arm}_s{seed}_*.json"))]
    thr_sets = {"0.5": None}
    if fits:
        win = min(fits, key=lambda d: d["best"])
        thr_sets["fitted"] = thr_of(np.array(win["c"]))
    out = {}
    for tag, th in thr_sets.items():
        rows = []
        for k, j in enumerate(ids):
            cst, u = f(k, j, th, 25)
            if u is None:
                rows.append(dict(j=j, fail=True)); continue
            rows.append(dict(j=j, gap=100*(cst-POOL[j][3])/POOL[j][3],
                             disc=100*float((np.asarray(u) != POOL[j][2]).mean()),
                             n_on=int(np.sum(u))))
        ok = [r for r in rows if "gap" in r]
        G = np.array([r["gap"] for r in ok])
        print(f"eval {arm:10s} s{seed} thr={tag:6s} n={len(ok)}/{len(ids)} |gap| {np.abs(G).mean():.4f}% "
              f"signed {G.mean():+.4f}%  disc {np.mean([r['disc'] for r in ok]):.2f}%", flush=True)
        out[tag] = rows
    json.dump(out, open(f"{OUT}/eval_{arm}_s{seed}.json", "w"))

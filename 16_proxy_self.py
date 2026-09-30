"""The decisive baseline: an NN proxy trained WITHOUT labels, like ours.

The supervised proxy beats our method (case118 0.063% vs 0.103%; case300
0.385% vs 0.454%) -- but it is trained on reference commitments, one QCAC solve
per instance, which our method never uses. That comparison therefore says
nothing about the contribution; it only says labels are valuable.

The like-for-like question is whether a proxy can learn the commitment directly
from the SAME signal ours uses: the deployment cost, through the same smoothed
gradient, same epochs, same restoration. If it can, the optimiser in the loop is
redundant. If it cannot, then keeping the relaxation is what makes label-free
learning possible, and that is the claim.

Arms, all with identical restoration (round, reserve top-up, upward and
downward repair) and identical AC pricing:

  proxy-self   (pd,qd) -> net -> u          trained on deployment cost, no labels
  proxy-sup    (pd,qd) -> net -> u          trained on u*  (labels: advantaged)
  ours         (pd,qd) -> net -> (V,cut) -> relaxation -> u   no labels
"""
# %% setup
import os, sys, re, time, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down, _init, _dep
from socp import Socp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sigma", type=float, default=0.15)   # on logits
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=48)
    ap.add_argument("--down", type=int, default=6)
    a = ap.parse_args()
    import multiprocessing as mp
    torch.manual_seed(0); rng = np.random.default_rng(0)
    set_cuts(8); set_down(a.down)
    g, nl = grid(); NG = g.n_gen

    POOL = []
    for f in sorted(os.listdir(f"{HERE}/results")):
        m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
        if not m:
            continue
        REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
        inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
        POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
                 for i in range(int(m.group(2))) if REF[i] is not None]
    perm = rng.permutation(len(POOL))
    te = [int(i) for i in perm[:a.n_test]]; tr = [int(i) for i in perm[a.n_test:]]
    base = float(g.pd.sum())
    X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    print(f"[setup] {CASE}: {len(POOL)} instances, {len(tr)} train / {len(te)} test",
          flush=True)


    # The proxy's u IS the fractional commitment handed to restoration, so it
    # rides the same _dep worker: V is flat and no cut is imposed. We pass the
    # proxy's u by fixing the relaxation to it -- u_lo = u_hi = u_proxy -- so
    # the solve only finds the dispatch, never re-decides the commitment.
    def dep_jobs(idx, U):
        return [(k, POOL[j][0], POOL[j][1], np.ones(g.n_bus), np.zeros(g.n_bus),
                 None, None, U[k]) for k, j in enumerate(idx)]

    S = Socp(g, nl, n_cuts=8)

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
        for k in [t for t in np.argsort(uf) if z[t] > 0.5][:a.down]:
            w = z.copy(); w[k] = 0.0
            if float(g.pmax @ w) < need:
                continue
            pr = S.price(pd_, qd_, w)
            if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
                z, cost, pg = w, pr["cost"], pr["pg"]
        return z, cost, pg

    def cost_of(args):
        j, uf = args
        out = restore_price(POOL[j][0], POOL[j][1], uf)
        return (j, 1e9) if out is None else (j, out[1])

    # A fork Pool snapshots the parent at fork time, so EVERY function a
    # worker runs must already be defined. Creating it earlier made all 12
    # workers die instantly with "Can't get attribute 'cost_of'" while the
    # parent blocked forever on results that were never coming -- 1h51m at
    # 0% CPU. Total CPU is the cheap check that catches this.
    pool = mp.get_context("fork").Pool(a.procs, initializer=_init)

    # Label-free per-instance scale. The earlier version ran a FULL deployment
    # per instance (a price() loop of up to 25 solves, plus 6 downward trials
    # -- ~175 solves) purely to set a gradient magnitude. The relaxation cost
    # at flat V is one solve, equally label-free, and serves the same purpose.
    SCALE = {}
    for j in range(len(POOL)):
        r = S.solve(POOL[j][0], POOL[j][1], np.ones(g.n_bus), np.zeros(g.n_bus),
                    rho=1e6)
        SCALE[j] = float(r["cost"]) if r else 1e5
    print(f"[scale] label-free relaxation cost, mean {np.mean(list(SCALE.values())):,.0f}",
          flush=True)

    net = torch.nn.Sequential(torch.nn.Linear(2*g.n_bus, 256), torch.nn.SiLU(),
                              torch.nn.Linear(256, 256), torch.nn.SiLU(),
                              torch.nn.Linear(256, NG))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    print(f"[proxy-self] {sum(p.numel() for p in net.parameters()):,} params | "
          f"smoothed gradient on deployment cost, NO labels", flush=True)
    best = None
    for ep in range(a.epochs):
        t0 = time.time()
        bidx = [int(i) for i in rng.permutation(tr)]
        Z = net(Xn[bidx])                       # logits
        Zd = Z.detach().numpy(); nb, dY = Z.shape
        E = rng.standard_normal((a.pairs, nb, dY))
        jobs, key = [], []
        for k, j in enumerate(bidx):
            jobs.append((j, 1/(1+np.exp(-Zd[k])))); key.append(("0", k))
            for p in range(a.pairs):
                for s_, sgn in ((f"+{p}", 1), (f"-{p}", -1)):
                    jobs.append((j, 1/(1+np.exp(-(Zd[k] + sgn*a.sigma*E[p, k])))))
                    key.append((s_, k))
        res = list(pool.imap(cost_of, jobs))
        C = np.array([r[1] for r in res], float)
        sc = np.array([SCALE[j] for j in bidx])
        Gest = np.zeros((nb, dY)); base_c = np.full(nb, np.nan)
        for t, (tag, k) in enumerate(key):
            if tag == "0":
                base_c[k] = C[t]
            elif tag.startswith("+"):
                p = int(tag[1:]); cp_, cm_ = C[t], C[t+1]
                if cp_ < 1e8 and cm_ < 1e8:
                    Gest[k] += (cp_-cm_)/(2*a.sigma*sc[k])*E[p, k]/a.pairs
        loss = (Z*torch.tensor(Gest, dtype=torch.float32)).sum()/nb
        opt.zero_grad(); loss.backward(); opt.step()
        ok = base_c < 1e8
        sel = float(np.mean(base_c[ok]/sc[ok])) if ok.any() else np.inf
        refc = np.array([POOL[j][3] for j in bidx])
        gp_ = 100*float(np.mean((base_c[ok]-refc[ok])/refc[ok])) if ok.any() else np.nan
        if best is None or sel < best[0]:
            best = (sel, {k: v.clone() for k, v in net.state_dict().items()})
        if ep % 5 == 0 or ep == a.epochs-1:
            print(f"[ep {ep:3d}] train gap {gp_:+7.3f}%  fail {int((~ok).sum())}/{nb}"
                  f"  {time.time()-t0:5.1f}s", flush=True)
    net.load_state_dict(best[1])

    PGS = {j: (lambda r: r["pg"] if r else None)(
        S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))) for j in te}
    with torch.no_grad():
        PU = torch.sigmoid(net(Xn[te])).numpy()
    rows = []
    for k, j in enumerate(te):
        pd_, qd_, u_s, c_s = POOL[j]
        out = restore_price(pd_, qd_, PU[k])
        if out is None or PGS[j] is None:
            continue
        z, cost, pg = out
        rows.append((100*float((z != u_s).mean()),
                     100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()),
                     100*(cost-c_s)/c_s))
    R = np.array(rows)
    print(f"\n{'='*66}\n[TEST] {CASE}, {len(R)} held out")
    print(f"  NN proxy (SELF-SUPERVISED, no labels):  "
          f"discrete {R[:,0].mean():.2f}%  continuous {R[:,1].mean():.2f}%  "
          f"gap {R[:,2].mean():+.3f}%\n{'='*66}", flush=True)
    torch.save(net.state_dict(), f"{HERE}/results/net_proxyself_{CASE}.pt")
    pool.close()

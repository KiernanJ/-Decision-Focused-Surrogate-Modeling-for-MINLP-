"""
Train the surrogate: predict BOTH the cuts AND the linearisation point, so ONE
continuous solve plus feasibility restoration replaces QCAC's iteration.

Reference is QCAC iterative (Gurobi, 54 binaries), which reproduces the global
MINLP optimum exactly on the instances where both were run. The global solver is
retired -- 600 s per instance, and we already have its answer on file.

Everything the surrogate does goes through src/socp.py (Clarabel, 0.08 s a
solve). Gurobi keeps only the reference.
"""
# %% imports
import os, sys, time, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from acopf_data import load
from qcac import no_load_cost
from socp import Socp
from netw import SurrogateNet

from pipeline import (CASE, RESERVE, grid, sample, deploy, set_cuts,
                      _init, _ref, _dep)

HERE = os.path.dirname(os.path.abspath(__file__))

# %% main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--cuts", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run1")
    # Exactly one head differs between arms, so the 7.6-point gain can be
    # attributed. "both" is the abstract's method; "cuts" freezes the
    # linearisation point at flat V (v_scale=0, head_v gets no gradient);
    # "v" predicts the linearisation point with no cuts at all.
    ap.add_argument("--arm", choices=["both", "cuts", "v"], default="both")
    a = ap.parse_args()
    set_cuts(a.cuts)

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    torch.manual_seed(a.seed)

    g, nl = grid()
    N = a.n_train + a.n_test
    inst = sample(g, N, seed=a.seed)
    tr, te = list(range(a.n_train)), list(range(a.n_train, N))
    print(f"[setup] {CASE}: {g.n_bus} bus, {g.n_gen} gen, {g.n_branch} branch | "
          f"{a.n_train} train / {a.n_test} test | K={a.cuts} cuts", flush=True)
    tot = np.array([float(p.sum()) for p, _ in inst])
    print(f"[setup] demand {tot.min():.1f}-{tot.max():.1f} pu, CV {100*tot.std()/tot.mean():.2f}%",
          flush=True)

    pool = ctx.Pool(a.procs, initializer=_init)

    # %% references (cached)
    rp = f"{HERE}/results/ref_{CASE}_s{a.seed}_n{N}.npz"
    if os.path.exists(rp):
        z = np.load(rp, allow_pickle=True); REF = list(z["ref"])
        print(f"[ref ] cached {rp}", flush=True)
    else:
        t0 = time.time(); REF = [None]*N
        for i, r in pool.imap_unordered(_ref, [(i, *inst[i]) for i in range(N)]):
            REF[i] = r
            print(f"[ref ] {sum(x is not None for x in REF):2d}/{N}", flush=True)
        np.savez(rp, ref=np.array(REF, dtype=object))
        print(f"[ref ] {time.time()-t0:.0f}s total", flush=True)
    ok = [i for i in range(N) if REF[i] is not None]
    uniq = len({tuple(REF[i]["u"]) for i in ok})
    print(f"[ref ] {len(ok)}/{N} solved, {uniq} distinct commitments, "
          f"mean {np.mean([REF[i]['secs'] for i in ok]):.1f}s, "
          f"mean units {np.mean([REF[i]['u'].sum() for i in ok]):.1f}", flush=True)
    tr = [i for i in tr if REF[i] is not None]
    te = [i for i in te if REF[i] is not None]

    # %% network inputs
    base = float(g.pd.sum())
    X = torch.tensor(np.stack([np.concatenate([p, q])/base for p, q in inst]),
                     dtype=torch.float32)
    Vr0 = torch.ones(N, g.n_bus); Vi0 = torch.zeros(N, g.n_bus)

    # anchor for the cut right-hand sides: the uncut relaxation at flat V
    S0 = Socp(g, nl, n_cuts=a.cuts)
    W = np.zeros((N, 2*g.n_gen))
    for i in range(N):
        r = S0.solve(*inst[i], np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e6)
        W[i] = np.concatenate([r["pg"], r["u"]])
    Wt = torch.tensor(W, dtype=torch.float32)

    # %% BASELINE -- the falsifying control, run BEFORE training
    def evaluate(idx, net=None, tag=""):
        jobs = []
        for i in idx:
            if net is None:
                jobs.append((i, *inst[i], np.ones(g.n_bus), np.zeros(g.n_bus), None, None))
            else:
                with torch.no_grad():
                    vr, vi, A, b = net(X[i:i+1], Vr0[i:i+1], Vi0[i:i+1], Wt[i:i+1])
                jobs.append((i, *inst[i], vr[0].numpy().astype(float),
                             vi[0].numpy().astype(float),
                             A[0].numpy().astype(float) if USE_CUTS else None,
                             b[0].numpy().astype(float) if USE_CUTS else None))
        out = {}
        for i, c, u, s in pool.imap_unordered(_dep, jobs):
            out[i] = (c, u, s)
        gaps, agr, secs, nfail = [], [], [], 0
        for i in idx:
            c, u, s = out[i]; secs.append(s)
            if c >= 1e8 or u is None:
                nfail += 1; continue
            gaps.append(100*(c - REF[i]["cost"])/REF[i]["cost"])
            agr.append(100*float((u == REF[i]["u"]).mean()))
        return dict(tag=tag, gap=float(np.mean(gaps)) if gaps else float("nan"),
                    gap_max=float(np.max(gaps)) if gaps else float("nan"),
                    agree=float(np.mean(agr)) if agr else float("nan"),
                    secs=float(np.mean(secs)), nfail=nfail, n=len(idx))

    t0 = time.time()
    USE_CUTS = True
    B = evaluate(te, None, "flat V, no cuts")
    print(f"\n[BASE] {B['tag']}: gap {B['gap']:+.3f}% (max {B['gap_max']:+.3f}%)  "
          f"agree {B['agree']:.1f}%  {B['secs']:.2f}s/inst  fail {B['nfail']}/{B['n']}"
          f"   [{time.time()-t0:.0f}s]\n", flush=True)

    # %% TRAIN -- smoothed (derivative-free) gradient on the network outputs.
    # Rounding makes the deployment cost piecewise constant in the outputs, so
    # there is no pathwise derivative to take; antithetic Gaussian smoothing
    # gives an unbiased gradient of the smoothed objective.
    net = SurrogateNet(g.n_bus, g.n_gen, n_cuts=a.cuts)
    USE_CUTS = a.arm in ("both", "cuts")
    if a.arm == "cuts":
        net.v_scale = 0.0                       # linearisation point stays flat
    params = [q for nm, q in net.named_parameters()
              if not (a.arm == "cuts" and nm.startswith("head_v"))
              and not (a.arm == "v" and (nm.startswith("head_dir")
                                         or nm.startswith("head_rhs")))]
    opt = torch.optim.Adam(params, lr=a.lr)
    print(f"[net ] {net.n_params():,} params | smoothed grad, sigma={a.sigma}, "
          f"pairs={a.pairs} | ARM={a.arm} "
          f"(V {'learned' if a.arm in ('both','v') else 'FLAT'}, "
          f"cuts {'on' if USE_CUTS else 'OFF'})", flush=True)
    hist, best = [], None
    for ep in range(a.epochs):
        t0 = time.time()
        perm = np.random.permutation(tr)
        vr, vi, A, b = net(X[perm], Vr0[perm], Vi0[perm], Wt[perm])
        Y = torch.cat([vr, vi, A.reshape(len(perm), -1), b], 1)   # all outputs
        Yd = Y.detach().numpy()
        nb, dY = len(perm), Y.shape[1]
        E = np.random.randn(a.pairs, nb, dY)
        jobs, key = [], []

        def pack(i_local, yv):
            i = int(perm[i_local]); nb_ = g.n_bus; G = g.n_gen
            return (i, *inst[i], yv[:nb_], yv[nb_:2*nb_],
                    yv[2*nb_:2*nb_+a.cuts*2*G].reshape(a.cuts, 2*G) if USE_CUTS else None,
                    yv[2*nb_+a.cuts*2*G:] if USE_CUTS else None)
        for j in range(nb):
            jobs.append(pack(j, Yd[j])); key.append(("0", 0, j))
            for p in range(a.pairs):
                jobs.append(pack(j, Yd[j] + a.sigma*E[p, j])); key.append(("+", p, j))
                jobs.append(pack(j, Yd[j] - a.sigma*E[p, j])); key.append(("-", p, j))
        res = list(pool.imap(_dep, jobs))
        C = np.array([r[1] for r in res], float)
        base_c = np.array([C[k] for k, kk in enumerate(key) if kk[0] == "0"])
        scale = np.array([REF[int(perm[j])]["cost"] for j in range(nb)])
        Gest = np.zeros((nb, dY))
        for k, (s, p, j) in enumerate(key):
            if s == "+":
                cp_, cm_ = C[k], C[k+1]
                if cp_ < 1e8 and cm_ < 1e8:
                    Gest[j] += (cp_-cm_)/(2*a.sigma*scale[j]) * E[p, j] / a.pairs
        loss = (Y * torch.tensor(Gest, dtype=torch.float32)).sum() / nb
        opt.zero_grad(); loss.backward(); opt.step()
        gp_ = 100*np.mean([(base_c[j]-scale[j])/scale[j] for j in range(nb)
                           if base_c[j] < 1e8])
        nf = int((base_c >= 1e8).sum())
        hist.append(dict(ep=ep, train_gap=float(gp_), nfail=nf, secs=time.time()-t0))
        if best is None or gp_ < best[0]:
            best = (gp_, {k: v.clone() for k, v in net.state_dict().items()})
        if ep % 5 == 0 or ep == a.epochs-1:
            print(f"[ep {ep:3d}] train gap {gp_:+7.3f}%  fail {nf}/{nb}  "
                  f"|g| {np.abs(Gest).mean():.2e}  {time.time()-t0:5.1f}s", flush=True)

    net.load_state_dict(best[1])
    T = evaluate(te, net, "learned V + cuts")
    print(f"\n{'='*66}\n[TEST] reference = QCAC iterative (== global MINLP)")
    print(f"  {'baseline  flat V, no cuts':32s} gap {B['gap']:+7.3f}%  "
          f"agree {B['agree']:5.1f}%  fail {B['nfail']}/{B['n']}")
    print(f"  {'learned   arm='+a.arm:32s} gap {T['gap']:+7.3f}%  "
          f"agree {T['agree']:5.1f}%  fail {T['nfail']}/{T['n']}")
    print(f"  best train gap {best[0]:+.3f}%\n{'='*66}", flush=True)
    json.dump(dict(base=B, learned=T, hist=hist, args=vars(a)),
              open(f"{HERE}/results/train_{a.tag}.json", "w"), indent=1)
    torch.save(net.state_dict(), f"{HERE}/results/net_{a.tag}.pt")
    pool.close()

# %% SUPERVISED NN PROXY under the current protocol (2026-10-09). Labels = reference commitments
#    (converged sequential QCAC) of the TRAINING instances; validation labels pick the training step.
#    MLP 2x256 SiLU (same size as our nets), BCE, Adam 1e-3. Deploy: round at 0.5 -> strict pricing with
#    upward merit repair (reserve top-up). Scored on the same test instances / cached strict references
#    as OUR METHOD (learned V + learned cuts, results/full_method_{case}.pkl), paired.
#    Usage: python 36_nn_proxy_sup.py <case> [workers]
import os, sys, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE = sys.argv[1]; NW = int(sys.argv[2]) if len(sys.argv) > 2 else 11
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0; SUF = "" if SEED == 0 else f"_s{SEED}"
OURS = "learned V @1e5 + cuts"


def main():
    import combopt_ac, data2
    from scipy.stats import wilcoxon
    from pipeline import grid
    g, nl = grid(CASE); G = g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    RP = data2.refs(CASE, os.path.join(HERE, "results", "pool2"))
    ltr = [i for i in tr if data2.converged(RP.get(i))]; lva = [i for i in va if data2.converged(RP.get(i))]
    if not lva:   # no validation labels computed (PGLib): hold out 20% of the labelled training set for early stopping
        k = max(1, len(ltr)//5); lva, ltr = ltr[-k:], ltr[:-k]
        print(f"[proxy] no validation labels -> early-stopping hold-out = {len(lva)} of the labelled training instances", flush=True)
    Ytr = torch.tensor(np.stack([RP[i]["u"] for i in ltr]), dtype=torch.float32); Yva = torch.tensor(np.stack([RP[i]["u"] for i in lva]), dtype=torch.float32)
    torch.manual_seed(SEED)
    m = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
    o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); best = (np.inf, None, 0); t0 = time.time()
    bce = nn.functional.binary_cross_entropy_with_logits
    for it in range(3000):
        o.zero_grad(); l = bce(m(Xn[ltr]), Ytr); l.backward(); o.step()
        if it % 25 == 0:
            with torch.no_grad(): v = float(bce(m(Xn[lva]), Yva))
            if v < best[0]: best = (v, {k: t.clone() for k, t in m.state_dict().items()}, it)
    m.load_state_dict(best[1]); m.eval(); train_s = time.time() - t0
    print(f"[proxy] supervised on {len(ltr)} reference-labelled training instances; best val BCE {best[0]:.4f} at step {best[2]}; train {train_s:.1f}s", flush=True)
    F = pickle.load(open(os.path.join(HERE, "results", f"vnet_layer_{CASE}{SUF}.pkl"), "rb"))   # OUR METHOD: label-free learned V + learned cuts
    SR = data2.strict_refs(CASE); tid = F["tid"]; J = {i: SR[i][0] for i in tid}
    t1 = time.time()
    with torch.no_grad(): P = torch.sigmoid(m(Xn[tid])).numpy()
    t_fwd = (time.time() - t1)/len(tid)
    with mp.get_context("spawn").Pool(NW, initializer=combopt_ac.init, initargs=(CASE, True)) as pool:
        PX = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, (P[k] > 0.5).astype(float)) for k, i in enumerate(tid)]))
        # like-for-like timing (Kiernan audit): our time = MIQCQP solve + the SAME strict pricing step as the proxy
        OP = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, F["sol"][i]["u"]) for i in tid if F["sol"].get(i)]))
    rows = {}
    for name, get in (("OUR METHOD (learned V + cuts)", lambda i: (F["pr"].get(i), F["sol"][i]["t_mip"] + OP[i]["secs"] if OP.get(i) else np.nan)),
                      ("supervised NN proxy", lambda i: ((PX[i]["cost"], PX[i]["pg"], None, None, PX[i]["repaired"], PX[i]["u"]) if PX.get(i) else None, (PX[i]["secs"] + t_fwd) if PX.get(i) else np.nan))):
        r = []
        for i in tid:
            p_, secs = get(i)
            if p_ is None: r.append((np.nan,)*4 + (1,)); continue
            r.append((100*abs(p_[0]-J[i])/J[i], 100*np.mean(p_[5] != np.asarray(SR[i][5])), 100*np.abs(p_[1]-SR[i][1]).sum()/SR[i][1].sum(), secs, 0))
        rows[name] = np.array(r, float)
        v = rows[name]
        print(f"  {name:30s} n={len(tid)} | gap mean {np.nanmean(v[:,0]):.4f}% median {np.nanmedian(v[:,0]):.4f}% | disc {np.nanmean(v[:,1]):.2f}% | "
              f"cont {np.nanmean(v[:,2]):.2f}% | time median {np.nanmedian(v[:,3]):.2f}s | failures {int(v[:,4].sum())}", flush=True)
    x, y = rows["OUR METHOD (learned V + cuts)"][:, 0], rows["supervised NN proxy"][:, 0]; mk = ~np.isnan(x) & ~np.isnan(y); d = x[mk] - y[mk]
    rng = np.random.default_rng(0); bs = [rng.choice(d, d.size).mean() for _ in range(10000)]
    print(f"  OURS - PROXY: {d.mean():+.4f} pp, 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}], "
          f"p {wilcoxon(x[mk], y[mk]).pvalue:.3g}, ours better/tie/worse {(d < -1e-6).sum()}/{(abs(d) <= 1e-6).sum()}/{(d > 1e-6).sum()}", flush=True)
    pickle.dump(dict(tid=tid, PX=PX, rows=rows, train_s=train_s, best=best[2]), open(os.path.join(HERE, "results", f"proxy_sup_{CASE}{SUF}.pkl"), "wb"))


if __name__ == "__main__":
    main()

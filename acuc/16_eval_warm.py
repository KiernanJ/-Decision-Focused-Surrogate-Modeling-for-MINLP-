# %% Evaluate a trained cut model with the cut-guided warm start (W/2 with cuts -> W/2 without,
#    started from it; cheaper priced kept) vs no cut at W, same total budget. Eval only.
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, multiprocessing as mp
ap = argparse.ArgumentParser(); ap.add_argument("--case", default="case118"); ap.add_argument("--ckpt", required=True)
ap.add_argument("--work", type=float, default=10.0); ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()


def main():
    import combopt_ac, data2
    from pipeline import grid
    from cutnet_k import CutNetK
    from combopt_net import CutNet
    ck = torch.load(a.ckpt, weights_only=False); cfg = ck["args"]
    g, nl = grid(a.case); G = g.n_gen; INST = data2.instances(g, a.case); te, va, tr = data2.split()
    REF = data2.refs(a.case, os.path.join(HERE, "results", "pool2")); ids = [i for i in te if data2.converged(REF.get(i))]
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
    vnet = CutNetK(g.n_bus, G, k_rows=2, v_scale=0.5, hidden=256, seed=0, learn_thr=False)
    vnet.load_state_dict(torch.load(os.path.join(HERE, "checkpoints2", f"{a.case}_s0.pt"), weights_only=False)["net"]); vnet.eval()
    with torch.no_grad(): VR, VI = [t.numpy().astype(float) for t in vnet(Xn)[:2]]
    pool = mp.get_context("spawn").Pool(a.workers, initializer=combopt_ac.init, initargs=(a.case,))
    UH = {i: r[0] for i, r in pool.imap_unordered(combopt_ac.relax, [(i, (VR[i], VI[i])) for i in ids]) if r is not None}
    net = CutNet(X.shape[1], G, cfg["rows"], seed=cfg["seed"]); net.load_state_dict(ck["net"]); net.eval()
    with torch.no_grad():
        A, d = net(Xn[ids]); uh = torch.tensor(np.stack([UH[i] for i in ids]), dtype=torch.float32)
        b = torch.einsum("nrg,ng->nr", A, uh) + cfg["margin"] - d
    A, b = A.numpy().astype(float), b.numpy().astype(float)
    Rw = dict(pool.imap_unordered(combopt_ac.miqcqp_warm, [(i, (VR[i], VI[i]), A[k], b[k], a.work/2, a.work/2, cfg["rho"]) for k, i in enumerate(ids)]))
    Rn = dict(pool.imap_unordered(combopt_ac.miqcqp, [(i, (VR[i], VI[i]), None, None, a.work, cfg["rho"]) for i in ids]))
    Jr = dict(pool.imap_unordered(combopt_ac.price_ref, [(i, REF[i]["u"]) for i in ids]))
    rows = []
    for i in ids:
        if Jr.get(i) is None or Rw.get(i) is None or Rn.get(i) is None: continue
        j, pref = Jr[i]; ur = np.asarray(REF[i]["u"])
        f = lambda R: dict(gap=100*abs(R["cost"]-j)/j, disc=100*float((R["u"] != ur).mean()), cont=100*float(np.abs(R["pg"]-pref).sum()/pref.sum()), secs=R["secs"])
        rows.append(dict(idx=i, warm=f(Rw[i]), nocut=f(Rn[i])))
    for k in ("warm", "nocut"):
        G_ = np.array([r[k]["gap"] for r in rows])
        print(f"[{a.case} W={a.work}] {k:6s} n={len(rows)} gap mean {G_.mean():.4f}% median {np.median(G_):.4f}% max {G_.max():.2f}% | disc {np.mean([r[k]['disc'] for r in rows]):.2f}% | "
              f"cont {np.mean([r[k]['cont'] for r in rows]):.2f}% | time median {np.median([r[k]['secs'] for r in rows]):.1f}s", flush=True)
    w = np.array([r["warm"]["gap"] for r in rows]); n = np.array([r["nocut"]["gap"] for r in rows])
    print(f"better {int((w<n-1e-9).sum())} worse {int((w>n+1e-9).sum())}")
    json.dump(rows, open(a.ckpt.replace(".pt", f"_warm_w{int(a.work)}.json"), "w"), indent=1); pool.close()


if __name__ == "__main__":
    main()

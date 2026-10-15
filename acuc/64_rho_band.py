# %% RHO HEADROOM in the sparse low-load regime (2026-10-13, DIAGNOSTIC, not a test): deploy the clean case118 pipeline (V_nc + cuts/rows _lfc, the pre-registered final method) on the
#    60 unlabeled band draws (data2.band_ids, load 0.78-0.88) at every deployment rho of the grid. Label-free quantities: strict-priced cost per rho, cost relative to the best rho per instance.
#    Reference quantities (report only): gap vs the converged reference where one exists (results/band_refs_case118.pkl). Question: is the per-instance best rho materially better than a single rho,
#    and are the failures (all-units-on, repaired infeasible answers) rho-dependent?   Usage: python 64_rho_band.py
import os, sys, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE, VSUF, TAG, W = "case118", "_env_nc", "_lfc", 10.0
RHOS = (1e4, 3e4, 1e5, 3e5, 1e6)


# %% main
def main():
    import combopt_ac, cuttrain, data2, runlog
    from pipeline import grid
    from cutnet_k import CutNetK, rows_torch
    from vnetio import vnet_out
    RUN = runlog.start("rho_band_case118")
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split(); ids = data2.band_ids()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    OUT, _, _ = vnet_out(os.path.join(HERE, "results", f"vnet_layer_{CASE}{VSUF}.pt"), Xn, n); V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    UC = torch.load(os.path.join(HERE, "results", f"unit_cuts_layer_{CASE}_soft{TAG}.pt"), weights_only=False); c = UC["c"]
    cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G)); cm.load_state_dict(UC["state"]); cm.eval()
    with torch.no_grad(): Mc = torch.sigmoid(cm(Xn)).numpy()
    LO = np.maximum((Mc - c)/(1 - c), 0); HI = 1 - np.maximum(((1 - c) - Mc)/(1 - c), 0)
    GR = torch.load(os.path.join(HERE, "results", f"general_rows_{CASE}{TAG}.pt"), weights_only=False); K = GR["K"]
    gnet = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False); gnet.load_state_dict(GR["selected"]); gnet.eval()
    pool = mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True)); ap = mp.get_context("spawn").Pool(11, initializer=cuttrain.init6, initargs=(CASE, 1000.0, K))
    ANT = dict(ap.imap_unordered(cuttrain.anchor6_t, [(i, *V(i), LO[i], HI[i]) for i in ids])); ap.close()
    base_jobs = {}
    for i in ids:
        Au = np.zeros((2*G, 2*G)); bu = np.ones(2*G)
        for k in range(G):
            if LO[i, k] > 0: Au[k, G + k] = -1.0; bu[k] = -1.0
            if HI[i, k] < 1: Au[G + k, G + k] = 1.0; bu[G + k] = 0.0
        if ANT.get(i) is not None and ANT[i][0] is not None:
            with torch.no_grad():
                _, _, a, al, a2, d, d2, _ = gnet(Xn[[i]]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.tensor(ANT[i][0], dtype=torch.float32)[None])
            Au = np.vstack([Au, A_[0, :K].numpy().astype(float)]); bu = np.r_[bu, b_[0, :K].numpy().astype(float)]
        base_jobs[i] = (Au, bu)
    RES = {}
    for rho in RHOS:
        t0 = time.time(); sol = dict(pool.imap_unordered(combopt_ac.miqcqp, [(i, V(i), *base_jobs[i], W, rho) for i in ids]))
        OP = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, sol[i]["u"]) for i in ids if sol.get(i)])); RES[rho] = (sol, OP)
        print(f"[rho {rho:.0e}] deployed {len(OP)}/{len(ids)}  [{time.time()-t0:.0f}s]", flush=True)
    REF = pickle.load(open(os.path.join(HERE, "results", "band_refs_case118.pkl"), "rb")) if os.path.exists(os.path.join(HERE, "results", "band_refs_case118.pkl")) else {}
    conv = [i for i in ids if REF.get(i, {}).get("converged")]
    SR = dict(pool.imap_unordered(combopt_ac.price_strict, [(i, REF[i]["u"]) for i in conv])); pool.close()
    SR = {i: r for i, r in SR.items() if r is not None and r[4] == 0}
    runlog.save(RUN, "rho_band.pkl", dict(ids=ids, RHOS=RHOS, RES=RES, SR=SR))
    ok = [i for i in ids if all(RES[r][1].get(i) for r in RHOS)]
    C = np.array([[RES[r][1][i]["cost"] for r in RHOS] for i in ok]); rel = C/C.min(1, keepdims=True); sc = {i: INST[i][0].sum()/base for i in ok}
    print(f"\n[RHO HEADROOM, sparse regime] {len(ok)} band draws (load {min(sc.values()):.3f}-{max(sc.values()):.3f}); {len(SR)} have a usable reference")
    print("single rho | mean cost vs the per-instance best rho | worst | #instances >1% worse | gap mean | gap median | discrete err | continuous err   (gap metrics on draws with references)")
    gi = [i for i in ok if i in SR]
    for k, r in enumerate(RHOS):
        v = np.array([(100*abs(RES[r][1][i]["cost"] - SR[i][0])/SR[i][0], 100*np.mean(RES[r][1][i]["u"] != np.asarray(SR[i][5])), 100*np.abs(RES[r][1][i]["pg"] - SR[i][1]).sum()/SR[i][1].sum()) for i in gi])
        print(f"  {r:.0e}   | {rel[:, k].mean():.4f} | {rel[:, k].max():.3f} | {(rel[:, k] > 1.01).sum():2d} | {v[:,0].mean():.4f}% | {np.median(v[:,0]):.4f}% | {v[:,1].mean():.2f}% | {v[:,2].mean():.2f}%")
    best = C.argmin(1)
    v = np.array([(100*abs(C[ok.index(i)].min() - SR[i][0])/SR[i][0],) for i in gi])
    print(f"  ORACLE  | 1.0000 | 1.000 |  0 | {v[:,0].mean():.4f}% | {np.median(v[:,0]):.4f}% | (needs {len(RHOS)} deployments per instance)")
    print("oracle-chosen rho (ties -> first):", {f"{r:.0e}": int((best == k).sum()) for k, r in enumerate(RHOS)})
    print("instances where rho matters (>0.3% cost spread):")
    for j, i in enumerate(ok):
        if rel[j].max() > 1.003:
            u = [int(np.sum(RES[r][1][i]["u"])) for r in RHOS]
            print(f"  #{i} load {sc[i]:.3f} ref units {int(np.sum(SR[i][5])) if i in SR else '-':>2} | cost vs best: " + " ".join(f"{r:.0e}:{rel[j,k]:.3f}" for k, r in enumerate(RHOS)) + f" | units on: {u} | best {RHOS[best[j]]:.0e}")


if __name__ == "__main__":
    main()

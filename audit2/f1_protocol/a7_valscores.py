# Re-compute the LABEL-FREE validation scores that drove selection (case118, all 24 validation draws), exactly as 46/52/61/65 compute them:
# score = mean over draws of strict price / flat-V relaxation cost, failure = 10. Also four metrics on the 21 draws with converged refs.
import os, sys, pickle, time, numpy as np, warnings
warnings.filterwarnings("ignore")
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src"); os.chdir(R + "/acuc")
import torch, torch.nn as nn, multiprocessing as mp

if __name__ == "__main__":
    import combopt_ac, cuttrain, data2
    from pipeline import grid
    from cutnet_k import CutNetK, rows_torch
    from vnetio import vnet_out
    CASE = "case118"; W = 10.0
    os.environ.pop("DEPLOY_RHO", None)
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    OUT, RHO_V, VSEL = vnet_out(f"{R}/acuc/results/vnet_layer_{CASE}_env_nc.pt", Xn, n); V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    UC = torch.load(f"{R}/acuc/results/unit_cuts_layer_{CASE}_soft_lfc.pt", weights_only=False); c = UC["c"]
    cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G)); cm.load_state_dict(UC["state"]); cm.eval()
    with torch.no_grad(): Mc = torch.sigmoid(cm(Xn)).numpy()
    LO = np.maximum((Mc - c)/(1 - c), 0); HI = 1 - np.maximum(((1 - c) - Mc)/(1 - c), 0)
    GR = torch.load(f"{R}/acuc/results/general_rows_{CASE}_lfc.pt", weights_only=False); K = GR["K"]
    gnet = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False); gnet.load_state_dict(GR["selected"]); gnet.eval()
    SR = data2.strict_refs(CASE); vl = list(va); vv = [i for i in va if SR.get(i) is not None and SR[i][4] == 0]
    nfix = np.mean([(LO[i] > 0).sum() + (HI[i] < 1).sum() for i in vl])
    print(f"[config] V-net stored rho {RHO_V:.0e}; c={c}; cut epoch {UC['epoch']}; rows epoch {GR['selected_epoch']}; mean units fixed on validation {nfix:.1f}/{G}", flush=True)
    ap = mp.get_context("spawn").Pool(2, initializer=cuttrain.init6, initargs=(CASE, 1000.0, K))
    AN = dict(ap.imap_unordered(cuttrain.anchor6, [(i, *V(i), LO[i], HI[i]) for i in vl])); ap.close(); ap.join()
    pool = mp.get_context("spawn").Pool(2, initializer=combopt_ac.init, initargs=(CASE, True))
    RL = dict(pool.imap_unordered(combopt_ac.relax, vl))
    def jobs(arm, rho):
        J = []
        for i in vl:
            if arm == "nocuts": J.append((i, V(i), None, None, W, rho)); continue
            Au = np.zeros((2*G, 2*G)); bu = np.ones(2*G)
            for k in range(G):
                if LO[i, k] > 0: Au[k, G + k] = -1.0; bu[k] = -1.0
                if HI[i, k] < 1: Au[G + k, G + k] = 1.0; bu[G + k] = 0.0
            if arm == "full" and AN.get(i) is not None:
                with torch.no_grad():
                    _, _, a, al, a2, d, d2, _ = gnet(Xn[[i]]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.tensor(AN[i], dtype=torch.float32)[None])
                Au = np.vstack([Au, A_[0, :K].numpy().astype(float)]); bu = np.r_[bu, b_[0, :K].numpy().astype(float)]
            J.append((i, V(i), Au, bu, W, rho))
        return J
    out = {}
    for arm, rho, expect in (("cuts", 1e4, "0.966475 (= UC['val'] = rows OFF in sel_hist)"), ("full", 1e4, "0.966189 (rows sel_hist epoch 6)"),
                             ("nocuts", 1e4, "0.961692 (65_vnet_vs_flat_rho, V-net, no cuts)"), ("full", 3e5, "0.961770 (RESULTS comparator) / 0.1286% | 0.0523% | 4.76% | 8.71%"),
                             ("nocuts", 3e5, "1.004816 (65 / 67 'none')")):
        t0 = time.time()
        sol = dict(pool.imap_unordered(combopt_ac.miqcqp, jobs(arm, rho)))
        OP = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, sol[i]["u"]) for i in vl if sol.get(i)]))
        lf = float(np.mean([OP[i]["cost"]/RL[i][1] if (OP.get(i) and RL.get(i)) else 10.0 for i in vl]))
        v = np.array([(100*abs(OP[i]["cost"] - SR[i][0])/SR[i][0], 100*np.mean(OP[i]["u"] != np.asarray(SR[i][5])), 100*np.abs(OP[i]["pg"] - SR[i][1]).sum()/SR[i][1].sum(),
                       100*(OP[i]["cost"] - SR[i][0])/SR[i][0]) if OP.get(i) else (np.nan,)*4 for i in vv])
        fb = sum(bool(sol[i].get("fallback")) for i in vl if sol.get(i)); rep = sum(int(OP[i]["repaired"] > 0) for i in vl if OP.get(i))
        print(f"[{arm} @ rho {rho:.0e}] label-free score {lf:.6f} (expected {expect}) | four metrics on {len(vv)} val draws: {np.nanmean(v[:,0]):.4f}% | {np.nanmedian(v[:,0]):.4f}% | {np.nanmean(v[:,1]):.2f}% | {np.nanmean(v[:,2]):.2f}% | signed {np.nanmean(v[:,3]):+.4f}% | worst {np.nanmax(v[:,0]):.3f}% | fallbacks {fb} strict repairs {rep}  [{time.time()-t0:.0f}s]", flush=True)
        out[(arm, rho)] = dict(lf=lf, v=v, sol=sol, OP=OP)
    pool.close()
    pickle.dump(dict(vl=vl, vv=vv, out={k: dict(lf=d['lf'], v=d['v']) for k, d in out.items()}), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "valscores.pkl"), "wb"))

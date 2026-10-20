# Solver-noise floor of the label-free validation score (case118, 24 validation draws): same deployed artifacts, same rho/W, only Gurobi's
# random Seed parameter changes (0 = default). If the spread across solver seeds is as large as the score differences used for
# selection (c, epochs, rows, rho), those selections were made on noise.
import os, sys, pickle, time, numpy as np, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src"); sys.path.insert(0, HERE); os.chdir(R + "/acuc")
import torch, torch.nn as nn, multiprocessing as mp
if __name__ == "__main__":
    import combopt_ac, cuttrain, data2, f1util
    from pipeline import grid
    from cutnet_k import CutNetK, rows_torch
    from vnetio import vnet_out
    CASE = "case118"; W = 10.0; os.environ.pop("DEPLOY_RHO", None)
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    OUT, _, _ = vnet_out(f"{R}/acuc/results/vnet_layer_{CASE}_env_nc.pt", Xn, n); OUT = OUT + 1e-7*np.random.default_rng(123).standard_normal(OUT.shape); V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    UC = torch.load(f"{R}/acuc/results/unit_cuts_layer_{CASE}_soft_lfc.pt", weights_only=False); c = UC["c"]
    cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G)); cm.load_state_dict(UC["state"]); cm.eval()
    with torch.no_grad(): Mc = torch.sigmoid(cm(Xn)).numpy()
    LO = np.maximum((Mc - c)/(1 - c), 0); HI = 1 - np.maximum(((1 - c) - Mc)/(1 - c), 0)
    GR = torch.load(f"{R}/acuc/results/general_rows_{CASE}_lfc.pt", weights_only=False); K = GR["K"]
    gnet = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False); gnet.load_state_dict(GR["selected"]); gnet.eval()
    SR = data2.strict_refs(CASE); vl = list(va); vv = [i for i in va if SR.get(i) is not None and SR[i][4] == 0]
    ap = mp.get_context("spawn").Pool(2, initializer=cuttrain.init6, initargs=(CASE, 1000.0, K))
    AN = dict(ap.imap_unordered(cuttrain.anchor6, [(i, *V(i), LO[i], HI[i]) for i in vl])); ap.close(); ap.join()
    def jobs(arm, rho):
        J = []
        for i in vl:
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
    res = {}
    for seed in (0,):
        pool = mp.get_context("spawn").Pool(2, initializer=f1util.init_seed, initargs=(CASE, seed))
        RL = dict(pool.imap_unordered(combopt_ac.relax, vl))
        for arm, rho in (("full", 3e5), ("cuts", 1e4)):
            t0 = time.time()
            sol = dict(pool.imap_unordered(combopt_ac.miqcqp, jobs(arm, rho)))
            OP = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, sol[i]["u"]) for i in vl if sol.get(i)]))
            lf = float(np.mean([OP[i]["cost"]/RL[i][1] if (OP.get(i) and RL.get(i)) else 10.0 for i in vl]))
            v = np.array([(100*abs(OP[i]["cost"] - SR[i][0])/SR[i][0], 100*np.mean(OP[i]["u"] != np.asarray(SR[i][5]))) if OP.get(i) else (np.nan,)*2 for i in vv])
            fb = sum(bool(sol[i].get("fallback")) for i in vl if sol.get(i))
            print(f"[V perturbed by 1e-7] {arm} @ rho {rho:.0e}: label-free {lf:.6f} | val gap mean {np.nanmean(v[:,0]):.4f}% median {np.nanmedian(v[:,0]):.4f}% disc {np.nanmean(v[:,1]):.2f}% worst {np.nanmax(v[:,0]):.3f}% | fallbacks {fb}  [{time.time()-t0:.0f}s]", flush=True)
            res[(seed, arm, rho)] = dict(lf=lf, v=v, u={i: OP[i]["u"] for i in OP})
        pool.close(); pool.join()
    pickle.dump(res, open(os.path.join(HERE, "perturb.pkl"), "wb"))

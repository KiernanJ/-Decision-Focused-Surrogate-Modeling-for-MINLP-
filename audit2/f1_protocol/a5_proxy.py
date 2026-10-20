# Proxy fairness check (case118, VALIDATION split only): replicate 55_final_eval's proxy exactly, then quantify
# per-unit accuracy, under/over-commitment, repair, gap; plus two cheap variants: val-tuned threshold (in-sample) and a
# label-free downward pass (switch off committed units in increasing proxy-probability order if the AC price drops).
import os, sys, pickle, time, numpy as np, warnings
warnings.filterwarnings("ignore")
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src"); os.chdir(R + "/acuc")
import torch, torch.nn as nn, multiprocessing as mp

def down_strict(args):
    import combopt_ac as C
    i, u0, prob, ndown = args
    _, r = C.price_strict((i, u0))
    if r is None: return i, None
    u, c = np.asarray(r[5], float).copy(), r[0]
    g = C._W["g"]; need = C._W["need_f"]*float(C._W["inst"][i][0].sum()); tried = 0
    for k in [t for t in np.argsort(prob) if u[t] > 0.5][:ndown]:
        w = u.copy(); w[k] = 0.0
        if float(g.pmax @ w) < need: continue
        _, rr = C.price_strict((i, w)); tried += 1
        if rr is not None and rr[4] == 0 and rr[0] < c - 1e-9: u, c = w, rr[0]
    _, rf = C.price_strict((i, u))
    return i, dict(cost=rf[0], u=rf[5], tried=tried)

if __name__ == "__main__":
    import combopt_ac, data2
    from pipeline import grid
    CASE = "case118"
    g, nl = grid(CASE); G = g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    RP = data2.refs(CASE, R + "/acuc/results/pool2")
    ltr = [i for i in tr if data2.converged(RP.get(i))]; lva = [i for i in va if data2.converged(RP.get(i))]
    Ytr = torch.tensor(np.stack([RP[i]["u"] for i in ltr]), dtype=torch.float32); Yva = torch.tensor(np.stack([RP[i]["u"] for i in lva]), dtype=torch.float32)
    SR = data2.strict_refs(CASE); vv = [i for i in va if SR.get(i) is not None and SR[i][4] == 0]
    print(f"labels: train {len(ltr)}, val (early stopping) {len(lva)}; scored val draws {len(vv)}", flush=True)
    pool = mp.get_context("spawn").Pool(2, initializer=combopt_ac.init, initargs=(CASE, True))
    out = {}
    for s in (0, 1, 2):
        torch.manual_seed(s)
        pm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
        po = torch.optim.Adam(pm.parameters(), lr=1e-3, weight_decay=1e-4); best = (np.inf, None, -1); bce = nn.functional.binary_cross_entropy_with_logits
        for it in range(3000):
            po.zero_grad(); l = bce(pm(Xn[ltr]), Ytr); l.backward(); po.step()
            if it % 25 == 0:
                with torch.no_grad(): v_ = float(bce(pm(Xn[lva]), Yva))
                if v_ < best[0]: best = (v_, {k: t.clone() for k, t in pm.state_dict().items()}, it)
        pm.load_state_dict(best[1]); pm.eval()
        with torch.no_grad():
            P = torch.sigmoid(pm(Xn[vv])).numpy(); Ptr = torch.sigmoid(pm(Xn[ltr])).numpy()
        U = np.stack([np.asarray(SR[i][5], float) for i in vv])
        # thresholds tuned on val labels (in-sample): min Hamming
        ths = np.round(np.arange(0.1, 0.91, 0.05), 2); ham = [np.mean((P > t) != (U > 0.5)) for t in ths]; tstar = float(ths[int(np.argmin(ham))])
        res = {}
        for nm, th in (("round 0.5", 0.5), (f"val-tuned thr {tstar}", tstar)):
            Z = (P > th).astype(float)
            PX = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, Z[k]) for k, i in enumerate(vv)]))
            rows = []
            for k, i in enumerate(vv):
                r = PX[i]; fn = int(((U[k] > 0.5) & (Z[k] < 0.5)).sum()); fp = int(((U[k] < 0.5) & (Z[k] > 0.5)).sum())
                rows.append((100*abs(r["cost"] - SR[i][0])/SR[i][0], 100*np.mean(r["u"] != U[k]), fn, fp, int(r["repaired"]), 100*(r["cost"] - SR[i][0])/SR[i][0], int(Z[k].sum()), int(U[k].sum())))
            res[nm] = np.array(rows, float)
        D = dict(pool.imap_unordered(down_strict, [(i, (P[k] > 0.5).astype(float), P[k], 25) for k, i in enumerate(vv)]))
        res["round 0.5 + label-free downward pass"] = np.array([(100*abs(D[i]["cost"] - SR[i][0])/SR[i][0], 100*np.mean(np.asarray(D[i]["u"]) != U[k]), np.nan, np.nan, np.nan, 100*(D[i]["cost"] - SR[i][0])/SR[i][0], np.sum(D[i]["u"]), U[k].sum()) for k, i in enumerate(vv)])
        acc_tr = np.mean((Ptr > 0.5) == (Ytr.numpy() > 0.5)); acc_va = np.mean((P > 0.5) == (U > 0.5))
        print(f"\n[seed {s}] early-stopped at step {best[2]} (val BCE {best[0]:.4f}); per-unit accuracy train {100*acc_tr:.2f}% val {100*acc_va:.2f}%; "
              f"exact val commitments {np.mean(np.all((P > 0.5) == (U > 0.5), 1))*100:.1f}%", flush=True)
        for nm, v in res.items():
            extra = "" if nm.endswith("pass") else f" | FN(ref on, proxy off) {v[:,2].mean():.2f}/inst FP {v[:,3].mean():.2f}/inst | needed repair {int((v[:,4]>0).sum())}/{len(v)} (units added mean {v[:,4].mean():.2f})"
            print(f"   {nm:38s}: gap mean {v[:,0].mean():.4f}% median {np.median(v[:,0]):.4f}% | signed mean {v[:,5].mean():+.4f}% | disc {v[:,1].mean():.2f}% | units on {v[:,6].mean():.1f} vs ref {v[:,7].mean():.1f}{extra}", flush=True)
        out[s] = res
    pool.close()
    pickle.dump(dict(vv=vv, out=out), open("proxy_val.pkl", "wb"))

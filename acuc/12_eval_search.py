# %% Test a trained CombOptNet cut model with the exact-layer-guided local search:
#    learned cut + search vs no cut + search, converged references, J_ref = price(u_ref).
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, multiprocessing as mp
ap = argparse.ArgumentParser(); ap.add_argument("--case", default="case118"); ap.add_argument("--ckpt", required=True)
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--k", type=int, default=6); ap.add_argument("--rounds", type=int, default=8)
a = ap.parse_args()


def main():
    import combopt_ac, data2
    from pipeline import grid
    from combopt_net import CutNet
    ck = torch.load(a.ckpt, weights_only=False); cfg = ck["args"]
    g, nl = grid(a.case); G = g.n_gen
    INST = data2.instances(g, a.case); te, va, tr = data2.split()
    REF = data2.refs(a.case, os.path.join(HERE, "results", "pool2"))
    pool = mp.get_context("spawn").Pool(a.workers, initializer=combopt_ac.init, initargs=(a.case,))
    base = float(g.pd.sum())
    X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
    UH = {i: r[0] for i, r in pool.imap_unordered(combopt_ac.relax, te) if r is not None}
    ids = [i for i in te if i in UH and data2.converged(REF.get(i))]
    net = CutNet(X.shape[1], G, cfg["rows"], seed=cfg["seed"]); net.load_state_dict(ck["net"]); net.eval()
    with torch.no_grad():
        A, d = net(Xn[ids]); uh = torch.tensor(np.stack([UH[i] for i in ids]), dtype=torch.float32)
        b = torch.einsum("nrg,ng->nr", A, uh) + cfg["margin"] - d
    A, b = A.numpy().astype(float), b.numpy().astype(float)
    Rc = dict(pool.imap_unordered(combopt_ac.final, [(i, UH[i], A[k], b[k], a.k, a.rounds) for k, i in enumerate(ids)]))
    Rn = dict(pool.imap_unordered(combopt_ac.final, [(i, UH[i], None, None, a.k, a.rounds) for i in ids]))
    Jr = dict(pool.imap_unordered(combopt_ac.price_ref, [(i, REF[i]["u"]) for i in ids]))
    rows = []
    for i in ids:
        if Jr[i] is None or Rc[i] is None or Rn[i] is None: continue
        j = Jr[i][0]; u_ref = np.asarray(REF[i]["u"])
        rows.append(dict(idx=i, cut=100*(Rc[i]["cost"]-j)/j, nocut=100*(Rn[i]["cost"]-j)/j,
                         cut_proj=100*(Rc[i]["cost_proj"]-j)/j, nocut_proj=100*(Rn[i]["cost_proj"]-j)/j,
                         disc_cut=100*float((Rc[i]["u"] != u_ref).mean()), disc_nocut=100*float((Rn[i]["u"] != u_ref).mean())))
    M = lambda k: np.mean([abs(r[k]) for r in rows])
    c = np.array([abs(r["cut"]) for r in rows]); n = np.array([abs(r["nocut"]) for r in rows])
    from scipy.stats import wilcoxon
    print(f"[{a.case}] n={len(rows)} | before search: cut {M('cut_proj'):.3f}% no-cut {M('nocut_proj'):.3f}% | "
          f"AFTER search: cut {M('cut'):.4f}% no-cut {M('nocut'):.4f}% (better {int((c<n-1e-9).sum())}, worse {int((c>n+1e-9).sum())}, "
          f"p={wilcoxon(c-n).pvalue if np.any(c!=n) else 1:.2g}) | disc cut {M('disc_cut'):.2f}% no-cut {M('disc_nocut'):.2f}%", flush=True)
    json.dump(rows, open(a.ckpt.replace(".pt", "_search.json"), "w"), indent=1); pool.close()


if __name__ == "__main__":
    main()

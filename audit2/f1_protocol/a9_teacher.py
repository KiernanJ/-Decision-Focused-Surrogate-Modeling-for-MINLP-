# Reproduce committed label-free targets (60_selfteacher.py logic) for a few case118 draws from label-free inputs only.
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src"); os.chdir(R + "/acuc")
import torch, multiprocessing as mp
if __name__ == "__main__":
    import combopt_ac, selfsup, data2
    from pipeline import grid
    from vnetio import vnet_out
    os.environ.pop("DEPLOY_RHO", None)
    CASE = "case118"; W = 10.0; K, ROUNDS, SWAPS = 6, 8, 4
    g, nl = grid(CASE); n = g.n_bus; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    OUT, RHO, VSEL = vnet_out(f"{R}/acuc/results/vnet_layer_{CASE}_env_nc.pt", Xn, n); V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    T = {int(k): np.array(v, float) for k, v in json.load(open(f"{R}/acuc/results/selfteacher_{CASE}_bestof_extra_nc.json")).items()}
    ids = tr[:3] + data2.extra_ids()[:2]
    pool = mp.get_context("spawn").Pool(2, initializer=combopt_ac.init, initargs=(CASE, True))
    SI = dict(pool.imap_unordered(selfsup.deploy_improve, [(i, V(i), None, None, W, RHO, K, ROUNDS, SWAPS) for i in ids]))
    RC = dict(pool.imap_unordered(selfsup.restore_candidates, [(i, V(i)) for i in ids]))
    cand = [(i, "W10 + improve", SI[i]["u"]) for i in ids if SI.get(i)] + [(i, nm, u) for i in ids for nm, u in RC[i]["cands"].items()]
    PR = [r for _, r in pool.imap(combopt_ac.price_strict, [(i, u) for i, _, u in cand])]; pool.close()
    best = {}
    for (i, nm, _), r in zip(cand, PR):
        if r is not None and (i not in best or r[0] < best[i][1]): best[i] = (nm, r[0], np.asarray(r[5], float))
    for i in ids:
        b = best.get(i)
        print(f"draw {i}: reproduced target ({b[0] if b else None}) == committed target: {bool(b is not None and np.array_equal(b[2], T[i]))}; units differ {int(np.sum(b[2] != T[i])) if b else 'n/a'}", flush=True)

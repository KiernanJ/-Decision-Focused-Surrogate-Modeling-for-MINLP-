import sys, numpy as np, torch, os, json, pickle, multiprocessing as mp
from common import *
case = sys.argv[1]; N = int(sys.argv[2]); NP = int(sys.argv[3])
D = setup(case); n = D["n"]
OUT, RHO, SEL = vnet(D, case)          # teacher used the V file's stored rho (60_selfteacher.py, no DEPLOY_RHO in the chain)
W = {"case118": 10.0, "case300": 30.0}.get(case, 10.0)
TG = json.load(open(os.path.join(RES, f"selfteacher_{case}_bestof_extra_nc.json")))
trn = [i for i in D["tr"] if str(i) in TG]
ids = list(np.random.default_rng(7).choice(trn, N, replace=False))
if __name__ == "__main__":
    import g_rank
    pool = mp.get_context("spawn").Pool(NP, initializer=g_rank.init, initargs=(case,))
    R = list(pool.imap_unordered(g_rank.work, [(int(i), (1.0 + OUT[i, :n], OUT[i, n:]), W, RHO, 6, 8, 4) for i in ids])); pool.close()
    pickle.dump(R, open(f"g_rank_{case}.pkl", "wb"))
    R = [r for r in R if r.get("strict_layer") and r.get("strict_rand")]
    bl = [min([r["strict_layer"]] + [v for v in r["restore"].values() if v]) for r in R]; br = [min([r["strict_rand"]] + [v for v in r["restore"].values() if v]) for r in R]
    win_l = sum(r["strict_layer"] <= min([v for v in r["restore"].values() if v] or [np.inf]) + 1e-9 for r in R)
    d = np.array([(r["strict_layer"] - r["strict_rand"])/r["strict_rand"] for r in R])
    print(f"[{case}] {len(R)} training draws, rho {RHO:.0e}, W {W}: improve with LAYER ranking vs RANDOM ranking (same k=6, rounds=8, swaps=4):")
    print(f"   strict cost layer - random: mean {100*d.mean():+.4f}%, layer better/tie/worse {(d < -1e-7).sum()}/{(np.abs(d) <= 1e-7).sum()}/{(d > 1e-7).sum()}")
    print(f"   pricings: layer {np.mean([r['n_layer'] for r in R]):.1f}, random {np.mean([r['n_rand'] for r in R]):.1f}; moves: layer {np.mean([r['moves_layer'] for r in R]):.2f}, random {np.mean([r['moves_rand'] for r in R]):.2f}; "
          f"improvement over the W{W:.0f} start: layer {100*np.mean([(r['strict_u0'] - r['strict_layer'])/r['strict_u0'] for r in R]):.3f}%, random {100*np.mean([(r['strict_u0'] - r['strict_rand'])/r['strict_u0'] for r in R]):.3f}%")
    db = np.array([(a - b)/b for a, b in zip(bl, br)])
    print(f"   best-of TARGET (with the two restorations): layer-ranked vs random-ranked label cost mean {100*db.mean():+.4f}%, differ on {(np.abs(db) > 1e-7).sum()}/{len(R)}; 'W10+improve(layer)' is the cheapest candidate on {win_l}/{len(R)}")

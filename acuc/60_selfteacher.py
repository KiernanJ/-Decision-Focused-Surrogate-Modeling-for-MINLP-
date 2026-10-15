# %% LABEL-FREE BEST-OF TEACHER for any set of training draws (2026-10-11), same rule as 58 (which passed its bar on case118):
#    candidates per instance = own short MIQCQP W (no cuts) + exact-layer improve, and the two restorations of the relaxation
#    at learned V (merit 0.8 / confidence 0.5); target = cheapest under the strict evaluator. No reference, no long solve.
#    Usage: python 60_selfteacher.py <case> <draws: extra> <merge-with json or ->   -> results/selfteacher_<case>_bestof_<draws>.json
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE, DRAWS = sys.argv[1], sys.argv[2]; MERGE = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
VTAG = sys.argv[4] if len(sys.argv) > 4 else ""   # (2026-10-12) V-net file suffix, "_nc" = clean V-net trained on the uncut relaxation
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0); K, ROUNDS, SWAPS, NPROC = 6, 8, 4, 10


# %% main
def main():
    import combopt_ac, selfsup, data2, runlog
    from pipeline import grid
    RUN = runlog.start(f"selfteacher_{CASE}_{DRAWS}")
    g, nl = grid(CASE); n = g.n_bus; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    ids = tr if DRAWS == "train" else getattr(data2, f"{DRAWS}_ids")()
    assert not set(ids) & set(te + va + data2.fresh_ids() + data2.fresh2_ids() + data2.fresh3_ids() + data2.fresh4_ids() + data2.fresh5_ids() + data2.fresh6_ids() + data2.band_ids()), "draws overlap val/test"
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg            # normalisation of the ORIGINAL training split (V-net input)
    from vnetio import vnet_out
    OUT, RHO, VSEL = vnet_out(os.path.join(HERE, "results", f"vnet_layer_{CASE}_env{VTAG}.pt"), Xn, n)
    V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    sc = np.array([INST[i][0].sum()/g.pd.sum() for i in ids])
    print(f"[config] {CASE}: {len(ids)} '{DRAWS}' draws (load scale {sc.min():.3f}-{sc.max():.3f}), W={W}, V-net s0 file suffix '{VTAG}' (selected {VSEL}, rho {RHO:.0e}), {NPROC} workers", flush=True)
    pool = mp.get_context("spawn").Pool(NPROC, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    SI = {}
    for k_, (i, r) in enumerate(pool.imap_unordered(selfsup.deploy_improve, [(i, V(i), None, None, W, RHO, K, ROUNDS, SWAPS) for i in ids])):
        SI[i] = r
        if (k_ + 1) % 50 == 0: print(f"   deploy+improve {k_+1}/{len(ids)}  [{time.time()-t0:.0f}s]", flush=True)
    RC = dict(pool.imap_unordered(selfsup.restore_candidates, [(i, V(i)) for i in ids]))
    print(f"   restorations done  [{time.time()-t0:.0f}s]", flush=True)
    cand = [(i, "W10 + improve", SI[i]["u"]) for i in ids if SI.get(i)] + [(i, nm, u) for i in ids for nm, u in RC[i]["cands"].items()]
    PR = [r for _, r in pool.imap(combopt_ac.price_strict, [(i, u) for i, _, u in cand])]; pool.close()
    best = {}
    for (i, nm, _), r in zip(cand, PR):
        if r is not None and (i not in best or r[0] < best[i][1]): best[i] = (nm, r[0], np.asarray(r[5], float))
    print(f"[done] targets for {len(best)}/{len(ids)} draws  [{time.time()-t0:.0f}s]; picked: "
          + ", ".join(f"{k} {sum(v[0] == k for v in best.values())}" for k in ("W10 + improve", "restore merit0.8", "restore conf0.5")), flush=True)
    T = {str(i): v[2].tolist() for i, v in best.items()}
    if MERGE:
        M = json.load(open(MERGE)); assert not set(M) & set(T); T = {**M, **T}
    out = os.path.join(HERE, "results", f"selfteacher_{CASE}_bestof_{DRAWS}{VTAG}.json"); json.dump(T, open(out, "w"))
    print(f"[saved] {out}: {len(T)} training targets ({'merged with ' + os.path.basename(MERGE) if MERGE else 'new draws only'})", flush=True)
    runlog.add_result(RUN, f"{CASE} label-free best-of teacher for {len(best)} '{DRAWS}' draws -> {os.path.basename(out)} ({len(T)} targets)")


if __name__ == "__main__":
    main()

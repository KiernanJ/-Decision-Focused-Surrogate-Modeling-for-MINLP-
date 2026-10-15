# %% V-NET vs FLAT V as a function of the DEPLOYMENT rho (2026-10-14, DIAGNOSTIC on the 24 VALIDATION draws only; label-free; no test/fresh draw, no reference).
#    Question (Parth): is the V-net's small edge over flat V (label-free score 0.9617 vs 0.9779 at rho 1e4) a consequence of the rho it was scored at?
#    For each deployment rho of the grid, deploy ONE short MIQCQP (W, no cuts, no rows) at (a) the clean V-net V_nc and (b) flat V, price strictly,
#    score = strict price / flat-V relaxation cost (the label-free score used everywhere).  Paired per-draw comparison, V-net minus flat.
#    Usage: python 65_vnet_vs_flat_rho.py [case118]
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, multiprocessing as mp
CASE = sys.argv[1] if len(sys.argv) > 1 else "case118"; W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0)
RHOS = (1e4, 3e4, 1e5, 3e5, 1e6)


# %% main
def main():
    import combopt_ac, data2, runlog
    from pipeline import grid
    from vnetio import vnet_out
    RUN = runlog.start(f"vnet_vs_flat_rho_{CASE}")
    g, nl = grid(CASE); n = g.n_bus; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    OUT, _, SEL = vnet_out(os.path.join(HERE, "results", f"vnet_layer_{CASE}_env_nc.pt"), Xn, n)
    Vnet = lambda i: (1.0 + OUT[i, :n], OUT[i, n:]); flat = lambda i: (np.ones(n), np.zeros(n))
    pool = mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    SC = {i: r[1] for i, r in pool.imap_unordered(combopt_ac.relax, va) if r is not None}
    ids = [i for i in va if i in SC]
    print(f"[config] {CASE}: {len(ids)}/{len(va)} validation draws, V-net {SEL}, W={W}", flush=True)
    S = {}
    for rho in RHOS:
        for name, Vf in (("vnet", Vnet), ("flat", flat)):
            sol = dict(pool.imap_unordered(combopt_ac.miqcqp_fix, [(i, Vf(i), [], [], W, rho) for i in ids]))
            pr = dict(pool.imap_unordered(combopt_ac.price_strict, [(i, sol[i]["u"]) for i in ids if sol.get(i)]))
            S[rho, name] = np.array([pr[i][0]/SC[i] if pr.get(i) else 10.0 for i in ids])
            print(f"[rho {rho:.0e}] {name}: score {S[rho, name].mean():.6f}  [{time.time()-t0:.0f}s]", flush=True)
    pool.close()
    print("\n[V-NET vs FLAT V by deployment rho, 24 validation draws, label-free score = strict price / flat-V relaxation cost (lower is better)]")
    print("rho    | V-net    | flat V   | V-net minus flat | V-net better on (of n) | worst V-net minus flat on one draw")
    lines = []
    for rho in RHOS:
        d = S[rho, "vnet"] - S[rho, "flat"]
        line = f"{rho:.0e} | {S[rho,'vnet'].mean():.6f} | {S[rho,'flat'].mean():.6f} | {d.mean():+.6f} ({100*d.mean()/S[rho,'flat'].mean():+.2f}%) | {(d < -1e-9).sum()} / {(abs(d) > 1e-9).sum()} | {d.max():+.4f}"
        print(line, flush=True); lines.append(line)
    runlog.add_result(RUN, f"{CASE} V-NET vs FLAT V by deployment rho (validation, label-free score, no cuts): " + " || ".join(lines))
    runlog.save(RUN, "vnet_vs_flat.pkl", dict(ids=ids, RHOS=RHOS, S=S))


if __name__ == "__main__":
    main()

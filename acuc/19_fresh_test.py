# %% FROZEN protocol, scored ONCE on a FRESH test set (after the 2026-10-08 audit).
#    Training, the teacher, the confidence c and the safeguard threshold tau are exactly
#    17_fix_cuts.py's (same splits, same seed, validation only). Only the test instances are
#    new: 48 draws from the same sampler/band with a new seed (data2.fresh_ids), never seen.
#    Arms (all paired on the same instances, all at the same Gurobi work budget W):
#      ours          learned V + learned fixing cuts
#      ours+guard    + the label-free safeguard (post-hoc design; timed incl. its relaxation)
#      seqQCAC@W     sequential QCAC (the reference algorithm) stopped at the same budget
#      flatV+cuts    same cuts, flat V            -> does learning V help?
#      no-cut        learned V, no cuts           -> do the cuts help?
#    Reference: sequential QCAC to convergence, MIP gap 1e-4 (10x tighter than before).
#    Usage: python 19_fresh_test.py --case case118 --work 10 --work-long 120 --workers 11
import os, sys, json, time, pickle, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
ap = argparse.ArgumentParser(); ap.add_argument("--case", default="case118"); ap.add_argument("--work", type=float, default=10.0)
ap.add_argument("--work-long", type=float, default=120.0); ap.add_argument("--rho", type=float, default=1e4)
ap.add_argument("--workers", type=int, default=11)
ap.add_argument("--ref-gap", type=float, default=1e-4); ap.add_argument("--ref-tl", type=float, default=600.0)
ap.add_argument("--no-refs", action="store_true")   # arms only; references via 20_fresh_refs.py
ap.add_argument("--test", default="fresh", choices=["fresh", "split"])   # split = data2.split() test (never-seen case)
ap.add_argument("--vnet", default="learned", choices=["learned", "none"])   # none: flat V everywhere (no V-net)
a = ap.parse_args()
OUT = os.path.join(HERE, "results", f"{a.test}_{a.case}{'_flatV' if a.vnet == 'none' else ''}.pkl")
ARMS = ("ours", "ours+guard", "seqQCAC@W", "flatV+cuts", "no-cut")


# %% scoring (also runnable alone on the saved pickle)
def score(D):
    from scipy.stats import wilcoxon
    ids, REF, JR, R = D["ids"], D["ref"], D["jref"], D["arms"]
    ok = [i for i in ids if REF.get(i, {}).get("converged") and JR.get(i) is not None]
    s = D["scale"]
    bad = [i for i in ids if i not in ok]
    print(f"\n[{a.case}] fresh test: {len(ids)} drawn, {len(ok)} with a converged reference "
          f"(excluded {len(bad)}, load scale {np.round(sorted(s[i] for i in bad), 3).tolist()})")
    mg = np.array([REF[i]["mip_gaps"][-1] for i in ok])*100
    print(f"  reference: final MIP gap median {np.median(mg):.4f}% max {mg.max():.4f}%; "
          f"time median {np.median([REF[i]['secs'] for i in ok]):.0f}s")
    tab = {}
    for name in ARMS:
        rows = []
        for i in ok:
            r = R[name].get(i); j, pref = JR[i]; ur = np.asarray(REF[i]["u"])
            if r is None:   # failure: counted, never dropped
                rows.append((np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, 1.0)); continue
            rows.append((100*(r["cost"]-j)/j, 100*abs(r["cost"]-j)/j, 100*float((r["u"] != ur).mean()),
                         100*float(np.abs(r["pg"]-pref).sum()/pref.sum()), r["secs"], float(r.get("fallback", False)), 0.0))
        v = np.array(rows); tab[name] = v
        print(f"  {name:11s} n={len(ok)} |gap| {np.nanmean(v[:,1]):.4f}% (median {np.nanmedian(v[:,1]):.4f}%) "
              f"signed {np.nanmean(v[:,0]):+.4f}% | disc {np.nanmean(v[:,2]):.2f}% | cont {np.nanmean(v[:,3]):.2f}% | "
              f"time median {np.nanmedian(v[:,4]):.1f}s | fallbacks {int(np.nansum(v[:,5]))} | failures {int(v[:,6].sum())}")
    rng = np.random.default_rng(0); x = tab["ours"][:, 1]
    for base in ("seqQCAC@W", "flatV+cuts", "no-cut"):
        y = tab[base][:, 1]; m = ~np.isnan(x) & ~np.isnan(y); d = x[m] - y[m]
        bs = [rng.choice(d, d.size).mean() for _ in range(5000)]
        p = wilcoxon(x[m], y[m]).pvalue if np.any(d != 0) else 1.0
        print(f"  ours - {base:11s}: mean diff {d.mean():+.4f} pp, 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}], "
              f"Wilcoxon p {p:.3g}, ours better/tie/worse {(d < -1e-6).sum()}/{(abs(d) <= 1e-6).sum()}/{(d > 1e-6).sum()}")
    return tab


def main():
    import combopt_ac, data2
    from pipeline import grid
    from cutnet_k import CutNetK
    g, nl = grid(a.case); G = g.n_gen; INST = data2.all_instances(g, a.case); te, va, tr = data2.split()
    fresh = data2.fresh_ids() if a.test == "fresh" else te; base = float(g.pd.sum())
    X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
    if a.vnet == "none":   # flat V: the headroom checks showed no V beats it on these grids
        V = lambda i: None; t_vnet = 0.0
    else:
        vnet = CutNetK(g.n_bus, G, k_rows=2, v_scale=0.5, hidden=256, seed=0, learn_thr=False)
        vnet.load_state_dict(torch.load(os.path.join(HERE, "checkpoints2", f"{a.case}_s0.pt"), weights_only=False)["net"]); vnet.eval()
        t0 = time.time()
        with torch.no_grad(): VR, VI = [t.numpy().astype(float) for t in vnet(Xn)[:2]]
        t_vnet = (time.time()-t0)/len(INST)
        V = lambda i: (VR[i], VI[i])
    pool = mp.get_context("spawn").Pool(a.workers, initializer=combopt_ac.init, initargs=(a.case, True))
    # ---- frozen protocol: identical to 17_fix_cuts.py ----
    SC = {i: r[1] for i, r in pool.imap_unordered(combopt_ac.relax, tr + va) if r is not None}
    cache = os.path.join(HERE, "results", f"teacher_{a.case}_w{int(a.work_long)}.json")
    if not os.path.exists(cache):   # identical to 17_fix_cuts.py: own long solves on TRAINING instances
        t0 = time.time()
        R = dict(pool.imap_unordered(combopt_ac.miqcqp, [(i, V(i), None, None, a.work_long, a.rho) for i in tr]))
        json.dump({str(i): R[i]["u"].tolist() for i in tr if R.get(i) is not None and not R[i].get("fallback")}, open(cache, "w"))
        print(f"[teacher] built in {time.time()-t0:.0f}s", flush=True)
    TG = {int(k): np.array(v) for k, v in json.load(open(cache)).items()}
    print(f"[teacher] {len(TG)} self-found targets on {len(tr)} training instances", flush=True)
    ids = [i for i in tr if i in TG]
    Y = torch.tensor(np.stack([TG[i] for i in ids]), dtype=torch.float32)
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
    o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(1500):
        o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(m(Xn[ids]), Y); l.backward(); o.step()
    m.eval(); t0 = time.time()
    with torch.no_grad(): Pr = torch.sigmoid(m(Xn)).numpy()
    t_mlp = (time.time()-t0)/len(INST)
    fixes = lambda i, c: ([k for k in range(G) if Pr[i, k] >= c], [k for k in range(G) if Pr[i, k] <= 1 - c])
    run = lambda ids_, c, Vf: dict(pool.imap_unordered(combopt_ac.miqcqp_fix,
                                   [(i, Vf(i), *(fixes(i, c) if c is not None else ([], [])), a.work, a.rho) for i in ids_]))
    sc = lambda R, ids_: float(np.mean([10.0 if R.get(i) is None else R[i]["cost"]/SC[i] for i in ids_]))
    table = {}
    for c in (None, 0.999, 0.99, 0.95, 0.9):
        table[str(c)] = sc(run(va, c, V), va)
    c = float(min((k for k in table if k != "None"), key=table.get))
    Rv = run(va, c, V); tau = 1.02*max(Rv[i]["cost"]/SC[i] for i in va if Rv.get(i) is not None)
    print(f"[frozen] validation table {table} -> c {c}, tau {tau:.4f}  (17_fix_cuts.py logs: case118 0.9/1.0112, case300 0.95/1.0727)", flush=True)
    print(f"[frozen] inference per instance: V-net {1e3*t_vnet:.2f} ms, cut-net {1e3*t_mlp:.2f} ms", flush=True)
    # ---- fresh test: arms first (short), then the references (long), one pool, all cores ----
    t0 = time.time()
    jobs = {"relax": [pool.apply_async(combopt_ac.relax_t, (i,)) for i in fresh]}
    jobs["ours"] = [pool.apply_async(combopt_ac.miqcqp_fix, ((i, V(i), *fixes(i, c), a.work, a.rho),)) for i in fresh]
    jobs["no-cut"] = [pool.apply_async(combopt_ac.miqcqp_fix, ((i, V(i), [], [], a.work, a.rho),)) for i in fresh]
    jobs["flatV+cuts"] = [] if a.vnet == "none" else [pool.apply_async(combopt_ac.miqcqp_fix, ((i, None, *fixes(i, c), a.work, a.rho),)) for i in fresh]
    jobs["seqQCAC@W"] = [pool.apply_async(combopt_ac.seq_budget, ((i, a.work),)) for i in fresh]
    refj = [] if a.no_refs else [pool.apply_async(combopt_ac.ref_tight, ((i, a.ref_gap, a.ref_tl),)) for i in fresh]
    arms = {}
    for name in ("relax", "ours", "no-cut", "flatV+cuts", "seqQCAC@W"):
        arms[name] = dict(j.get() for j in jobs[name]) if jobs[name] else dict(arms["ours"])   # vnet none: identical to ours
        print(f"[arm] {name} done ({time.time()-t0:.0f}s)", flush=True)
    # safeguard: deployed cost / relaxation > tau -> the no-cut solve at W (identical to a
    # re-solve without fixes, WorkLimit is deterministic); keep the cheaper. Its time counts
    # the relaxation (every instance) and, when flagged, the extra solve.
    RG, nflag = {}, 0
    for i in fresh:
        r, rx, rn = arms["ours"].get(i), arms["relax"].get(i), arms["no-cut"].get(i)
        if r is None: RG[i] = None; continue
        r = dict(r); r["secs"] += rx[2] if rx else 0.0
        if rx is not None and r["cost"]/rx[1] > tau:
            nflag += 1; r["secs"] += rn["secs"] if rn else 0.0
            if rn is not None and rn["cost"] < r["cost"]:
                r.update(u=rn["u"], cost=rn["cost"], pg=rn["pg"], fallback=rn.get("fallback", False))
        RG[i] = r
    arms["ours+guard"] = RG
    print(f"[guard] tau {tau:.4f}: flagged {nflag}/{len(fresh)}", flush=True)
    s = {i: float(INST[i][0].sum()/base) for i in fresh}
    D = dict(case=a.case, args=vars(a), ids=fresh, c=c, tau=tau, val=table, scale=s, arms=arms, ref={}, jref={})
    pickle.dump(D, open(OUT, "wb"))
    if a.no_refs:
        pool.close(); print("[arms saved]", OUT, flush=True); return
    for k, j in enumerate(refj):
        i, r = j.get(); D["ref"][i] = r
        print(f"[ref {k+1}/{len(fresh)}] {i} {r.get('status', r.get('err'))} slack {r.get('slack', np.nan):.1e} "
              f"gap {100*(r.get('mip_gaps') or [np.nan])[-1]:.3f}% {r.get('secs', np.nan):.0f}s  [{time.time()-t0:.0f}s]", flush=True)
        pickle.dump(D, open(OUT, "wb"))
    ok = [i for i in fresh if D["ref"][i].get("converged")]
    D["jref"] = dict(pool.imap_unordered(combopt_ac.price_ref, [(i, D["ref"][i]["u"]) for i in ok]))
    pickle.dump(D, open(OUT, "wb")); pool.close()
    score(D)


if __name__ == "__main__":
    main()

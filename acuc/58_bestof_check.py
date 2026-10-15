# %% LABEL-FREE teacher check, option 1 (2026-10-11): BEST OF several cheap own candidates, TRAINING instances only.
#    Candidates per instance: (a) own short MIQCQP W (no cuts) + exact-layer improve  [from run 57, reused]
#                             (b) restoration of the relaxation at learned V, round 0.8 + merit order
#                             (c) restoration of the relaxation at learned V, round 0.5 + confidence order
#    Pick = the cheapest under the SAME strict evaluator (label-free). Nothing solved longer than deployment.
#    Same PRE-REGISTERED bar as 57 vs the long-solve teacher: mean <= 0.05% AND median <= 0.01%.
#    Four metrics vs the converged reference = [diag only] report.   Usage: python 58_bestof_check.py <case> <run-57 folder>
import os, sys, glob, json, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE = sys.argv[1] if len(sys.argv) > 1 else "case118"
R57 = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(HERE, "runs", f"*_selfimprove_check_{CASE}")))[-1]
NPROC = 10


# %% main
def main():
    import combopt_ac, selfsup, data2, runlog
    from pipeline import grid
    RUN = runlog.start(f"bestof_check_{CASE}")
    D = pickle.load(open(os.path.join(R57, "selfimprove.pkl"), "rb")); SI, PR57, ids = D["SI"], D["PR"], D["both"]
    g, nl = grid(CASE); n = g.n_bus; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    VS = torch.load(os.path.join(HERE, "results", f"vnet_layer_{CASE}_env.pt"), weights_only=False)
    vm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2*n)); vm.load_state_dict(VS["state"]); vm.eval()
    with torch.no_grad(): OUT = vm(Xn).numpy().astype(float)
    V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    print(f"[config] {CASE}: {len(ids)} training instances; candidates reused from {os.path.basename(R57)} + 2 restorations at V-net s0", flush=True)
    pool = mp.get_context("spawn").Pool(NPROC, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    RC = dict(pool.imap_unordered(selfsup.restore_candidates, [(i, V(i)) for i in ids]))
    keys = [(nm, i) for i in ids for nm in RC[i]["cands"]]
    PR = dict(zip(keys, [r for _, r in pool.imap(combopt_ac.price_strict, [(i, RC[i]["cands"][nm]) for nm, i in keys])])); pool.close()
    for i in ids:
        PR[("W10 + improve", i)] = PR57[("self", i)]; PR[("W10", i)] = PR57[("own", i)]; PR[("teacher", i)] = PR57[("teacher", i)]
    pick = {}
    for i in ids:
        c = {nm: PR[(nm, i)] for nm in ("W10 + improve", "restore merit0.8", "restore conf0.5") if PR.get((nm, i)) is not None}
        if c: best = min(c, key=lambda k: c[k][0]); PR[("best of", i)] = c[best]; pick[i] = best
    ids = [i for i in ids if i in pick]
    print(f"[done] {len(ids)} instances with a priced pick  [{time.time()-t0:.0f}s]; picked: "
          + ", ".join(f"{k} {sum(v == k for v in pick.values())}" for k in ("W10 + improve", "restore merit0.8", "restore conf0.5")), flush=True)

    # %% label-free decision: vs the long-solve teacher (same strict evaluator)
    d = np.array([100*(PR[("best of", i)][0] - PR[("teacher", i)][0])/PR[("teacher", i)][0] for i in ids])
    passed = d.mean() <= 0.05 and np.median(d) <= 0.01
    print(f"   best of vs teacher W=120: mean {d.mean():+.4f}% | median {np.median(d):+.4f}% | p90 {np.percentile(d, 90):+.4f}% | max {d.max():+.4f}% | "
          f"cheaper {np.mean(d < -1e-6):.2f}", flush=True)
    print(f"[BAR] mean <= 0.05% and median <= 0.01%: {'PASSED -> long-solve teacher can be replaced' if passed else 'FAILED'}", flush=True)

    # %% [diag only] four metrics vs the converged reference (report only; decides nothing)
    SR = data2.strict_refs(CASE); dg = [i for i in ids if SR.get(i) is not None and SR[i][4] == 0]
    print(f"   [diag only] four metrics vs the converged reference, n={len(dg)} training instances", flush=True)
    print("   version                       | gap mean | gap median | discrete err | continuous err", flush=True)
    lines = {}
    for a in ("teacher", "W10", "W10 + improve", "restore merit0.8", "restore conf0.5", "best of"):
        sub = [i for i in dg if PR.get((a, i)) is not None]
        v = np.array([(100*abs(PR[(a, i)][0] - SR[i][0])/SR[i][0], 100*np.mean(np.asarray(PR[(a, i)][5]) != np.asarray(SR[i][5])),
                       100*np.abs(PR[(a, i)][1] - SR[i][1]).sum()/SR[i][1].sum()) for i in sub])
        lines[a] = f"{v[:,0].mean():.4f}% | {np.median(v[:,0]):.4f}% | {v[:,1].mean():.2f}% | {v[:,2].mean():.2f}%" + ("" if len(sub) == len(dg) else f" (n={len(sub)})")
        print(f"   {a:29s} | {lines[a]}", flush=True)
    runlog.add_result(RUN, f"{CASE} TRAIN label-free best-of teacher [diag four metrics vs ref: {lines['best of']}] vs teacher mean {d.mean():+.4f}% median "
                           f"{np.median(d):+.4f}% -> bar {'PASSED' if passed else 'FAILED'}")
    runlog.save(RUN, "bestof.pkl", dict(ids=ids, PR=PR, pick=pick, RC=RC))
    json.dump({str(i): np.asarray(PR[("best of", i)][5]).tolist() for i in ids}, open(os.path.join(RUN, f"selfteacher_{CASE}_bestof.json"), "w"))


if __name__ == "__main__":
    main()

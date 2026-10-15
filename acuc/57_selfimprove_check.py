# %% FALSIFYING CHECK for a TRULY label-free cut teacher (2026-10-11), TRAINING instances only.
#    Today's integer cuts imitate our own LONG solves (teacher_<case>_w120/180.json) -- self-generated near-optimal labels.
#    Candidate replacement: own DEPLOYMENT (no cuts, short MIQCQP at W, learned V seed 0) + exact-layer-guided local search
#    (src/selfsup.deploy_improve; accepts a move only if the priced cost drops). Nothing is solved with more budget than deployment.
#    Question: are the self-improved commitments as cheap as the long-solve teacher? Both priced with the SAME strict evaluator.
#    PRE-REGISTERED bar: mean (self - teacher)/teacher <= 0.05% AND median <= 0.01% -> the long-solve teacher can be dropped.
#    The four-metric line vs the reference is [diag only] (report, never used to decide anything). No validation/test instance is touched.
#    Usage: python 57_selfimprove_check.py <case>
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE = sys.argv[1] if len(sys.argv) > 1 else "case118"
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0); WLONG = {"case118": 120, "case300": 180}.get(CASE, 120)
K, ROUNDS, SWAPS, NPROC = 6, 8, 4, 10


# %% main
def main():
    import combopt_ac, selfsup, data2, runlog
    from pipeline import grid
    RUN = runlog.start(f"selfimprove_check_{CASE}")
    g, nl = grid(CASE); n = g.n_bus; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    VS = torch.load(os.path.join(HERE, "results", f"vnet_layer_{CASE}_env.pt"), weights_only=False); RHO = VS["rho"]
    vm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2*n)); vm.load_state_dict(VS["state"]); vm.eval()
    with torch.no_grad(): OUT = vm(Xn).numpy().astype(float)
    V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    TG = {int(k): np.array(v, float) for k, v in json.load(open(os.path.join(HERE, "results", f"teacher_{CASE}_w{WLONG}.json"))).items()}
    ids = [i for i in tr if i in TG]
    print(f"[config] {CASE}: {len(ids)} training instances, own deployment W={W} (no cuts) at V-net s0 (rho {RHO:.0e}) + improve(k={K}, "
          f"rounds={ROUNDS}, swaps={SWAPS}) vs long-solve teacher W={WLONG}; {NPROC} workers", flush=True)
    pool = mp.get_context("spawn").Pool(NPROC, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    SI = {}
    for k_, (i, r) in enumerate(pool.imap_unordered(selfsup.deploy_improve, [(i, V(i), None, None, W, RHO, K, ROUNDS, SWAPS) for i in ids])):
        SI[i] = r
        if (k_ + 1) % 12 == 0 or k_ + 1 == len(ids):
            print(f"   [{k_+1}/{len(ids)}] {time.time()-t0:.0f}s", flush=True)
    ok = [i for i in ids if SI.get(i) is not None]
    jobs = [(i, SI[i]["u0"]) for i in ok] + [(i, SI[i]["u"]) for i in ok] + [(i, TG[i]) for i in ok]
    keys = [("own", i) for i in ok] + [("self", i) for i in ok] + [("teacher", i) for i in ok]
    PR = dict(zip(keys, [r for _, r in pool.imap(combopt_ac.price_strict, jobs)])); pool.close()
    both = [i for i in ok if PR[("self", i)] is not None and PR[("teacher", i)] is not None and PR[("own", i)] is not None]
    print(f"[done] {len(ok)}/{len(ids)} deployed, {len(both)} strictly priced in all three versions  [{time.time()-t0:.0f}s]", flush=True)

    # %% label-free comparison: our own commitments vs the long-solve teacher (same strict evaluator)
    def rel(a):
        return np.array([100*(PR[(a, i)][0] - PR[("teacher", i)][0])/PR[("teacher", i)][0] for i in both])
    lines = {}
    for a, nm in (("own", f"own deployment W={W} (no cuts)"), ("self", f"own deployment + exact-layer improve")):
        d = rel(a); dis = np.mean([100*np.mean(np.asarray(PR[(a, i)][5]) != np.asarray(PR[("teacher", i)][5])) for i in both])
        lines[a] = (f"vs teacher W={WLONG}: mean {d.mean():+.4f}% | median {np.median(d):+.4f}% | p90 {np.percentile(d, 90):+.4f}% | "
                    f"max {d.max():+.4f}% | within 0.01% {np.mean(np.abs(d) <= 0.01):.2f} | cheaper {np.mean(d < -1e-6):.2f} | unit disagreement {dis:.2f}%")
        print(f"   {nm:40s} {lines[a]}", flush=True)
    d = rel("self"); passed = d.mean() <= 0.05 and np.median(d) <= 0.01
    sec = np.array([SI[i]["secs"] for i in both]); mv = np.array([SI[i]["n_moves"] for i in both])
    print(f"   time per instance (deploy + improve): median {np.median(sec):.1f}s, max {sec.max():.1f}s | units changed by improve: mean {mv.mean():.2f}, "
          f"instances changed {np.mean(mv > 0):.2f} | fallbacks {sum(SI[i]['fallback'] for i in both)}", flush=True)
    print(f"[BAR] mean <= 0.05% and median <= 0.01%: {'PASSED -> long-solve teacher can be replaced' if passed else 'FAILED'}", flush=True)

    # %% [diag only] four metrics vs the converged reference (report only; decides nothing)
    SR = data2.strict_refs(CASE); dg = [i for i in both if SR.get(i) is not None and SR[i][4] == 0]
    for a in ("teacher", "own", "self"):
        v = np.array([(100*abs(PR[(a, i)][0] - SR[i][0])/SR[i][0], 100*np.mean(np.asarray(PR[(a, i)][5]) != np.asarray(SR[i][5])),
                       100*np.abs(PR[(a, i)][1] - SR[i][1]).sum()/SR[i][1].sum()) for i in dg])
        print(f"   [diag only] {a:8s} vs reference (n={len(dg)}): gap mean {v[:,0].mean():.4f}% | median {np.median(v[:,0]):.4f}% | "
              f"disc {v[:,1].mean():.2f}% | cont {v[:,2].mean():.2f}%", flush=True)
    runlog.add_result(RUN, f"{CASE} TRAIN label-free teacher check: own deploy+improve {lines['self']} -> bar {'PASSED' if passed else 'FAILED'}")
    runlog.save(RUN, "selfimprove.pkl", dict(ids=ids, both=both, SI=SI, PR=PR, cfg=dict(W=W, RHO=RHO, K=K, ROUNDS=ROUNDS, SWAPS=SWAPS)))
    # self-improved targets for the next step (training), same format as the teacher json
    json.dump({str(i): SI[i]["u"].tolist() for i in ok}, open(os.path.join(RUN, f"selfteacher_{CASE}_r0.json"), "w"))


if __name__ == "__main__":
    main()

# %% LABEL-FREE CUT-VALIDITY CHECK at deployment (2026-10-11), VALIDATION, integer cuts only.
#    Deployment today: learned V -> learned per-unit integer cuts (hard-rounded rows) -> one short MIQCQP (W) -> strict pricing.
#    Check (src/selfsup.check_rows): before the MIQCQP, solve the convex relaxation at V with the cut rows; rows the relaxation
#    must violate are released. Diagnosis on case118: the label-free-teacher cuts force a violation on exactly the 3 validation
#    instances where they lose (141, 149, 185); the old cuts on none.
#    Arms differ ONLY by (teacher of the cuts) x (check on/off). Same V-net s0, c, W, rho. General rows not included.
#    Four metrics vs the converged reference = report only.   Usage: python 59_cut_check_val.py <case>
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE = sys.argv[1] if len(sys.argv) > 1 else "case118"
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0); NPROC, TOL, PASSES = 10, 1e-4, 3
CUTS = {"long-solve teacher": f"unit_cuts_layer_{CASE}_soft.pt", "label-free teacher": f"unit_cuts_layer_{CASE}_soft_lf.pt",
        "label-free + 300 extra draws": f"unit_cuts_layer_{CASE}_soft_lfx.pt"}
CUTS = {k: f for k, f in CUTS.items() if os.path.exists(os.path.join(HERE, "results", f))}


# %% main
def main():
    import combopt_ac, selfsup, data2, runlog
    from pipeline import grid
    RUN = runlog.start(f"cut_check_val_{CASE}")
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    VS = torch.load(os.path.join(HERE, "results", f"vnet_layer_{CASE}_env.pt"), weights_only=False); RHO = VS["rho"]
    vm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2*n)); vm.load_state_dict(VS["state"]); vm.eval()
    with torch.no_grad(): OUT = vm(Xn).numpy().astype(float)
    V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    SR = data2.strict_refs(CASE); vv = [i for i in va if SR.get(i) is not None and SR[i][4] == 0]
    print(f"[config] {CASE}: {len(vv)} validation instances with references, W={W}, V-net s0 rho {RHO:.0e}, check tol {TOL}", flush=True)

    def rows(f):                                  # hard-rounded integer cut rows exactly as in 52/55
        UC = torch.load(os.path.join(HERE, "results", f), weights_only=False); c = UC["c"]
        cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G)); cm.load_state_dict(UC["state"]); cm.eval()
        with torch.no_grad(): M = torch.sigmoid(cm(Xn)).numpy()
        LO = np.maximum((M - c)/(1 - c), 0); HI = 1 - np.maximum(((1 - c) - M)/(1 - c), 0); out = {}
        for i in vv:
            A = np.zeros((2*G, 2*G)); b = np.ones(2*G)
            for k in range(G):
                if LO[i, k] > 0: A[k, G + k] = -1.0; b[k] = -1.0
                if HI[i, k] < 1: A[G + k, G + k] = 1.0; b[G + k] = 0.0
            out[i] = (A, b)
        return out
    pool = mp.get_context("spawn").Pool(NPROC, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    res, info = {}, {}
    for tname, f in CUTS.items():
        Rw = rows(f)
        CK = dict(pool.imap_unordered(selfsup.check_rows, [(i, V(i), RHO, *Rw[i], TOL, PASSES) for i in vv]))
        rel = {i: CK[i][2] for i in vv if CK[i][2]}
        info[tname] = dict(released=rel, secs=np.median([CK[i][3] for i in vv]))
        print(f"[check] {tname}: rows released on {len(rel)}/{len(vv)} instances "
              + ", ".join(f"#{i}: {len(r)} rows" for i, r in rel.items()) + f" | check time median {info[tname]['secs']:.2f}s  [{time.time()-t0:.0f}s]", flush=True)
        for chk, RR in (("no check", Rw), ("with check", {i: CK[i][:2] for i in vv})):
            sol = dict(pool.imap_unordered(combopt_ac.miqcqp, [(i, V(i), *RR[i], W, RHO) for i in vv]))
            pr = dict(pool.imap_unordered(combopt_ac.price_strict, [(i, sol[i]["u"]) for i in vv if sol.get(i)]))
            res[(tname, chk)] = (sol, pr)
            print(f"   deployed {tname}, {chk}  [{time.time()-t0:.0f}s]", flush=True)
    pool.close()

    # %% REPORT: four metrics vs the converged reference (report only)
    print(f"\n[REPORT] {CASE} VALIDATION, {len(vv)} instances, integer cuts only, W={W}", flush=True)
    print("   version                                    | gap mean | gap median | discrete err | continuous err | failures", flush=True)
    for (tname, chk), (sol, pr) in res.items():
        v = np.array([(100*abs(pr[i][0]-SR[i][0])/SR[i][0], 100*np.mean(np.asarray(pr[i][5]) != np.asarray(SR[i][5])),
                       100*np.abs(pr[i][1]-SR[i][1]).sum()/SR[i][1].sum()) if pr.get(i) else (np.nan,)*3 for i in vv])
        line = f"{np.nanmean(v[:,0]):.4f}% | {np.nanmedian(v[:,0]):.4f}% | {np.nanmean(v[:,1]):.2f}% | {np.nanmean(v[:,2]):.2f}% | {int(np.isnan(v[:,0]).sum())}"
        print(f"   {tname + ', ' + chk:42s} | {line}", flush=True)
        worst = np.argsort(-np.nan_to_num(v[:, 0]))[:3]
        print(f"      worst: " + ", ".join(f"#{vv[k]} {v[k,0]:.3f}%" for k in worst), flush=True)
        runlog.add_result(RUN, f"{CASE} VALIDATION integer cuts only, {tname}, {chk}: gap mean | median | disc | cont | failures = {line}")
    runlog.save(RUN, "cut_check.pkl", dict(vv=vv, res=res, info=info))


if __name__ == "__main__":
    main()

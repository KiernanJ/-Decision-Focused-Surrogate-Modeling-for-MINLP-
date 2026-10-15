# %% FULL-METHOD VALIDATION REPORT (2026-10-12): learned V + integer cuts + general rows, deployed EXACTLY as 55_final_eval.py does
#    (anchor solve -> rows; one short MIQCQP W; strict pricing), on the validation draws that have references. Four metrics vs the
#    reference + time + fallbacks. Report only (selects nothing).   Usage: python 61_val_full.py <case> <vnet suffix> <cut/row tag> [label]
#    e.g. 61_val_full.py case118 _env_r2 _lfc  -> V-net results/vnet_layer_case118_env_r2.pt, cuts unit_cuts_layer_case118_soft_lfc.pt,
#         rows general_rows_case118_lfc.pt
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE, VSUF, TAG = sys.argv[1], sys.argv[2], sys.argv[3]; LABEL = sys.argv[4] if len(sys.argv) > 4 else f"V{VSUF} cuts/rows{TAG}"
RHO_OVR = float(sys.argv[7]) if len(sys.argv) > 7 else None   # (2026-10-13 diagnostic) deployment penalty rho override (default: the V file's stored rho)
CHECK = len(sys.argv) > 6 and sys.argv[6] == "check"   # (2026-10-13 diagnostic) label-free relaxation cut-validity check (selfsup.check_rows) before the MIQCQP
NOROWS = len(sys.argv) > 5 and sys.argv[5] == "norows"   # (2026-10-13 diagnostic) integer cuts only: general rows NOT deployed (report-only; never the final method)
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0)


# %% main
def main():
    import combopt_ac, cuttrain, data2, runlog, selfsup
    from pipeline import grid
    from cutnet_k import CutNetK, rows_torch
    from vnetio import vnet_out
    RUN = runlog.start(f"val_full_{CASE}{VSUF}{TAG}")
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    SR = data2.strict_refs(CASE); vv = [i for i in va if SR.get(i) is not None and SR[i][4] == 0]
    vl = list(va)   # (audit 2026-10-13) the label-free rho/selection SCORE uses ALL validation draws; the four-metric REPORT stays on vv (those with converged references)
    OUT, RHO, VSEL = vnet_out(os.path.join(HERE, "results", f"vnet_layer_{CASE}{VSUF}.pt"), Xn, n)
    V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    if RHO_OVR is not None: RHO = RHO_OVR
    UC = torch.load(os.path.join(HERE, "results", f"unit_cuts_layer_{CASE}_soft{TAG}.pt"), weights_only=False); c = UC["c"]
    cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G)); cm.load_state_dict(UC["state"]); cm.eval()
    with torch.no_grad(): Mc = torch.sigmoid(cm(Xn)).numpy()
    LO = np.maximum((Mc - c)/(1 - c), 0); HI = 1 - np.maximum(((1 - c) - Mc)/(1 - c), 0)
    assert "last" in UC and UC["state"] is not None and UC["epoch"] >= 0, f"{CASE}: stale/untrained unit-cut file"            # (audit) as 55
    if NOROWS:
        GR, K, GSEL = {"selected_epoch": "OFF (diagnostic)"}, 2, None
    else:
        GR = torch.load(os.path.join(HERE, "results", f"general_rows_{CASE}{TAG}.pt"), weights_only=False); K = GR["K"]; GSEL = GR.get("selected")
        assert GR.get("trained") is not None and GSEL is not None, f"{CASE}: general rows not trained/selected -> not the full method"   # (audit) as 55
    gnet = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False)
    if GSEL is not None: gnet.load_state_dict(GSEL); gnet.eval()
    print(f"[config] {CASE} {LABEL}: {len(vv)} validation instances, W={W}, V-net {VSUF} ({VSEL}, rho {RHO:.0e}), c={c}, cut epoch {UC['epoch']}, "
          f"rows epoch {GR.get('selected_epoch')}", flush=True)
    pool = mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True))
    ap = mp.get_context("spawn").Pool(11, initializer=cuttrain.init6, initargs=(CASE, 1000.0, K)); t0 = time.time()
    ANT = dict(ap.imap_unordered(cuttrain.anchor6_t, [(i, *V(i), LO[i], HI[i]) for i in vl])) if GSEL is not None else {}
    ap.close()
    BASE = {}
    for i in vl:
        Au = np.zeros((2*G, 2*G)); bu = np.ones(2*G)
        for k in range(G):
            if LO[i, k] > 0: Au[k, G + k] = -1.0; bu[k] = -1.0
            if HI[i, k] < 1: Au[G + k, G + k] = 1.0; bu[G + k] = 0.0
        BASE[i] = (Au, bu)
    t_chk, rel = {}, {}
    if CHECK:
        CK = dict(pool.imap_unordered(selfsup.check_rows, [(i, V(i), RHO, *BASE[i], 1e-4, 3) for i in vl]))
        for i in vl: BASE[i] = CK[i][:2]; t_chk[i] = CK[i][3]; rel[i] = CK[i][2]
        print(f"[check] rows released on {sum(bool(r) for r in rel.values())}/{len(vv)} instances: " + ", ".join(f"#{i}:{len(r)}" for i, r in rel.items() if r), flush=True)
    jobs = []
    for i in vl:
        Au, bu = BASE[i]
        if ANT.get(i) is not None and ANT[i][0] is not None:
            with torch.no_grad():
                _, _, a, al, a2, d, d2, _ = gnet(Xn[[i]]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.tensor(ANT[i][0], dtype=torch.float32)[None])
            Au = np.vstack([Au, A_[0, :K].numpy().astype(float)]); bu = np.r_[bu, b_[0, :K].numpy().astype(float)]
        jobs.append((i, V(i), Au, bu, W, RHO))
    pool2 = pool
    sol = dict(pool.imap_unordered(combopt_ac.miqcqp, jobs))
    OP = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, sol[i]["u"]) for i in vl if sol.get(i)]))
    v = np.array([(100*abs(OP[i]["cost"] - SR[i][0])/SR[i][0], 100*np.mean(OP[i]["u"] != np.asarray(SR[i][5])), 100*np.abs(OP[i]["pg"] - SR[i][1]).sum()/SR[i][1].sum(),
                   (ANT[i][1] if ANT.get(i) else 0.0) + t_chk.get(i, 0.0) + sol[i]["secs"] + OP[i]["secs"]) if OP.get(i) else (np.nan,)*4 for i in vv])
    fb = sum(bool(sol[i].get("fallback")) for i in vl if sol.get(i))
    RL = dict(pool2.imap_unordered(combopt_ac.relax, vl)); lf = float(np.mean([OP[i]["cost"]/RL[i][1] if (OP.get(i) and RL.get(i)) else 10.0 for i in vl]))
    print(f"[label-free score of the deployed method (strict price / flat-V relaxation cost), rho {RHO:.0e}, ALL {len(vl)} validation draws] {lf:.6f}", flush=True)
    pool.close()
    if not vv:   # (2026-10-13) PGLib has no validation references: label-free score only, no four-metric validation report
        runlog.add_result(RUN, f"{CASE} VALIDATION FULL METHOD ({LABEL}): rho {RHO:.0e}, label-free score {lf:.6f} (no validation references -> no four-metric report), fallbacks {fb}")
        runlog.save(RUN, "val_full.pkl", dict(vv=vv, sol=sol, OP=OP, lf=lf)); return
    line = f"{np.nanmean(v[:,0]):.4f}% | {np.nanmedian(v[:,0]):.4f}% | {np.nanmean(v[:,1]):.2f}% | {np.nanmean(v[:,2]):.2f}% | {np.nanmedian(v[:,3]):.1f}s (max {np.nanmax(v[:,3]):.1f}s) | fallbacks {fb} | failures {int(np.isnan(v[:,0]).sum())}"
    worst = np.argsort(-np.nan_to_num(v[:, 0]))[:3]
    print(f"[REPORT] {CASE} VALIDATION full method ({LABEL}): gap mean | median | disc | cont | time median | fallbacks | failures = {line}", flush=True)
    print("   worst: " + ", ".join(f"#{vv[k]} {v[k,0]:.3f}%" for k in worst) + f"  [{time.time()-t0:.0f}s]", flush=True)
    runlog.add_result(RUN, f"{CASE} VALIDATION FULL METHOD ({LABEL}): gap mean | median | disc | cont | time | fallbacks | failures = {line}")
    import hashlib
    files = (f"vnet_layer_{CASE}{VSUF}.pt", f"unit_cuts_layer_{CASE}_soft{TAG}.pt") + (() if NOROWS else (f"general_rows_{CASE}{TAG}.pt",))
    hsh = {f: hashlib.sha1(open(os.path.join(HERE, "results", f), "rb").read()).hexdigest()[:12] for f in files}
    print("   files: " + ", ".join(f"{k} {v}" for k, v in hsh.items()), flush=True)
    runlog.save(RUN, "val_full.pkl", dict(vv=vv, v=v, sol=sol, OP=OP, files=hsh))


if __name__ == "__main__":
    main()

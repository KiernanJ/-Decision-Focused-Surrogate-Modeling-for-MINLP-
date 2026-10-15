# %% (2026-10-10) GENERAL mixed cut rows  alpha'pg + a'u <= b  ON TOP of the per-unit INTEGER cuts (both categories of
#    the abstract), case118, label-free. Integer cuts = the layer-trained cut-net (results/unit_cuts_layer_case118_soft.pt,
#    frozen). General rows: CutNetK general category (K rows), anchored at the relaxation WITH the integer cuts at the
#    learned V, trained THROUGH THE SMOOTHED EXACT LAYER (gamma=1000) with tracking (own long solves) + validity.
#    Deploy: one short MIQCQP (W=10) at the learned V with integer cuts + general rows. VALIDATION report: four metrics.
#    (audit 2026-10-11) LABEL-FREE selection on ALL validation instances (deployed strict price / flat-V relaxation cost) among
#    {rows OFF, init rows, every epoch}; every epoch's weights saved; 'selected' = chosen state or None (= OFF). If the FD gate
#    fails the rows are OFF (never deploy untrained rows). The reference-based four-metric report is REPORT ONLY.  FD gate first.
import os, sys, json, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE = sys.argv[1] if len(sys.argv) > 1 else "case118"
K, GAMMA, EPOCHS, BATCH, LR, LAM = 2, 1000.0, 10, 24, 2e-3, 10.0
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0); WLONG = {"case118": 120, "case300": 180}.get(CASE, 120)
# (2026-10-11) optional: argv[2] = teacher json, argv[3] = tag (reads unit_cuts_layer_<case>_soft<tag>.pt, writes general_rows_<case><tag>.pt)
TEACHER = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", f"teacher_{CASE}_w{WLONG}.json")
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
EXTRA = len(sys.argv) > 4 and sys.argv[4] == "extra"
VTAG = sys.argv[5] if len(sys.argv) > 5 else ""   # (2026-10-12) V-net file suffix ("_nc" = clean V-net, uncut relaxation)
VALUEH = len(sys.argv) > 6 and sys.argv[6] == "valueh"   # (2026-10-13) HARDENED value loss (cuttrain.value_gen_h) + LINEAR validity, gamma=0
VALUE = (len(sys.argv) > 6 and sys.argv[6] == "value") or VALUEH   # (2026-10-12, option A) value loss + exact envelope gradient (cuttrain.value_gen), gamma=0


def main():
    import combopt_ac, cuttrain, data2, runlog
    from pipeline import grid
    from cutnet_k import CutNetK, rows_torch
    RUN = runlog.start(f"general_rows_on_top_{CASE}{TAG}")
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    from vnetio import vnet_out
    OUT, RHO, VSEL = vnet_out(os.path.join(HERE, "results", f"vnet_layer_{CASE}_env{VTAG}.pt"), Xn, n)
    print(f"[config] {CASE}: W={W}, teacher {os.path.basename(TEACHER)}, V-net '{VTAG}' (selected {VSEL}) rho {RHO:.0e}, tag '{TAG}'", flush=True)
    V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    UC = torch.load(os.path.join(HERE, "results", f"unit_cuts_layer_{CASE}_soft{TAG}.pt"), weights_only=False); c = UC["c"]
    assert "last" in UC and UC["state"] is not None and UC["epoch"] >= 0, "stale/untrained unit-cut file"   # (pre-final audit)
    cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G)); cm.load_state_dict(UC["state"]); cm.eval()
    with torch.no_grad(): M = torch.sigmoid(cm(Xn)).numpy()
    LO = np.maximum((M - c)/(1 - c), 0); HI = 1 - np.maximum(((1 - c) - M)/(1 - c), 0)
    def unit_fix_rows(i):        # deployment: integer rounding of the per-unit cuts (u_k >= l_k > 0 -> u_k = 1 ; u_k <= h_k < 1 -> u_k = 0)
        A = np.zeros((2*G, 2*G)); b = np.ones(2*G)
        for k in range(G):
            if LO[i, k] > 0: A[k, G + k] = -1.0; b[k] = -1.0
            if HI[i, k] < 1: A[G + k, G + k] = 1.0; b[G + k] = 0.0
        return A, b
    TG = {int(k): np.array(v, float) for k, v in json.load(open(TEACHER)).items()}
    dp = mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    trn = [i for i in tr + (data2.extra_ids() if EXTRA else []) if i in TG]
    print(f"[config] training instances with a target: {len(trn)} ({'original split + extra unlabeled draws' if EXTRA else 'original split'})", flush=True)
    PO = dict(dp.imap_unordered(combopt_ac.price_strict, [(i, TG[i]) for i in trn])); trn = [i for i in trn if PO.get(i) is not None and PO[i][4] == 0]
    WOWN = {i: torch.tensor(np.r_[PO[i][1], TG[i]], dtype=torch.float32) for i in trn}
    SR = data2.strict_refs(CASE); vv = list(va)                                   # label-free selection uses ALL validation instances
    SC = {i: r[1] for i, r in dp.imap_unordered(combopt_ac.relax, vv) if r is not None}; vv = [i for i in vv if i in SC]
    lp = mp.get_context("spawn").Pool(11, initializer=cuttrain.init6, initargs=(CASE, 0.0 if VALUE else GAMMA, K))
    TRG = cuttrain.value_gen_h if VALUEH else (cuttrain.value_gen if VALUE else cuttrain.track_gen)
    print(f"[config] row loss: {'VALUE (gap closed, capped) + LAM*validity, exact envelope gradient, no smoothing' if VALUE else 'tracking (KKT VJP)'}", flush=True)
    # (audit 2026-10-12) the row ANCHORS must be computed exactly as deployment (55/61: init6 with gamma=1000), even in value mode
    if VALUE:
        apool = mp.get_context("spawn").Pool(11, initializer=cuttrain.init6, initargs=(CASE, GAMMA, K))
        AN = dict(apool.imap_unordered(cuttrain.anchor6, [(i, *V(i), LO[i], HI[i]) for i in trn + vv])); apool.close()
    else:
        AN = dict(lp.imap_unordered(cuttrain.anchor6, [(i, *V(i), LO[i], HI[i]) for i in trn + vv]))
    trn = [i for i in trn if AN.get(i) is not None]; vv = [i for i in vv if AN.get(i) is not None]
    Wr = {i: torch.tensor(AN[i], dtype=torch.float32) for i in trn + vv}
    net = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False)
    init_state = {k: t.clone() for k, t in net.state_dict().items()}
    def grows(idx, grad):
        with torch.set_grad_enabled(grad):
            _, _, a, al, a2, d, d2, _ = net(Xn[idx]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.stack([Wr[i] for i in idx]))
        return A_[:, :K], b_[:, :K]                              # rows 0..K-1 = the GENERAL (mixed) category
    # ---- FD gate (general-row parameters) ----
    rng = np.random.default_rng(0); gate = []
    for i in trn[:3]:
        with torch.no_grad(): Ag, bg = [t[0].numpy().astype(float) for t in grows([i], False)]
        _, R0 = lp.apply(TRG, ((i, *V(i), LO[i], HI[i], Ag, bg, TG[i], True),))
        if R0 is None: print(f"[gate] inst {i}: failed", flush=True); continue
        L0, gA, gb = R0
        dirs = []
        for _ in range(8):
            dA = rng.standard_normal(Ag.shape); db = rng.standard_normal(bg.shape); nr = np.sqrt((dA**2).sum() + (db**2).sum()); dA /= nr; db /= nr; dirs.append((dA, db))
        # (2026-10-11) at h=1e-2 one-sided FDs disagree through CURVATURE (case300/PG: self-consistency 0.93-0.95 while central FD vs analytic
        # corr 0.95-1.00) -> retry the SAME directions at smaller h; same pass bar (self-consistency >= 0.99, corr >= 0.95)
        for h in (1e-2, 3e-3, 1e-3):
            jobs = []
            for dA, db in dirs: jobs += [(i, *V(i), LO[i], HI[i], Ag + h*dA, bg + h*db, TG[i], False), (i, *V(i), LO[i], HI[i], Ag - h*dA, bg - h*db, TG[i], False)]
            res = [r for _, r in lp.imap(TRG, jobs)]; okd = [k for k in range(8) if res[2*k] is not None and res[2*k+1] is not None]
            fd = np.array([(res[2*k][0] - res[2*k+1][0])/(2*h) for k in okd]); fw = np.array([(res[2*k][0] - L0)/h for k in okd]); bw = np.array([(L0 - res[2*k+1][0])/h for k in okd])
            an = np.array([(gA*dirs[k][0]).sum() + gb @ dirs[k][1] for k in okd]); cfd = np.corrcoef(fw, bw)[0, 1]; ca = np.corrcoef(fd, an)[0, 1]
            print(f"[gate] inst {i} h={h:.0e}: {len(okd)}/8 | FD self-consistency {cfd:.3f} | analytic vs FD corr {ca:.4f} | |FD| {np.abs(fd).mean():.2e}", flush=True)
            if np.isfinite(cfd) and cfd >= 0.99: break
        gate.append((cfd, ca))
    def deploy_jobs(st):
        jobs = []
        for i in vv:
            Au, bu = unit_fix_rows(i)
            if st is None: jobs.append((i, V(i), Au, bu, W, RHO)); continue
            net.load_state_dict(st)
            with torch.no_grad(): Ag, bg = [t[0].numpy().astype(float) for t in grows([i], False)]
            jobs.append((i, V(i), np.vstack([Au, Ag]), np.r_[bu, bg], W, RHO))
        return jobs
    def lf_score(st):                                       # LABEL-FREE validation score (no reference read)
        cur = {k: t.clone() for k, t in net.state_dict().items()}
        sol = dict(dp.imap_unordered(combopt_ac.miqcqp, deploy_jobs(st))); net.load_state_dict(cur)
        pr = dict(dp.imap_unordered(combopt_ac.price_strict, [(i, sol[i]["u"]) for i in vv if sol.get(i)]))
        return float(np.mean([pr[i][0]/SC[i] if pr.get(i) else 10.0 for i in vv])), sum(bool(sol[i].get("fallback")) for i in vv if sol.get(i))
    s_off = lf_score(None); s_init = lf_score(init_state); epoch_states, sel_hist = {}, [("off", *s_off), (-1, *s_init)]
    print(f"[val] label-free: rows OFF {s_off[0]:.6f} (fallbacks {s_off[1]}) | init rows {s_init[0]:.6f} (fallbacks {s_init[1]})", flush=True)
    good = [ca for cfd, ca in gate if np.isfinite(cfd) and np.isfinite(ca) and cfd >= 0.99]   # (audit 2026-10-13) NaN-blind rule fixed
    print(f"[gate] self-consistent FD on {len(good)}/{len(gate)} instances", flush=True)
    if not good or min(good) < 0.95:
        print("[gate] FAILED or signal too small -> no training; report integer cuts only", flush=True); trained = None
    else:
        print("[gate] PASSED -> train general rows", flush=True)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        for ep in range(EPOCHS):
            order = rng.permutation(trn); Lt, Lv, nf = [], [], 0
            for s0 in range(0, len(order), BATCH):
                b = [int(i) for i in order[s0:s0+BATCH]]
                A_t, b_t = grows(b, True); An, bn = A_t.detach().numpy().astype(float), b_t.detach().numpy().astype(float)
                R = dict(lp.imap_unordered(TRG, [(i, *V(i), LO[i], HI[i], An[k], bn[k], TG[i], True) for k, i in enumerate(b)]))
                GA, GB, ok = np.zeros_like(An), np.zeros_like(bn), 0
                for k, i in enumerate(b):
                    if R.get(i) is None: nf += 1; continue
                    Lt.append(R[i][0]); GA[k], GB[k] = R[i][1], R[i][2]; ok += 1
                if ok == 0: continue
                viol = torch.relu(torch.einsum("nkd,nd->nk", A_t, torch.stack([WOWN[i] for i in b])) - b_t); lval = (LAM*viol.sum(1).mean() if VALUEH else LAM*(viol**2).sum(1).mean()); Lv.append(float(lval))
                opt.zero_grad(); ((A_t*torch.tensor(GA, dtype=torch.float32)).sum() + (b_t*torch.tensor(GB, dtype=torch.float32)).sum()).div(ok).add(lval).backward(); opt.step()
            epoch_states[ep] = {k: t.clone() for k, t in net.state_dict().items()}; se = lf_score(epoch_states[ep]); sel_hist.append((ep, *se))
            print(f"[ep {ep}] tracking {np.mean(Lt):.5f} validity {np.mean(Lv):.5f} (failed {nf}) | label-free val {se[0]:.6f} (fallbacks {se[1]})  [{time.time()-t0:.0f}s]", flush=True)
        trained = {k: t.clone() for k, t in net.state_dict().items()}
    lp.close()
    # (2026-10-11, Parth) both cut categories must be LEARNED: label-free selection among TRAINED epochs only (OFF / init scores are report-only)
    cand = [(e, sc_) for e, sc_, _ in sel_hist[2:]] if trained is not None else [("off", s_off[0])]
    if trained is None: print("[select] FD GATE FAILED -> rows NOT trained; must be diagnosed (not deployable as learned rows)", flush=True)
    pick = min(cand, key=lambda t: t[1]); selected = None if pick[0] == "off" else (init_state if pick[0] == -1 else epoch_states[pick[0]])
    print(f"[select] label-free: {'rows OFF' if pick[0] == 'off' else f'epoch {pick[0]}'} (score {pick[1]:.6f}; off {s_off[0]:.6f})", flush=True)
    G_ = dict(init=init_state, trained=trained, epochs=epoch_states, selected=selected, selected_epoch=pick[0], sel_hist=sel_hist, K=K)
    torch.save(G_, os.path.join(RUN, "general_rows_weights.pt")); torch.save(G_, os.path.join(HERE, "results", f"general_rows_{CASE}{TAG}.pt"))
    vv = [i for i in vv if SR.get(i) is not None and SR[i][4] == 0]               # the four-metric REPORT needs references
    if not vv:
        print("[REPORT] no validation references for this case -> report skipped", flush=True); dp.close(); return
    # ---- VALIDATION REPORT (four metrics) ----
    arms = {"integer cuts only": None}
    if selected is not None: arms[f"+ general rows (selected: epoch {pick[0]})"] = selected
    print(f"[REPORT] {CASE} VALIDATION, {len(vv)} instances with references, W={W}", flush=True)
    print("   version                              | gap mean | gap median | discrete err | continuous err", flush=True)
    allres = {}
    for nm, st in arms.items():
        jobs = []
        for i in vv:
            Au, bu = unit_fix_rows(i)
            if st is None:
                jobs.append((i, V(i), Au, bu, W, RHO))
            else:
                net.load_state_dict(st)
                with torch.no_grad(): Ag, bg = [t[0].numpy().astype(float) for t in grows([i], False)]
                jobs.append((i, V(i), np.vstack([Au, Ag]), np.r_[bu, bg], W, RHO))
        sol = dict(dp.imap_unordered(combopt_ac.miqcqp, jobs))
        pr = dict(dp.imap_unordered(combopt_ac.price_strict, [(i, sol[i]["u"]) for i in vv if sol.get(i)]))
        v = np.array([(100*abs(pr[i][0]-SR[i][0])/SR[i][0], 100*np.mean(pr[i][5] != np.asarray(SR[i][5])), 100*np.abs(pr[i][1]-SR[i][1]).sum()/SR[i][1].sum())
                      if pr.get(i) else (np.nan,)*3 for i in vv])
        line = f"{np.nanmean(v[:,0]):.4f}% | {np.nanmedian(v[:,0]):.4f}% | {np.nanmean(v[:,1]):.2f}% | {np.nanmean(v[:,2]):.2f}%"
        print(f"   {nm:38s} | {line}", flush=True); allres[nm] = (sol, pr, v)
        runlog.add_result(RUN, f"{CASE} VALIDATION {nm}: gap mean | median | disc | cont = {line}")
    runlog.save(RUN, "results.pkl", dict(vv=vv, res=allres)); dp.close()


if __name__ == "__main__":
    main()

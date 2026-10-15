# %% FINAL EVALUATION, ONCE, on the UNTOUCHED draw (data2.fresh2_ids) (2026-10-10).
#    OUR METHOD = learned V (V-net, exact layer) + per-unit INTEGER cuts (layer-trained) + GENERAL mixed rows (layer-trained),
#                 one short MIQCQP (W) at the learned V.  All label-free.
#    SUPERVISED NN PROXY = MLP on reference commitments of TRAINING instances (validation labels or a 20% hold-out for
#                 early stopping), round 0.5 + the same strict pricing/repair.
#    Seeds vary the V-net and the proxy (cuts: seed 0). Times like-for-like (own solve/forward + the SAME strict pricing).
#    Reference time = standard sequential QCAC only (polish excluded).  Usage: python 55_final_eval.py <case> <seeds e.g. 0 or 0,1,2>
import os, sys, json, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE = sys.argv[1]; SEEDS = [int(s) for s in sys.argv[2].split(",")]; DRAW = sys.argv[3] if len(sys.argv) > 3 else "fresh2"
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
VTAG = sys.argv[5] if len(sys.argv) > 5 else ""
# (2026-10-13, PRE-REGISTERED) RETRY safeguard, label-free: env RETRY_RHO (e.g. 1e4) and RETRY_TAU (default 0.05). If the MIQCQP solve at the deployment rho ends with MIP gap > RETRY_TAU (or no incumbent), the SAME
# learned pipeline is deployed once more at RETRY_RHO and the CHEAPER strictly-priced answer is kept (no reference, no heuristic; each solve is still W). Time = both solves + both pricings.
RETRY_RHO = float(os.environ["RETRY_RHO"]) if os.environ.get("RETRY_RHO") else None; RETRY_TAU = float(os.environ.get("RETRY_TAU", "0.05"))
VFIX = len(sys.argv) > 6 and sys.argv[6] == "vseed0"   # (2026-10-13, audit) V-net ALWAYS seed 0 (cuts/rows were trained AT that V); seeds then vary the PROXY only   # (2026-10-12) V-net file suffix ("_nc" = clean V-nets trained on the uncut relaxation)   # (audit 2026-10-12) "_lfx" = the label-free method's cut files; "" = the old long-solve-teacher method
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0)

TIE_PP = 1e-3   # (audit 2026-10-12, PRE-REGISTERED) |difference| <= 1e-3 pp = tie (pricing precision: strict pricing slack <= 1e-4, physics 1e-3)


def paired(xa, ya, lab):
    """PRE-REGISTERED paired comparison ours (xa) vs proxy (ya), lower = better. Failures are NOT dropped: an instance where only
    one method failed counts as a loss for that method in the better/tie/worse counts and the sign test. Wilcoxon (zero_method
    'wilcox') runs on differences thresholded at TIE_PP over instances where both succeeded; n_nonzero printed. Bootstrap CI of the mean."""
    from scipy.stats import wilcoxon, binomtest
    fa, fb = np.isnan(xa), np.isnan(ya); ok = ~fa & ~fb
    d = xa[ok] - ya[ok]; dt = np.where(np.abs(d) <= TIE_PP, 0.0, d)
    better = int((dt < 0).sum() + (fb & ~fa).sum()); worse = int((dt > 0).sum() + (fa & ~fb).sum()); tie = int((dt == 0).sum() + (fa & fb).sum())
    nz = int((dt != 0).sum())
    pw = wilcoxon(dt, zero_method="wilcox").pvalue if nz > 0 else float("nan")
    ps = binomtest(better, better + worse).pvalue if better + worse > 0 else float("nan")
    rng = np.random.default_rng(0); bs = [rng.choice(d, d.size).mean() for _ in range(10000)] if d.size else [np.nan]
    return (f"{lab}: ours - proxy {d.mean():+.4f} pp, 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] (n={int(ok.sum())}), "
            f"Wilcoxon p {pw:.3g} (n_nonzero {nz}), better/tie/worse {better}/{tie}/{worse} (tie <= {TIE_PP} pp), sign test p {ps:.3g} | "
            f"failures ours {int(fa.sum())}, proxy {int(fb.sum())}")


def main():
    import combopt_ac, cuttrain, data2, runlog
    from pipeline import grid
    from cutnet_k import CutNetK, rows_torch
    from scipy.stats import wilcoxon
    import subprocess; print("[machine] " + subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(), flush=True)   # (audit) load at start
    RUN = runlog.start(f"FINAL_{CASE}_{DRAW}{TAG}{VTAG}{'_vseed0' if VFIX else ''}{'_retry' if RETRY_RHO else ''}_seeds{'-'.join(map(str, SEEDS))}")
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    REF = pickle.load(open(os.path.join(HERE, "results", f"{DRAW}_refs_{CASE}.pkl"), "rb")); ids2 = getattr(data2, f"{DRAW}_ids")()
    assert set(ids2) <= set(REF), f"{DRAW} references incomplete: {len(set(ids2) & set(REF))}/{len(ids2)} (56 is resumable -> finish it first)"
    pool = mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    conv = [i for i in ids2 if REF.get(i, {}).get("converged")]
    SR = dict(pool.imap_unordered(combopt_ac.price_strict, [(i, REF[i]["u"]) for i in conv]))
    tid = [i for i in conv if SR.get(i) is not None and SR[i][4] == 0]; exc = [i for i in ids2 if i not in tid]
    sc = {i: float(INST[i][0].sum()/base) for i in ids2}
    t_ref = np.median([REF[i]["secs"] - REF[i].get("secs_polish", 0.0) for i in tid])
    print(f"[{CASE}] UNTOUCHED test: {len(ids2)} drawn, {len(tid)} with a usable reference; excluded {len(exc)} at load scale "
          f"{np.round(sorted(sc[i] for i in exc), 3).tolist()}; reference (sequential QCAC) median {t_ref:.0f}s  [{time.time()-t0:.0f}s]", flush=True)
    # ---- frozen learned cuts (seed 0): per-unit integer cuts + general rows
    UC = torch.load(os.path.join(HERE, "results", f"unit_cuts_layer_{CASE}_soft{TAG}.pt"), weights_only=False); c = UC["c"]
    cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G)); cm.load_state_dict(UC["state"]); cm.eval()
    with torch.no_grad(): Mc = torch.sigmoid(cm(Xn)).numpy()
    LO = np.maximum((Mc - c)/(1 - c), 0); HI = 1 - np.maximum(((1 - c) - Mc)/(1 - c), 0)
    assert "last" in UC and UC["state"] is not None and UC["epoch"] >= 0, f"{CASE}: stale/untrained unit-cut file"   # (pre-final audit) layer-trained cuts only
    GR = torch.load(os.path.join(HERE, "results", f"general_rows_{CASE}{TAG}.pt"), weights_only=False); K = GR["K"]
    assert GR.get("trained") is not None and GR.get("selected") is not None, f"{CASE}: general rows not trained/selected -> diagnose"
    gnet = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False)
    GSEL = GR.get("selected")                          # (audit 2026-10-11) label-free selected rows, None = OFF; never untrained rows
    if GSEL is not None: gnet.load_state_dict(GSEL); gnet.eval()
    print(f"[{CASE}] general rows: {'OFF (label-free selection or failed gate)' if GSEL is None else 'epoch ' + str(GR.get('selected_epoch'))}", flush=True)
    ap = mp.get_context("spawn").Pool(11, initializer=cuttrain.init6, initargs=(CASE, 1000.0, K))
    # ---- supervised proxy labels
    RP = data2.refs(CASE, os.path.join(HERE, "results", "pool2"))
    ltr = [i for i in tr if data2.converged(RP.get(i))]; lva = [i for i in va if data2.converged(RP.get(i))]
    if not lva: k_ = max(1, len(ltr)//5); lva, ltr = ltr[-k_:], ltr[:-k_]
    Ytr = torch.tensor(np.stack([RP[i]["u"] for i in ltr]), dtype=torch.float32); Yva = torch.tensor(np.stack([RP[i]["u"] for i in lva]), dtype=torch.float32)
    rows_out, per_seed = {}, {}
    for s in SEEDS:
        vf = os.path.join(HERE, "results", f"vnet_layer_{CASE}{'' if (s == 0 or VFIX) else f'_s{s}'}_env{VTAG}.pt")
        from vnetio import vnet_out
        OUT, RHO, VSEL = vnet_out(vf, Xn[tid], n)
        assert os.environ.get("DEPLOY_RHO") or VTAG != "_nc", "clean-V run without DEPLOY_RHO would silently deploy at the V file's stored rho (the #209 failure mode)"   # (audit 2026-10-13)
        print(f"[deployment] seed {s}: V-net {os.path.basename(vf)} ({VSEL}), deployment rho {RHO:.0e}, DEPLOY_RHO env {os.environ.get('DEPLOY_RHO')}", flush=True)
        Vd = {i: (1.0 + OUT[k, :n], OUT[k, n:]) for k, i in enumerate(tid)}
        ANT = dict(ap.imap_unordered(cuttrain.anchor6_t, [(i, *Vd[i], LO[i], HI[i]) for i in tid])) if GSEL is not None else {}
        AN = {i: v[0] for i, v in ANT.items()}; t_an = {i: v[1] for i, v in ANT.items()}
        print(f"   seed {s}: anchor solve failed (rows dropped) on {sum(AN.get(i) is None for i in tid)} instances", flush=True)
        jobs = []
        for i in tid:
            Au = np.zeros((2*G, 2*G)); bu = np.ones(2*G)
            for k in range(G):
                if LO[i, k] > 0: Au[k, G + k] = -1.0; bu[k] = -1.0
                if HI[i, k] < 1: Au[G + k, G + k] = 1.0; bu[G + k] = 0.0
            if AN.get(i) is not None:
                with torch.no_grad():
                    _, _, a, al, a2, d, d2, _ = gnet(Xn[[i]]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.tensor(AN[i], dtype=torch.float32)[None])
                Au = np.vstack([Au, A_[0, :K].numpy().astype(float)]); bu = np.r_[bu, b_[0, :K].numpy().astype(float)]
            jobs.append((i, Vd[i], Au, bu, W, RHO))
        sol = dict(pool.imap_unordered(combopt_ac.miqcqp, jobs))
        OP = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, sol[i]["u"]) for i in tid if sol.get(i)]))
        retried, kept2, extra_t = [], [], {}
        sol0, OP0 = dict(sol), dict(OP)          # (audit 2026-10-13) first attempts at the deployment rho = the single-rho ablation, derived in the SAME run
        if RETRY_RHO is not None:
            gap_ = lambda i: (np.nan_to_num(sol[i].get("mipgap", np.nan), nan=1.0) if sol.get(i) else 1.0)
            retried = [i for i in tid if (not sol.get(i)) or sol[i].get("fallback") or gap_(i) > RETRY_TAU or not OP.get(i)]
            jobs2 = [(j[0], j[1], j[2], j[3], j[4], RETRY_RHO) for j in jobs if j[0] in set(retried)]
            sol2 = dict(pool.imap_unordered(combopt_ac.miqcqp, jobs2))
            OP2 = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, sol2[i]["u"]) for i in retried if sol2.get(i)]))
            att = lambda S_, O_, i: (S_[i]["secs"] if S_.get(i) else 0.0) + (O_[i]["secs"] if O_.get(i) else 0.0)     # solve + pricing time of one attempt
            for i in retried:
                t1, t2 = att(sol, OP, i), att(sol2, OP2, i)
                if OP2.get(i) is not None and (OP.get(i) is None or OP2[i]["cost"] < OP[i]["cost"]):
                    sol[i], OP[i] = sol2[i], OP2[i]; kept2.append(i); extra_t[i] = t1        # kept the retried answer; the first attempt's time is added
                else:
                    extra_t[i] = t2                                                          # kept the first answer; the retry's time is added
            print(f"   seed {s}: RETRY at rho {RETRY_RHO:.0e} (MIP gap > {RETRY_TAU}): retried {len(retried)}/{len(tid)} instances, kept the retried answer on {len(kept2)}", flush=True)
        torch.manual_seed(s)
        pm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
        po = torch.optim.Adam(pm.parameters(), lr=1e-3, weight_decay=1e-4); best = (np.inf, None); bce = nn.functional.binary_cross_entropy_with_logits
        for it in range(3000):
            po.zero_grad(); l = bce(pm(Xn[ltr]), Ytr); l.backward(); po.step()
            if it % 25 == 0:
                with torch.no_grad(): v_ = float(bce(pm(Xn[lva]), Yva))
                if v_ < best[0]: best = (v_, {k: t.clone() for k, t in pm.state_dict().items()})
        pm.load_state_dict(best[1]); pm.eval(); t1 = time.time()
        with torch.no_grad(): PP = torch.sigmoid(pm(Xn[tid])).numpy()
        t_fwd = (time.time() - t1)/len(tid)
        PX = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, (PP[k] > 0.5).astype(float)) for k, i in enumerate(tid)]))
        res = {}
        # time (audit 2026-10-11): ours = anchor solve + the WHOLE miqcqp call (incl. its internal pricing) + strict pricing; proxy = forward + strict pricing/repair
        ARMS = [("OUR METHOD", OP, lambda i: t_an.get(i, 0.0) + sol[i]["secs"] + OP[i]["secs"] + extra_t.get(i, 0.0)), ("supervised NN proxy", PX, lambda i: t_fwd + PX[i]["secs"])]
        if RETRY_RHO is not None: ARMS.append(("OUR METHOD, single rho (ablation)", OP0, lambda i: t_an.get(i, 0.0) + sol0[i]["secs"] + OP0[i]["secs"]))
        for nm, R, tm in ARMS:
            res[nm] = np.array([(100*abs(R[i]["cost"]-SR[i][0])/SR[i][0], 100*np.mean(R[i]["u"] != np.asarray(SR[i][5])),
                                 100*np.abs(R[i]["pg"]-SR[i][1]).sum()/SR[i][1].sum(), tm(i), 100*(R[i]["cost"]-SR[i][0])/SR[i][0], float(R[i]["repaired"] > 0))
                                if R.get(i) else (np.nan,)*6 for i in tid])
        per_seed[s] = res; rows_out[s] = dict(sol=sol, OP=OP, PX=PX, retried=retried, kept_retry=kept2, retry_rho=RETRY_RHO, retry_tau=RETRY_TAU, sol0=sol0 if RETRY_RHO is not None else None, OP0=OP0 if RETRY_RHO is not None else None)
        for col, lab in ((0, "|gap|"), (4, "signed gap")):
            print(f"--- seed {s} " + paired(res["OUR METHOD"][:, col], res["supervised NN proxy"][:, col], lab), flush=True)
        print(f"    seed {s}: our fallbacks {sum(bool(sol[i].get('fallback')) for i in tid if sol.get(i))}, our strict repairs "
              f"{sum(int(OP[i]['repaired'] > 0) for i in tid if OP.get(i))}, proxy repairs {sum(int(PX[i]['repaired'] > 0) for i in tid if PX.get(i))}", flush=True)
        runlog.save(RUN, "partial.pkl", dict(tag=TAG, vtag=VTAG, tid=tid, exc=exc, SR=SR, per_seed=per_seed, rows=rows_out, t_ref=t_ref))   # (audit) save every seed
    ap.close(); pool.close()
    # (audit 2026-10-11) instance-level paired test on SEED-AVERAGED |gap| (n = instances), signed gaps, and the split by whether the proxy needed repair
    xo = np.nanmean([per_seed[s]["OUR METHOD"][:, 0] for s in SEEDS], 0); yp = np.nanmean([per_seed[s]["supervised NN proxy"][:, 0] for s in SEEDS], 0)
    # (audit 2026-10-12) seed-averaged; an instance counts as a FAILURE if the method failed on ANY seed (not dropped)
    fo = np.any([np.isnan(per_seed[s]["OUR METHOD"][:, 0]) for s in SEEDS], 0); fp = np.any([np.isnan(per_seed[s]["supervised NN proxy"][:, 0]) for s in SEEDS], 0)
    xo = np.where(fo, np.nan, xo); yp = np.where(fp, np.nan, yp); mk = ~np.isnan(xo) & ~np.isnan(yp)
    print("[paired, seed-averaged, PRE-REGISTERED] " + paired(xo, yp, "|gap|"), flush=True)
    xs = np.where(fo, np.nan, np.nanmean([per_seed[s]["OUR METHOD"][:, 4] for s in SEEDS], 0)); ys = np.where(fp, np.nan, np.nanmean([per_seed[s]["supervised NN proxy"][:, 4] for s in SEEDS], 0))
    print("[paired, seed-averaged, PRE-REGISTERED] " + paired(xs, ys, "signed gap (= paired cost difference)"), flush=True)
    rep_ = np.nanmax([per_seed[s]["supervised NN proxy"][:, 5] for s in SEEDS], 0) > 0
    for lab, sub in (("[descriptive, post hoc] proxy needed NO repair", mk & ~rep_), ("[descriptive, post hoc] proxy needed repair (any seed)", mk & rep_)):
        if sub.sum(): print(f"   {lab}: n={sub.sum()}, ours - proxy {(xo[sub] - yp[sub]).mean():+.4f} pp", flush=True)
    if RETRY_RHO is not None:
        AB = "OUR METHOD, single rho (ablation)"
        fa = np.any([np.isnan(per_seed[s][AB][:, 0]) for s in SEEDS], 0); xa = np.where(fa, np.nan, np.nanmean([per_seed[s][AB][:, 0] for s in SEEDS], 0))
        print("[ABLATION, same run, seed-averaged, PRE-REGISTERED] single rho vs proxy " + paired(xa, yp, "|gap|"), flush=True)
        xs_ = np.where(fa, np.nan, np.nanmean([per_seed[s][AB][:, 4] for s in SEEDS], 0)); print("[ABLATION, same run, seed-averaged, PRE-REGISTERED] single rho vs proxy " + paired(xs_, ys, "signed gap"), flush=True)
        rr = rows_out[SEEDS[0]]; ri = rr["retried"]; print(f"   [descriptive] retry fired on {len(ri)}/{len(tid)} instances (seed {SEEDS[0]}; deterministic), kept the retried answer on {len(rr['kept_retry'])}: ids {ri}", flush=True)
        if ri:
            ix = [tid.index(i) for i in ri]; print(f"   [descriptive] on the retried instances: |gap| with retry {np.nanmean(xo[ix]):.4f}% vs single rho {np.nanmean(xa[ix]):.4f}%; time per retried instance {np.nanmean(np.nanmean([per_seed[s]['OUR METHOD'][ix, 3] for s in SEEDS], 0)):.1f}s vs {np.nanmean(np.nanmean([per_seed[s][AB][ix, 3] for s in SEEDS], 0)):.1f}s", flush=True)
    for nm in ("OUR METHOD", "supervised NN proxy"):
        sg_ = np.nanmean([per_seed[s][nm][:, 4] for s in SEEDS], 0)
        print(f"   signed gap {nm}: mean {np.nanmean(sg_):+.4f}%, min {np.nanmin(sg_):+.4f}% (negative = cheaper than the reference)", flush=True)
    print(f"   scored load range [{min(sc[i] for i in tid):.3f}, {max(sc[i] for i in tid):.3f}] of drawn [{min(sc.values()):.3f}, {max(sc.values()):.3f}]", flush=True)
    print(f"\n[FINAL {CASE}] untouched test, {len(tid)} instances, seeds {SEEDS}, W={W}; reference median {t_ref:.0f}s", flush=True)
    print("   method               | gap mean | gap median | discrete err | continuous err | time median | failures", flush=True)
    for nm in ("OUR METHOD", "supervised NN proxy") + (("OUR METHOD, single rho (ablation)",) if RETRY_RHO is not None else ()):
        M = np.array([[np.nanmean(v[:, 0]), np.nanmedian(v[:, 0]), np.nanmean(v[:, 1]), np.nanmean(v[:, 2]), np.nanmedian(v[:, 3]), np.isnan(v[:, 0]).sum()]
                      for v in (per_seed[s][nm] for s in SEEDS)])
        f = lambda j, dd=4: f"{M[:, j].mean():.{dd}f}" + (f"+-{M[:, j].std(ddof=1):.{dd}f}" if len(M) > 1 else "")
        line = f"{f(0)}% | {f(1)}% | {f(2, 2)}% | {f(3, 2)}% | {f(4, 2)}s | {int(M[:, 5].sum())}"
        print(f"   {nm:20s} | {line}", flush=True)
        runlog.add_result(RUN, f"FINAL {CASE} untouched ({len(tid)} inst, seeds {SEEDS}) {nm}: gap mean | median | disc | cont | time | failures = {line}")
    import hashlib
    hsh = {f: hashlib.sha1(open(os.path.join(HERE, "results", f), "rb").read()).hexdigest()[:12] for f in (f"unit_cuts_layer_{CASE}_soft{TAG}.pt", f"general_rows_{CASE}{TAG}.pt")
           + tuple(f"vnet_layer_{CASE}{'' if (s_ == 0 or VFIX) else f'_s{s_}'}_env{VTAG}.pt" for s_ in (SEEDS[:1] if VFIX else SEEDS)) + (f"{DRAW}_refs_{CASE}.pkl",)}   # (audit 2026-10-12) prove which V-nets were used
    print("[machine at end] " + subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(), flush=True)
    json.dump({k: os.environ.get(k) for k in ("DEPLOY_RHO", "RETRY_RHO", "RETRY_TAU")}, open(os.path.join(RUN, "run_env.json"), "w"))
    runlog.save(RUN, "final.pkl", dict(tag=TAG, vtag=VTAG, rho=RHO, deploy_rho_env=os.environ.get("DEPLOY_RHO"), retry_rho=RETRY_RHO, retry_tau=RETRY_TAU, vseed0=VFIX, files=hsh, tid=tid, exc=exc, SR=SR, per_seed=per_seed, rows=rows_out, t_ref=t_ref))


if __name__ == "__main__":
    main()

# %% GENERALIZED INTEGER CUTS (2026-10-15) -- VALIDATION ONLY, label-free selection, NO per-unit fixings anywhere.
#    Question (PI): can the integer cuts be GENERAL linear rows over the integer variables, a'u <= b with (a, b) produced by the network, instead of the
#    one-variable cuts u_k >= l_k / u_k <= h_k?  The old per-unit cut-net and its confidence threshold c are NOT used: unit rows are all inert (lo = 0, hi = 1).
#    Row families (all SOFT in the layer and in the deployed MIQCQP, violation priced 1e6; rows are over [pg; u], 2G columns, integer rows have zero pg part):
#      none : no rows (control: V-net + one short solve)
#      rows : K_I learned INTEGER rows a2'u <= b2 (CutNetK integer category; b = <A, w_rlx> - depth, anchored at the UNCUT relaxation) [+ M mixed rows alpha'pg + a'u <= b]
#      ham  : ONE learned Hamming-ball row  sum_k [p_k (1 - u_k) + (1 - p_k) u_k] <= r  (p = learned centre in (0,1), a linear row with a = 1 - 2p, b = r - sum p;
#             the local-branching / no-good family; r chosen label-free from RGRID) [+ M mixed rows]
#    Targets = the label-free best-of targets at the clean V-net (`_nc`); V-net = clean `_env_nc`; deployment rho from DEPLOY_RHO (3e5 for clean case118).
#    Training: tracking through the exact layer + validity (the own target must satisfy every row), FD gate first; epoch chosen by the LABEL-FREE score on ALL 24 validation
#    draws among TRAINED epochs; four-metric report vs the reference is REPORT ONLY.  No test / fresh draw is read.
#    Usage: DEPLOY_RHO=3e5 python 67_generalized_integer_cuts.py <case> <none|rows|ham> <K_I> <M mixed rows, 0 or 2> <tag>      (SMOKE=1: tiny run to test the plumbing)
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE, MODE, KI, MIX, TAG = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
SMOKE = os.environ.get("SMOKE") == "1"
SEED = int(os.environ.get("SEED", 0))
GAMMA, EPOCHS, BATCH, LR, LAM = 1000.0, (1 if SMOKE else 10), 24, 2e-3, 10.0
RGRID = (3, 8) if SMOKE else (1, 2, 3, 5, 8, 12, 20)
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0)
TEACHER = os.path.join(HERE, "results", f"selfteacher_{CASE}_bestof_extra_nc.json"); VTAG = "_nc"
assert os.environ.get("DEPLOY_RHO"), "set DEPLOY_RHO=3e5 (otherwise the V file's own rho 1e4 would be deployed silently)"
assert MODE in ("none", "rows", "ham") and (MODE != "rows" or KI >= 1) and MIX in (0, 2)


# %% main
def main():
    import combopt_ac, cuttrain, data2, runlog
    from pipeline import grid
    from cutnet_k import CutNetK, rows_torch
    from vnetio import vnet_out
    RUN = runlog.start(f"geninteger_{CASE}_{MODE}_K{KI}_M{MIX}{TAG}")
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    OUT, RHO, VSEL = vnet_out(os.path.join(HERE, "results", f"vnet_layer_{CASE}_env{VTAG}.pt"), Xn, n); V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    LOz, HIz = np.zeros(G), np.ones(G)                                      # NO per-unit fixings: every unit row is inert (A = 0, b = 1)
    TG = {int(k): np.array(v, float) for k, v in json.load(open(TEACHER)).items()}
    dp = mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    trn = [i for i in tr + data2.extra_ids() if i in TG]
    NTRAIN = int(os.environ.get("NTRAIN", 0))                                # (2026-10-16) match the supervised proxy's data size: NTRAIN loads from the ORIGINAL training split only (seeded subset, no extra loads, no reference used)
    if NTRAIN:
        pool_ids = [i for i in tr if i in TG]; trn = sorted(np.random.default_rng(1000 + SEED).choice(pool_ids, NTRAIN, replace=False).tolist())
        assert set(trn) <= set(tr) and not set(trn) & set(va) and not set(trn) & set(te), "subset must be training-split loads only"
    if SMOKE: trn = trn[:24]
    PO = dict(dp.imap_unordered(combopt_ac.price_strict, [(i, TG[i]) for i in trn])); trn = [i for i in trn if PO.get(i) is not None and PO[i][4] == 0]
    WOWN = {i: torch.tensor(np.r_[PO[i][1], TG[i]], dtype=torch.float32) for i in trn}
    SR = data2.strict_refs(CASE); vl = list(va)                             # label-free selection on ALL validation draws
    SC = {i: r[1] for i, r in dp.imap_unordered(combopt_ac.relax, vl) if r is not None}; vl = [i for i in vl if i in SC]
    if SMOKE: vl = vl[:6]
    print(f"[config] SEED={SEED} NTRAIN={NTRAIN or 'all'} (training targets after the strict-pricing filter: {len(trn)})", flush=True)
    print(f"[config] {CASE} mode={MODE} K_I={KI} mixed={MIX} tag={TAG}: W={W}, V-net {VTAG} ({VSEL}) deployment rho {RHO:.0e}, {len(trn)} training targets, {len(vl)} validation draws, NO unit fixings", flush=True)
    K = max(KI, MIX, 1); R = (KI if MODE == "rows" else (1 if MODE == "ham" else 0)) + MIX
    # ---- networks
    torch.manual_seed(SEED)                                                 # (audit) CutNetK weights were unseeded
    net = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False) if (MODE == "rows" or MIX) else None
    pnet = None
    if MODE == "ham":
        torch.manual_seed(SEED); pnet = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
        Ytr = torch.tensor(np.stack([TG[i] for i in trn]), dtype=torch.float32); o = torch.optim.Adam(pnet.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(1500 if not SMOKE else 200):
            o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(pnet(Xn[trn]), Ytr); l.backward(); o.step()   # imitation warm start of the CENTRE only
    Rr = {"r": float(RGRID[0])}
    # ---- anchors: the UNCUT relaxation at V (no unit cuts exist), exactly the point the rows are anchored at
    if R:
        lp = mp.get_context("spawn").Pool(11, initializer=cuttrain.init6, initargs=(CASE, GAMMA, R))
        if net is not None:                                                  # anchors are only used by the CutNetK rows (a Hamming-only row does not need them)
            AN = dict(lp.imap_unordered(cuttrain.anchor6_t, [(i, *V(i), LOz, HIz) for i in trn + vl]))
            trn = [i for i in trn if AN.get(i) is not None and AN[i][0] is not None]; vl = [i for i in vl if AN.get(i) is not None and AN[i][0] is not None]
            Wr = {i: torch.tensor(AN[i][0], dtype=torch.float32) for i in trn + vl}; t_an = {i: AN[i][1] for i in vl}
        else:
            Wr, t_an = {}, {i: 0.0 for i in vl}
        print(f"[config] after anchors: {len(trn)} training targets, {len(vl)} validation draws (must be all 24 for a full run)", flush=True)
        assert SMOKE or len(vl) == len(va), "an anchor failure changed the validation set -> not comparable"
    else:
        lp, t_an = None, {i: 0.0 for i in vl}

    def rows(idx, grad):                                                    # -> (A (n, R, 2G), b (n, R)) torch
        parts_A, parts_b = [], []
        with torch.set_grad_enabled(grad):
            if MODE == "ham":
                p = torch.sigmoid(pnet(Xn[idx])); A_h = torch.cat([torch.zeros(len(idx), 1, G), (1 - 2*p)[:, None, :]], 2); parts_A.append(A_h); parts_b.append((Rr["r"] - p.sum(1, keepdim=True)))
            if net is not None:
                _, _, a, al, a2, d, d2, _ = net(Xn[idx]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.stack([Wr[i] for i in idx]))
                if MIX: parts_A.append(A_[:, :MIX]); parts_b.append(b_[:, :MIX])
                if MODE == "rows": parts_A.append(A_[:, K:K + KI]); parts_b.append(b_[:, K:K + KI])
        return torch.cat(parts_A, 1), torch.cat(parts_b, 1)

    def deploy_jobs(idx):
        if R == 0: return [(i, V(i), None, None, W, RHO) for i in idx]
        with torch.no_grad(): A_, b_ = rows(idx, False)
        return [(i, V(i), A_[k].numpy().astype(float), b_[k].numpy().astype(float), W, RHO) for k, i in enumerate(idx)]

    def deploy(idx):
        sol = dict(dp.imap_unordered(combopt_ac.miqcqp, deploy_jobs(idx)))
        pr = dict(dp.imap_unordered(combopt_ac.price_strict, [(i, sol[i]["u"]) for i in idx if sol.get(i)])); return sol, pr

    def lf_score():                                                          # LABEL-FREE: deployed strict price / flat-V relaxation cost, ALL validation draws; no reference read
        sol, pr = deploy(vl); return float(np.mean([pr[i][0]/SC[i] if pr.get(i) else 10.0 for i in vl])), sum(bool(sol[i].get("fallback")) for i in vl if sol.get(i))

    sel_hist, epoch_states, trained = [], {}, None
    s_none = None
    if R:
        # ---- r selection for the Hamming row (label-free, warm-start centre) -- validation score only
        if MODE == "ham":
            rs = {}
            for r_ in RGRID:
                Rr["r"] = float(r_); rs[r_] = lf_score(); print(f"[r] {r_}: label-free val {rs[r_][0]:.6f} (fallbacks {rs[r_][1]})  [{time.time()-t0:.0f}s]", flush=True)
            Rr["r"] = float(min(RGRID, key=lambda k: rs[k][0])); print(f"[r] selected r = {Rr['r']:.0f} (label-free)", flush=True)
        # ---- FD gate on the row parameters at the initial rows
        TRG = cuttrain.track_gen; rng = np.random.default_rng(SEED); gate = []
        for i in trn[:3]:
            with torch.no_grad(): A0, b0 = rows([i], False)
            Ag, bg = A0[0].numpy().astype(float), b0[0].numpy().astype(float)
            _, R0 = lp.apply(TRG, ((i, *V(i), LOz, HIz, Ag, bg, TG[i], True),))
            if R0 is None: print(f"[gate] inst {i}: failed", flush=True); continue
            L0, gA, gb = R0; dirs = []
            for _ in range(8):
                dA = rng.standard_normal(Ag.shape); db = rng.standard_normal(bg.shape); nr = np.sqrt((dA**2).sum() + (db**2).sum()); dirs.append((dA/nr, db/nr))
            for h in (1e-2, 3e-3, 1e-3):
                jobs = []
                for dA, db in dirs: jobs += [(i, *V(i), LOz, HIz, Ag + h*dA, bg + h*db, TG[i], False), (i, *V(i), LOz, HIz, Ag - h*dA, bg - h*db, TG[i], False)]
                res = [r_ for _, r_ in lp.imap(TRG, jobs)]; okd = [k for k in range(8) if res[2*k] is not None and res[2*k+1] is not None]
                fd = np.array([(res[2*k][0] - res[2*k+1][0])/(2*h) for k in okd]); fw = np.array([(res[2*k][0] - L0)/h for k in okd]); bw = np.array([(L0 - res[2*k+1][0])/h for k in okd])
                an = np.array([(gA*dirs[k][0]).sum() + gb @ dirs[k][1] for k in okd]); cfd = np.corrcoef(fw, bw)[0, 1]; ca = np.corrcoef(fd, an)[0, 1]
                print(f"[gate] inst {i} h={h:.0e}: {len(okd)}/8 | FD self-consistency {cfd:.3f} | analytic vs FD corr {ca:.4f} | |FD| {np.abs(fd).mean():.2e}", flush=True)
                if np.isfinite(cfd) and cfd >= 0.99: break                                            # identical to 52: retry at a smaller h only while the FD is not self-consistent; a low corr then FAILS the gate
            gate.append((cfd, ca))
        good = [ca for cfd, ca in gate if np.isfinite(cfd) and np.isfinite(ca) and cfd >= 0.99]; print(f"[gate] self-consistent FD on {len(good)}/{len(gate)} instances", flush=True)
        s_init = lf_score(); print(f"[val] label-free: INITIAL rows {s_init[0]:.6f} (fallbacks {s_init[1]})  [{time.time()-t0:.0f}s]", flush=True)
        if not good or min(good) < 0.95:
            print("[gate] FAILED or signal too small -> NO TRAINING (diagnose; not deployable as learned rows)", flush=True)
        else:
            print("[gate] PASSED -> train", flush=True)
            params = list(net.parameters() if net is not None else []) + list(pnet.parameters() if pnet is not None else [])
            opt = torch.optim.Adam(([{'params': list(net.parameters()), 'lr': LR}] if net is not None else []) + ([{'params': list(pnet.parameters()), 'lr': 1e-4}] if pnet is not None else []))   # centre net as the old cut-net fine-tune (1e-4); rows as the old row-net (2e-3)
            for ep in range(EPOCHS):
                order = rng.permutation(trn); Lt, Lv, nf = [], [], 0
                for s0 in range(0, len(order), BATCH):
                    b = [int(i) for i in order[s0:s0 + BATCH]]
                    A_t, b_t = rows(b, True); An, bn = A_t.detach().numpy().astype(float), b_t.detach().numpy().astype(float)
                    Rs = dict(lp.imap_unordered(TRG, [(i, *V(i), LOz, HIz, An[k], bn[k], TG[i], True) for k, i in enumerate(b)]))
                    GA, GB, ok = np.zeros_like(An), np.zeros_like(bn), 0
                    for k, i in enumerate(b):
                        if Rs.get(i) is None: nf += 1; continue
                        Lt.append(Rs[i][0]); GA[k], GB[k] = Rs[i][1], Rs[i][2]; ok += 1
                    if ok == 0: continue
                    viol = torch.relu(torch.einsum("nkd,nd->nk", A_t, torch.stack([WOWN[i] for i in b])) - b_t); lval = LAM*(viol**2).sum(1).mean(); Lv.append(float(lval))
                    opt.zero_grad(); ((A_t*torch.tensor(GA, dtype=torch.float32)).sum() + (b_t*torch.tensor(GB, dtype=torch.float32)).sum()).div(ok).add(lval).backward(); opt.step()
                epoch_states[ep] = ({k: t.clone() for k, t in net.state_dict().items()} if net is not None else None, {k: t.clone() for k, t in pnet.state_dict().items()} if pnet is not None else None)
                se = lf_score(); sel_hist.append((ep, *se)); print(f"[ep {ep}] tracking {np.mean(Lt):.5f} validity {np.mean(Lv):.5f} (failed {nf}) | label-free val {se[0]:.6f} (fallbacks {se[1]})  [{time.time()-t0:.0f}s]", flush=True)
            trained = True
    else:
        s0_ = lf_score(); print(f"[val] label-free: NO ROWS (control) {s0_[0]:.6f} (fallbacks {s0_[1]})  [{time.time()-t0:.0f}s]", flush=True); sel_hist.append(("none", *s0_))
    # ---- label-free epoch selection among TRAINED epochs only
    pick = min(sel_hist, key=lambda t: t[1]) if sel_hist else None
    if trained and pick is not None:
        st_n, st_p = epoch_states[pick[0]]
        if net is not None: net.load_state_dict(st_n)
        if pnet is not None: pnet.load_state_dict(st_p)
        print(f"[select] label-free: epoch {pick[0]} (score {pick[1]:.6f})", flush=True)
    elif R:
        print("[select] NOT TRAINED -> variant reported with its initial rows only (diagnose)", flush=True)
    if R: lp.close()
    torch.save(dict(mode=MODE, KI=KI, MIX=MIX, r=Rr["r"], rho=RHO, sel_hist=sel_hist, trained=trained, net=(net.state_dict() if net is not None else None), pnet=(pnet.state_dict() if pnet is not None else None)),
               os.path.join(HERE, "results", f"geninteger_{CASE}_{MODE}_K{KI}_M{MIX}{TAG}.pt"))
    # ---- REPORT ONLY: four metrics vs the reference on the validation draws that have one (never used for any selection)
    vv = [i for i in vl if SR.get(i) is not None and SR[i][4] == 0]
    assert vv, "no validation draw has a usable reference"
    sol, pr = deploy(vv)
    v = np.array([(100*abs(pr[i][0] - SR[i][0])/SR[i][0], 100*np.mean(pr[i][5] != np.asarray(SR[i][5])), 100*np.abs(pr[i][1] - SR[i][1]).sum()/SR[i][1].sum(), t_an.get(i, 0.0) + sol[i]["secs"])
                  if pr.get(i) and sol.get(i) else (np.nan,)*4 for i in vv])
    fin = lf_score(); line = f"{np.nanmean(v[:,0]):.4f}% | {np.nanmedian(v[:,0]):.4f}% | {np.nanmean(v[:,1]):.2f}% | {np.nanmean(v[:,2]):.2f}% | {np.nanmedian(v[:,3]):.1f}s | fallbacks {fin[1]}"
    print(f"[REPORT] {CASE} VALIDATION generalized integer cuts mode={MODE} K_I={KI} mixed={MIX}{(' r=%d' % Rr['r']) if MODE == 'ham' else ''} rho={RHO:.0e} ({len(vv)} draws with refs): gap mean | median | disc | cont | time | label-free {fin[0]:.6f}"
          f" = {line}", flush=True)
    if not SMOKE: runlog.add_result(RUN, f"{CASE} VALIDATION gen-int-cuts {MODE} K_I={KI} M={MIX} r={Rr['r']:.0f} rho={RHO:.0e}: gap mean | median | disc | cont | time | fallbacks = {line}; label-free score {fin[0]:.6f}")
    dp.close()


if __name__ == "__main__":
    main()

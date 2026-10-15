# %% LABEL-FREE learned V through the EXACT conic layer, end to end (2026-10-09). All four pillars:
#    learned cuts (cut-net, self-supervised, frozen) + learned V (V-net), no labels, V trained by the exact
#    layer's implicit gradient, nonconvex AC (QCAC).
#    V-net: V = flat + MLP(load), last layer zero-init (starts at flat V). Loss per instance = cut-constrained
#    relaxed QCAC objective at V-net's V (normalised by its flat-V value); dL/dV from the ConeLayer (workers),
#    backpropagated into the weights. rho in {1e4, 1e5}: one net each; selection by LABEL-FREE validation =
#    mean deployed (strict) cost / flat-V relaxation cost of OUR METHOD (V-net V + cuts, one MIQCQP, W).
#    Test (fresh set, never used): OUR METHOD vs the SUPERVISED NN PROXY (results/proxy_sup_{case}.pkl).
#    Usage: python 40_vnet_layer.py <case> <W>
import os, sys, json, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
CASE = sys.argv[1]; W = float(sys.argv[2]); WLONG = {"case118": 120, "case300": 180}.get(CASE, 120)
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
GRAD = sys.argv[4] if len(sys.argv) > 4 else "vjp"   # "env": CORRECTED envelope gradient (2026-10-10 audit; vgrad VJP wrong off flat V)
# (2026-10-12, clean label-free redo) argv[5] == "nocuts": train the V-net on the UNCUT relaxation -- no cut-net, no teacher, no c are
# read at all (the old run used fixings from a cut-net fitted on our own W=120/180 long solves). Output suffix "_nc".
NOCUT = len(sys.argv) > 5 and sys.argv[5] == "nocuts"
# (2026-10-12) argv[5] == "lfcuts", argv[6] = cut tag (e.g. _lfc): V-net ROUND 2 trained through the relaxation with the fixings of the
# LABEL-FREE cut-net results/unit_cuts_layer_<case>_soft<tag>.pt (its own c); never the long-solve teacher. Output suffix argv[7] (default _r2).
LFCUTS = len(sys.argv) > 6 and sys.argv[5] == "lfcuts"; CUTTAG = sys.argv[6] if LFCUTS else ""
SUF = ("" if SEED == 0 else f"_s{SEED}") + ("_env" if GRAD == "env" else "") + ("_nc" if NOCUT else "") + ((sys.argv[7] if len(sys.argv) > 7 else "_r2") if LFCUTS else "")
PG = CASE.startswith("pglib")      # PGLib: test = never-seen split test set; cut confidence from its frozen run
EPOCHS, BATCH, LR = 40, 24, 1e-3


def main():
    import combopt_ac, data2
    from scipy.stats import wilcoxon
    from pipeline import grid
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    # ---- learned cuts: cut-net exactly as 17/19 (teacher = own long solves), c frozen ----
    if LFCUTS:
        UC = torch.load(os.path.join(HERE, "results", f"unit_cuts_layer_{CASE}_soft{CUTTAG}.pt"), weights_only=False); c = UC["c"]
        cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G)); cm.load_state_dict(UC["state"]); cm.eval()
        with torch.no_grad(): Pr = torch.sigmoid(cm(Xn)).numpy()
        fx = lambda i: ([k for k in range(G) if Pr[i, k] > c], [k for k in range(G) if Pr[i, k] < 1 - c])   # = the deployed rounding (LO > 0 / HI < 1)
        print(f"[config] {CASE} seed {SEED}: V-net round 2 through the relaxation with LABEL-FREE cut fixings {CUTTAG} (c={c}, cut epoch {UC['epoch']})", flush=True)
    elif NOCUT:
        fx = lambda i: ([], [])                       # uncut relaxation: nothing from any teacher reaches the V-net
        print(f"[config] {CASE} seed {SEED}: V-net trained on the UNCUT relaxation (no cut-net, no teacher, no c)", flush=True)
    else:
        TG = {int(k): np.array(v) for k, v in json.load(open(os.path.join(HERE, "results", f"teacher_{CASE}_w{WLONG}.json"))).items()}
        ids = [i for i in tr if i in TG]; Y = torch.tensor(np.stack([TG[i] for i in ids]), dtype=torch.float32)
        torch.manual_seed(0)
        cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
        o = torch.optim.Adam(cm.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(1500):
            o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(cm(Xn[ids]), Y); l.backward(); o.step()
        with torch.no_grad(): Pr = torch.sigmoid(cm.eval()(Xn)).numpy()
        c = pickle.load(open(os.path.join(HERE, "results", f"split_{CASE}_flatV.pkl" if PG else f"fresh_{CASE}.pkl"), "rb"))["c"]
        fx = lambda i: ([k for k in range(G) if Pr[i, k] >= c], [k for k in range(G) if Pr[i, k] <= 1 - c])
    pool = mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    SC = {i: r[1] for i, r in pool.imap_unordered(combopt_ac.relax, va) if r is not None}
    def Vof(m, idx):
        with torch.no_grad(): out = m(Xn[idx]).numpy().astype(float)
        return {i: (1.0 + out[k, :n], out[k, n:]) for k, i in enumerate(idx)}
    def deploy(Vd, idx, rho):
        sol = dict(pool.imap_unordered(combopt_ac.miqcqp_fix, [(i, Vd[i], *fx(i), W, rho) for i in idx]))
        pr = dict(pool.imap_unordered(combopt_ac.price_strict, [(i, sol[i]["u"]) for i in idx if sol.get(i)]))
        return sol, pr
    vscore = lambda pr: float(np.mean([pr[i][0]/SC[i] if pr.get(i) else 10.0 for i in va]))
    flat = {i: (np.ones(n), np.zeros(n)) for i in tr + va + te + data2.fresh_ids()}
    L0 = {}                                                   # per-instance flat-V loss: normaliser
    cands = {}
    for rho in (1e4, 1e5):
        GF = combopt_ac.vgrad_env if GRAD == "env" else combopt_ac.vgrad
        r0 = dict(pool.imap_unordered(GF, [(i, *flat[i], rho, fx(i)) for i in tr]))
        L0 = {i: r0[i][0] for i in tr if r0.get(i)}
        torch.manual_seed(SEED)
        m = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2*n))
        nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        opt = torch.optim.Adam(m.parameters(), lr=LR); rng = np.random.default_rng(SEED)
        trn = [i for i in tr if i in L0]; best = None
        for ep in range(EPOCHS):
            order = rng.permutation(trn); Ls = []
            for s0 in range(0, len(order), BATCH):
                b = [int(i) for i in order[s0:s0+BATCH]]
                out = m(Xn[b]); Vr_t, Vi_t = 1.0 + out[:, :n], out[:, n:]
                Vn = Vr_t.detach().numpy().astype(float), Vi_t.detach().numpy().astype(float)
                R = dict(pool.imap_unordered(GF, [(i, Vn[0][k], Vn[1][k], rho, fx(i)) for k, i in enumerate(b)]))
                gr_, gi_, ok = np.zeros_like(Vn[0]), np.zeros_like(Vn[1]), 0
                for k, i in enumerate(b):
                    if R.get(i) is None: continue
                    Ls.append(R[i][0]/L0[i]); gr_[k] = R[i][1]/L0[i]; gi_[k] = R[i][2]/L0[i]; ok += 1
                if ok == 0: continue
                opt.zero_grad()
                ((Vr_t*torch.tensor(gr_, dtype=torch.float32)).sum() + (Vi_t*torch.tensor(gi_, dtype=torch.float32)).sum()).div(ok).backward()
                opt.step()
            if ep % 5 == 4 or ep == EPOCHS - 1:
                _, pr = deploy(Vof(m, va), va, rho); vs = vscore(pr)
                if best is None or vs < best[0]: best = (vs, {k: t.clone() for k, t in m.state_dict().items()}, ep)
                print(f"[rho {rho:.0e} ep {ep:2d}] train loss (vs flat V) {np.mean(Ls):.5f} | label-free val score {vs:.6f}  [{time.time()-t0:.0f}s]", flush=True)
        m.load_state_dict(best[1]); m.eval(); cands[rho] = (best[0], m, best[2])
    _, prf = deploy({i: flat[i] for i in va}, va, 1e4); cands["flat"] = (vscore(prf), None, -1)
    for k, v in cands.items(): print(f"[val] {('flat V @1e4' if k == 'flat' else f'V-net @{k:.0e} (ep {v[2]})'):22s} label-free score {v[0]:.6f}", flush=True)
    sel = min(cands, key=lambda k: cands[k][0]); print(f"[select] {sel}", flush=True)
    try:
        SRv = data2.strict_refs(CASE); vv = [i for i in va if SRv.get(i) is not None and SRv[i][4] == 0]
        for k, v in cands.items():
            if v[1] is None: continue
            Vv = Vof(v[1], vv); e = np.mean([np.abs(Vv[i][0] + 1j*Vv[i][1] - (SRv[i][2] + 1j*SRv[i][3])).max() for i in vv])
            print(f"[diag only] V-net @{k:.0e}: V error vs reference voltages on {len(vv)} VALIDATION instances {e:.4f} "
                  f"(flat {np.mean([np.abs(1 - (SRv[i][2] + 1j*SRv[i][3])).max() for i in vv]):.4f})", flush=True)
    except Exception as ex:
        print(f"[diag only] skipped: {ex}", flush=True)
    rho_s = 1e4 if sel == "flat" else sel; msel = cands[sel][1]
    if os.environ.get("NOTEST") == "1":   # final evaluation is a separate, single run on untouched draws
        torch.save(dict(sel=str(sel), rho=rho_s, state=None if msel is None else msel.state_dict()), os.path.join(HERE, "results", f"vnet_layer_{CASE}{SUF}.pt"))
        print(f"[saved] vnet_layer_{CASE}{SUF}.pt (no test evaluation)", flush=True); pool.close(); return
    # ---- TEST: OUR METHOD vs SUPERVISED NN PROXY, same fresh instances / cached strict references ----
    SR = data2.strict_refs(CASE); PF = os.path.join(HERE, "results", f"proxy_sup_{CASE}.pkl")
    P = pickle.load(open(PF, "rb")) if os.path.exists(PF) else None
    tid = P["tid"] if P else [i for i in (te if PG else data2.fresh_ids()) if SR.get(i) is not None and SR[i][4] == 0]
    J = {i: SR[i][0] for i in tid}; VO = {i: SR[i][2] + 1j*SR[i][3] for i in tid}
    Vt = {i: flat[i] for i in tid} if msel is None else Vof(msel, tid)
    err = np.array([np.abs(Vt[i][0] + 1j*Vt[i][1] - VO[i]).max() for i in tid])
    sol, pr = deploy(Vt, tid, rho_s); pool.close()
    rows = {"OUR METHOD (learned V + learned cuts)": [(100*abs(pr[i][0]-J[i])/J[i], 100*np.mean(pr[i][5] != np.asarray(SR[i][5])),
              100*np.abs(pr[i][1]-SR[i][1]).sum()/SR[i][1].sum(), sol[i]["secs"], int(bool(sol[i].get("fallback")))) if pr.get(i) else (np.nan,)*4 + (1,) for i in tid],
            "supervised NN proxy": [] if P is None else [(100*abs(P["PX"][i]["cost"]-J[i])/J[i], 100*np.mean(P["PX"][i]["u"] != np.asarray(SR[i][5])),
              100*np.abs(P["PX"][i]["pg"]-SR[i][1]).sum()/SR[i][1].sum(), P["PX"][i]["secs"], 0) if P["PX"].get(i) else (np.nan,)*4 + (1,) for i in tid]}
    print(f"\n[TEST {CASE}] n={len(tid)} fresh | V error vs reference voltages (diagnostic): learned {err.mean():.4f}, flat {np.mean([np.abs(1-VO[i]).max() for i in tid]):.4f}", flush=True)
    for k, r in rows.items():
        if not r: continue
        v = np.array(r, float)
        print(f"  {k:38s} gap mean {np.nanmean(v[:,0]):.4f}% median {np.nanmedian(v[:,0]):.4f}% | disc {np.nanmean(v[:,1]):.2f}% | cont {np.nanmean(v[:,2]):.2f}% | "
              f"time median {np.nanmedian(v[:,3]):.2f}s | fallbacks {int(np.nansum(v[:,4]))}", flush=True)
    if P is not None:
        x = np.array(rows["OUR METHOD (learned V + learned cuts)"], float)[:, 0]; y = np.array(rows["supervised NN proxy"], float)[:, 0]
        mk = ~np.isnan(x) & ~np.isnan(y); d = x[mk] - y[mk]; rng = np.random.default_rng(0); bs = [rng.choice(d, d.size).mean() for _ in range(10000)]
        print(f"  OURS - PROXY: {d.mean():+.4f} pp, 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}], p {wilcoxon(x[mk], y[mk]).pvalue:.3g}, "
              f"ours better/tie/worse {(d < -1e-6).sum()}/{(abs(d) <= 1e-6).sum()}/{(d > 1e-6).sum()}", flush=True)
    torch.save(dict(sel=str(sel), rho=rho_s, state=None if msel is None else msel.state_dict()), os.path.join(HERE, "results", f"vnet_layer_{CASE}{SUF}.pt"))
    pickle.dump(dict(tid=tid, sol=sol, pr=pr, rows=rows, err=err, cands={str(k): (v[0], v[2]) for k, v in cands.items()}), open(os.path.join(HERE, "results", f"vnet_layer_{CASE}{SUF}.pkl"), "wb"))


if __name__ == "__main__":
    main()

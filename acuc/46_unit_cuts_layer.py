# %% PER-UNIT INTEGER CUTS learned through the smoothed EXACT LAYER (2026-10-10), case118, label-free.
#    Cut for each unit k: u_k >= l_k and u_k <= h_k, with m = cut-net confidence and c = 0.9 (validation-chosen):
#        l = relu((m - c)/(1 - c)),   h = 1 - relu(((1 - c) - m)/(1 - c))       (l <= h always)
#    In the INTEGER problem u_k >= l_k > 0 => u_k = 1 and u_k <= h_k < 1 => u_k = 0 (rounding the cut's RHS),
#    so at the WARM START (the BCE cut-net) the cuts are exactly the current method's fixings.
#    Fine-tuned through the layer: (l, h) are the smoothed relaxation's bound parameters (gamma = 1000), exact VJP.
#    Loss = ||u_relax - u_own||^2/G + LAM * validity (our own long-solve commitment must satisfy every cut).
#    Validation only (label-free score + fallbacks), deploy = one short MIQCQP (W=10) at the learned V with the
#    cuts' integer rounding. Epoch -1 = the current method. Step 0 = FD gate.
import os, sys, json, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
SOFT = len(sys.argv) > 1 and sys.argv[1] == "soft"   # cuts as SOFT rows in the layer, as in deployment
CASE = sys.argv[2] if len(sys.argv) > 2 else "case118"; PG = CASE.startswith("pglib")
GAMMA, EPOCHS, BATCH, LR, LAM = 1000.0, 15, 24, 1e-4, 10.0
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0); WLONG = {"case118": 120, "case300": 180}.get(CASE, 120)
# (2026-10-11) optional: argv[3] = teacher json (e.g. the label-free best-of teacher from 58), argv[4] = output tag (default: unchanged files)
TEACHER = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "results", f"teacher_{CASE}_w{WLONG}.json")
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
EXTRA = len(sys.argv) > 5 and sys.argv[5] == "extra"
VTAG = sys.argv[6] if len(sys.argv) > 6 else ""        # (2026-10-12) V-net file suffix ("_nc" = clean V-net, uncut relaxation)
SELC = len(sys.argv) > 7 and sys.argv[7] == "selc"    # re-select c by the label-free validation score (the old c came from long-teacher cuts)
VALUEH = len(sys.argv) > 8 and sys.argv[8] == "valueh"  # (2026-10-13) HARDENED value loss (cuttrain.value_soft_h) + LINEAR validity, gamma=0
VALUE = (len(sys.argv) > 8 and sys.argv[8] == "value") or VALUEH  # (2026-10-12, option A) value loss + exact envelope gradient (cuttrain.value_soft), gamma=0
CGRID = (0.999, 0.99, 0.95, 0.9)   # candidate confidence thresholds for SELC (same grid as 17_fix_cuts.py)


def main():
    import combopt_ac, cuttrain, data2
    from pipeline import grid
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    from vnetio import vnet_out
    OUT, RHO, VSEL = vnet_out(os.path.join(HERE, "results", f"vnet_layer_{CASE}_env{VTAG}.pt"), Xn, n)
    V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
    TG = {int(k): np.array(v, float) for k, v in json.load(open(TEACHER)).items()}
    TRN = tr + (data2.extra_ids() if EXTRA else [])
    ids = [i for i in TRN if i in TG]; Y = torch.tensor(np.stack([TG[i] for i in ids]), dtype=torch.float32)
    torch.manual_seed(0)                                      # warm start = the current BCE cut-net, exactly
    cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
    o = torch.optim.Adam(cm.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(1500):
        o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(cm(Xn[ids]), Y); l.backward(); o.step()
    c = CGRID[0] if SELC else pickle.load(open(os.path.join(HERE, "results", f"split_{CASE}_flatV.pkl" if PG else f"fresh_{CASE}.pkl"), "rb"))["c"]
    def cuts(idx, grad=False):
        with torch.set_grad_enabled(grad):
            m = torch.sigmoid(cm(Xn[idx])); lo = torch.relu((m - c)/(1 - c)); hi = 1 - torch.relu(((1 - c) - m)/(1 - c))
        return lo, hi
    def fixes(lo, hi):
        return [k for k in range(G) if lo[k] > 0], [k for k in range(G) if hi[k] < 1]
    dp = mp.get_context("spawn").Pool(11, initializer=combopt_ac.init, initargs=(CASE, True)); t0 = time.time()
    SC = {i: r[1] for i, r in dp.imap_unordered(combopt_ac.relax, va) if r is not None}; vv = [i for i in va if i in SC]
    print(f"[config] {CASE}: W={W}, teacher {os.path.basename(TEACHER)} ({len(TG)} inst), c={'re-selected below' if SELC else c}, V-net '{VTAG}' (selected {VSEL}) rho {RHO:.0e}, output tag '{TAG}'", flush=True)
    trn = [i for i in TRN if i in TG]
    print(f"[config] training instances: {len(trn)} ({'original split + extra unlabeled draws' if EXTRA else 'original split'})", flush=True)
    lp = mp.get_context("spawn").Pool(11, initializer=(cuttrain.init4 if SOFT else cuttrain.init3), initargs=(CASE, 0.0 if VALUE else GAMMA))
    TR = cuttrain.value_soft_h if VALUEH else (cuttrain.value_soft if VALUE else (cuttrain.track_soft if SOFT else cuttrain.track_bounds))
    print(f"[config] cut loss: {'VALUE: -(relaxation gap closed, capped at our own answer) + LAM*validity, exact envelope gradient, no smoothing' if VALUE else 'tracking ||u_relax - u_own||^2 (KKT VJP)'}", flush=True)
    print(f"[config] per-unit integer cuts as {'SOFT rows (as deployment)' if SOFT else 'HARD bounds'} in the layer", flush=True)
    def validate():
        lo, hi = cuts(vv); lo, hi = lo.numpy(), hi.numpy()
        sol = dict(dp.imap_unordered(combopt_ac.miqcqp_fix, [(i, V(i), *fixes(lo[k], hi[k]), W, RHO) for k, i in enumerate(vv)]))
        pr = dict(dp.imap_unordered(combopt_ac.price_strict, [(i, sol[i]["u"]) for i in vv if sol.get(i)]))
        nfix = np.mean([sum(map(len, fixes(lo[k], hi[k]))) for k in range(len(vv))])
        return float(np.mean([pr[i][0]/SC[i] if pr.get(i) else 10.0 for i in vv])), sum(bool(sol[i].get("fallback")) for i in vv if sol.get(i)), nfix
    if SELC:                                    # (2026-10-12) c chosen by the LABEL-FREE validation score with the warm-start cuts
        csc = {}
        for cc in CGRID:
            c = cc; csc[cc] = validate()
            print(f"[c] {cc}: label-free val {csc[cc][0]:.6f} (fallbacks {csc[cc][1]}, units fixed {csc[cc][2]:.1f}/{G})", flush=True)
        c = min(CGRID, key=lambda k: csc[k][0]); print(f"[c] selected c = {c} (label-free)", flush=True)
    # ---- step 0: FD GATE (directions in (l, h) space, at the warm-start cuts) ----
    rng = np.random.default_rng(0); gate = []
    for i in trn[:3]:
        lo, hi = [t[0].numpy().astype(float) for t in cuts([i])]
        _, (L0, gl, gh) = lp.apply(TR, ((i, *V(i), lo, hi, TG[i], True),))
        jobs, dirs, h = [], [], 1e-2
        for _ in range(8):
            dl = rng.standard_normal(G); dh = rng.standard_normal(G)
            dl[(lo + 2*h > np.minimum(hi, 1.0))] = 0.0; dh[(hi - 2*h < np.maximum(lo, 0.0))] = 0.0   # keep 0 <= l <= h <= 1
            nr = np.sqrt((dl**2).sum() + (dh**2).sum()); dl /= nr; dh /= nr
            dirs.append((dl, dh)); jobs += [(i, *V(i), lo + h*dl, hi + h*dh, TG[i], False), (i, *V(i), lo - h*dl, hi - h*dh, TG[i], False)]
        res = [r for _, r in lp.imap(TR, jobs)]
        okd = [k for k in range(8) if res[2*k] is not None and res[2*k+1] is not None]
        fd = np.array([(res[2*k][0] - res[2*k+1][0])/(2*h) for k in okd])
        fw = np.array([(res[2*k][0] - L0)/h for k in okd]); bw = np.array([(L0 - res[2*k+1][0])/h for k in okd])
        an = np.array([dirs[k][0] @ gl + dirs[k][1] @ gh for k in okd]); cfd = np.corrcoef(fw, bw)[0, 1]; ca = np.corrcoef(fd, an)[0, 1]; gate.append((cfd, ca))
        print(f"[gate] inst {i}: {len(okd)}/8 directions solved, interior coords: l {(np.abs(dirs[0][0]) > 0).sum()} h {(np.abs(dirs[0][1]) > 0).sum()}", flush=True)
        print(f"[gate] inst {i}: FD self-consistency {cfd:.3f} | analytic vs FD corr {ca:.4f}, signs {(np.sign(fd) == np.sign(an)).sum()}/{len(okd)}", flush=True)
    good = [ca for cfd, ca in gate if np.isfinite(cfd) and np.isfinite(ca) and cfd >= 0.99]   # (audit 2026-10-13) NaN-blind rule fixed: non-finite correlations never count as a pass
    if not good or min(good) < 0.95:
        print("[gate] FAILED -> stop", flush=True); lp.close(); dp.close(); return
    print("[gate] PASSED -> train", flush=True)
    v0 = validate()
    print(f"[val] epoch -1 = CURRENT METHOD (warm start): {v0[0]:.6f} (fallbacks {v0[1]}, units fixed {v0[2]:.1f}/{G})  [{time.time()-t0:.0f}s]", flush=True)
    # (2026-10-11, Parth) deploy cuts LEARNED THROUGH THE LAYER: label-free selection among trained epochs only (warm start not eligible)
    opt = torch.optim.Adam(cm.parameters(), lr=LR); best = (np.inf, None, None)
    for ep in range(EPOCHS):
        order = rng.permutation(trn); Lt, Lv, nf = [], [], 0
        for s0 in range(0, len(order), BATCH):
            b = [int(i) for i in order[s0:s0+BATCH]]
            lo_t, hi_t = cuts(b, True); lon, hin = lo_t.detach().numpy().astype(float), hi_t.detach().numpy().astype(float)
            R = dict(lp.imap_unordered(TR, [(i, *V(i), lon[k], hin[k], TG[i], True) for k, i in enumerate(b)]))
            GL, GH, ok = np.zeros_like(lon), np.zeros_like(hin), 0
            for k, i in enumerate(b):
                if R.get(i) is None: nf += 1; continue
                Lt.append(R[i][0]); GL[k], GH[k] = R[i][1], R[i][2]; ok += 1
            if ok == 0: continue
            uo = torch.tensor(np.stack([TG[i] for i in b]), dtype=torch.float32)
            lval = (LAM*(torch.relu(lo_t - uo).sum(1) + torch.relu(uo - hi_t).sum(1)).mean() if VALUEH else
                    LAM*((torch.relu(lo_t - uo)**2).sum(1) + (torch.relu(uo - hi_t)**2).sum(1)).mean()); Lv.append(float(lval))
            opt.zero_grad()
            ((lo_t*torch.tensor(GL, dtype=torch.float32)).sum() + (hi_t*torch.tensor(GH, dtype=torch.float32)).sum()).div(ok).add(lval).backward(); opt.step()
        msg = f"[ep {ep:2d}] tracking {np.mean(Lt):.5f} validity {np.mean(Lv):.5f} (failed {nf})  [{time.time()-t0:.0f}s]"
        if ep % 3 == 2 or ep == EPOCHS - 1:
            v = validate(); msg += f" | val {v[0]:.6f} (fallbacks {v[1]}, fixed {v[2]:.1f}) vs current {v0[0]:.6f}"
            if v[0] < best[0]: best = (v[0], {k: t.clone() for k, t in cm.state_dict().items()}, ep)
        print(msg, flush=True)
    lp.close(); dp.close()
    print(f"[best] epoch {best[2]} val {best[0]:.6f} vs current method {v0[0]:.6f} -> "
          f"{'LAYER-LEARNED CUTS BETTER' if best[0] < v0[0] - 1e-6 else 'no improvement over the current method'}", flush=True)
    torch.save(dict(state=best[1], epoch=best[2], val=best[0], v_cur=v0[0], c=c, last={k: t.clone() for k, t in cm.state_dict().items()}), os.path.join(HERE, "results", f"unit_cuts_layer_{CASE}{'_soft' if SOFT else ''}{TAG}.pt"))


if __name__ == "__main__":
    main()

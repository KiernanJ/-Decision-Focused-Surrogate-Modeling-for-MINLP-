# F3 audit: deployment ablations / missing baselines for the AC pipeline (second audit, main e64170f).
# Usage: python abl.py <case> <set: val | fresh5 | fresh6> <arms comma-separated>
# Deployment machinery = acuc/55_final_eval.py (anchor6_t -> rows_torch -> combopt_ac.miqcqp -> proxy_deploy), scoring = J_ref = price_strict(u_ref).
# Results are cached per (case, set) in res_<case>_<set>.pkl ; arms already present are skipped.
import os, sys, json, pickle, time
REPO = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); HERE = os.path.join(REPO, "acuc"); OUTD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, OUTD)
import numpy as np, torch, torch.nn as nn, multiprocessing as mp

CASE, SET = sys.argv[1], sys.argv[2]; ARMS = sys.argv[3].split(",")
RHO = {"case118": 3e5, "case300": 1e4, "pglib_opf_case118_ieee__api": 1e4}[CASE]
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0)
NW = 6
CACHE = os.path.join(OUTD, f"res_{CASE}_{SET}.pkl")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def mlp(d_in, G):
    return nn.Sequential(nn.Linear(d_in, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))


def fixings(M, c):
    LO = np.maximum((M - c)/(1 - c), 0); HI = 1 - np.maximum(((1 - c) - M)/(1 - c), 0)
    return LO, HI


def fix_rows(lo, hi, G):
    Au = np.zeros((2*G, 2*G)); bu = np.ones(2*G)
    for k in range(G):
        if lo[k] > 0: Au[k, G + k] = -1.0; bu[k] = -1.0
        if hi[k] < 1: Au[G + k, G + k] = 1.0; bu[G + k] = 0.0
    return Au, bu


def main():
    import combopt_ac, cuttrain, data2, f3work
    from pipeline import grid
    from cutnet_k import CutNetK, rows_torch
    from vnetio import vnet_out
    os.environ["DEPLOY_RHO"] = f"{RHO:g}"
    st = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else dict(arms={}, meta={})
    g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    pool = mp.get_context("spawn").Pool(NW, initializer=combopt_ac.init, initargs=(CASE, True))
    # ---------------- instance set + references (exactly as 61 / 55)
    if "SR" not in st["meta"]:
        if SET == "val":
            SRall = data2.strict_refs(CASE); ids = list(va)
            scored = [i for i in va if SRall.get(i) is not None and SRall[i][4] == 0]
            SR = {i: SRall[i] for i in scored}; exc = [i for i in ids if i not in scored]
        else:
            REF = pickle.load(open(os.path.join(HERE, "results", f"{SET}_refs_{CASE}.pkl"), "rb")); ids = getattr(data2, f"{SET}_ids")()
            conv = [i for i in ids if REF.get(i, {}).get("converged")]
            SRc = dict(pool.imap_unordered(combopt_ac.price_strict, [(i, REF[i]["u"]) for i in conv]))
            scored = [i for i in conv if SRc.get(i) is not None and SRc[i][4] == 0]
            SR = {i: SRc[i] for i in scored}; exc = [i for i in ids if i not in scored]
        RL = dict(pool.imap_unordered(f3work.relax_flat, ids))
        st["meta"].update(SR=SR, ids=ids, scored=scored, exc=exc, RLflat={i: (None if RL[i] is None else RL[i][1]) for i in ids})
        pickle.dump(st, open(CACHE, "wb"))
        log(f"[{CASE} {SET}] {len(ids)} draws, {len(scored)} scored (converged ref, no repair); excluded {exc}")
    ids, scored, SR = st["meta"]["ids"], st["meta"]["scored"], st["meta"]["SR"]
    # the deployed instance list: val -> all 24 draws (label-free score on all, report on scored); fresh -> scored only (as 55)
    dep = ids if SET == "val" else scored
    # ---------------- learned components (as 55 / 61)
    vf = os.path.join(HERE, "results", f"vnet_layer_{CASE}_env_nc.pt")
    if SET == "val":
        OUT, rho_, VSEL = vnet_out(vf, Xn, n); Vd = {i: (1.0 + OUT[i, :n], OUT[i, n:]) for i in dep}            # 61: full Xn
    else:
        OUT, rho_, VSEL = vnet_out(vf, Xn[dep], n); Vd = {i: (1.0 + OUT[k, :n], OUT[k, n:]) for k, i in enumerate(dep)}   # 55: Xn[tid]
    assert abs(rho_ - RHO) < 1e-6
    flat = {i: (np.ones(n), np.zeros(n)) for i in dep}
    UC = torch.load(os.path.join(HERE, "results", f"unit_cuts_layer_{CASE}_soft_lfc.pt"), weights_only=False); c = UC["c"]
    cm = mlp(X.shape[1], G); cm.load_state_dict(UC["state"]); cm.eval()
    with torch.no_grad(): Mc = torch.sigmoid(cm(Xn)).numpy()
    GR = torch.load(os.path.join(HERE, "results", f"general_rows_{CASE}_lfc.pt"), weights_only=False); K = GR["K"]
    gnet = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False)
    gnet.load_state_dict(GR["selected"]); gnet.eval()
    # A5 warm start (46 recipe, self labels, 444), A6 (46 recipe, reference labels), A6b (46 recipe, self labels on the proxy's instances)
    TG = {int(k): np.array(v, float) for k, v in json.load(open(os.path.join(HERE, "results", f"selfteacher_{CASE}_bestof_extra_nc.json"))).items()}
    RP = data2.refs(CASE, os.path.join(HERE, "results", "pool2"))
    ltr = [i for i in tr if data2.converged(RP.get(i))]; lva = [i for i in va if data2.converged(RP.get(i))]
    if not lva: k_ = max(1, len(ltr)//5); lva, ltr = ltr[-k_:], ltr[:-k_]

    def bce46(idx, Y):
        torch.manual_seed(0); m = mlp(X.shape[1], G); o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(1500):
            o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(m(Xn[idx]), Y); l.backward(); o.step()
        m.eval()
        with torch.no_grad(): return torch.sigmoid(m(Xn)).numpy()
    nets = {}
    need = set(ARMS)
    need |= {a[:-1] for a in ARMS if a in ("A5h", "A6h", "A6bh")} | ({"A7"} if "A7h" in ARMS else set()) | ({"A5"} if "A12i" in ARMS else set())
    if need & {"A5"}:
        ids5 = [i for i in tr + data2.extra_ids() if i in TG]
        nets["A5"] = bce46(ids5, torch.tensor(np.stack([TG[i] for i in ids5]), dtype=torch.float32)); st["meta"]["n_A5"] = len(ids5)
    if need & {"A6", "A6r"}:
        nets["A6"] = bce46(ltr, torch.tensor(np.stack([RP[i]["u"] for i in ltr]), dtype=torch.float32)); st["meta"]["n_A6"] = len(ltr)
    if need & {"A6b"}:
        ids6b = [i for i in ltr if i in TG]
        nets["A6b"] = bce46(ids6b, torch.tensor(np.stack([TG[i] for i in ids6b]), dtype=torch.float32)); st["meta"]["n_A6b"] = len(ids6b)
    if need & {"A7", "A8", "A10"}:
        Ytr = torch.tensor(np.stack([RP[i]["u"] for i in ltr]), dtype=torch.float32); Yva = torch.tensor(np.stack([RP[i]["u"] for i in lva]), dtype=torch.float32)
        torch.manual_seed(0)
        pm = mlp(X.shape[1], G); po = torch.optim.Adam(pm.parameters(), lr=1e-3, weight_decay=1e-4); best = (np.inf, None); bce = nn.functional.binary_cross_entropy_with_logits
        for it in range(3000):
            po.zero_grad(); l = bce(pm(Xn[ltr]), Ytr); l.backward(); po.step()
            if it % 25 == 0:
                with torch.no_grad(): v_ = float(bce(pm(Xn[lva]), Yva))
                if v_ < best[0]: best = (v_, {k: t.clone() for k, t in pm.state_dict().items()})
        pm.load_state_dict(best[1]); pm.eval()
        with torch.no_grad():
            PPdep = torch.sigmoid(pm(Xn[dep])).numpy()      # 55: pm(Xn[tid])
        nets["proxy"] = {i: PPdep[k] for k, i in enumerate(dep)}
        st["meta"]["proxy_val_bce"] = best[0]; st["meta"]["n_proxy"] = (len(ltr), len(lva))
    log(f"[{CASE} {SET}] c={c}, cut epoch {UC['epoch']}, rows epoch {GR['selected_epoch']}, V-net {VSEL}, RHO {RHO:g}, W {W}; proxy labels {len(ltr)}/{len(lva)}")
    # ---------------- anchors for arms with rows
    LOc, HIc = fixings(Mc, c)
    anchors = st["meta"].setdefault("anchors", {})
    need_anchor = {"A0": Vd, "A3b": flat}
    if "A0h" in ARMS and "A0" not in anchors and "A0" not in ARMS: ARMS.insert(0, "A0")
    ap = None
    for arm, VV in need_anchor.items():
        if arm in ARMS and arm not in st["arms"] and arm not in anchors:
            if ap is None:
                ap = mp.get_context("spawn").Pool(NW, initializer=cuttrain.init6, initargs=(CASE, 1000.0, K))
            ANT = dict(ap.imap_unordered(cuttrain.anchor6_t, [(i, *VV[i], LOc[i], HIc[i]) for i in dep]))
            anchors[arm] = ANT
            log(f"   anchors {arm}: failed on {sum(ANT[i][0] is None for i in dep)}")
    if ap is not None: ap.close()

    def rows_for(i, ANT):
        Au, bu = fix_rows(LOc[i], HIc[i], G)
        if ANT.get(i) is not None and ANT[i][0] is not None:
            with torch.no_grad():
                _, _, a, al, a2, d, d2, _ = gnet(Xn[[i]]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.tensor(ANT[i][0], dtype=torch.float32)[None])
            Au = np.vstack([Au, A_[0, :K].numpy().astype(float)]); bu = np.r_[bu, b_[0, :K].numpy().astype(float)]
        return Au, bu

    def fix_only(i, M):
        lo, hi = fixings(M, c)
        return fix_rows(lo, hi, G)

    for arm in ARMS:
        if arm in st["arms"]:
            log(f"skip {arm} (cached)"); continue
        t0 = time.time()
        if arm == "A0":   jobs = [(i, Vd[i], *rows_for(i, anchors["A0"]), W, RHO) for i in dep]
        elif arm == "A1": jobs = [(i, flat[i], None, None, W, RHO) for i in dep]
        elif arm == "A2": jobs = [(i, Vd[i], None, None, W, RHO) for i in dep]
        elif arm == "A3a": jobs = [(i, flat[i], *fix_only(i, Mc[i]), W, RHO) for i in dep]
        elif arm == "A3b": jobs = [(i, flat[i], *rows_for(i, anchors["A3b"]), W, RHO) for i in dep]
        elif arm == "A4": jobs = [(i, Vd[i], *fix_only(i, Mc[i]), W, RHO) for i in dep]
        elif arm == "A5": jobs = [(i, Vd[i], *fix_only(i, nets["A5"][i]), W, RHO) for i in dep]
        elif arm == "A6": jobs = [(i, Vd[i], *fix_only(i, nets["A6"][i]), W, RHO) for i in dep]
        elif arm == "A6b": jobs = [(i, Vd[i], *fix_only(i, nets["A6b"][i]), W, RHO) for i in dep]
        elif arm == "A7": jobs = [(i, Vd[i], *fix_only(i, nets["proxy"][i]), W, RHO) for i in dep]
        elif arm in ("A0h", "A3ah", "A4h", "A5h", "A6h", "A6bh", "A7h"):
            Msrc = {"A0h": Mc, "A3ah": Mc, "A4h": Mc}.get(arm)
            Mget = (lambda i: Msrc[i]) if Msrc is not None else (lambda i: nets["proxy"][i] if arm == "A7h" else nets[arm[:-1]][i])
            jobs = []
            for i in dep:
                lo, hi = fixings(Mget(i), c); on = [k for k in range(G) if lo[k] > 0]; off = [k for k in range(G) if hi[k] < 1]
                Vh = flat[i] if arm == "A3ah" else Vd[i]
                if arm == "A0h":
                    Afull, bfull = rows_for(i, anchors["A0"])      # soft fallback = exactly A0's rows
                    Ar = Afull[2*G:] if Afull.shape[0] > 2*G else None; br = bfull[2*G:] if Afull.shape[0] > 2*G else None
                else:
                    Afull, bfull = fix_rows(lo, hi, G); Ar, br = None, None
                jobs.append((i, Vh, on, off, Ar, br, W, RHO, Afull, bfull))
        elif arm == "A9f": jobs = [(i, flat[i], None, None, 4*W, RHO) for i in dep]
        elif arm == "A9v": jobs = [(i, Vd[i], None, None, 4*W, RHO) for i in dep]
        elif arm in ("A12", "A12i"):   # OUR cut-net (layer-trained / imitation) deployed like the proxy: round 0.5 + strict upward repair
            Msrc = Mc if arm == "A12" else nets["A5"]
            R = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, (Msrc[i] > 0.5).astype(float)) for i in dep]))
            st["arms"][arm] = {i: (None if R[i] is None else dict(cost=R[i]["cost"], u=R[i]["u"], pg=R[i]["pg"], secs=R[i]["secs"], repaired=R[i]["repaired"], fallback=False, mipgap=np.nan, work=0.0)) for i in dep}
            pickle.dump(st, open(CACHE, "wb")); log(f"done {arm} [{time.time()-t0:.0f}s]"); continue
        elif arm == "A8":
            R = dict(pool.imap_unordered(combopt_ac.proxy_deploy, [(i, (nets["proxy"][i] > 0.5).astype(float)) for i in dep]))
            st["arms"][arm] = {i: (None if R[i] is None else dict(cost=R[i]["cost"], u=R[i]["u"], pg=R[i]["pg"], secs=R[i]["secs"], repaired=R[i]["repaired"], fallback=False, mipgap=np.nan, work=0.0)) for i in dep}
            pickle.dump(st, open(CACHE, "wb")); log(f"done {arm} [{time.time()-t0:.0f}s]"); continue
        elif arm == "A10":
            R = dict(pool.imap_unordered(f3work.proxy_polish, [(i, (nets["proxy"][i] > 0.5).astype(float), 6, 8, 4) for i in dep]))
            st["arms"][arm] = {i: (None if R[i] is None else dict(cost=R[i]["cost"], u=R[i]["u"], pg=R[i]["pg"], secs=R[i]["secs_total"], repaired=R[i]["repaired"], fallback=False, mipgap=np.nan, work=0.0, n_moves=R[i]["n_moves"], cost_start=R[i]["cost_start"])) for i in dep}
            pickle.dump(st, open(CACHE, "wb")); log(f"done {arm} [{time.time()-t0:.0f}s]"); continue
        elif arm == "A11":
            R = dict(pool.imap_unordered(f3work.teacher_method, [(i, Vd[i], W, RHO, 6, 8, 4) for i in dep]))
            st["arms"][arm] = {i: (None if R[i] is None else dict(cost=R[i]["cost"], u=R[i]["u"], pg=R[i]["pg"], secs=R[i]["secs"], repaired=R[i]["repaired"], fallback=R[i]["fallback"], mipgap=np.nan, work=R[i]["work"], winner=R[i]["winner"])) for i in dep}
            pickle.dump(st, open(CACHE, "wb")); log(f"done {arm} [{time.time()-t0:.0f}s]"); continue
        else:
            raise ValueError(arm)
        # fixing statistics for this arm (units fixed on/off)
        nfix = {}
        for j in jobs:
            if arm.endswith("h"):
                nfix[j[0]] = (len(j[2]), len(j[3])); continue
            A_ = j[2]
            if A_ is None: nfix[j[0]] = (0, 0); continue
            on = int(sum(A_[k, G + k] < 0 for k in range(G))); off = int(sum(A_[G + k, G + k] > 0 for k in range(G)))
            nfix[j[0]] = (on, off)
        R = dict(pool.imap_unordered(f3work.deploy_strict_hard if arm.endswith("h") else f3work.deploy_strict, jobs))
        out = {}
        for i in dep:
            r = R.get(i)
            if r is None or r["op"] is None:
                out[i] = None; continue
            so, op = r["sol"], r["op"]
            out[i] = dict(cost=op["cost"], u=op["u"], pg=op["pg"], secs=so["secs"] + op["secs"] + (anchors.get(arm, {}).get(i, (None, 0.0))[1] if arm in anchors else 0.0),
                          repaired=op["repaired"], fallback=bool(so.get("fallback")), mipgap=so.get("mipgap", np.nan), work=so.get("work", np.nan),
                          u_mip=so["u"], nfix=nfix[i], hard_fallback=bool(so.get("hard_fallback", False)))
        st["arms"][arm] = out
        pickle.dump(st, open(CACHE, "wb"))
        nf = sum(v is None for v in out.values()); fb = sum(bool(v and v["fallback"]) for v in out.values())
        log(f"done {arm}: failures {nf}, fallbacks {fb} [{time.time()-t0:.0f}s]")
    pool.close()
    # store fixing probabilities for later analysis (A0 = committed cut-net)
    st["meta"]["Mc"] = {i: Mc[i] for i in dep}; st["meta"]["c"] = c
    for k_ in ("A5", "A6", "A6b"):
        if k_ in nets: st["meta"][f"M_{k_}"] = {i: nets[k_][i] for i in dep}
    if "proxy" in nets: st["meta"]["M_proxy"] = nets["proxy"]
    pickle.dump(st, open(CACHE, "wb"))
    log("ALL DONE")


if __name__ == "__main__":
    main()

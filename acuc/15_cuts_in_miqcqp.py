# %% Phase 1: learned cut rows INSIDE the time-limited MIQCQP (CombOptNet setting: the cuts are
#    constraints of the integer program the solver optimises). Self-supervised.
#    Deployment: MIQCQP at the learned V (exact conic layer) + cut rows, work limit W, AC pricing.
#    --mode teacher : target = OUR OWN long solve (work W_long, no cut), priced; cuts learn to make
#                     the short solve reach it (no reference solver, no labels).
#    --mode descent : CombOptNet driven by exact-layer-ranked single flips priced with AC.
#    Selection on deployed validation cost. Test: cut vs no-cut at the same W.
import os, sys, json, time, argparse, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch, multiprocessing as mp
ap = argparse.ArgumentParser()
ap.add_argument("--case", default="case300"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", default="teacher", choices=["teacher", "descent"])
ap.add_argument("--work", type=float, default=10.0); ap.add_argument("--work-long", type=float, default=120.0)
ap.add_argument("--rho", type=float, default=1e4); ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--rows", type=int, default=4); ap.add_argument("--margin", type=float, default=1.0)
ap.add_argument("--kmoves", type=int, default=8); ap.add_argument("--workers", type=int, default=11)
ap.add_argument("--tag", default="")
a = ap.parse_args()


def main():
    import combopt_ac, data2
    from pipeline import grid
    from cutnet_k import CutNetK
    from combopt_net import CutNet
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    g, nl = grid(a.case); G = g.n_gen; INST = data2.instances(g, a.case); te, va, tr = data2.split()
    REF = data2.refs(a.case, os.path.join(HERE, "results", "pool2"))
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
    vnet = CutNetK(g.n_bus, G, k_rows=2, v_scale=0.5, hidden=256, seed=0, learn_thr=False)
    vnet.load_state_dict(torch.load(os.path.join(HERE, "checkpoints2", f"{a.case}_s0.pt"), weights_only=False)["net"]); vnet.eval()
    with torch.no_grad(): VR, VI = [t.numpy().astype(float) for t in vnet(Xn)[:2]]
    V = {i: (VR[i], VI[i]) for i in range(len(INST))}
    pool = mp.get_context("spawn").Pool(a.workers, initializer=combopt_ac.init, initargs=(a.case,))
    ids_all = tr + va + te
    UH, SC = {}, {}
    for i, r in pool.imap_unordered(combopt_ac.relax, [(i, V[i]) for i in ids_all]):
        if r is not None: UH[i] = r[0]
    for i, r in pool.imap_unordered(combopt_ac.relax, [i for i in ids_all]):     # flat-V normaliser (label-free)
        if r is not None: SC[i] = r[1]
    tr = [i for i in tr if i in UH and i in SC]; va = [i for i in va if i in UH and i in SC]; te = [i for i in te if i in UH and i in SC]
    net = CutNet(X.shape[1], G, a.rows, seed=a.seed)

    def rows(ids):
        A, d = net(Xn[ids]); uh = torch.tensor(np.stack([UH[i] for i in ids]), dtype=torch.float32)
        return A, torch.einsum("nrg,ng->nr", A, uh) + a.margin - d

    def solve(ids, A=None, b=None, work=a.work):
        jobs = [(i, V[i], None if A is None else A[k], None if b is None else b[k], work, a.rho) for k, i in enumerate(ids)]
        return dict(pool.imap_unordered(combopt_ac.miqcqp, jobs))

    def score(R, ids):
        return float(np.mean([10.0 if R.get(i) is None else R[i]["cost"]/SC[i] for i in ids]))

    if a.mode == "teacher":
        t0 = time.time(); RL = solve(tr, work=a.work_long)
        TGT = {i: RL[i]["u"] for i in tr if RL.get(i) is not None and not RL[i].get("fallback")}
        print(f"[teacher] {len(TGT)} self-found targets from W={a.work_long:.0f} solves in {time.time()-t0:.0f}s", flush=True)

    def mismatch(A_t, b_t, u, ut):
        diff = [k for k in range(G) if u[k] != ut[k]]
        if not diff: return None
        nrm = torch.linalg.norm(A_t, dim=1) + 1e-9
        dist = lambda pt: torch.abs(A_t @ torch.tensor(pt, dtype=torch.float32) - b_t)/nrm
        P = torch.zeros(()); w = u.copy()
        for k in diff[:a.kmoves]:
            w = w.copy(); w[k] = ut[k]
            v = (A_t @ torch.tensor(w, dtype=torch.float32) - b_t) > 0
            P = P + ((dist(w)*v.float()).sum() if v.any() else -0.1*torch.logsumexp(-dist(u)/0.1, 0))
        v = (A_t @ torch.tensor(ut, dtype=torch.float32) - b_t) > 0
        if v.any(): P = P + (dist(ut)*v.float()).sum()
        return P

    def descent_target(i, res):
        """best single flip among exact-layer-ranked candidates (priced); target = flipped u."""
        u = res["u"]; order = combopt_ac._rank_flips(i, u, res["pg"], res["vr"], res["vi"])[:6]
        best = None
        for k in order:
            w = u.copy(); w[k] = 1 - w[k]; r = combopt_ac._price(i, w)
            if r is not None and r[0] < res["cost"] and (best is None or r[0] < best[1]): best = (w, r[0])
        return None if best is None else best[0]

    NC_va = solve(va); v_nc = score(NC_va, va)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    with torch.no_grad(): A0, b0 = rows(va)
    v0 = score(solve(va, A0.numpy().astype(float), b0.numpy().astype(float)), va)
    best = (v0, {k: v.clone() for k, v in net.state_dict().items()}, -1); hist = []
    print(f"[{a.case} {a.mode} W={a.work}] init val {v0:.6f}  no-cut val {v_nc:.6f}", flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        order = rng.permutation(tr); D = []
        for s0 in range(0, len(order), a.batch):
            ids = [int(i) for i in order[s0:s0+a.batch]]
            A_t, b_t = rows(ids)
            R = solve(ids, A_t.detach().numpy().astype(float), b_t.detach().numpy().astype(float))
            P, n = 0.0, 0
            for k, i in enumerate(ids):
                if R.get(i) is None: continue
                ut = TGT.get(i) if a.mode == "teacher" else descent_target(i, R[i])
                if ut is None: continue
                D.append(float(np.abs(R[i]["u"] - ut).sum()))
                p_ = mismatch(A_t[k], b_t[k], R[i]["u"], ut)
                if p_ is not None and p_.requires_grad: P = P + p_; n += 1
            if n: opt.zero_grad(); (P/n).backward(); opt.step()
        with torch.no_grad(): Av, bv = rows(va)
        Rv = solve(va, Av.numpy().astype(float), bv.numpy().astype(float)); v = score(Rv, va)
        nfb = sum(1 for i in va if Rv.get(i) is not None and Rv[i].get("fallback"))
        if v < best[0]: best = (v, {k: x.clone() for k, x in net.state_dict().items()}, ep)
        hist.append(dict(ep=ep, dist=float(np.mean(D)) if D else np.nan, val=v))
        print(f"[ep {ep}] dist-to-target {np.mean(D) if D else np.nan:.2f}  val {v:.6f} (no-cut {v_nc:.6f})  val fallbacks {nfb}/{len(va)}  {time.time()-t0:.0f}s", flush=True)
    net.load_state_dict(best[1])
    ids = [i for i in te if data2.converged(REF.get(i))]
    with torch.no_grad(): At, bt = rows(ids)
    Rc = solve(ids, At.numpy().astype(float), bt.numpy().astype(float)); Rn = solve(ids)
    Jr = dict(pool.imap_unordered(combopt_ac.price_ref, [(i, REF[i]["u"]) for i in ids]))
    out = []
    for i in ids:
        if Jr.get(i) is None or Rc.get(i) is None or Rn.get(i) is None: continue
        j, pref = Jr[i]; ur = np.asarray(REF[i]["u"])
        f = lambda R: dict(gap=100*abs(R["cost"]-j)/j, disc=100*float((R["u"] != ur).mean()),
                           cont=100*float(np.abs(R["pg"]-pref).sum()/pref.sum()), secs=R["secs"])
        out.append(dict(idx=i, cut=dict(f(Rc[i]), fb=bool(Rc[i].get("fallback"))), nocut=dict(f(Rn[i]), fb=bool(Rn[i].get("fallback")))))
    M = lambda k, m: np.mean([o[k][m] for o in out]); Md = lambda k: np.median([o[k]["secs"] for o in out])
    c = np.array([o["cut"]["gap"] for o in out]); n_ = np.array([o["nocut"]["gap"] for o in out])
    print(f"[TEST] {a.case} {a.mode} W={a.work} n={len(out)}: CUT gap {M('cut','gap'):.4f}% disc {M('cut','disc'):.2f}% cont {M('cut','cont'):.2f}% t {Md('cut'):.1f}s | "
          f"NO-CUT gap {M('nocut','gap'):.4f}% disc {M('nocut','disc'):.2f}% cont {M('nocut','cont'):.2f}% t {Md('nocut'):.1f}s | better {int((c<n_-1e-9).sum())} worse {int((c>n_+1e-9).sum())} | fallbacks cut {sum(o['cut']['fb'] for o in out)} no-cut {sum(o['nocut']['fb'] for o in out)} | best ep {best[2]}", flush=True)
    os.makedirs(os.path.join(HERE, "results", "phase1"), exist_ok=True)
    json.dump(dict(rows=out, hist=hist, best_epoch=best[2], args=vars(a)), open(os.path.join(HERE, "results", "phase1", f"{a.case}_{a.mode}_w{int(a.work)}_s{a.seed}{a.tag}.json"), "w"), indent=1)
    torch.save(dict(net=best[1], args=vars(a)), os.path.join(HERE, "results", "phase1", f"{a.case}_{a.mode}_w{int(a.work)}_s{a.seed}{a.tag}.pt"))
    pool.close()


if __name__ == "__main__":
    main()

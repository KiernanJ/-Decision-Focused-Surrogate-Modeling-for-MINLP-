# %% CombOptNet-style learned cuts for AC-UC (post-audit), one seed per process.
#    cut network -> integer-only rows A u <= b (anchored SLACK at the relaxed u_hat)
#    -> restoration MILP with the rows -> AC pricing -> deployed cost (self-supervised).
#    Gradient: CombOptNet mismatch on the rows, driven by feasible single-flip descent.
#    Selection: deployed validation cost (label-free). Test: learned cut vs no cut (same
#    restoration without rows), against converged references priced by the same evaluator.
#    --check runs the descent test only.   Workers: all cores (multiprocessing, spawn).
import os, sys, json, time, argparse, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
import multiprocessing as mp

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="case300"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--batch", type=int, default=24); ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--margin", type=float, default=1.0); ap.add_argument("--kcand", type=int, default=12)
ap.add_argument("--kmoves", type=int, default=4); ap.add_argument("--workers", type=int, default=11)
ap.add_argument("--check", action="store_true"); ap.add_argument("--tag", default="")
# u_hat from the relaxation at the LEARNED V (exact-layer trained CutNetK in checkpoints2/)
ap.add_argument("--learned-v", action="store_true")
# deployment = cut projection + exact-layer-guided search (validation selection and test)
ap.add_argument("--search-k", type=int, default=15); ap.add_argument("--search-rounds", type=int, default=40)
a = ap.parse_args()


from combopt_net import CutNet


def main():
    import combopt_ac, data2
    from pipeline import grid
    os.environ["QCAC_CASE"] = a.case
    g, nl = grid(a.case); G = g.n_gen
    INST = data2.instances(g, a.case); te, va, tr = data2.split()
    REF = data2.refs(a.case, os.path.join(HERE, "results", "pool2"))
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.workers, initializer=combopt_ac.init, initargs=(a.case,))
    base = float(g.pd.sum())
    X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
    UH, SC = {}, {}
    jobs = tr + va + te
    if a.learned_v:
        from cutnet_k import CutNetK
        vnet = CutNetK(g.n_bus, G, k_rows=2, v_scale=0.5, hidden=256, seed=0, learn_thr=False)
        vnet.load_state_dict(torch.load(os.path.join(HERE, "checkpoints2", f"{a.case}_s0.pt"), weights_only=False)["net"]); vnet.eval()
        with torch.no_grad(): Vr, Vi = [t.numpy().astype(float) for t in vnet(Xn[jobs])[:2]]
        jobs = [(i, (Vr[k], Vi[k])) for k, i in enumerate(jobs)]
    for i, r in pool.imap_unordered(combopt_ac.relax, jobs):
        if r is not None: UH[i], SC[i] = r
    if a.learned_v:
        SC = {i: r[1] for i, r in pool.imap_unordered(combopt_ac.relax, list(UH)) if r is not None}
    tr = [i for i in tr if i in UH and i in SC]; va = [i for i in va if i in UH and i in SC]; te = [i for i in te if i in UH and i in SC]
    net = CutNet(X.shape[1], G, a.rows, seed=a.seed)

    def rows(ids):
        A, d = net(Xn[ids])
        uh = torch.tensor(np.stack([UH[i] for i in ids]), dtype=torch.float32)
        return A, torch.einsum("nrg,ng->nr", A, uh) + a.margin - d

    def mismatch(A_t, b_t, res):
        u = res["u"]; order = res["order"]
        if not order: return None
        mags = np.abs(res["desc"][order]); lam = np.r_[mags[:-1] - mags[1:], mags[-1]]
        nrm = torch.linalg.norm(A_t, dim=1) + 1e-9
        dist = lambda pt: torch.abs(A_t @ torch.tensor(pt, dtype=torch.float32) - b_t)/nrm
        P = torch.zeros(())
        for k, (un, feas) in enumerate(res["neigh"]):
            if not feas: continue                                   # infeasible for the problem itself
            v = (A_t @ torch.tensor(un, dtype=torch.float32) - b_t) > 0
            if v.any():
                P = P + float(lam[k])*(dist(un)*v.float()).sum()    # cut off by rows -> push past it
            else:
                P = P - float(lam[k])*0.1*torch.logsumexp(-dist(u)/0.1, 0)   # pull nearest row to u
        return P

    def run(ids, A, b, kc=a.kcand, km=a.kmoves):
        jobs = [(i, UH[i], A[k], b[k], kc, km) for k, i in enumerate(ids)]
        return dict(pool.imap_unordered(combopt_ac.work, jobs))

    def final(ids, A=None, b=None):
        jobs = [(i, UH[i], None if A is None else A[k], None if b is None else b[k], a.search_k, a.search_rounds)
                for k, i in enumerate(ids)]
        return dict(pool.imap_unordered(combopt_ac.final, jobs))

    def val_score(ids):
        with torch.no_grad(): A, b = rows(ids)
        R_ = final(ids, A.numpy().astype(float), b.numpy().astype(float))
        return float(np.mean([10.0 if R_[i] is None else R_[i]["cost"]/SC[i] for i in ids]))

    nocut = final(va)
    v_nc = float(np.mean([10.0 if nocut[i] is None else nocut[i]["cost"]/SC[i] for i in va]))

    if a.check:                                                     # descent test
        ids = va[:16]
        with torch.no_grad(): A, b = rows(ids)
        An, bn = A.numpy().astype(float), b.numpy().astype(float)
        R0 = run(ids, An, bn)
        out = []
        for lr in (0.05, 0.2, 0.5):
            Ps = []
            st = {k: v.clone() for k, v in net.state_dict().items()}
            A_t, b_t = rows(ids); P = 0.0; n = 0
            for k, i in enumerate(ids):
                if R0[i] is None: continue
                p_ = mismatch(A_t[k], b_t[k], R0[i])
                if p_ is not None and p_.requires_grad: P = P + p_; n += 1
            if not n: print("no signal"); return
            net.zero_grad(); (P/n).backward()
            gn = torch.sqrt(sum((p.grad**2).sum() for p in net.parameters() if p.grad is not None)) + 1e-12
            with torch.no_grad():
                for p in net.parameters():
                    if p.grad is not None: p -= lr*p.grad/gn
            with torch.no_grad(): A2, b2 = rows(ids)
            R1 = run(ids, A2.numpy().astype(float), b2.numpy().astype(float), 0, 0)
            dlt = [100*(R1[i]["cost"] - R0[i]["cost"])/SC[i] for i in ids if R0[i] and R1[i]]
            net.load_state_dict(st)
            print(f"step {lr}: deployed cost lowered on {np.mean(np.array(dlt) < -1e-6):.0%}, raised on "
                  f"{np.mean(np.array(dlt) > 1e-6):.0%}, mean {np.mean(dlt):+.4f} pp  (n={len(dlt)}; signal on {n})", flush=True)
        print(f"init cut vs no-cut on these: {np.mean([R0[i]['cost']/SC[i] for i in ids if R0[i]]):.6f} vs "
              f"{np.mean([nocut[i]['cost']/SC[i] for i in ids if nocut[i]]):.6f}")
        pool.close(); return

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    v0 = val_score(va); best = (v0, {k: v.clone() for k, v in net.state_dict().items()}, -1); hist = []
    print(f"[{a.case} s{a.seed}] init val {v0:.6f}  no-cut val {v_nc:.6f}", flush=True)
    rng = np.random.default_rng(a.seed); t0 = time.time()
    for ep in range(a.epochs):
        order = rng.permutation(tr); Ls = []
        for s0 in range(0, len(order), a.batch):
            ids = [int(i) for i in order[s0:s0+a.batch]]
            A_t, b_t = rows(ids)
            R_ = run(ids, A_t.detach().numpy().astype(float), b_t.detach().numpy().astype(float))
            P, n = 0.0, 0
            for k, i in enumerate(ids):
                if R_[i] is None: continue
                Ls.append(R_[i]["cost"]/SC[i])
                p_ = mismatch(A_t[k], b_t[k], R_[i])
                if p_ is not None and p_.requires_grad: P = P + p_; n += 1
            if n:
                opt.zero_grad(); (P/n).backward(); opt.step()
        v = val_score(va)
        if v < best[0]: best = (v, {k: x.clone() for k, x in net.state_dict().items()}, ep)
        hist.append(dict(ep=ep, train=float(np.mean(Ls)), val=v))
        print(f"[ep {ep:2d}] train {np.mean(Ls):.6f}  val {v:.6f} (no-cut {v_nc:.6f})  {time.time()-t0:.0f}s", flush=True)
    net.load_state_dict(best[1])
    # test vs no-cut, converged references, J_ref = price(u_ref)
    with torch.no_grad(): A, b = rows(te)
    Rc = final(te, A.numpy().astype(float), b.numpy().astype(float))
    Rn = final(te)
    conv = [i for i in te if data2.converged(REF.get(i))]
    Jr = dict(pool.imap_unordered(combopt_ac.price_ref, [(i, REF[i]["u"]) for i in conv]))
    rows_ = []
    for i in conv:
        if Jr[i] is None or Rc[i] is None or Rn[i] is None: continue
        jr = Jr[i][0]
        rows_.append(dict(idx=i, gap=100*(Rc[i]["cost"]-jr)/jr, gap_nocut=100*(Rn[i]["cost"]-jr)/jr,
                          disc=100*float((Rc[i]["u"] != np.asarray(REF[i]["u"])).mean()),
                          disc_nocut=100*float((Rn[i]["u"] != np.asarray(REF[i]["u"])).mean())))
    Gm = np.array([[abs(r["gap"]), abs(r["gap_nocut"])] for r in rows_])
    print(f"[TEST] {a.case} s{a.seed} n={len(rows_)}: learned cut |gap| {Gm[:,0].mean():.4f}% vs no-cut {Gm[:,1].mean():.4f}% | "
          f"better {int((Gm[:,0] < Gm[:,1]-1e-9).sum())}, worse {int((Gm[:,0] > Gm[:,1]+1e-9).sum())} | best epoch {best[2]}", flush=True)
    os.makedirs(os.path.join(HERE, "results", "combopt"), exist_ok=True)
    json.dump(dict(rows=rows_, hist=hist, best_epoch=best[2], args=vars(a)),
              open(os.path.join(HERE, "results", "combopt", f"{a.case}_s{a.seed}{a.tag}.json"), "w"), indent=1)
    torch.save(dict(net=best[1], args=vars(a)), os.path.join(HERE, "results", "combopt", f"{a.case}_s{a.seed}{a.tag}.pt"))
    pool.close()


if __name__ == "__main__":
    main()

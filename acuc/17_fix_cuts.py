# %% Learned FIXING cuts (u_k >= 1 / u_k <= 0) for the time-limited MIQCQP at the learned V.
#    Self-supervised: targets = OUR OWN long solves (W_long) on training instances (no reference
#    solver, no labels). A small MLP predicts P(u_k = 1 | load); units with confidence beyond a
#    threshold are fixed; the threshold is chosen on VALIDATION deployed cost (label-free).
#    Test: fixing cuts at W vs no cut at W (and no cut at W_long for reference).
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
ap = argparse.ArgumentParser(); ap.add_argument("--case", default="case118"); ap.add_argument("--work", type=float, default=10.0)
ap.add_argument("--work-long", type=float, default=120.0); ap.add_argument("--rho", type=float, default=1e4)
ap.add_argument("--workers", type=int, default=6)
a = ap.parse_args()


def main():
    import combopt_ac, data2
    from pipeline import grid
    from cutnet_k import CutNetK
    g, nl = grid(a.case); G = g.n_gen; INST = data2.instances(g, a.case); te, va, tr = data2.split()
    REF = data2.refs(a.case, os.path.join(HERE, "results", "pool2"))
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
    vnet = CutNetK(g.n_bus, G, k_rows=2, v_scale=0.5, hidden=256, seed=0, learn_thr=False)
    vnet.load_state_dict(torch.load(os.path.join(HERE, "checkpoints2", f"{a.case}_s0.pt"), weights_only=False)["net"]); vnet.eval()
    with torch.no_grad(): VR, VI = [t.numpy().astype(float) for t in vnet(Xn)[:2]]
    V = lambda i: (VR[i], VI[i])
    pool = mp.get_context("spawn").Pool(a.workers, initializer=combopt_ac.init, initargs=(a.case,))
    SC = {i: r[1] for i, r in pool.imap_unordered(combopt_ac.relax, tr + va + te) if r is not None}
    cache = os.path.join(HERE, "results", f"teacher_{a.case}_w{int(a.work_long)}.json")
    if os.path.exists(cache):
        TG = {int(k): np.array(v) for k, v in json.load(open(cache)).items()}
    else:
        t0 = time.time()
        R = dict(pool.imap_unordered(combopt_ac.miqcqp, [(i, V(i), None, None, a.work_long, a.rho) for i in tr]))
        TG = {i: R[i]["u"] for i in tr if R.get(i) is not None and not R[i].get("fallback")}
        json.dump({str(k): v.tolist() for k, v in TG.items()}, open(cache, "w"))
        print(f"[teacher] {len(TG)} self-found targets in {time.time()-t0:.0f}s", flush=True)
    ids = [i for i in tr if i in TG]
    Y = torch.tensor(np.stack([TG[i] for i in ids]), dtype=torch.float32)
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
    o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(1500):
        o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(m(Xn[ids]), Y); l.backward(); o.step()
    m.eval()
    with torch.no_grad(): Pr = torch.sigmoid(m(Xn)).numpy()

    def fixes(i, c):
        return [k for k in range(G) if Pr[i, k] >= c], [k for k in range(G) if Pr[i, k] <= 1 - c]

    def run(ids_, c, work):
        jobs = [(i, V(i), *(fixes(i, c) if c is not None else ([], [])), work, a.rho) for i in ids_]
        return dict(pool.imap_unordered(combopt_ac.miqcqp_fix, jobs))

    sc = lambda R, ids_: float(np.mean([10.0 if R.get(i) is None else R[i]["cost"]/SC[i] for i in ids_]))
    table = {}
    for c in (None, 0.999, 0.99, 0.95, 0.9):
        R = run(va, c, a.work); table[str(c)] = sc(R, va)
        nf = np.mean([sum(map(len, fixes(i, c))) for i in va]) if c else 0
        print(f"[val] confidence {c}: fixed {nf:.1f}/{G} units, score {table[str(c)]:.6f}, fallbacks {sum(1 for i in va if R.get(i) and R[i].get('fallback'))}", flush=True)
    cbest = min((k for k in table if k != "None"), key=table.get); c = float(cbest)
    print(f"[select] confidence {c} (label-free validation)", flush=True)
    tids = [i for i in te if data2.converged(REF.get(i))]
    Rf = run(tids, c, a.work); Rn = run(tids, None, a.work); RL = run(tids, None, 60.0)
    # label-free safeguard: deployed cost vs the uncut relaxation (flat V). Instances whose ratio
    # exceeds the largest VALIDATION ratio (+2%) are re-solved WITHOUT fixes; keep the cheaper.
    Rv = run(va, c, a.work)
    tau = 1.02*max(Rv[i]["cost"]/SC[i] for i in va if Rv.get(i) is not None)
    flag = [i for i in tids if Rf.get(i) is not None and Rf[i]["cost"]/SC[i] > tau]
    Rg = dict(Rf)
    if flag:
        Rre = run(flag, None, a.work)
        for i in flag:
            if Rre.get(i) is not None and Rre[i]["cost"] < Rf[i]["cost"]:
                Rg[i] = dict(Rre[i]); Rg[i]["secs"] = Rf[i]["secs"] + Rre[i]["secs"]
            else:
                Rg[i] = dict(Rf[i]); Rg[i]["secs"] = Rf[i]["secs"] + (Rre[i]["secs"] if Rre.get(i) else 0)
    print(f"[guard] threshold {tau:.4f} (validation, label-free); flagged {len(flag)}/{len(tids)} test instances", flush=True)
    Jr = dict(pool.imap_unordered(combopt_ac.price_ref, [(i, REF[i]["u"]) for i in tids]))
    allrows = {}
    for name, R in (("fix-cuts W", Rf), ("fix+guard W", Rg), ("no-cut W", Rn), ("no-cut W=60", RL)):
        rows = []
        for i in tids:
            if Jr.get(i) is None or R.get(i) is None: continue
            j, pref = Jr[i]; ur = np.asarray(REF[i]["u"])
            rows.append((100*abs(R[i]["cost"]-j)/j, 100*float((R[i]["u"] != ur).mean()),
                         100*float(np.abs(R[i]["pg"]-pref).sum()/pref.sum()), R[i]["secs"], bool(R[i].get("fallback"))))
        v = np.array(rows); allrows[name] = rows
        print(f"[TEST {a.case}] {name:11s} (W={a.work if 'W=' not in name else 60}) n={len(v)} gap {v[:,0].mean():.4f}% median {np.median(v[:,0]):.4f}% | "
              f"disc {v[:,1].mean():.2f}% | cont {v[:,2].mean():.2f}% | time median {np.median(v[:,3]):.1f}s | fallbacks {int(v[:,4].sum())}", flush=True)
    json.dump(dict(conf=c, val=table, rows={k: [list(map(float, r)) for r in v] for k, v in allrows.items()}, ids=tids),
              open(os.path.join(HERE, "results", f"fixcuts_{a.case}.json"), "w"), indent=1)
    pool.close()


if __name__ == "__main__":
    main()

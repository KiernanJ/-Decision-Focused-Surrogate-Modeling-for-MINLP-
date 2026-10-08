# %% Phase 0: deployment by ONE time-limited MIQCQP at the learned V (vs flat V), no cut.
#    Gap vs converged references (J_ref = price(u_ref)), discrete / continuous error, time.
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, multiprocessing as mp
ap = argparse.ArgumentParser(); ap.add_argument("--case", default="case118"); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--rho", type=float, default=1e4)
a = ap.parse_args()


def main():
    import combopt_ac, data2
    from pipeline import grid
    from cutnet_k import CutNetK
    g, nl = grid(a.case); INST = data2.instances(g, a.case); te, va, tr = data2.split()
    REF = data2.refs(a.case, os.path.join(HERE, "results", "pool2"))
    ids = [i for i in te if data2.converged(REF.get(i))]
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
    net = CutNetK(g.n_bus, g.n_gen, k_rows=2, v_scale=0.5, hidden=256, seed=0, learn_thr=False)
    net.load_state_dict(torch.load(os.path.join(HERE, "checkpoints2", f"{a.case}_s0.pt"), weights_only=False)["net"]); net.eval()
    with torch.no_grad(): Vr, Vi = [t.numpy().astype(float) for t in net(Xn[ids])[:2]]
    pool = mp.get_context("spawn").Pool(a.workers, initializer=combopt_ac.init, initargs=(a.case,))
    Jr = dict(pool.imap_unordered(combopt_ac.price_ref, [(i, REF[i]["u"]) for i in ids]))
    out = {}
    for name, work in (("learnedV", 10), ("learnedV", 30), ("learnedV", 60), ("flatV", 30)):
        jobs = [(i, (Vr[k], Vi[k]) if name == "learnedV" else None, None, None, work, a.rho) for k, i in enumerate(ids)]
        R = dict(pool.imap_unordered(combopt_ac.miqcqp, jobs))
        rows = []
        for i in ids:
            if R.get(i) is None or Jr[i] is None: rows.append(None); continue
            j, pref = Jr[i]; ur = np.asarray(REF[i]["u"])
            rows.append(dict(gap=100*abs(R[i]["cost"]-j)/j, disc=100*float((R[i]["u"] != ur).mean()),
                             cont=100*float(np.abs(R[i]["pg"]-pref).sum()/pref.sum()), secs=R[i]["secs"]))
        ok = [r for r in rows if r]
        key = f"{name}_w{work}"; out[key] = ok
        print(f"[{a.case}] {key:13s} n={len(ok)}/{len(ids)} | gap {np.mean([r['gap'] for r in ok]):.4f}% (median {np.median([r['gap'] for r in ok]):.4f}%) | "
              f"disc {np.mean([r['disc'] for r in ok]):.2f}% | cont {np.mean([r['cont'] for r in ok]):.2f}% | time median {np.median([r['secs'] for r in ok]):.1f}s", flush=True)
    ref_t = [REF[i]["secs"] for i in ids]
    print(f"[{a.case}] reference QCAC iterative: median {np.median(ref_t):.0f}s", flush=True)
    json.dump(out, open(os.path.join(HERE, "results", f"phase0_{a.case}.json"), "w"), indent=1); pool.close()


if __name__ == "__main__":
    main()

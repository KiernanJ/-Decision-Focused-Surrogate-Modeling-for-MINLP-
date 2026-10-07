# %% Diagnostic: no-cut + guided search with u_hat from FLAT V vs the LEARNED V
#    (audit-trained CutNetK, checkpoints2), and search length 8 vs 30 rounds.
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch, multiprocessing as mp
ap = argparse.ArgumentParser(); ap.add_argument("--case", default="case118"); ap.add_argument("--workers", type=int, default=7)
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
    UH = {i: r[0] for i, r in pool.imap_unordered(combopt_ac.relax, [(i, (Vr[k], Vi[k])) for k, i in enumerate(ids)]) if r is not None}
    for mode in ("guided15", "down", "down+guided"):
        R = dict(pool.imap_unordered(combopt_ac.final2, [(i, UH[i], None, None, mode) for i in ids if i in UH]))
        gp_ = [abs(100*(R[i]["cost"]-Jr[i][0])/Jr[i][0]) for i in ids if R.get(i) and Jr[i]]
        print(f"[{a.case}] no cut, learned V, search={mode:12s}: {np.mean(gp_):.4f}% (median {np.median(gp_):.4f}%)", flush=True)
    pool.close()


if __name__ == "__main__":
    main()

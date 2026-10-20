# G: does the exact layer do work in LABEL GENERATION? The self-teacher's local search (combopt_ac.improve) ranks flips/swaps by the
# conic layer's VJP (_rank_flips/_pred). Compare against the same search with RANDOM ranking (same k, rounds, swaps, same start).
import os, sys, time
sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "src"))
import numpy as np

def init(case):
    os.environ["OMP_NUM_THREADS"] = "1"
    import combopt_ac; combopt_ac.init(case, True)

def work(args):
    import combopt_ac as C, selfsup
    i, V, W, rho, K, ROUNDS, SWAPS = args
    out = dict(i=i); t0 = time.time()
    _, R = C.miqcqp((i, V, None, None, W, rho))
    if R is None: return out
    out.update(c0=float(R["cost"]), t_mip=time.time() - t0, fb=bool(R.get("fallback")))
    u0 = np.asarray(R["u"], float)
    cnt = {"n": 0}; orig_price = C._price
    def counting_price(i_, u_):
        cnt["n"] += 1; return orig_price(i_, u_)
    C._price = counting_price
    # (a) layer-ranked (the teacher as committed)
    t1 = time.time(); u_l, c_l, _ = C.improve(i, u0, float(R["cost"]), R["pg"], R["vr"], R["vi"], k=K, rounds=ROUNDS, swaps=SWAPS)
    out.update(c_layer=float(c_l), n_layer=cnt["n"], moves_layer=int((u_l != u0).sum()), t_layer=time.time() - t1)
    # (b) random ranking, same budget structure
    orig_rank, orig_pred = C._rank_flips, C._pred
    rng = np.random.default_rng(1000 + i)
    def rand_rank(i_, u, pg, vr, vi):
        g = C._W["g"]; p = C._W["inst"][i_][0]; need = C._W["need_f"]*float(p.sum()); cap = float(g.pmax @ u)
        return [k for k in rng.permutation(len(u)) if u[k] < 0.5 or cap - g.pmax[k] >= need]
    def rand_pred(i_, u, pg, vr, vi):
        return rng.standard_normal(len(u))
    C._rank_flips, C._pred = rand_rank, rand_pred; cnt["n"] = 0
    t1 = time.time(); u_r, c_r, _ = C.improve(i, u0, float(R["cost"]), R["pg"], R["vr"], R["vi"], k=K, rounds=ROUNDS, swaps=SWAPS)
    out.update(c_rand=float(c_r), n_rand=cnt["n"], moves_rand=int((u_r != u0).sum()), t_rand=time.time() - t1)
    C._rank_flips, C._pred, C._price = orig_rank, orig_pred, orig_price
    # the two restorations (the other best-of candidates), priced strictly like the teacher does
    rc = selfsup.restore_candidates((i, V))[1]["cands"]
    out["restore"] = {nm: (None if (r := C.price_strict((i, u))[1]) is None else float(r[0])) for nm, u in rc.items()}
    for nm, u in (("strict_layer", u_l), ("strict_rand", u_r), ("strict_u0", u0)):
        r = C.price_strict((i, u))[1]; out[nm] = None if r is None else float(r[0])
    return out

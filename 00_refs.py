"""Generate and cache one seed's QCAC-iterative references, once.

Both ablation arms of a seed need the SAME references. Launching the arms
together makes each regenerate all of them -- roughly 100 minutes of duplicated
solver work per seed -- and race on the cache file.

    SEED=1 NINST=40 PROCS=6 python 00_refs.py
"""
# %% setup
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import CASE, grid, sample, set_cuts, _init, _ref

# %% main
if __name__ == "__main__":
    import multiprocessing as mp
    seed = int(os.environ.get("SEED", "1"))
    n = int(os.environ.get("NINST", "40"))
    procs = int(os.environ.get("PROCS", "12"))
    rp = f"{HERE}/results/ref_{CASE}_s{seed}_n{n}.npz"
    if os.path.exists(rp):
        print(f"[ref ] already cached {rp}", flush=True); sys.exit(0)
    g, nl = grid()
    inst = sample(g, n, seed=seed)
    set_cuts(8)
    pool = mp.get_context("fork").Pool(procs, initializer=_init)
    t0 = time.time(); REF = [None]*n
    for i, r in pool.imap_unordered(_ref, [(i, *inst[i]) for i in range(n)]):
        REF[i] = r
        print(f"[ref s{seed}] {sum(x is not None for x in REF):2d}/{n}", flush=True)
    np.savez(rp, ref=np.array(REF, dtype=object))
    ok = [i for i in range(n) if REF[i] is not None]
    print(f"[ref s{seed}] {len(ok)}/{n} solved, "
          f"{len({tuple(REF[i]['u']) for i in ok})} distinct, {time.time()-t0:.0f}s",
          flush=True)
    pool.close()

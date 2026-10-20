import os, sys, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src")
os.chdir(R + "/acuc")
import data2
from pipeline import grid
for case in ("case118", "case300", "pglib_opf_case118_ieee__api"):
    os.environ["QCAC_CASE"] = case
    g, nl = grid(case)
    te, va, tr = data2.split()
    RP = data2.refs(case, R + "/acuc/results/pool2")
    conv = {i for i, r in RP.items() if data2.converged(r)}
    # which gen is the ext_grid? ppc order [ext_grid rows, gen rows]
    k0 = 0
    print(f"== {case}: G={g.n_gen}, pmax[0]={g.pmax[0]:.3f} pu (max pmax {g.pmax.max():.3f}), qmin/qmax[0]={g.qmin[0]:.3f}/{g.qmax[0]:.3f}, nl[0]={nl[0]:.1f} median nl {np.median(nl):.1f}, always_on any {np.any(g.always_on)}")
    print(f"   pool2 refs: {len(RP)} solved entries (None: {sum(r is None for r in RP.values())}), converged {len(conv)}")
    for nm, ids in (("test", te), ("val", va), ("train", tr)):
        sol = [i for i in ids if i in RP]; cv = [i for i in ids if i in conv]
        print(f"   {nm}: {len(ids)} draws, {len(sol)} with pool2 entry, {len(cv)} converged")
    ltr = [i for i in tr if i in conv]; lva = [i for i in va if i in conv]
    print(f"   proxy labels (55: ltr/lva): train {len(ltr)}, val {len(lva)}" + ("  -> no val labels: 20% hold-out of train" if not lva else ""))
    U = np.array([RP[i]["u"] for i in sorted(conv)])
    if len(U):
        print(f"   ext_grid unit (gen 0) committed in {int(U[:,0].sum())}/{len(U)} converged refs; mean units on {U.sum(1).mean():.1f}")
    # extra keys
    any_r = next(r for r in RP.values() if r is not None)
    print(f"   pool2 entry keys: {sorted(any_r.keys())}")
    st = [r.get("status") for r in RP.values() if r is not None]
    from collections import Counter; print(f"   statuses: {Counter(st)}")
    # train pool entries outside the split?
    extra = [i for i in RP if i not in set(te+va+tr)]
    print(f"   pool2 indices outside te/va/tr: {len(extra)} {sorted(extra)[:10]}")

import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src"))
from pipeline import grid
for case in ("case118", "case300", "pglib_opf_case118_ieee__api"):
    g, nl = grid(case)
    z = g.pmax <= 1e-9; zc = (g.c1 == 0) & (g.c2 == 0)
    print(f"{case}: G={g.n_gen}, pmax=0 units {int(z.sum())}, zero-cost units {int(zc.sum())}, nl<1e-3 units {int((nl<1e-3).sum())}; "
          f"nl quantiles {np.round(np.quantile(nl,[0,.25,.5,.75,1]),2).tolist()}; pmax quantiles {np.round(np.quantile(g.pmax,[0,.25,.5,.75,1]),2).tolist()}")
    print("   c1 quantiles", np.round(np.quantile(g.c1,[0,.25,.5,.75,1]),1).tolist(), " c2 quantiles", np.round(np.quantile(g.c2,[0,.25,.5,.75,1]),2).tolist())
    if z.any(): print("   pmax=0 units: qmin/qmax", np.c_[g.qmin[z], g.qmax[z]].round(2).tolist()[:12], "...")

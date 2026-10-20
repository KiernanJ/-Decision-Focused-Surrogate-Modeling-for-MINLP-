# F2: why is the strict price above the PYPOWER optimum on PGLib? residual sums, losses, iterations to the physics check, and pricing to a tighter tolerance
import sys, os, pickle, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combopt_ac as C
case = "pglib_opf_case118_ieee__api"; C.init(case, True); g, nl, S = C._W["g"], C._W["nl"], C._W["S"]
from evalac import opf_cost, check_point
from pypower.makeYbus import makeYbus
from evalac import build_ppc
F = pickle.load(open(glob.glob(os.path.join(R, "audit2", "repro", "runs", "*FINAL_pglib*fresh5*", "final.pkl"))[0], "rb")); tid = F["tid"]; SR = F["SR"]
T = pickle.load(open(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t3_pglib*.pkl"))[0], "rb"))
def resid(i, u, vr, vi, pg):
    p, q = C._W["inst"][i]; ppc = build_ppc(g, p, q, u); bi, bri = ppc["bus"].copy(), ppc["branch"].copy(); bi[:, 0] -= 1; bri[:, 0:2] -= 1
    Y, _, _ = makeYbus(100.0, bi, bri); V = vr + 1j*vi; Sinj = V*np.conj(Y @ V)
    Cg = np.zeros((g.n_bus, g.n_gen)); Cg[g.gen_bus, np.arange(g.n_gen)] = 1
    return Cg @ (pg*u) - p - Sinj.real
rows = []
for i in tid[:14]:
    p, q = C._W["inst"][i]
    for nm, u in (("ref", np.asarray(SR[i][5], float)), ("ours", np.asarray(F["rows"][0]["OP"][i]["u"], float))):
        r = S.price(p, q, u, max_iter=200, check=lambda r_: C._phys(i, r_))           # = strict pricing (first passing iterate)
        r2 = S.price(p, q, u, max_iter=400, eps=1e-9, check=lambda r_: (C._phys(i, r_) and max(abs(resid(i, u, r_["vr"], r_["vi"], r_["pg"]))) < 1e-6))   # tighter: residual 1e-6
        res = resid(i, u, r["vr"], r["vi"], r["pg"]); ok, Jo, _ = opf_cost(g, nl, p, q, u)
        J1 = float(r["cost"]); J2 = float(r2["cost"]) if r2 else np.nan
        rows.append((nm, 100*(J1-Jo)/Jo, 100*(J2-Jo)/Jo, res.sum(), r["iters"], r2["iters"] if r2 else -1))
        print(f"{i} {nm}: strict iters {r['iters']}, sum P residual {res.sum():+.2e} pu (max |.| {np.abs(res).max():.1e}); strict - OPF {100*(J1-Jo)/Jo:+.4f}%; tight-tolerance price - OPF {100*(J2-Jo)/Jo:+.4f}% (iters {r2['iters'] if r2 else -1})", flush=True)
for nm in ("ref", "ours"):
    a = np.array([x[1:] for x in rows if x[0] == nm], float)
    print(f"[{nm}] strict-OPF mean {a[:,0].mean():+.4f}%; tight-tol price-OPF mean {np.nanmean(a[:,1]):+.4f}%; sum P residual mean {a[:,2].mean():+.2e}")

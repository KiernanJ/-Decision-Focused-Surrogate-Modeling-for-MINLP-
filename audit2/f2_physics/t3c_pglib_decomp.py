import sys, os, pickle, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data2
from pipeline import grid
from evalac import opf_cost
case = "pglib_opf_case118_ieee__api"; g, nl = grid(case); INST = data2.all_instances(g, case)
F = pickle.load(open(glob.glob(os.path.join(R, "audit2", "repro", "runs", "*FINAL_pglib*fresh5*", "final.pkl"))[0], "rb")); tid = F["tid"]; SR = F["SR"]
real = g.pmax > 0
U = {"ref": np.array([SR[i][5] for i in tid]), "ours": np.array([F["rows"][0]["OP"][i]["u"] for i in tid]), "proxy": np.array([F["rows"][0]["PX"][i]["u"] for i in tid])}
for nm, u in U.items():
    print(f"{nm}: real units on {u[:, real].sum(1).mean():.2f}/{real.sum()}, condensers on {u[:, ~real].sum(1).mean():.2f}/{(~real).sum()}, no-load cost share {np.mean([nl @ u[k] for k in range(len(tid))]):.0f}")
dr = (U["ours"][:, real] != U["ref"][:, real]).sum(1); dc = (U["ours"][:, ~real] != U["ref"][:, ~real]).sum(1)
print(f"ours vs ref: differing real units per instance mean {dr.mean():.2f} (max {dr.max()}), differing condensers mean {dc.mean():.2f} (max {dc.max()})")
print(f"discrete-error share from condensers (ours vs ref): {dc.sum()/(dc.sum()+dr.sum()):.2f}; proxy vs ref condenser share: "
      f"{(U['proxy'][:, ~real] != U['ref'][:, ~real]).sum()/max(1,(U['proxy'] != U['ref']).sum()):.2f}")
# decomposition under PYPOWER for a sample: no-load vs dispatch difference ours - ref
nlo, dis = [], []
for k, i in enumerate(tid[:20]):
    p, q = INST[i]
    out = {}
    for nm in ("ref", "ours"):
        ok, Jo, r = opf_cost(g, nl, p, q, U[nm][k].astype(float)); out[nm] = (nl @ U[nm][k], Jo - nl @ U[nm][k])
    nlo.append(100*(out["ours"][0]-out["ref"][0])/(out["ref"][0]+out["ref"][1])); dis.append(100*(out["ours"][1]-out["ref"][1])/(out["ref"][0]+out["ref"][1]))
print(f"PYPOWER decomposition (20 inst) ours - ref as % of ref total: no-load {np.mean(nlo):+.3f}%, dispatch {np.mean(dis):+.3f}%")

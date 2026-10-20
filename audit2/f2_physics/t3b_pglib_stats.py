# F2: PGLib fresh5 -- paired stats under strict vs PYPOWER-OPF evaluators; PYPOWER solution feasibility; REF QCAC cost vs prices
import sys, os, pickle, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import wilcoxon, binomtest
import data2
from pipeline import grid
from evalac import opf_cost, check_point
case = "pglib_opf_case118_ieee__api"; g, nl = grid(case); INST = data2.all_instances(g, case)
fin = glob.glob(os.path.join(R, "audit2", "repro", "runs", "*FINAL_pglib*fresh5*", "final.pkl"))[0]
F = pickle.load(open(fin, "rb")); tid = F["tid"]; SR = F["SR"]
T = pickle.load(open(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t3_pglib*.pkl"))[0], "rb"))
REF = pickle.load(open(os.path.join(R, "acuc/results/fresh5_refs_pglib_opf_case118_ieee__api.pkl"), "rb"))
d = np.array([100*(REF[i]["cost"] - SR[i][0])/SR[i][0] for i in tid]); do = np.array([100*(REF[i]["cost"] - T[i]["ref"][2])/T[i]["ref"][2] for i in tid])
print(f"REF QCAC cost minus strict price of own u: mean {d.mean():+.3f}% [min {d.min():+.3f}, max {d.max():+.3f}]; minus PYPOWER-OPF of own u: mean {do.mean():+.3f}% [{do.min():+.3f}, {do.max():+.3f}]")
ch = [i for i in tid if REF[i].get("changed")]; print(f"   polish changed u on {len(ch)}/{len(tid)}; QCAC-minus-strict on changed {np.mean([d[tid.index(i)] for i in ch]):+.3f}% vs unchanged {np.mean([d[k] for k,i in enumerate(tid) if i not in ch]):+.3f}%")
for ev, col in (("strict", 0), ("PYPOWER-OPF", 2)):
    J = {nm: np.array([T[i][nm][col] for i in tid]) for nm in ("ref", "ours", "proxy")}
    so, sp = 100*(J["ours"]-J["ref"])/J["ref"], 100*(J["proxy"]-J["ref"])/J["ref"]
    diff = so - sp; dt = np.where(np.abs(diff) <= 1e-3, 0, diff)
    ga, gp = np.abs(so), np.abs(sp); dg = ga - gp; dgt = np.where(np.abs(dg) <= 1e-3, 0, dg)
    print(f"[{ev}] ours vs ref signed mean {so.mean():+.4f}% (cheaper than ref on {(so < -1e-3).sum()}/{len(so)}, worse on {(so > 1e-3).sum()}); proxy vs ref {sp.mean():+.4f}% (cheaper {(sp < -1e-3).sum()}, worse {(sp > 1e-3).sum()})")
    print(f"     ours-proxy SIGNED {diff.mean():+.4f} pp, Wilcoxon p {wilcoxon(dt[dt!=0]).pvalue:.2g}, better/worse {(dt<0).sum()}/{(dt>0).sum()}; |gap| ours {ga.mean():.4f}% vs proxy {gp.mean():.4f}% -> diff {dg.mean():+.4f} pp, Wilcoxon p {wilcoxon(dgt[dgt!=0]).pvalue:.2g}, better/worse {(dgt<0).sum()}/{(dgt>0).sum()}")
# PYPOWER solutions feasible in the project's model?  check a sample
worst = {}
for i in tid[:12]:
    p, q = INST[i]; u = np.asarray(F["rows"][0]["OP"][i]["u"], float)
    ok, Jo, r = opf_cost(g, nl, p, q, u)
    Vm, Va = r["bus"][:, 7], np.deg2rad(r["bus"][:, 8]); V = Vm*np.exp(1j*Va)
    pg = r["gen"][:, 1]/100.0
    c = check_point(g, p, q, u, V.real, V.imag, pg)
    for k, v in c.items(): worst[k] = max(worst.get(k, 0), v)
    # recompute cost of the PYPOWER point
    Jc = float(g.c2 @ (pg*u)**2 + g.c1 @ (pg*u) + nl @ u)
    worst["cost_recompute_rel"] = max(worst.get("cost_recompute_rel", 0), abs(Jc - Jo)/Jo)
print("PYPOWER-OPF solutions (ours, 12 inst) checked in the project's model:", {k: f"{v:.2e}" for k, v in worst.items()})

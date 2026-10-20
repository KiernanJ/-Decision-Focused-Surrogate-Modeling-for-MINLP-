# Independent re-tabulation of the ablation per-instance results (f3_ablations/res_*.pkl) + cross-check against the
# re-run of 55_final_eval.py (repro/runs/*/final.pkl).  Usage: python audit2/f3_ablations/verify_ablations.py
import os, pickle, glob, numpy as np, sys
from scipy.stats import wilcoxon, binomtest
S = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # = audit2/
def load(case, s): return pickle.load(open(f"{S}/f3_ablations/res_{case}_{s}.pkl", "rb"))
def gaps(st, arm):
    SR = st["meta"]["SR"]; out = {}
    for i in st["meta"]["scored"]:
        r = st["arms"].get(arm, {}).get(i)
        out[i] = np.nan if r is None else 100*abs(r["cost"] - SR[i][0])/SR[i][0]
    return out
def signed(st, arm):
    SR = st["meta"]["SR"]; return {i: (np.nan if st["arms"].get(arm, {}).get(i) is None else 100*(st["arms"][arm][i]["cost"] - SR[i][0])/SR[i][0]) for i in st["meta"]["scored"]}
def cmp(xs, ys, la, lb):
    x = np.array(xs); y = np.array(ys); ok = ~np.isnan(x) & ~np.isnan(y); d = x[ok] - y[ok]; dt = np.where(np.abs(d) <= 1e-3, 0, d)
    b, w, t = int((dt < 0).sum()), int((dt > 0).sum()), int((dt == 0).sum())
    pw = wilcoxon(dt, zero_method="wilcox").pvalue if (dt != 0).any() else np.nan
    ps = binomtest(b, b + w).pvalue if b + w else np.nan
    rng = np.random.default_rng(1); bs = [rng.choice(d, d.size).mean() for _ in range(5000)]
    print(f"   {la:>5s} vs {lb:<5s}: diff mean {d.mean():+.3f} pp [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  {la} better/tie/worse {b}/{t}/{w}  Wilcoxon p={pw:.3g} sign p={ps:.3g}  n={ok.sum()}")
# 1) cross-check A0 / A8 with the 55 re-run final.pkl (seed 0)
for case, draw in (("case118", "fresh5"), ("case118", "fresh6"), ("case300", "fresh5")):
    st = load(case, draw); f = glob.glob(f"{S}/repro/runs/*FINAL_{case}_{draw}*/final.pkl")[0]; F = pickle.load(open(f, "rb"))
    tid = F["tid"]; SR = F["SR"]; ps = F["per_seed"][0]
    g0 = gaps(st, "A0"); g8 = gaps(st, "A8")
    a0 = np.array([g0[i] for i in tid]); r0 = ps["OUR METHOD"][:, 0]; a8 = np.array([g8[i] for i in tid]); r8 = ps["supervised NN proxy"][:, 0]
    print(f"[{case} {draw}] scored sets equal: {sorted(tid) == sorted(st['meta']['scored'])}; A0 vs 55-rerun ours max|diff| {np.nanmax(np.abs(a0 - r0)):.2e}; A8 vs 55-rerun proxy(seed0) max|diff| {np.nanmax(np.abs(a8 - r8)):.2e}")
# 2) tables
def table(case, draws, arms):
    sts = [load(case, d) for d in draws]
    print(f"\n=== {case} {'+'.join(draws)}  (|gap| % vs J_ref = strict price of u_ref)")
    print("   arm   |  n | mean   | median | >2% | signed mean | fallbacks")
    for a in arms:
        if not all(a in st["arms"] for st in sts): print(f"   {a:5s} | (missing)"); continue
        g = np.array([v for st in sts for v in gaps(st, a).values()]); sg = np.array([v for st in sts for v in signed(st, a).values()])
        fb = sum(bool(st["arms"][a][i].get("fallback")) for st in sts for i in st["meta"]["scored"] if st["arms"][a].get(i))
        print(f"   {a:5s} | {np.sum(~np.isnan(g)):2d} | {np.nanmean(g):.3f}  | {np.nanmedian(g):.3f}  | {int(np.nansum(g > 2)):2d}  | {np.nanmean(sg):+.3f}      | {fb}")
    return sts
def pairs(sts, prs):
    for a, b in prs:
        if not all(a in st["arms"] and b in st["arms"] for st in sts): print(f"   {a} vs {b}: missing"); continue
        xa = [v for st in sts for v in gaps(st, a).values()]; xb = [v for st in sts for v in gaps(st, b).values()]
        cmp(xa, xb, a, b)
arms_all = ["A0", "A4", "A3a", "A3b", "A5", "A6", "A6h", "A6b", "A7", "A7h", "A8", "A10", "A11", "A12", "A12i", "A2", "A9v", "A1", "A9f"]
sts = table("case118", ["fresh5", "fresh6"], arms_all)
pairs(sts, [("A0", "A8"), ("A12", "A4"), ("A12", "A8"), ("A3a", "A4"), ("A3a", "A0"), ("A5", "A4"), ("A0", "A4"), ("A6", "A5"), ("A6b", "A6"), ("A5", "A6b"), ("A7h", "A8"), ("A11", "A0"), ("A2", "A0")])
sts = table("case300", ["fresh5"], arms_all)
pairs(sts, [("A0", "A8"), ("A6", "A4"), ("A12", "A4"), ("A12", "A8"), ("A2", "A0"), ("A2", "A1"), ("A3a", "A4")])
sts = table("case118", ["val"], arms_all)

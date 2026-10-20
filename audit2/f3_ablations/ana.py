# F3 analysis: tables + paired tests vs A0 from res_<case>_<set>.pkl. Usage: python ana.py <case> <set>[+<set>...] [arms]
import os, sys, pickle
import numpy as np
from scipy.stats import wilcoxon, binomtest
OUTD = os.path.dirname(os.path.abspath(__file__))
CASE = sys.argv[1]; SETS = sys.argv[2].split("+")
NAMES = dict(A0="A0 ours as deployed (V_nc+fix+rows)", A1="A1 flat V, no cuts, W", A2="A2 V-net, no cuts, W", A3a="A3a flat V + fixings (rows off)",
             A3b="A3b flat V + fixings + rows (re-anchored)", A4="A4 V-net + fixings, rows off", A5="A5 V-net + IMITATION fixings (warm start)",
             A6="A6 V-net + SUPERVISED fixings (ref labels)", A6b="A6b V-net + self-label fixings, proxy's 118 inst", A7="A7 V-net + PROXY probs as fixings",
             A8="A8 proxy (round 0.5 + repair)", A9f="A9f flat V, no cuts, 4W", A9v="A9v V-net, no cuts, 4W", A10="A10 proxy + improve()", A11="A11 label generator as method", A0h="A0h = A0 with HARD fixings (rows soft)", A3ah="A3ah flat V + HARD fixings", A4h="A4h V-net + HARD fixings, rows off", A5h="A5h imitation fixings, HARD", A6h="A6h supervised fixings, HARD", A6bh="A6bh self-label 118 inst, HARD", A7h="A7h proxy probs as HARD fixings", A12="A12 OUR cut-net round 0.5 + repair (no MIQCQP)", A12i="A12i imitation net round 0.5 + repair")
TIE = 1e-3

ST = [pickle.load(open(os.path.join(OUTD, f"res_{CASE}_{s}.pkl"), "rb")) for s in SETS]
arms_all = [a for a in NAMES if all(a in st["arms"] for st in ST)]
ARMS = sys.argv[3].split(",") if len(sys.argv) > 3 else arms_all


def metrics(arm):
    """per scored instance across sets: |gap|, disc, cont, signed, secs, work, fallback, mipgap"""
    rows = []
    for st in ST:
        SR = st["meta"]["SR"]
        for i in st["meta"]["scored"]:
            r = st["arms"][arm].get(i)
            if r is None:
                rows.append((np.nan,)*8); continue
            J, pgr, ur = SR[i][0], SR[i][1], np.asarray(SR[i][5])
            rows.append((100*abs(r["cost"] - J)/J, 100*np.mean(np.asarray(r["u"]) != ur), 100*np.abs(np.asarray(r["pg"]) - pgr).sum()/pgr.sum(),
                         100*(r["cost"] - J)/J, r["secs"], r.get("work", np.nan), float(bool(r.get("fallback"))), r.get("mipgap", np.nan)))
    return np.array(rows)


def lfscore(arm):
    out = []
    for st in ST:
        RL = st["meta"]["RLflat"]
        for i in st["meta"]["ids"]:
            r = st["arms"][arm].get(i)
            out.append(r["cost"]/RL[i] if (r is not None and RL.get(i)) else 10.0)
    return float(np.mean(out))


def paired(x, y):
    fa, fb = np.isnan(x), np.isnan(y); ok = ~fa & ~fb
    d = x[ok] - y[ok]; dt = np.where(np.abs(d) <= TIE, 0.0, d)
    better = int((dt < 0).sum() + (fb & ~fa).sum()); worse = int((dt > 0).sum() + (fa & ~fb).sum()); tie = int((dt == 0).sum() + (fa & fb).sum())
    nz = int((dt != 0).sum())
    pw = wilcoxon(dt, zero_method="wilcox").pvalue if nz > 0 else float("nan")
    ps = binomtest(better, better + worse).pvalue if better + worse > 0 else float("nan")
    rng = np.random.default_rng(0); bs = [rng.choice(d, d.size).mean() for _ in range(10000)] if d.size else [np.nan]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), pw, ps, better, tie, worse


print(f"=== {CASE} {'+'.join(SETS)}: n scored = {sum(len(st['meta']['scored']) for st in ST)}" + (f" (label-free score on all {sum(len(st['meta']['ids']) for st in ST)} draws)" if SETS == ['val'] else ""))
print(f"{'arm':52s} | gap mean | gap med | disc  | cont  | signed  | >2% | fb | fail | t med(s)* | work mean | " + ("LF score" if SETS == ['val'] else ""))
M = {a: metrics(a) for a in ARMS}
for a in ARMS:
    v = M[a]
    lf = f"{lfscore(a):.6f}" if SETS == ['val'] else ""
    print(f"{NAMES[a]:52s} | {np.nanmean(v[:,0]):7.4f}% | {np.nanmedian(v[:,0]):6.4f}% | {np.nanmean(v[:,1]):5.2f} | {np.nanmean(v[:,2]):5.2f} | {np.nanmean(v[:,3]):+7.4f}% | {int(np.nansum(v[:,0] > 2)):3d} | {int(np.nansum(v[:,6])):2d} | {int(np.isnan(v[:,0]).sum()):4d} | {np.nanmedian(v[:,4]):8.1f} | {np.nanmean(v[:,5]):9.2f} | {lf}")
print("* wall-clock under machine contention (other audit jobs running): relative only")
if "A0" in M:
    print("\npaired vs A0 (arm - A0; negative = arm better), |gap| and signed gap, tie <= 1e-3 pp; Wilcoxon on thresholded diffs; sign test; bootstrap 95% CI")
    for a in ARMS:
        if a == "A0": continue
        for col, lab in ((0, "|gap|"), (3, "signed")):
            m, lo, hi, pw, ps, b, t, w = paired(M[a][:, col], M["A0"][:, col])
            print(f"  {a:4s} {lab:6s}: {m:+.4f} pp [{lo:+.4f}, {hi:+.4f}]  Wilcoxon p {pw:.3g}  sign p {ps:.3g}  arm better/tie/worse {b}/{t}/{w}")
    if "A8" in M:
        print("\npaired vs A8 proxy (arm - proxy)")
        for a in ARMS:
            if a == "A8": continue
            m, lo, hi, pw, ps, b, t, w = paired(M[a][:, 0], M["A8"][:, 0])
            print(f"  {a:4s} |gap|: {m:+.4f} pp [{lo:+.4f}, {hi:+.4f}]  Wilcoxon p {pw:.3g}  sign p {ps:.3g}  arm better/tie/worse {b}/{t}/{w}")
# A0 fixing diagnostics
print("\nfixing diagnostics on scored instances (units fixed ON / OFF by the arm's fixings; disagreement with u_ref)")
for a in ARMS:
    on_, off_, bad_on, bad_off, nsc = [], [], [], [], 0
    for st in ST:
        SR = st["meta"]["SR"]
        for i in st["meta"]["scored"]:
            r = st["arms"][a].get(i)
            if r is None or "nfix" not in r: continue
            ur = np.asarray(SR[i][5])
            Mkey = {"A0": "Mc", "A3a": "Mc", "A3b": "Mc", "A4": "Mc", "A5": "M_A5", "A6": "M_A6", "A6b": "M_A6b", "A7": "M_proxy", "A0h": "Mc", "A3ah": "Mc", "A4h": "Mc", "A5h": "M_A5", "A6h": "M_A6", "A6bh": "M_A6b", "A7h": "M_proxy"}.get(a)
            if Mkey is None or Mkey not in st["meta"]: continue
            Mi = np.asarray(st["meta"][Mkey][i]); c = st["meta"]["c"]
            fon = Mi > c; foff = Mi < 1 - c
            on_.append(fon.sum()); off_.append(foff.sum()); bad_on.append((fon & (ur < 0.5)).sum()); bad_off.append((foff & (ur > 0.5)).sum()); nsc += 1
    if nsc:
        G = len(ur)
        print(f"  {a:4s}: fixed on {np.mean(on_):.1f}, off {np.mean(off_):.1f}, free {G - np.mean(on_) - np.mean(off_):.1f} of {G}; wrong fixings per inst: on->ref off {np.mean(bad_on):.2f}, off->ref on {np.mean(bad_off):.2f}; inst with >=1 wrong fixing {np.mean(np.array(bad_on) + np.array(bad_off) > 0)*100:.0f}%")
print("\nMIQCQP MIP gap at the end of the W budget (scored instances): median / share > 1e-3 / share > 0.05 / fallbacks")
for a in ARMS:
    v = M[a][:, 7]
    if np.all(np.isnan(v)): continue
    print(f"  {a:4s}: median {np.nanmedian(v):.2e}, >1e-3 {np.nanmean(v > 1e-3)*100:.0f}%, >0.05 {np.nanmean(v > 0.05)*100:.0f}%, work mean {np.nanmean(M[a][:,5]):.2f}")

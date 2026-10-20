import os, sys, pickle, numpy as np
from scipy.stats import wilcoxon, binomtest
D = os.path.dirname(os.path.abspath(__file__))
MILP = [f"milp_v{v}" for v in (0, 1, 2, 4, 6, 7, 8, 9)]
B = {n: pickle.load(open(os.path.join(D, f"bl_{n}.pkl"), "rb")) for n in MILP + ["minlp"]}
ARMS = list(B["milp_v0"]["arms"].keys())
TIE = 1e-3


def per_inst(b, arm, ids, col):
    """seed-averaged metric per instance; NaN if failed on any seed. col: gap|disc|signed|secs"""
    seeds = b["arms"][arm]
    vals = []
    for s, R in seeds.items():
        v = []
        for i in ids:
            r, ref = R.get(i), b["ref"][i]
            if r is None: v.append(np.nan); continue
            if col == "gap": v.append(100*abs(r["cost"] - ref["cost"])/abs(ref["cost"]))
            elif col == "signed": v.append(100*(r["cost"] - ref["cost"])/abs(ref["cost"]))
            elif col == "disc": v.append(100*np.abs(r["z"] - ref["z"]).sum()/max(ref["z"].sum(), 1e-9))
            elif col == "secs": v.append(1000*r["secs"])
            elif col == "cost": v.append(r["cost"])
        vals.append(v)
    A = np.array(vals); return np.where(np.isnan(A).any(0), np.nan, A.mean(0))


def paired(x, y):
    ok = ~np.isnan(x) & ~np.isnan(y); d = x[ok] - y[ok]; dt = np.where(np.abs(d) <= TIE, 0, d)
    bt, wt = int((dt < 0).sum() + (np.isnan(x) == False).__and__(np.isnan(y)).sum()), int((dt > 0).sum() + (np.isnan(x) & ~np.isnan(y)).sum())
    pw = wilcoxon(dt, zero_method="wilcox").pvalue if (dt != 0).sum() else np.nan
    rng = np.random.default_rng(0); bs = [rng.choice(d, d.size).mean() for _ in range(4000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), pw, bt, int((dt == 0).sum()), wt


print("=" * 30, "MILP, 8 vehicles pooled, fresh2 test, converged refs (n = %d)" % sum(len(B[n]["tid"]) for n in MILP))
print(f"{'arm':72s} | gap mean | median | disc  | time med ms | fails | vs OURS: diff pp [CI] p_w b/t/w | vehicles where arm beats OURS / proxy")
rows = {}
for arm in ARMS:
    g = np.concatenate([per_inst(B[n], arm, B[n]["tid"], "gap") for n in MILP]); rows[arm] = g
for arm in ARMS:
    g = rows[arm]; dsc = np.concatenate([per_inst(B[n], arm, B[n]["tid"], "disc") for n in MILP]); tm = np.concatenate([per_inst(B[n], arm, B[n]["tid"], "secs") for n in MILP])
    m, lo, hi, pw, bt, tt, wt = paired(g, rows["OURS (deployed)"])
    vb_o = sum(np.nanmean(per_inst(B[n], arm, B[n]["tid"], "gap")) < np.nanmean(per_inst(B[n], "OURS (deployed)", B[n]["tid"], "gap")) - TIE for n in MILP)
    vb_p = sum(np.nanmean(per_inst(B[n], arm, B[n]["tid"], "gap")) < np.nanmean(per_inst(B[n], "PROXY (deployed)", B[n]["tid"], "gap")) - TIE for n in MILP)
    print(f"{arm:72s} | {np.nanmean(g):.4f}% | {np.nanmedian(g):.4f}% | {np.nanmean(dsc):5.2f}% | {np.nanmedian(tm):7.1f} | {int(np.isnan(g).sum()):3d} | {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={pw:.2g} {bt}/{tt}/{wt} | {vb_o}/8, {vb_p}/8")

print("\nper-vehicle mean |gap| (seed-averaged), converged-ref test instances")
hdr = ["OURS (deployed)", "PROXY (deployed)", "nocut W=0.01", "nocut W=0.02", "nocut W=0.05", "ours: imitation-only integer cuts (no layer, no rows)",
       "ours: layer-trained integer cuts only (rows off)", "SUPERVISED cut-net (reference labels) + same short solve", "ours-imitation on original train profiles only (100/144, label-free)",
       "proxy band [floor,ceil] + W=0.01 solve", "proxy as warm start + W=0.01 solve", "LF teacher procedure (no learning)", "NN lookup of our label-free targets (400/444)", "NN lookup of the proxy's reference labels"]
short = ["ours", "proxy", "nc.01", "nc.02", "nc.05", "imit", "int-only", "sup-cut", "imit100", "pxband", "pxws", "teacher", "NN-LF", "NN-ref"]
print("vehicle | n  | " + " | ".join(f"{s:>8s}" for s in short))
for n in MILP:
    print(f"{n:7s} | {len(B[n]['tid']):3d} | " + " | ".join(f"{np.nanmean(per_inst(B[n], a, B[n]['tid'], 'gap')):8.4f}" for a in hdr))
# vehicle-clustered sign tests of OURS vs key arms
print("\nvehicle-clustered (8 vehicles) per-vehicle mean |gap| difference OURS - arm (pp); sign test")
for a in hdr[1:]:
    vm = np.array([np.nanmean(per_inst(B[n], "OURS (deployed)", B[n]["tid"], "gap") - per_inst(B[n], a, B[n]["tid"], "gap")) for n in MILP])
    vb, vw = int((vm < -TIE).sum()), int((vm > TIE).sum())
    print(f"  OURS vs {a:72s}: {np.round(vm, 3).tolist()} -> ours better {vb}/8, worse {vw}/8, sign p {binomtest(vb, vb+vw).pvalue if vb+vw else np.nan:.3g}")

print("\n" + "=" * 30, "MINLP fresh2 test (n = %d)" % len(B["minlp"]["tid"]))
b = B["minlp"]; ro = per_inst(b, "OURS (deployed)", b["tid"], "gap"); rp = per_inst(b, "PROXY (deployed)", b["tid"], "gap")
for arm in ARMS:
    g = per_inst(b, arm, b["tid"], "gap"); dsc = per_inst(b, arm, b["tid"], "disc"); tm = per_inst(b, arm, b["tid"], "secs")
    m, lo, hi, pw, bt, tt, wt = paired(g, ro); m2, lo2, hi2, pw2, bt2, tt2, wt2 = paired(g, rp)
    print(f"{arm:72s} | {np.nanmean(g):.4f}% | {np.nanmedian(g):.4f}% | {np.nanmean(dsc):5.2f}% | {np.nanmedian(tm):7.1f} ms | fails {int(np.isnan(g).sum())} | vs OURS {m:+.3f} [{lo:+.3f},{hi:+.3f}] p={pw:.2g} {bt}/{tt}/{wt} | vs PROXY {m2:+.3f} p={pw2:.2g}")

print("\n" + "=" * 30, "EXCLUDED MILP test instances (reference MIPGap > 1e-4): raw costs, signed vs the 120 s incumbent")
for arm in ("OURS (deployed)", "PROXY (deployed)", "nocut W=0.01", "LF teacher procedure (no learning)"):
    s = np.concatenate([per_inst(B[n], arm, B[n]["exc"], "signed") for n in MILP]); print(f"  {arm:40s}: n={np.sum(~np.isnan(s))}, signed mean {np.nanmean(s):+.4f}%, median {np.nanmedian(s):+.4f}%, min {np.nanmin(s):+.4f}%")
co = np.concatenate([per_inst(B[n], "OURS (deployed)", B[n]["exc"], "signed") for n in MILP]); cp_ = np.concatenate([per_inst(B[n], "PROXY (deployed)", B[n]["exc"], "signed") for n in MILP])
d = co - cp_; print(f"  ours - proxy on excluded: mean {np.nanmean(d):+.4f} pp, better/tie/worse {(d < -TIE).sum()}/{(np.abs(d) <= TIE).sum()}/{(d > TIE).sum()} (seed-averaged; RESULTS' 126/50 counts instance-seeds)")
print("  excluded per vehicle:", {n: len(B[n]["exc"]) for n in MILP})
for n in MILP:
    b = B[n]; sec_ex = [b["ref"][i].get("secs", np.nan) for i in b["exc"]]
print("  warm-start reproduction checks:", {n: [tuple(round(x, 6) for x in B[n]["info"][f"warm_check_s{s}"]) for s in (0, 1, 2)] for n in ("milp_v1",)})
print("  supervised cut-net chosen c:", {n: [B[n]["info"][f"sup_c_s{s}"] for s in (0, 1, 2)] for n in MILP + ["minlp"]})

"""Summaries of the vehicle diagnostics from out/*.json (no solves):
  1. where the uncut relaxation + qz projection goes wrong (near-tie analysis)
  2. per-epoch training table and epoch-level correlations for each run
Usage: python summarise.py   (from this folder or anywhere)"""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REF, OUT
import numpy as np
from scipy.stats import spearmanr

d = json.load(open(f"{OUT}/diag_init.json"))
rows = d["uncut_val_rows"]
print("=== 1. Uncut relaxation + qz projection on validation: where are the errors? ===")
fr, wrong, net = [], [], []
for r in rows:
    zf = np.array(r["zf"]); zq = np.array(r["zq"]); zr = np.array(REF[r["i"]]["z"], float)
    fr += list(zf - np.floor(zf)); wrong += list(zq != zr); net.append((zq - zr).sum())
fr, wrong, net = np.array(fr), np.array(wrong), np.array(net)
print(f"steps {len(fr)}, wrong {wrong.sum()}")
bins = [0, .05, .2, .35, .45, .55, .65, .8, .95, 1.0001]
h, _ = np.histogram(fr, bins); hw, _ = np.histogram(fr[wrong], bins)
for a, b, n, w in zip(bins[:-1], bins[1:], h, hw):
    print(f"  frac(z_relax) in [{a:.2f},{b:.2f}): {n:4d} steps, {w:3d} wrong")
near = np.abs(fr - 0.5) < 0.15
print(f"near-tie steps (|frac-0.5|<0.15): {near.sum()} ({near.mean():.0%}); projection accuracy on them "
      f"{1 - wrong[near].mean():.1%}; wrong steps outside near-ties: {wrong[~near].sum()}")
print(f"net mode count error sum(zq - zref) per instance: mean {net.mean():+.2f} "
      f"(#neg {(net < 0).sum()}, #pos {(net > 0).sum()}, #zero {(net == 0).sum()})")
print(f"integrality gap (ref - relax)/ref on val: {d['intgap_val']:.3f}%; uncut+round/repair |gap| {d['uncut_roundrepair_val']:.4f}%")

print("\n=== 2. Training runs: per-epoch validation diagnostics ===")
for f in sorted(glob.glob(f"{OUT}/train_*.json")):
    t = json.load(open(f)); L = t["log"]
    print(f"\n-- {os.path.basename(f)}  (w_int={t['args']['w_int']}, seed={t['args']['seed']})")
    print(" ep fails dep_cost |gap|% ham  int_pen nfrac act% val_obj  g_int/g_cost g_cos soft-vs-qz")
    for s in L:
        print(f"{s['ep']:3d} {s['fails']:4d} {s.get('dep_cost', np.nan):8.5f} {s.get('absgap', np.nan):6.3f} "
              f"{s.get('ham', np.nan):4.2f} {s.get('int_pen', np.nan):7.2f} {s.get('nfrac', np.nan):5.2f} "
              f"{100*s.get('rows_active_frac', np.nan):4.0f} {s['val_Ltrain'] or np.nan:8.5f} "
              f"{s.get('g_int', np.nan)/s.get('g_cost', np.nan):8.1f} {s.get('g_cos', np.nan):+6.2f} "
              f"{s.get('soft_vs_qz_l1', np.nan):6.2f}")
    E = [s for s in L if s["ep"] >= 0]
    best = min(E, key=lambda s: s["dep_cost"])
    print(f"spearman over epochs (val training objective, val deployed cost) = "
          f"{spearmanr([s['val_Ltrain'] for s in E], [s['dep_cost'] for s in E])[0]:+.2f}; "
          f"(train objective, val deployed) = {spearmanr([s['train_Ltot'] for s in E], [s['dep_cost'] for s in E])[0]:+.2f}; "
          f"label-free selection -> ep{best['ep']} |gap| {best['absgap']:.4f}%")

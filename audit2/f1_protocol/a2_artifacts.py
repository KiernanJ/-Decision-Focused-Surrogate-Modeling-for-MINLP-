import os, sys, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src")
import data2
RES = R + "/acuc/results"
te, va, tr = data2.split()
held = set(te + va + data2.fresh_ids() + data2.fresh2_ids() + data2.fresh3_ids() + data2.fresh4_ids() + data2.fresh5_ids() + data2.fresh6_ids() + data2.band_ids())
for case in ("case118", "case300", "pglib_opf_case118_ieee__api"):
    print(f"===== {case}")
    V = torch.load(f"{RES}/vnet_layer_{case}_env_nc.pt", weights_only=False)
    print(f" V-net: keys {sorted(V.keys())}, sel={V.get('sel')}, stored rho={V.get('rho')}, state None? {V['state'] is None}")
    UC = torch.load(f"{RES}/unit_cuts_layer_{case}_soft_lfc.pt", weights_only=False)
    print(f" unit cuts: keys {sorted(UC.keys())}, c={UC['c']}, epoch={UC['epoch']}, val={UC['val']:.6f}, v_cur(warm start)={UC['v_cur']:.6f}")
    # how different is selected state from last?
    d = max(float((UC['state'][k] - UC['last'][k]).abs().max()) for k in UC['state'])
    print(f"   max |selected - last| weight diff {d:.3e}")
    GR = torch.load(f"{RES}/general_rows_{case}_lfc.pt", weights_only=False)
    print(f" rows: keys {sorted(GR.keys())}, K={GR['K']}, selected_epoch={GR['selected_epoch']}, trained None? {GR['trained'] is None}, n epochs saved {len(GR['epochs'])}")
    print("   sel_hist (epoch, score, fallbacks): " + ", ".join(f"({e}, {s:.6f}, {f})" for e, s, f in GR['sel_hist']))
    for nm in ("train_nc", "extra_nc"):
        f = f"{RES}/selfteacher_{case}_bestof_{nm}.json"
        T = {int(k): v for k, v in json.load(open(f)).items()}
        ids = set(T); ov = ids & held
        print(f" teacher {nm}: {len(T)} targets; in train {len(ids & set(tr))}, in extra {len(ids & set(data2.extra_ids()))}, overlap held-out {len(ov)}; mean units on {np.mean([sum(v) for v in T.values()]):.1f}")

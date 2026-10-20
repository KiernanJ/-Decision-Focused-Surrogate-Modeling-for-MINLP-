import os, sys, pickle, numpy as np
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src")
import data2
sc = pickle.load(open("loadscales.pkl", "rb"))
for case, draw in (("case118", "fresh5"), ("case118", "fresh6"), ("case300", "fresh5"), ("pglib_opf_case118_ieee__api", "fresh5"), ("case118", "fresh4")):
    REF = pickle.load(open(f"{R}/acuc/results/{draw}_refs_{case}.pkl", "rb")); ids = getattr(data2, f"{draw}_ids")()
    conv = [i for i in ids if REF.get(i, {}).get("converged")]; nc = [i for i in ids if i not in conv]
    st = {}
    for i in ids:
        s = REF.get(i, {}).get("status", REF.get(i, {}).get("err", "missing")); st[s] = st.get(s, 0) + 1
    pol = [i for i in conv if REF[i].get("secs_polish") is not None]
    changed = sum(bool(REF[i].get("changed")) for i in conv)
    gaps_after = [REF[i].get("gap_after") for i in conv if REF[i].get("gap_after") is not None]
    print(f"{case} {draw}: {len(conv)}/{len(ids)} converged; statuses {st}; non-converged load scale {np.round(sorted(sc[case][nc]),3).tolist()}; "
          f"polished {len(pol)}, polish changed u on {changed}; polish MIP gap after: median {np.nanmedian(gaps_after) if gaps_after else float('nan'):.2e}, max {np.nanmax(gaps_after) if gaps_after else float('nan'):.2e}")

# %% Write paper/numbers.tex from the result files (run after figures/make_figures.py).
import json, glob, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..")
S = json.load(open(f"{R}/figures/summary.json"))
M = {}
for tag, c in (("A", "case118"), ("B", "case300"), ("V", "hybrid_vehicle")):
    o, p = S[c]["ours"], S[c]["proxy"]
    M[f"OURgap{tag}"] = f"{o['abs_gap']:.3f}"; M[f"OURdisc{tag}"] = f"{o['disc']:.2f}"
    M[f"OURcont{tag}"] = f"{o['cont']:.2f}"
    M[f"PXgap{tag}"] = f"{p['abs_gap']:.3f}"; M[f"PXdisc{tag}"] = f"{p['disc']:.2f}"
    M[f"PXcont{tag}"] = f"{p['cont']:.2f}"
    ft = (lambda x: f"{x:.3f}") if tag == "V" else (lambda x: f"{x:.1f}")
    M[f"OURtime{tag}"] = ft(o["median_secs"]); M[f"PXtime{tag}"] = ft(p["median_secs"])
    M[f"REFtime{tag}"] = f"{S[c]['reference_median_secs']:.1f}" if tag != "V" else f"{S[c]['reference_median_secs']:.2f}"
    M[f"SPD{tag}"] = f"{S[c]['reference_median_secs']/o['median_secs']:.0f}"
M["VRATIO"] = f"{S['hybrid_vehicle']['proxy']['abs_gap']/S['hybrid_vehicle']['ours']['abs_gap']:.0f}"
for tag, c in (("A", "case118"), ("B", "case300")):
    r = np.array(json.load(open(f"{R}/acuc/results/timingF_{c}.json"))["reference"])
    M[f"LABELH{tag}"] = f"{144*r.mean()/3600:.1f}"
    tr = [json.load(open(f))["train_secs"] for f in glob.glob(f"{R}/acuc/results/training/train_{c}_s?.json")]
    M[f"TRAIN{tag}"] = f"{np.mean(tr):,.0f}".replace(",", "{,}")
ref = [r["secs"] for f in glob.glob(f"{R}/vehicle/results/pool/pool_*.json")
       for r in json.load(open(f))["ref"].values() if r]
M["LABELSV"] = f"{144*np.mean(ref):.0f}"
M["TRAINV"] = "1{,}420"     # vehicle train JSONs carry per-epoch times only: 1403-1428 s per seed (logs)
with open(f"{H}/numbers.tex", "w") as f:
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(f"{H}/numbers.tex").read())

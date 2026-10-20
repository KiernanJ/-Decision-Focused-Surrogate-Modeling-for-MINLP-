# hard-fixing fallbacks: are the fixings reserve-infeasible (max capacity with fixed-off units excluded < 1.1 x load)?
import sys, pickle, numpy as np
import os
REPO = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO + "/src")
import data2
from pipeline import grid, RESERVE
g, nl = grid("case118"); INST = data2.all_instances(g, "case118")
key = {"A0h": "Mc", "A4h": "Mc", "A5h": "M_A5", "A6h": "M_A6", "A6bh": "M_A6b", "A7h": "M_proxy", "A3ah": "Mc"}
tot = {}
for s in ("val", "fresh5", "fresh6"):
    st = pickle.load(open(f"res_case118_{s}.pkl", "rb")); c = st["meta"]["c"]
    for a, mk in key.items():
        for i in st["meta"]["ids"] if s == "val" else st["meta"]["scored"]:
            r = st["arms"][a].get(i)
            if not r or not r.get("hard_fallback"): continue
            M = np.asarray(st["meta"][mk][i]); off = M < 1 - c
            cap = float(g.pmax[~off].sum()); need = (1 + RESERVE)*float(INST[i][0].sum())
            tot.setdefault(a, []).append((s, i, cap >= need, round(cap/need, 3)))
for a, v in tot.items():
    print(a, "hard fallbacks:", len(v), "| reserve-feasible under fixings:", sum(x[2] for x in v), "|", v)

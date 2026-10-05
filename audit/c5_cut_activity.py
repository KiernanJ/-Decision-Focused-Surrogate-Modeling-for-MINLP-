"""Critical 5: are the learned cut rows active where they are actually used?

b = <A, w_rlx> - depth anchors each row to the uncut relaxation at FLAT V
(acuc/02_train.py), but layer 1 is solved at the PREDICTED V. Prints the row
slack b - A w at both points for trained case118 nets on test instances.
Slack ~0 = active; slack > 0 = the row cuts nothing and gets zero gradient.
Usage: python audit/c5_cut_activity.py [case] [n_instances]   Runtime: ~1-2 min.
"""
import json
import sys
import _common
from _common import setup, load_net, predict, RHO
import numpy as np

case = sys.argv[1] if len(sys.argv) > 1 else "case118"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
E = setup(case)
g, S, POOL = E["g"], E["S"], E["POOL"]
out = {}
for seed in range(4):
    net = load_net(case, seed, "net", g)
    for j in E["te"][:n]:
        Vr, Vi, A, b = predict(E, net, j)
        rp = S.solve(POOL[j][0], POOL[j][1], Vr, Vi, rho=RHO, A=A, b=b)
        rf = S.solve(POOL[j][0], POOL[j][1], np.ones(g.n_bus), np.zeros(g.n_bus), rho=RHO, A=A, b=b)
        sp = b - A @ np.r_[rp["pg"], rp["u"]]
        sf = b - A @ np.r_[rf["pg"], rf["u"]]
        out[f"s{seed}_j{j}"] = dict(slack_predV=sp.tolist(), slack_flatV=sf.tolist())
        print(f"{case} s{seed} j{j:4d}  slack at predicted V {np.round(sp, 3)}   at flat V {np.round(sf, 4)}",
              flush=True)
json.dump(out, open(f"{_common.OUT}/c5_cut_activity_{case}.json", "w"), indent=1)

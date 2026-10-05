"""Critical 3 (contrast case) and 5: on case118, how much does TRAINING contribute?

Deploys the published case118 pipeline (pipeline.deploy, confidence order, NDOWN=25,
per-generator threshold from the committed thrlf_eval_case118_s<seed>_d25.json fit)
with either the trained checkpoint or an untrained network of the same seed.

Usage: python audit/c3_case118_untrained.py --weights init --seed 0 [--thr fitted|0.5] [--positions ...]
Runtime: ~5-20 s per instance. Writes audit/out/case118_<weights>_<thr>_s<seed>.json.
"""
import argparse
import json
import _common
from _common import setup, load_net, predict, REPO, RHO
import numpy as np
from pipeline import deploy, set_down

ap = argparse.ArgumentParser()
ap.add_argument("--weights", choices=["net", "init"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--thr", choices=["fitted", "0.5"], default="fitted")
ap.add_argument("--positions", default="")
a = ap.parse_args()

E = setup("case118"); set_down(25)
g, S, POOL, nl = E["g"], E["S"], E["POOL"], E["nl"]
zz = lambda v: (v - v.mean()) / (v.std() + 1e-12)
F = np.stack([zz(g.pmax), zz(g.c1 + g.c2 * g.pmax), zz(nl)])
c = np.array(json.load(open(f"{REPO}/acuc/results/thrlf_eval_case118_s{a.seed}_d25.json"))["c"])
th = None if a.thr == "0.5" else np.clip(0.5 + c[0] + c[1] * F[0] + c[2] * F[1] + c[3] * F[2], 0.05, 0.95)
net = load_net("case118", a.seed, a.weights, g)
positions = [int(p) for p in a.positions.split(",")] if a.positions else list(range(48))
path = f"{_common.OUT}/case118_{a.weights}_{a.thr}_s{a.seed}.json"
res = {}
for p in positions:
    j = E["te"][p]
    Vr, Vi, A, b = predict(E, net, j)
    cst, dd = deploy(S, g, POOL[j][0], POOL[j][1], Vr, Vi, A=A, b=b, rho=RHO, thr=th)
    res[str(p)] = None if dd is None else [100 * (cst - POOL[j][3]) / POOL[j][3],
                                           100 * float((dd["u"] != POOL[j][2]).mean())]
    print(f"pos {p:2d} j {j:3d}  " + ("FAILED" if dd is None else
          f"gap {res[str(p)][0]:+.4f}%  disc {res[str(p)][1]:.2f}%"), flush=True)
    json.dump(res, open(path, "w"), indent=1)
v = np.array([x for x in res.values() if x is not None])
print(f"case118 s{a.seed} {a.weights} thr={a.thr}: n={len(v)}  |gap| {np.abs(v[:, 0]).mean():.4f}%  "
      f"disc {v[:, 1].mean():.2f}%")

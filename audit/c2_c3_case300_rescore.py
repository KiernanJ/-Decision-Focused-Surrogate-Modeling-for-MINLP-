"""Critical 2 + 3: re-score case300 test instances with different weights, using the
published deployment (merit order, threshold 0.80, NDOWN=40; the restoration is
copied from acuc/04_eval_case300.py).

  --weights net    what the published numbers used (last epoch); reproduces c300pipe rows
  --weights best   the epoch selected on validation, which the paper says is used
  --weights init   an untrained network with the same seed (is the gain from learning?)

Usage: python audit/c2_c3_case300_rescore.py --weights best --seeds 0,1,2,3 [--positions 0,1,2]
Runtime: ~30-60 s per instance per seed on one core (all 48 x 4 seeds is several hours;
run seeds as separate processes). Writes audit/out/case300_<weights>_s<seed>.json as
{test position: [gap %, discrete %, continuous %, n_on]} (null = restoration failed).
"""
import argparse
import json
import time
import _common
from _common import setup, load_net, predict, merit, RHO
import numpy as np
from pipeline import RESERVE

ap = argparse.ArgumentParser()
ap.add_argument("--weights", choices=["net", "best", "init"], required=True)
ap.add_argument("--seeds", default="0,1,2,3")
ap.add_argument("--positions", default="")
ap.add_argument("--thr", type=float, default=0.80)
ap.add_argument("--down", type=int, default=40)
a = ap.parse_args()

E = setup("case300")
g, S, POOL, nl = E["g"], E["S"], E["POOL"], E["nl"]
MERIT = merit(g, nl)


def restore(j, uf, thr, order, ndown):
    """Verbatim logic of the local restore() in acuc/04_eval_case300.py."""
    pd_, qd_ = POOL[j][0], POOL[j][1]
    z = (uf > thr).astype(float)
    need = (1.0 + RESERVE) * float(pd_.sum())
    for t in order:
        if float(g.pmax @ z) >= need:
            break
        z[t] = 1.0
    best = None
    for _ in range(g.n_gen + 1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"], pr["pg"]); break
        off = [t for t in order if z[t] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, cost, pg = best
    for t in [t2 for t2 in order[::-1] if z[t2] > 0.5][:ndown]:
        w = z.copy(); w[t] = 0.0
        if float(g.pmax @ w) < need:
            continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w, pr["cost"], pr["pg"]
    return z, cost, pg


positions = [int(p) for p in a.positions.split(",")] if a.positions else list(range(48))
for seed in [int(s) for s in a.seeds.split(",")]:
    net = load_net("case300", seed, a.weights, g)
    path = f"{_common.OUT}/case300_{a.weights}_s{seed}.json"
    try:
        res = json.load(open(path))
    except FileNotFoundError:
        res = {}
    for p in positions:
        j = E["te"][p]; t0 = time.time()
        Vr, Vi, A, b = predict(E, net, j)
        r1 = S.solve(POOL[j][0], POOL[j][1], Vr, Vi, rho=RHO, A=A, b=b)
        r = None if r1 is None else restore(j, r1["u"], a.thr, MERIT, a.down)
        if r is None:
            res[str(p)] = None
            print(f"s{seed} pos {p:2d} j {j:3d}  FAILED", flush=True)
        else:
            z, c, pg = r
            ref = S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))
            res[str(p)] = [100 * (c - POOL[j][3]) / POOL[j][3],
                           100 * float((z.astype(int) != POOL[j][2].astype(int)).mean()),
                           100 * float(np.abs(pg - ref["pg"]).sum() / ref["pg"].sum()),
                           float(z.sum())]
            print(f"s{seed} pos {p:2d} j {j:3d}  gap {res[str(p)][0]:+.4f}%  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(path, "w"), indent=1)
    v = np.array([x for x in res.values() if x is not None])
    print(f"s{seed} {a.weights}: n={len(v)}  |gap| {np.abs(v[:, 0]).mean():.4f}  "
          f"signed {v[:, 0].mean():+.4f}  disc {v[:, 1].mean():.3f}  cont {v[:, 2].mean():.3f}")

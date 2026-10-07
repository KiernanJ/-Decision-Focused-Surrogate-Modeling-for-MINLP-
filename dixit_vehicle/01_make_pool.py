# %% Reference pool for the Dixit vehicle: per vehicle, 224 demand profiles
#    (u ~ rand(100:2000)/1000 as in their code), exact MILP + LP relaxation.
#    Split per vehicle: first 100 test, next 24 validation, last 100 train (their 100/100).
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from dixit_vehicle import Vehicle
ap = argparse.ArgumentParser(); ap.add_argument("--vid", type=int, required=True); a = ap.parse_args()
V = Vehicle(a.vid); rng = np.random.default_rng(a.vid)
U = rng.integers(100, 2001, 224)/1000
out = []
for k, u in enumerate(U):
    D = Vehicle.demand(u); ex = V.solve(D); rl = V.solve(D, integer=False)
    out.append(dict(u=float(u), cost=ex["cost"], z=ex["z"].round().tolist(), Pe=ex["Pe"].tolist(), secs=ex["secs"],
                    mipgap=ex["gap"], lp_cost=rl["cost"], lp_z=rl["z"].tolist(), lp_y=rl["y"].tolist()))
os.makedirs("results/pool", exist_ok=True)
json.dump(out, open(f"results/pool/vehicle_{a.vid}.json", "w"))
s = np.array([r["secs"] for r in out])
print(f"vehicle {a.vid}: {len(out)} instances, exact median {np.median(s)*1000:.0f} ms max {s.max():.1f} s, "
      f"max MIP gap {max(r['mipgap'] for r in out):.1e}", flush=True)

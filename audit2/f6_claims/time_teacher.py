# F6: time the label-free best-of teacher (60_selfteacher.py logic) per instance, single process, on a few case118 training draws.
import os, sys, time, json
import os; REPO = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import numpy as np, torch
CASE = sys.argv[1] if len(sys.argv) > 1 else "case118"; NI = int(sys.argv[2]) if len(sys.argv) > 2 else 4
W = {"case118": 10.0, "case300": 30.0}.get(CASE, 10.0); K, ROUNDS, SWAPS = 6, 8, 4
import combopt_ac, selfsup, data2
from pipeline import grid
from vnetio import vnet_out
combopt_ac.init(CASE, True)
g, nl = grid(CASE); n = g.n_bus; INST = data2.all_instances(g, CASE); te, va, tr = data2.split()
base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
OUT, RHO, VSEL = vnet_out(os.path.join(REPO, "acuc", "results", f"vnet_layer_{CASE}_env_nc.pt"), Xn, n)
V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
T = json.load(open(os.path.join(REPO, "acuc", "results", f"selfteacher_{CASE}_bestof_extra_nc.json")))
print(f"[config] {CASE} rho {RHO:.0e} sel {VSEL} W {W}", flush=True)
rows = []
for i in tr[:NI]:
    t0 = time.time(); _, r = selfsup.deploy_improve((i, V(i), None, None, W, RHO, K, ROUNDS, SWAPS)); t1 = time.time()
    _, rc = selfsup.restore_candidates((i, V(i))); t2 = time.time()
    cands = ([("W+improve", r["u"])] if r else []) + list(rc["cands"].items())
    best = None
    for nm, u in cands:
        _, pr = combopt_ac.price_strict((i, u))
        if pr is not None and (best is None or pr[0] < best[1]): best = (nm, pr[0], np.asarray(pr[5]))
    t3 = time.time()
    same = (best is not None and str(i) in T and np.array_equal(best[2], np.asarray(T[str(i)])))
    rows.append((i, t1-t0, t2-t1, t3-t2, t3-t0, best[0] if best else None, same))
    print(f"inst {i}: deploy+improve {t1-t0:.1f}s, restorations {t2-t1:.1f}s, strict pricing of cands {t3-t2:.1f}s, total {t3-t0:.1f}s, winner {best[0] if best else None}, matches committed target {same}", flush=True)
tot = np.array([r[4] for r in rows]); print(f"[summary] {CASE}: per-instance teacher time mean {tot.mean():.1f}s median {np.median(tot):.1f}s (n={len(rows)}); committed-target reproduced on {sum(r[6] for r in rows)}/{len(rows)}", flush=True)

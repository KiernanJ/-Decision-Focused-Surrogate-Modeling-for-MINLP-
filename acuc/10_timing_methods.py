# %% Per-instance deployment time on an IDLE machine (audit 2026-10-06), seed 0,
#    converged test instances, the setting each method selected on validation.
#    ours  = forward + anchor solve + cut relaxation + restoration
#    proxy = forward + restoration.        Single thread, no price cache.
#    Usage: python 10_timing_methods.py --case case300 --method ours --shard 0 --nshard 4
#           python 10_timing_methods.py --case case300 --merge
import os, sys, json, time, argparse, glob, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--method", default="ours")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
OUT = "results/eval2"
if a.merge:
    res = {}
    for m in ("ours", "proxy"):
        res[m] = [x for f in sorted(glob.glob(f"{OUT}/tmeth_{a.case}_{m}_*.json")) for x in json.load(open(f))]
        print(a.case, m, len(res[m]), "median", np.median(res[m]) if res[m] else None)
    json.dump(res, open(f"{OUT}/timing_methods_{a.case}.json", "w")); raise SystemExit
# reuse the evaluator's functions by running it as a module with a fixed method/seed
sys.argv = [sys.argv[0], "--case", a.case, "--seed", "0", "--method", a.method]
os.environ["TIMING_ONLY"] = "1"
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "07_select_eval.py")).read()
cut = src.index("# %% selection on validation")          # setup + uf only (proxy: reload model)
if a.method == "proxy":
    pre = src[:src.index("    # supervised proxy: labels")]
    exec(compile(pre + "\n    pass\n", "07_head", "exec"))
    import torch, torch.nn as nn
    ck = torch.load(f"{OUT}/proxy_{a.case}_s0.pt", weights_only=False)
    pm = nn.Sequential(nn.Linear(2*NB, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, NG))
    pm.load_state_dict(ck["state"]); pm.eval()
else:
    exec(compile(src[:cut], "07_head", "exec"))
choice = json.load(open(f"{OUT}/{a.case}_s0_{a.method}.json"))["choice"]
o_sel, t_sel = choice.split("|")[0], float(choice.split("|")[1])
REF = data2.refs(a.case)
ids = [j for j in te if data2.converged(REF.get(j))][a.shard::a.nshard]
secs = []
for j in ids:
    t0 = time.time()
    if a.method == "proxy":
        with torch.no_grad():
            uf = torch.sigmoid(pm(Xn[[j]]))[0].numpy().astype(float)
        t_uf = time.time() - t0
    else:
        u_, tt = net_uf(net, [j]); uf, t_uf = u_[j], tt[j]
    t1 = time.time()
    if uf is not None:
        restore(j, uf, t_sel, o_sel, cache=False)
    secs.append(t_uf + time.time() - t1); print(j, f"{secs[-1]:.2f}s", flush=True)
json.dump(secs, open(f"{OUT}/tmeth_{a.case}_{a.method}_{a.shard}.json", "w"))

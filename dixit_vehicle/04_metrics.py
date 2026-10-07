# %% Our method on the Dixit vehicle: gap, discrete error, continuous error and time per
#    instance (forward + cut projection + re-solve + dual-guided search), vs exact Gurobi.
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np, torch
out = {}
for v in range(10):
    ck = f"{HERE}/results/combopt_v{v}_search.pt"
    if not os.path.exists(ck): continue
    sys.argv = ["x", "--vid", str(v), "--search"]
    src = open(f"{HERE}/02_combopt_train.py").read()
    g = {"__file__": f"{HERE}/02_combopt_train.py"}; exec(src[:src.index("NC_va = score(va")], g)
    g["net"].load_state_dict(torch.load(ck, weights_only=False)["net"]); g["net"].eval()
    rows = json.load(open(f"{HERE}/results/combopt_v{v}_search.json"))["rows"]
    secs = []
    for i in g["te"]:
        t0 = time.time()
        with torch.no_grad(): A, b = g["rows"]([i])
        g["deploy"](i, A[0].numpy().astype(float), b[0].numpy().astype(float), True)
        secs.append(time.time() - t0)
    ex = [g["P"][i]["secs"] for i in g["te"]]
    c = [r["cut"] for r in rows if r["cut"]]
    out[v] = dict(gap=float(np.mean([x["gap"] for x in c])), disc=float(np.mean([x["disc"] for x in c])),
                  cont=float(np.mean([x["cont"] for x in c])), t_ours=float(np.median(secs)), t_exact=float(np.median(ex)))
    o = out[v]; print(f"veh {v}: gap {o['gap']:.3f}%  disc {o['disc']:.2f}%  cont {o['cont']:.2f}%  time ours {o['t_ours']*1000:.0f} ms vs exact {o['t_exact']*1000:.0f} ms", flush=True)
json.dump(out, open(f"{HERE}/results/metrics_search.json", "w"), indent=1)

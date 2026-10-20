# H: vehicle exact layer (src/vehicle_layer.py) -- is the per-step cut gradient anything more than the trivial imitation gradient
# 2(z_t - z_own_t)/T on binding rows? Same data/cut-net construction as dixit_vehicle/13_gate_layer.py (stage-1 imitation cuts).
import os, sys, json
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HERE = os.path.join(R, "dixit_vehicle"); sys.path.insert(0, os.path.join(R, "src")); sys.path.insert(0, HERE); os.chdir(HERE)
import numpy as np, torch
from importlib import import_module
S1 = import_module("12_fixcuts_stage1")
from dixit_vehicle import Vehicle, SW
from vehicle_layer import VehLayer
def run(label, layer, Dall, tr, TG, c, N=30):
    X = torch.tensor(Dall, dtype=torch.float32); mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    trk = [i for i in tr if TG.get(i) is not None]; net = S1.cutnet(Xn[trk], np.stack([TG[i]["z"] for i in trk]))
    with torch.no_grad(): Pr = torch.softmax(net(Xn).view(len(Dall), -1, SW+1), -1).numpy()
    cs, modes, nb = [], [], []
    for i in trk[:N]:
        conf, lev = Pr[i].max(-1), Pr[i].argmax(-1)
        lo = np.where(conf >= c, np.maximum(lev - (1 - conf)/(1 - c), 0), 0.0); hi = np.where(conf >= c, np.minimum(lev + (1 - conf)/(1 - c), SW), SW)
        zo = np.array(TG[i]["z"], float); r = layer.track(Dall[i], lo, hi, zo)
        if r is None: continue
        L0, gl, gh = r; z = layer.z.value; T = len(z); modes.append(layer.L.last_mode)
        tl = np.where((lo > 0) & (np.abs(z - lo) < 1e-5), 2*(z - zo)/T, 0.0); th = np.where((hi < SW) & (np.abs(z - hi) < 1e-5), 2*(z - zo)/T, 0.0)
        ga, gt = np.r_[gl, gh], np.r_[tl, th]; nb.append(int((tl != 0).sum() + (th != 0).sum()))
        if np.linalg.norm(ga) > 0 and np.linalg.norm(gt) > 0: cs.append(ga @ gt/(np.linalg.norm(ga)*np.linalg.norm(gt)))
        elif np.linalg.norm(ga) == 0 and np.linalg.norm(gt) == 0: cs.append(1.0)
    from collections import Counter
    cs = np.array(cs)
    print(f"[{label}] {len(cs)} training profiles: cos(VJP, trivial imitation on binding rows) median {np.median(cs):.4f}, min {cs.min():.4f}, frac >= 0.99 {np.mean(cs >= 0.99):.2f}; KKT modes {dict(Counter(modes))}; binding cut rows/profile {np.mean(nb):.1f}", flush=True)
vid = 4; P = json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json"))); V = Vehicle(vid)
TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", f"milp_v{vid}_w1.json"))).items()}
Dall = np.stack([Vehicle.demand(p["u"]) for p in P])
run(f"MILP vehicle {vid}", VehLayer("milp", V), Dall, list(range(124, 224)), TG, c=0.999)
from src_hv import load_pool, splits
VEH, Dm, REF = load_pool(os.path.join(HERE, "..", "vehicle", "results", "pool", "pool_*.json")); te, va, tr = splits(len(REF))
TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", "minlp_w1.json"))).items()}
run("MINLP vehicle", VehLayer("minlp", VEH), Dm, tr, TG, c=0.9)

# %% FD GATE for the vehicle exact layer (2026-10-10), both versions, 3 TRAINING profiles each, at the imitation cuts
#    (same cut-net as 12_fixcuts_stage1, confidence c -> l, h as in the AC method). Pass: corr >= 0.99, signs >= 7/8
#    where the finite differences are self-consistent (forward vs backward corr >= 0.99).
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, torch
from importlib import import_module
S1 = import_module("12_fixcuts_stage1")
from dixit_vehicle import Vehicle, SW
from vehicle_layer import VehLayer


def gate(label, layer, Dall, tr, TG, c, h=1e-2):
    X = torch.tensor(Dall, dtype=torch.float32); mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    trk = [i for i in tr if TG.get(i) is not None]; net = S1.cutnet(Xn[trk], np.stack([TG[i]["z"] for i in trk]))
    with torch.no_grad(): Pr = torch.softmax(net(Xn).view(len(Dall), -1, SW+1), -1).numpy()
    rng = np.random.default_rng(0); ok_all = True
    for i in trk[:3]:
        conf, lev = Pr[i].max(-1), Pr[i].argmax(-1)
        lo = np.where(conf >= c, np.maximum(lev - (1 - conf)/(1 - c), 0), 0.0)        # soft strength of z_t >= level
        hi = np.where(conf >= c, np.minimum(lev + (1 - conf)/(1 - c), SW), SW)       # soft strength of z_t <= level
        zo = np.array(TG[i]["z"], float); r = layer.track(Dall[i], lo, hi, zo)
        if r is None: print(f"[gate] {label} inst {i}: layer failed"); ok_all = False; continue
        L0, gl, gh = r; rows = []
        for _ in range(8):
            dl = rng.standard_normal(len(lo)); dh = rng.standard_normal(len(hi))
            dl[(lo <= 0) | (lo + 2*h > hi)] = 0.0; dh[(hi >= SW) | (hi - 2*h < lo)] = 0.0
            nr = np.sqrt((dl**2).sum() + (dh**2).sum())
            if nr == 0: continue
            dl /= nr; dh /= nr
            Lp = layer.track(Dall[i], lo + h*dl, hi + h*dh, zo, False)[0]; Lm = layer.track(Dall[i], lo - h*dl, hi - h*dh, zo, False)[0]
            rows.append(((Lp - Lm)/(2*h), (Lp - L0)/h, (L0 - Lm)/h, gl @ dl + gh @ dh))
        R = np.array(rows)
        if len(R) < 3: print(f"[gate] {label} inst {i}: too few active cut coordinates ({len(R)})"); continue
        cfd = np.corrcoef(R[:, 1], R[:, 2])[0, 1]; ca = np.corrcoef(R[:, 0], R[:, 3])[0, 1]; sg_ = int((np.sign(R[:, 0]) == np.sign(R[:, 3])).sum())
        passed = (cfd < 0.99) or (ca >= 0.99 and sg_ >= 7); ok_all &= passed
        print(f"[gate] {label} inst {i}: FD self-consistency {cfd:.3f} | analytic vs FD corr {ca:.4f}, signs {sg_}/{len(R)} | |FD| {np.abs(R[:,0]).mean():.2e} "
              f"| {'PASS' if (cfd >= 0.99 and ca >= 0.99 and sg_ >= 7) else ('FD unreliable' if cfd < 0.99 else 'FAIL')}", flush=True)
    print(f"[gate] {label}: {'PASSED' if ok_all else 'FAILED'}", flush=True)


if __name__ == "__main__":
    vid = 4; P = json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json"))); V = Vehicle(vid)
    TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", f"milp_v{vid}_w1.json"))).items()}
    Dall = np.stack([Vehicle.demand(p["u"]) for p in P])
    gate(f"MILP vehicle {vid}", VehLayer("milp", V), Dall, list(range(124, 224)), TG, c=0.999)
    from src_hv import load_pool, splits
    VEH, Dm, REF = load_pool(os.path.join(HERE, "..", "vehicle", "results", "pool", "pool_*.json")); te, va, tr = splits(len(REF))
    TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", "minlp_w1.json"))).items()}
    gate("MINLP vehicle", VehLayer("minlp", VEH), Dm, tr, TG, c=0.9)

# %% FD GATE for the vehicle GENERAL rows (2026-10-11): K=2 rows Ag [Pe; z] <= bg (CutNetK general category, anchored
#    b = <A, w_rlx> - depth at the relaxation with the integer cuts), exact VJP of the tracking loss w.r.t. (Ag, bg),
#    integer cuts frozen at the stage-1 imitation cuts. Checked at the INIT rows and at a PERTURBED (off-init) point.
#    3 training profiles per version; pass: corr >= 0.99 & signs >= 7/8 where FD is self-consistent.
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, torch
from importlib import import_module
S1 = import_module("12_fixcuts_stage1")
from dixit_vehicle import Vehicle, SW
from vehicle_layer import VehLayer
from cutnet_k import CutNetK, rows_torch
K = 2


def gate(label, layer, Dall, tr, TG, c, h=1e-2):
    T = Dall.shape[1]; X = torch.tensor(Dall, dtype=torch.float32); mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    trk = [i for i in tr if TG.get(i) is not None]; net = S1.cutnet(Xn[trk], np.stack([TG[i]["z"] for i in trk]))
    with torch.no_grad(): Pr = torch.softmax(net(Xn).view(len(Dall), -1, SW+1), -1).numpy()
    rng = np.random.default_rng(0); ok_all = True
    for where in ("init", "perturbed"):
        g = CutNetK(T//2, T, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False)
        if where == "perturbed":
            with torch.no_grad():
                for p_ in (g.head_a, g.head_al, g.head_d): p_.bias.add_(0.5*torch.randn_like(p_.bias))
        for i in trk[:3]:
            conf, lev = Pr[i].max(-1), Pr[i].argmax(-1); s = (1 - conf)/(1 - c); act = conf >= c
            lo = np.where(act, np.maximum(lev - s, 0), 0.0); hi = np.where(act, np.minimum(lev + s, SW), SW)
            w = layer.anchor(Dall[i], lo, hi)
            with torch.no_grad():
                _, _, a, al, a2, d, d2, _ = g(Xn[[i]]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.tensor(w, dtype=torch.float32)[None])
            Ag, bg = A_[0, :K].numpy().astype(float), b_[0, :K].numpy().astype(float); zo = np.array(TG[i]["z"], float)
            r = layer.track_gen(Dall[i], lo, hi, Ag, bg, zo)
            if r is None: print(f"[gate] {label} {where} inst {i}: layer failed", flush=True); ok_all = False; continue
            L0, gA, gb = r; rows = []
            for _ in range(8):
                dA = rng.standard_normal(Ag.shape); db = rng.standard_normal(bg.shape); nr = np.sqrt((dA**2).sum() + (db**2).sum()); dA /= nr; db /= nr
                Lp = layer.track_gen(Dall[i], lo, hi, Ag + h*dA, bg + h*db, zo, False); Lm = layer.track_gen(Dall[i], lo, hi, Ag - h*dA, bg - h*db, zo, False)
                if Lp is None or Lm is None: continue
                rows.append(((Lp[0] - Lm[0])/(2*h), (Lp[0] - L0)/h, (L0 - Lm[0])/h, (gA*dA).sum() + gb @ db))
            R = np.array(rows); cfd = np.corrcoef(R[:, 1], R[:, 2])[0, 1]; ca = np.corrcoef(R[:, 0], R[:, 3])[0, 1]; sg_ = int((np.sign(R[:, 0]) == np.sign(R[:, 3])).sum())
            vio = float(np.maximum(Ag @ w - bg, 0).sum())
            st = "PASS" if (cfd >= 0.99 and ca >= 0.99 and sg_ >= 7) else ("FD unreliable" if not cfd >= 0.99 else "FAIL")
            ok_all &= st != "FAIL"
            print(f"[gate] {label} {where} inst {i}: rows violated at anchor by {vio:.3f} | FD self-cons {cfd:.3f} | analytic vs FD corr {ca:.4f}, signs {sg_}/{len(R)} | |FD| {np.abs(R[:,0]).mean():.2e} | {st}", flush=True)
    print(f"[gate] {label}: {'PASSED' if ok_all else 'FAILED'}", flush=True)


if __name__ == "__main__":
    vid = 4; P = json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json"))); V = Vehicle(vid)
    TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", f"milp_v{vid}_w1.json"))).items()}
    Dall = np.stack([Vehicle.demand(p["u"]) for p in P])
    gate(f"MILP vehicle {vid}", VehLayer("milp", V, k_gen=K), Dall, list(range(124, 224)), TG, c=0.999)
    from src_hv import load_pool, splits
    VEH, Dm, REF = load_pool(os.path.join(HERE, "..", "vehicle", "results", "pool", "pool_*.json")); te, va, tr = splits(len(REF))
    TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", "minlp_w1.json"))).items()}
    gate("MINLP vehicle", VehLayer("minlp", VEH, k_gen=K), Dm, tr, TG, c=0.9)

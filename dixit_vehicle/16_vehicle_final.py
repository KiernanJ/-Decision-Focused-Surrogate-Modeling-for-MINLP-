# %% HYBRID VEHICLE — FINAL PIPELINE (2026-10-11), both versions: MILP (Dixit, 8 vehicles) and MINLP (convex MIQCQP).
#    OUR METHOD (label-free, no V), same as the AC method:
#      1. per-step INTEGER cuts: cut-net imitation of OUR OWN W=1 solves (11_teacher.py, training profiles only), confidence
#         threshold c chosen by the LABEL-FREE validation score, then fine-tuned through the EXACT LAYER (src/vehicle_layer.py),
#         epoch chosen label-free (epoch -1 = imitation cuts eligible);
#      2. K=2 GENERAL rows alpha'Pe + a'z <= b (CutNetK general category, anchored b = <A, w_rlx> - depth at the relaxation
#         with the integer cuts), trained through the exact layer (tracking + validity), epoch chosen label-free among
#         among the trained epochs;
#      3. deploy: ONE short solve (Gurobi WorkLimit W=0.01, Threads=1) with integer cuts (rounded RHS) + general rows (soft,
#         priced 1e3); fallback if no solution = same solve without cuts (counted). Scored cost = fixed-z re-solve.
#    SUPERVISED NN PROXY (labels allowed): MLP D -> z* (MSE) on the pool references of the TRAINING profiles, hidden/lr/epochs
#      chosen on VALIDATION MSE (as dixit_vehicle/03_nn_proxy.py), then L1 projection onto the integer-feasible set, rounding,
#      fixed-z re-solve (Threads=1). Failures counted.
#    Label-free validation score = deployed cost / relaxation cost.  References = pool exact solves (MIPGap 1e-4); a test
#    instance is excluded only if its reference did not reach MIPGap 1e-4 (listed).  Seeds 0,1,2 vary every network.
#    (2026-10-11, Parth) selection is among LAYER-TRAINED epochs only, for both cut categories (warm start / init / OFF not deployable).
#    Usage:  python 16_vehicle_final.py train     (training + VALIDATION only, saves everything; never reads test)
#            python 16_vehicle_final.py test      (untouched test, ONCE, loads the saved bundles)
import os, sys, json, time, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
from importlib import import_module
W, CS, K, SEEDS = 0.01, (0.999, 0.99, 0.95, 0.9, 0.8), 2, (0, 1, 2)
EP_INT, LR_INT, EP_GEN, LR_GEN, BATCH, LAM = 15, 1e-4, 10, 2e-3, 24, 10.0
CASES = [f"milp_v{v}" for v in (0, 1, 2, 4, 6, 7, 8, 9)] + ["minlp"]
# (2026-10-11 night) VEH_LF=1: TRULY label-free teacher (18_lf_teacher.py: own W=0.01 solve / rounded relaxation + local search) and
# 300 extra unlabeled training profiles per case; outputs to results/final_lf (default mode unchanged)
LF = os.environ.get("VEH_LF") == "1"
TESTDIR = "pool_fresh2" if LF else "pool_fresh"   # (audit 2026-10-12) LF method is tested ONCE on pool_fresh2 (pool_fresh was scored with the earlier method)
OUT = os.path.join(HERE, "results", "final_lf" if LF else "final"); os.makedirs(OUT, exist_ok=True)


def _lf_extend(name, Dall, TG):
    """LF mode: append the extra unlabeled profiles (training only) and swap in the label-free teacher. -> Dall, TG, extra ids."""
    from dixit_vehicle import Vehicle
    EX = json.load(open(os.path.join(HERE, "results", "extra", f"{name}.json")))
    Dx = np.stack([Vehicle.demand(u) for u in EX["u"]]) if "u" in EX else np.array(EX["D"])
    n0 = len(Dall); ex = list(range(n0, n0 + len(Dx)))
    LT = json.load(open(os.path.join(HERE, "results", "teacher", f"{name}_lf.json")))
    TG = {(n0 + int(k[1:]) if k.startswith("x") else int(k)): v for k, v in LT.items()}
    return np.vstack([Dall, Dx]), TG, ex


def load_case(name):
    S1 = import_module("12_fixcuts_stage1"); import gurobipy as gp; from gurobipy import GRB
    from dixit_vehicle import Vehicle
    if name.startswith("milp_v"):
        vid = int(name[6:]); P = json.load(open(os.path.join(HERE, "results", "pool", f"vehicle_{vid}.json"))); V = Vehicle(vid, env=S1.H.ENV)
        Dall = np.stack([Vehicle.demand(p["u"]) for p in P]); te, va, tr = list(range(100)), list(range(100, 124)), list(range(124, 224))
        TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", f"milp_v{vid}_w1.json"))).items()}
        REF = {i: dict(cost=P[i]["cost"], z=np.array(P[i]["z"], float), Pe=np.array(P[i]["Pe"]), ok=P[i]["mipgap"] <= 1e-4 + 1e-12) for i in range(len(P))}
        FP = os.path.join(HERE, "results", TESTDIR, f"vehicle_{vid}.json")       # (audit #4) FRESH test profiles, 17_fresh_test_pool.py
        if os.path.exists(FP):
            PF = json.load(open(FP)); n0 = len(P); Dall = np.vstack([Dall, np.stack([Vehicle.demand(p["u"]) for p in PF])]); te = list(range(n0, n0 + len(PF)))
            REF.update({n0 + k: dict(cost=p["cost"], z=np.array(p["z"], float), Pe=np.array(p["Pe"]), ok=p["mipgap"] <= 1e-4 + 1e-12) for k, p in enumerate(PF)})
        ex = []
        if LF: Dall, TG, ex = _lf_extend(name, Dall, TG)
        solve = lambda D, w, relax=False, bnd=None, rows=None: S1.milp(V, D, w, None, relax, bnd, rows)
        def project(D, zf):
            r = V.solve(D, target=zf); return None if r is None else np.round(r["z"])
        return dict(kind="milp", veh=V, D=Dall, te=te, va=va, tr=tr, ex=ex, TG=TG, REF=REF, solve=solve, project=project, SW=3)
    from src_hv import load_pool, splits
    VEH, Dm, RL = load_pool(os.path.join(HERE, "..", "vehicle", "results", "pool", "pool_*.json")); te, va, tr = splits(len(RL))
    TG = {int(k): v for k, v in json.load(open(os.path.join(HERE, "results", "teacher", "minlp_w1.json"))).items()}
    REF = {i: dict(cost=r["cost"], z=np.array(r["z"], float), Pe=np.array(r["Peng"]), ok=r["mipgap"] <= 1e-4 + 1e-12) for i, r in enumerate(RL) if r}
    FP = os.path.join(HERE, "results", TESTDIR, "minlp.json")                     # (audit #4) FRESH test profiles
    if os.path.exists(FP):
        F = json.load(open(FP)); n0 = len(RL); Dm = np.vstack([Dm, np.array(F["D"])]); te = list(range(n0, n0 + len(F["D"])))
        REF.update({n0 + int(k): dict(cost=r["cost"], z=np.array(r["z"], float), Pe=np.array(r["Peng"]), ok=r["mipgap"] <= 1e-4 + 1e-12) for k, r in F["ref"].items() if r})
    ex = []
    if LF: Dm, TG, ex = _lf_extend(name, Dm, TG)
    solve = lambda D, w, relax=False, bnd=None, rows=None: S1.minlp(VEH, D, w, None, relax, bnd, rows)
    T, S = VEH["T"], VEH["S_MODES"]
    def project(Dp, zf):                                  # = vehicle/03_nn_proxy.py qz, single-threaded
        m = gp.Model(env=S1.H.ENV); m.Params.TimeLimit = 120
        E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"]); Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
        z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER); w = m.addVars(T, lb=0); m.addConstr(E[0] == VEH["E_init"])
        for t in range(T):
            m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0); m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t] + Pe[t] >= Dp[t])
            m.addConstr(w[t] >= z[t] - float(zf[t])); m.addConstr(w[t] >= float(zf[t]) - z[t])
        m.setObjective(gp.quicksum(w[t] for t in range(T)), GRB.MINIMIZE); m.optimize()
        return None if m.SolCount == 0 else np.array([round(z[t].X) for t in range(T)], float)
    return dict(kind="minlp", veh=VEH, D=Dm, te=te, va=va, tr=tr, ex=ex, TG=TG, REF=REF, solve=solve, project=project, SW=S)


class Method:
    """our method's networks + deployment (shared by train and test)."""
    def __init__(self, C, Xn, cut_state, c, gen_state, seed):
        S1 = import_module("12_fixcuts_stage1"); from cutnet_k import CutNetK; from vehicle_layer import VehLayer
        T = C["D"].shape[1]; self.C, self.Xn, self.c, self.SW, self.T = C, Xn, c, C["SW"], T
        self.net = nn.Sequential(nn.Linear(T, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, T*(C["SW"]+1)))
        if cut_state is not None: self.net.load_state_dict(cut_state)
        self.gnet = CutNetK(T//2, T, k_rows=K, v_scale=0.5, hidden=256, seed=seed, init_depth=0.25, max_depth=2.0, learn_thr=False)
        if gen_state is not None: self.gnet.load_state_dict(gen_state)
        self.use_gen = gen_state is not None
        self.L0 = VehLayer(C["kind"], C["veh"]); self.L2 = VehLayer(C["kind"], C["veh"], k_gen=K)

    def cuts(self, idx, grad=False):
        with torch.set_grad_enabled(grad):
            P = torch.softmax(self.net(self.Xn[idx]).view(len(idx), -1, self.SW+1), -1); conf, lev = P.max(-1); s = (1 - conf)/(1 - self.c); act = conf >= self.c
            lo = torch.where(act, torch.clamp(lev - s, min=0.0), torch.zeros_like(s)); hi = torch.where(act, torch.clamp(lev + s, max=float(self.SW)), torch.full_like(s, float(self.SW)))
        return lo, hi

    def bounds(self, lo, hi):
        return np.where(lo > 0, np.ceil(lo - 1e-9), 0.0), np.where(hi < self.SW, np.floor(hi + 1e-9), float(self.SW))

    def grows(self, idx, wr, grad=False):
        from cutnet_k import rows_torch
        with torch.set_grad_enabled(grad):
            _, _, a, al, a2, d, d2, _ = self.gnet(self.Xn[idx]); A_, b_ = rows_torch(a, al, a2, d, d2, wr)
        return A_[:, :K], b_[:, :K]

    def deploy(self, i):
        """-> dict(cost, z, Pe, secs, fallback) or None.  Timed end to end: forward + anchor + short solve + fixed-z re-solve."""
        C = self.C; t0 = time.time()
        lo, hi = [t[0].numpy().astype(float) for t in self.cuts([i])]; bnd = self.bounds(lo, hi); rows = None
        if self.use_gen:
            w = self.L2.anchor(C["D"][i], lo, hi)
            if w is not None:
                A_, b_ = self.grows([i], torch.tensor(w, dtype=torch.float32)[None]); rows = (A_[0].detach().numpy().astype(float), b_[0].detach().numpy().astype(float))
        r = C["solve"](C["D"][i], W, bnd=bnd, rows=rows); fb = r is None
        if fb: r = C["solve"](C["D"][i], W)
        if r is None: return None
        p = C["solve"](C["D"][i], 1e6, relax=True, bnd=(r[1], r[1]))
        if p is None: return None
        return dict(cost=p[0], z=r[1].astype(float), Pe=p[2], secs=time.time() - t0, fallback=fb)


def proxy_train(C, Xn, seed):
    """(audit #4) the old grid selected at its edges -> wider grid (hidden up to 1024, weight decay) and validation-MSE EARLY STOPPING
    (best checkpoint every 25 steps, up to 3000), labels = CONVERGED references only."""
    T = C["D"].shape[1]; tr = [i for i in C["tr"] if i in C["REF"] and C["REF"][i]["ok"]]; va = [i for i in C["va"] if i in C["REF"] and C["REF"][i]["ok"]]
    Y = torch.tensor(np.stack([C["REF"][i]["z"] for i in tr + va]), dtype=torch.float32); Ytr, Yva = Y[:len(tr)], Y[len(tr):]; best = None
    for h in (16, 64, 256, 512, 1024):
        for lr in (1e-3, 1e-2):
            for wd in (0.0, 1e-4):
                torch.manual_seed(seed)
                m = nn.Sequential(nn.Linear(T, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, T)); o = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
                for it in range(3000):
                    o.zero_grad(); l = ((m(Xn[tr]) - Ytr)**2).mean(); l.backward(); o.step()
                    if it % 25 == 24:
                        with torch.no_grad(): v = float(((m(Xn[va]) - Yva)**2).mean())
                        if best is None or v < best[0]: best = (v, {k: t.clone() for k, t in m.state_dict().items()}, dict(hidden=h, lr=lr, wd=wd, epochs=it + 1))
    return best


def proxy_net(T, cfg, state):
    m = nn.Sequential(nn.Linear(T, cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], cfg["hidden"]), nn.ReLU(), nn.Linear(cfg["hidden"], T)); m.load_state_dict(state); return m.eval()


def proxy_deploy(C, m, Xn, i):
    t0 = time.time()
    with torch.no_grad(): zf = np.clip(m(Xn[[i]])[0].numpy(), 0, C["SW"])
    zq = C["project"](C["D"][i], zf)
    if zq is None: return None
    p = C["solve"](C["D"][i], 1e6, relax=True, bnd=(zq, zq))
    if p is None: return None
    return dict(cost=p[0], z=zq, Pe=p[2], secs=time.time() - t0, fallback=False)


def metrics(R, ref):
    if R is None: return (np.nan,)*4 + (0.0, np.nan)
    return (100*abs(R["cost"] - ref["cost"])/abs(ref["cost"]), 100*np.abs(R["z"] - ref["z"]).sum()/max(ref["z"].sum(), 1e-9),
            100*np.abs(R["Pe"] - ref["Pe"]).sum()/max(ref["Pe"].sum(), 1e-9), R["secs"], float(R["fallback"]), 100*(R["cost"] - ref["cost"])/abs(ref["cost"]))


def train_job(args):
    name, seed = args; os.environ["OMP_NUM_THREADS"] = "1"; torch.set_num_threads(1)
    S1 = import_module("12_fixcuts_stage1"); log = []; t0 = time.time()
    def say(m):
        log.append(m); print(f"[{name} s{seed}] {m}", flush=True)
    C = load_case(name); Dall = C["D"]; T = Dall.shape[1]; tr, va = C["tr"], C["va"]; SWn = C["SW"]
    X = torch.tensor(Dall, dtype=torch.float32); mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    trk = [i for i in tr + C.get("ex", []) if C["TG"].get(i) is not None]; ZO = {i: np.array(C["TG"][i]["z"], float) for i in trk}   # mu/sg stay on tr
    say(f"training profiles with a target: {len(trk)} ({'label-free teacher, original + extra unlabeled' if LF else 'W=1 teacher, original'})")
    WO = {i: torch.tensor(np.r_[C["TG"][i]["Pe"], C["TG"][i]["z"]], dtype=torch.float32) for i in trk}
    SCv = {i: abs(C["solve"](Dall[i], 1e6, relax=True)[0]) for i in va}            # label-free normaliser
    net = S1.cutnet(Xn[trk], np.stack([ZO[i] for i in trk]), seed=seed)
    M = Method(C, Xn, net.state_dict(), CS[0], None, seed)
    def score(): 
        R = {i: M.deploy(i) for i in va}; return float(np.mean([R[i]["cost"]/SCv[i] if R[i] else 10.0 for i in va])), R
    st = lambda mod: {k: t.clone() for k, t in mod.state_dict().items()}
    # ---- 1a. threshold c, label-free
    sc = {}
    for c in CS: M.c = c; sc[c] = score()[0]
    M.c = min(CS, key=lambda c: sc[c]); say(f"c = {M.c} (label-free scores {', '.join(f'{c}:{v:.6f}' for c, v in sc.items())})")
    # ---- 1b. integer cuts through the exact layer
    v0 = sc[M.c]; best = (np.inf, None, None); opt = torch.optim.Adam(M.net.parameters(), lr=LR_INT); rng = np.random.default_rng(seed); hist = []
    for ep in range(EP_INT):
        order = rng.permutation(trk); Lt, nf = [], 0
        for s0 in range(0, len(order), BATCH):
            b = [int(i) for i in order[s0:s0+BATCH]]; lo_t, hi_t = M.cuts(b, True)
            lon, hin = lo_t.detach().numpy().astype(float), hi_t.detach().numpy().astype(float); GL, GH, ok = np.zeros_like(lon), np.zeros_like(hin), 0
            for k, i in enumerate(b):
                r = M.L0.track(Dall[i], lon[k], hin[k], ZO[i])
                if r is None or len(r) < 3: nf += 1; continue
                Lt.append(r[0]); GL[k], GH[k] = r[1], r[2]; ok += 1
            if ok == 0: continue
            zo = torch.tensor(np.stack([ZO[i] for i in b]), dtype=torch.float32)
            lval = LAM*((torch.relu(lo_t - zo)**2).sum(1) + (torch.relu(zo - hi_t)**2).sum(1)).mean()
            opt.zero_grad(); ((lo_t*torch.tensor(GL, dtype=torch.float32)).sum() + (hi_t*torch.tensor(GH, dtype=torch.float32)).sum()).div(ok).add(lval).backward(); opt.step()
        h = dict(ep=ep, track=float(np.mean(Lt)) if Lt else np.nan, failed=nf)
        if ep % 3 == 2 or ep == EP_INT - 1:
            h["val"] = score()[0]
            if h["val"] < best[0] - 1e-9: best = (h["val"], st(M.net), ep)
        hist.append(h)
    int_last = st(M.net); M.net.load_state_dict(best[1]); int_ep = best[2]
    say(f"integer cuts (layer-trained): epoch {int_ep} selected (val {best[0]:.6f}; imitation warm start {v0:.6f})  [{time.time()-t0:.0f}s]")
    # ---- 2. general rows through the exact layer (integer cuts frozen)
    LOHI = {}
    for i in trk:
        lo, hi = [t[0].numpy().astype(float) for t in M.cuts([i])]; LOHI[i] = (lo, hi)
    AN = {i: M.L2.anchor(Dall[i], *LOHI[i]) for i in trk}; trg = [i for i in trk if AN[i] is not None]
    WR = {i: torch.tensor(AN[i], dtype=torch.float32) for i in trg}
    M.use_gen = True; g_init = st(M.gnet); vg0 = score()[0]
    bestg = (np.inf, None, None)                         # (Parth 2026-10-11) both categories LEARNED: trained epochs only; OFF/init are report-only
    opt = torch.optim.Adam(M.gnet.parameters(), lr=LR_GEN); hg = []
    for ep in range(EP_GEN):
        order = rng.permutation(trg); Lt, Lv, nf = [], [], 0
        for s0 in range(0, len(order), BATCH):
            b = [int(i) for i in order[s0:s0+BATCH]]; A_t, b_t = M.grows(b, torch.stack([WR[i] for i in b]), True)
            An, bn = A_t.detach().numpy().astype(float), b_t.detach().numpy().astype(float); GA, GB, ok = np.zeros_like(An), np.zeros_like(bn), 0
            for k, i in enumerate(b):
                r = M.L2.track_gen(Dall[i], *LOHI[i], An[k], bn[k], ZO[i])
                if r is None or len(r) < 3: nf += 1; continue
                Lt.append(r[0]); GA[k], GB[k] = r[1], r[2]; ok += 1
            if ok == 0: continue
            viol = torch.relu(torch.einsum("nkd,nd->nk", A_t, torch.stack([WO[i] for i in b])) - b_t); lval = LAM*(viol**2).sum(1).mean(); Lv.append(float(lval))
            opt.zero_grad(); ((A_t*torch.tensor(GA, dtype=torch.float32)).sum() + (b_t*torch.tensor(GB, dtype=torch.float32)).sum()).div(ok).add(lval).backward(); opt.step()
        v = score()[0]; hg.append(dict(ep=ep, track=float(np.mean(Lt)) if Lt else np.nan, valid=float(np.mean(Lv)) if Lv else np.nan, failed=nf, val=v))
        if v < bestg[0] - 1e-9: bestg = (v, st(M.gnet), ep)
    gen_last = st(M.gnet); gen_ep = bestg[2]
    if bestg[1] is None: M.use_gen = False
    else: M.gnet.load_state_dict(bestg[1])
    say(f"general rows: {'OFF (integer cuts only)' if gen_ep == 'off' else f'epoch {gen_ep}'} selected (val {bestg[0]:.6f}; init rows {vg0:.6f}; integer cuts only {best[0]:.6f}); "
        f"tracking {hg[0]['track']:.4f} -> {hg[-1]['track']:.4f}, validity {hg[0]['valid']:.4f} -> {hg[-1]['valid']:.4f}  [{time.time()-t0:.0f}s]")
    # ---- proxy (supervised; validation MSE)
    pv, pstate, pcfg = proxy_train(C, Xn, seed); pm = proxy_net(T, pcfg, pstate)
    # ---- VALIDATION REPORT (four metrics; report only)
    vr = [i for i in va if i in C["REF"] and C["REF"][i]["ok"]]
    ours = np.array([metrics(M.deploy(i), C["REF"][i]) for i in vr]); prox = np.array([metrics(proxy_deploy(C, pm, Xn, i), C["REF"][i]) for i in vr])
    bundle = dict(name=name, seed=seed, mu=mu, sg=sg, c=M.c, c_scores=sc, cut_state=st(M.net), cut_last=int_last, int_epoch=int_ep, int_hist=hist,
                  gen_state=(st(M.gnet) if M.use_gen else None), gen_init=g_init, gen_last=gen_last, gen_epoch=gen_ep, gen_hist=hg, val_scores=dict(imitation=v0, int_best=best[0], gen_init=vg0, gen_best=bestg[0]),
                  proxy_state=pstate, proxy_cfg=pcfg, proxy_val_mse=pv, val_ids=vr, val_ours=ours, val_proxy=prox, log=log,
                  config=CONFIG, lf=LF, n_targets=len(trk), teacher=(f"{name}_lf.json" if LF else "W=1 teacher"))   # (audit 2026-10-12) config needed by test_job
    torch.save(bundle, os.path.join(OUT, f"{name}_s{seed}.pt"))
    f = lambda v: f"gap mean {np.nanmean(v[:,0]):.3f}% median {np.nanmedian(v[:,0]):.3f}% disc {np.nanmean(v[:,1]):.2f}% cont {np.nanmean(v[:,2]):.2f}% | fails {int(np.isnan(v[:,0]).sum())}"
    say(f"VALIDATION ours: {f(ours)} | proxy {pcfg}: {f(prox)}  [{time.time()-t0:.0f}s]")
    return name, seed


CONFIG = dict(W=W, K=K, CS=CS, EP_INT=EP_INT, LR_INT=LR_INT, EP_GEN=EP_GEN, LR_GEN=LR_GEN, LAM=LAM, gamma=1.0, cut_price=1e3)


def proxy_job(args):
    """(audit #4) retrain the supervised proxy with the wider grid + early stopping; validation report; how often the layer fine-tune
    changed the DEPLOYED integer cuts vs the imitation warm start (validation); store CONFIG in the bundle."""
    name, seed = args; os.environ["OMP_NUM_THREADS"] = "1"; torch.set_num_threads(1); S1 = import_module("12_fixcuts_stage1")
    f = os.path.join(OUT, f"{name}_s{seed}.pt"); B = torch.load(f, weights_only=False); C = load_case(name); T = C["D"].shape[1]
    Xn = (torch.tensor(C["D"], dtype=torch.float32) - B["mu"])/B["sg"]
    pv, pstate, pcfg = proxy_train(C, Xn, seed); pm = proxy_net(T, pcfg, pstate)
    vr = B["val_ids"]; prox = np.array([metrics(proxy_deploy(C, pm, Xn, i), C["REF"][i]) for i in vr])
    trk = [i for i in C["tr"] if C["TG"].get(i) is not None]; warm = S1.cutnet(Xn[trk], np.stack([np.array(C["TG"][i]["z"], float) for i in trk]), seed=seed)
    Mw = Method(C, Xn, warm.state_dict(), B["c"], None, seed); Ms = Method(C, Xn, B["cut_state"], B["c"], None, seed); va = C["va"]
    bw = [Mw.bounds(*[t[0].numpy().astype(float) for t in Mw.cuts([i])]) for i in va]; bs = [Ms.bounds(*[t[0].numpy().astype(float) for t in Ms.cuts([i])]) for i in va]
    changed = float(np.mean([np.any(np.c_[a[0] != b[0], a[1] != b[1]], 1).mean() for a, b in zip(bw, bs)]))
    B.update(proxy_old_state=B["proxy_state"], proxy_old_cfg=B["proxy_cfg"], val_proxy_old=B["val_proxy"], proxy_state=pstate, proxy_cfg=pcfg,
             proxy_val_mse=pv, val_proxy=prox, steps_changed_by_layer=changed, config=CONFIG)
    torch.save(B, f)
    g = lambda v: f"gap mean {np.nanmean(v[:,0]):.3f}% disc {np.nanmean(v[:,1]):.2f}%"
    print(f"[{name} s{seed}] proxy {pcfg} val MSE {pv:.4f}: {g(prox)} (old grid {B['proxy_old_cfg']}: {g(B['val_proxy_old'])}) | ours {g(B['val_ours'])} | "
          f"layer fine-tune changed {100*changed:.1f}% of deployed cut steps vs imitation", flush=True)
    return name, seed


def test_job(args):
    name, seed = args; os.environ["OMP_NUM_THREADS"] = "1"; torch.set_num_threads(1)
    B = torch.load(os.path.join(OUT, f"{name}_s{seed}.pt"), weights_only=False); C = load_case(name); T = C["D"].shape[1]
    assert all(os.path.exists(os.path.join(HERE, "results", TESTDIR, f)) for f in [f"vehicle_{v}.json" for v in (0, 1, 2, 4, 6, 7, 8, 9)] + ["minlp.json"]), "fresh pool missing"
    assert B.get("config") == CONFIG, (B.get("config"), CONFIG)                     # (audit #4) the test must use the trained configuration
    assert bool(B.get("lf")) == LF and (not LF or B.get("teacher") == f"{name}_lf.json"), (name, seed, B.get("lf"), B.get("teacher"))   # (audit 2026-10-12) bundle = this mode
    Xn = (torch.tensor(C["D"], dtype=torch.float32) - B["mu"])/B["sg"]
    M = Method(C, Xn, B["cut_state"], B["c"], B["gen_state"], seed); pm = proxy_net(T, B["proxy_cfg"], B["proxy_state"])
    tid = [i for i in C["te"] if i in C["REF"] and C["REF"][i]["ok"]]
    exc = [(i, "no reference" if i not in C["REF"] else "reference MIPGap > 1e-4") for i in C["te"] if i not in tid]
    ours, prox = [], []
    for k, i in enumerate(tid):                                                      # (audit #4) arms INTERLEAVED per instance, alternating order
        if k % 2 == 0: ours.append(metrics(M.deploy(i), C["REF"][i])); prox.append(metrics(proxy_deploy(C, pm, Xn, i), C["REF"][i]))
        else: prox.append(metrics(proxy_deploy(C, pm, Xn, i), C["REF"][i])); ours.append(metrics(M.deploy(i), C["REF"][i]))
    return name, seed, tid, exc, np.array(ours), np.array(prox)


def test_job_safe(args):
    """(audit 2026-10-12) one crashing job must not kill the other 26 (Pool.__exit__ terminates) -> traceback returned instead."""
    import traceback
    try: return test_job(args)
    except Exception: return args[0], args[1], None, traceback.format_exc(), None, None


TIE = 1e-3   # (audit 2026-10-12, PRE-REGISTERED, = acuc/55 TIE_PP) |difference| <= 1e-3 pp = tie


def paired(xa, ya, lab):
    """= acuc/55_final_eval.py paired (PRE-REGISTERED): failures NOT dropped (one-sided failure = loss), Wilcoxon (zero_method wilcox)
    on the differences thresholded at TIE with n_nonzero, sign test on better/worse, bootstrap 95% CI of the mean."""
    from scipy.stats import wilcoxon, binomtest
    fa, fb = np.isnan(xa), np.isnan(ya); ok = ~fa & ~fb
    d = xa[ok] - ya[ok]; dt = np.where(np.abs(d) <= TIE, 0.0, d)
    better = int((dt < 0).sum() + (fb & ~fa).sum()); worse = int((dt > 0).sum() + (fa & ~fb).sum()); tie = int((dt == 0).sum() + (fa & fb).sum())
    nz = int((dt != 0).sum()); pw = wilcoxon(dt, zero_method="wilcox").pvalue if nz > 0 else float("nan")
    ps = binomtest(better, better + worse).pvalue if better + worse > 0 else float("nan")
    rng = np.random.default_rng(0); bs = [rng.choice(d, d.size).mean() for _ in range(10000)] if d.size else [np.nan]
    return (f"{lab}: ours - proxy {d.mean():+.4f} pp, 95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] (n={int(ok.sum())}), "
            f"Wilcoxon p {pw:.3g} (n_nonzero {nz}), better/tie/worse {better}/{tie}/{worse} (tie <= {TIE} pp), sign test p {ps:.3g} | "
            f"failures ours {int(fa.sum())}, proxy {int(fb.sum())}")


def savg(arrs, col):
    """seed-averaged column; an instance is a FAILURE (NaN, then a loss in paired) if the method failed on ANY seed."""
    A = np.array([a[:, col] for a in arrs]); return np.where(np.isnan(A).any(0), np.nan, A.mean(0))


if __name__ == "__main__":
    import runlog
    phase = sys.argv[1]
    if phase == "train":
        RUN = runlog.start("vehicle_final_train" + ("_lf" if LF else "")); import shutil; shutil.copy(os.path.join(HERE, "..", "src", "vehicle_layer.py"), RUN); shutil.copy(os.path.join(HERE, "12_fixcuts_stage1.py"), RUN)
        jobs = [(n, s) for s in SEEDS for n in CASES]
        with mp.get_context("spawn").Pool(11) as pool:
            for r in pool.imap_unordered(train_job, jobs): print(f"[done] {r}", flush=True)
        runlog.add_result(RUN, f"VEHICLE final TRAIN phase done ({len(jobs)} case-seed bundles in dixit_vehicle/results/final/; validation only)")
    elif phase == "proxy":
        RUN = runlog.start("vehicle_final_proxy")
        with mp.get_context("spawn").Pool(11) as pool:
            for r in pool.imap_unordered(proxy_job, [(n, s) for s in SEEDS for n in CASES]): pass
    elif phase == "test":
        import hashlib, glob
        from scipy.stats import wilcoxon, binomtest
        RUN = runlog.start("vehicle_final_TEST" + (f"_lf_{TESTDIR}" if LF else "")); TAGV = f"[{'LF' if LF else 'W1'} method, {TESTDIR}]"
        jobs = [(n, s) for s in SEEDS for n in CASES]; RES, ERR = [], []
        with mp.get_context("spawn").Pool(4) as pool:
            for r in pool.imap_unordered(test_job_safe, jobs):
                (ERR if r[2] is None else RES).append(r); runlog.save(RUN, "test_partial.pkl", dict(RES=RES, ERR=ERR))   # (audit) saved after EVERY job
                print(f"[done] {r[0]} s{r[1]} " + ("ERROR" if r[2] is None else f"{len(r[2])} scored"), flush=True)
        runlog.save(RUN, "provenance.pkl", {os.path.relpath(p, HERE): hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]
                                            for p in sorted(glob.glob(os.path.join(OUT, "*.pt")) + glob.glob(os.path.join(HERE, "results", TESTDIR, "*.json")))})
        if ERR:
            for e in ERR: print(f"[ERROR] {e[0]} s{e[1]}:\n{e[3]}", flush=True)
            sys.exit(f"{len(ERR)} test job(s) crashed; finished jobs saved in {RUN}/test_partial.pkl; re-run ONLY the crashed (unscored) jobs")
        RES.sort(key=lambda r: (CASES.index(r[0]), r[1])); runlog.save(RUN, "test_results.pkl", RES)
        for ver, names in (("MILP (8 vehicles pooled)", CASES[:-1]), ("MINLP", ["minlp"])):
            per = {s: {nm: np.vstack([r[4 if nm == 'ours' else 5] for r in RES if r[1] == s and r[0] in names]) for nm in ("ours", "proxy")} for s in SEEDS}
            ntid = sum(len(r[2]) for r in RES if r[1] == 0 and r[0] in names); exc = [e for r in RES if r[1] == 0 and r[0] in names for e in r[3]]
            ex_ = {k: sum(e[1] == k for e in exc) for k in ("no reference", "reference MIPGap > 1e-4")}
            print(f"\n[FINAL VEHICLE {ver}] {TAGV} FRESH untouched test, {ntid} instances (excluded: {ex_}), seeds {list(SEEDS)}, W={W}", flush=True)
            for s in SEEDS:
                for col, lab in ((0, "|gap|"), (5, "signed gap")):
                    print(f"--- seed {s} " + paired(per[s]["ours"][:, col], per[s]["proxy"][:, col], lab), flush=True)
                print(f"    seed {s}: our fallbacks {int(np.nansum(per[s]['ours'][:, 4]))}", flush=True)
            for col, lab in ((0, "|gap|"), (5, "signed gap (= paired cost difference)")):
                print("[paired, seed-averaged, PRE-REGISTERED] " + paired(savg([per[s]["ours"] for s in SEEDS], col), savg([per[s]["proxy"] for s in SEEDS], col), lab), flush=True)
            print("   method               | gap mean | gap median | discrete err | continuous err | time median | failures | signed gap mean", flush=True)
            for nm, lab in (("ours", "OUR METHOD"), ("proxy", "supervised NN proxy")):
                Mx = np.array([[np.nanmean(v[:, 0]), np.nanmedian(v[:, 0]), np.nanmean(v[:, 1]), np.nanmean(v[:, 2]), 1000*np.nanmedian(v[:, 3]), np.isnan(v[:, 0]).sum(), np.nanmean(v[:, 5])]
                               for v in (per[s][nm] for s in SEEDS)])
                g = lambda j, dd=4: f"{Mx[:, j].mean():.{dd}f}+-{Mx[:, j].std(ddof=1):.{dd}f}"
                line = f"{g(0)}% | {g(1)}% | {g(2, 2)}% | {g(3, 2)}% | {g(4, 1)} ms | {int(Mx[:, 5].sum())} | {g(6)}%"
                print(f"   {lab:20s} | {line}", flush=True)
                runlog.add_result(RUN, f"FINAL VEHICLE {ver} {TAGV} FRESH untouched ({ntid} inst, seeds {list(SEEDS)}) {lab}: gap mean | median | disc | cont | time | failures | signed = {line}")
            if len(names) > 1:                                 # (audit) VEHICLE-CLUSTERED: the 8 vehicles are the independent units
                print("   per vehicle (seed-averaged, strict failures): vehicle | n | ours gap mean/median, disc, cont | proxy gap mean/median, disc, cont | paired |gap|", flush=True)
                vm = []
                for n_ in names:
                    A = [np.vstack([r[4] for r in RES if r[1] == s and r[0] == n_]) for s in SEEDS]; B = [np.vstack([r[5] for r in RES if r[1] == s and r[0] == n_]) for s in SEEDS]
                    xo, yp = savg(A, 0), savg(B, 0); a, b = np.nanmean(A, 0), np.nanmean(B, 0); ok = ~np.isnan(xo) & ~np.isnan(yp); vm.append((xo[ok] - yp[ok]).mean())
                    f_ = lambda v: f"{np.nanmean(v[:,0]):.3f}/{np.nanmedian(v[:,0]):.3f}%, {np.nanmean(v[:,1]):.2f}%, {np.nanmean(v[:,2]):.2f}%"
                    print(f"   {n_:8s} | {len(a)} | {f_(a)} | {f_(b)} | " + paired(xo, yp, "|gap|"), flush=True)
                vm = np.array(vm); vb, vw = int((vm < -TIE).sum()), int((vm > TIE).sum())
                print(f"[vehicle-clustered, PRE-REGISTERED] per-vehicle mean |gap| difference (ours - proxy, pp): {np.round(vm, 4).tolist()}; "
                      f"vehicles better/tie/worse {vb}/{len(vm) - vb - vw}/{vw}, sign test p {binomtest(vb, vb + vw).pvalue if vb + vw else float('nan'):.3g} "
                      f"(min attainable {2*0.5**len(vm):.3g}), Wilcoxon over vehicles p {wilcoxon(vm).pvalue:.3g}", flush=True)

# F5 audit: baselines for the vehicle final test (pool_fresh2), run in the clone repo_veh, read-only on the committed bundles.
# Usage: python veh_baselines.py <case|all> [nproc]
import os, sys, time, pickle, json
os.environ["VEH_LF"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
import os; REPO = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
HERE = os.path.join(REPO, "dixit_vehicle"); sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, HERE)
OUTD = os.path.dirname(os.path.abspath(__file__))
import numpy as np, torch, multiprocessing as mp
from importlib import import_module
CASES = [f"milp_v{v}" for v in (0, 1, 2, 4, 6, 7, 8, 9)] + ["minlp"]
SEEDS = (0, 1, 2)


def solve_ws(C, D, work, z0=None, lb=None, ub=None):
    """short solve (no cuts) with optional warm start z0 and z bounds; replicates 12_fixcuts_stage1.milp/minlp."""
    import gurobipy as gp
    from gurobipy import GRB
    S1 = import_module("12_fixcuts_stage1")
    if C["kind"] == "milp":
        from dixit_vehicle import T, SW, TAU, E0, EMAX, PMAX
        V = C["veh"]; m = gp.Model(env=S1.H.ENV); m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
        Pb = m.addVars(T, lb=-GRB.INFINITY); Pe = m.addVars(T, lb=0, ub=PMAX); E = m.addVars(T+1, lb=0, ub=EMAX)
        z = m.addVars(T, lb=0, ub=SW, vtype=GRB.INTEGER); m.addConstr(E[0] == E0)
        for i in range(T):
            m.addConstr(E[i+1] == E[i] - TAU*Pb[i]); m.addConstr(Pb[i] + Pe[i] >= D[i]); m.addConstr(Pe[i] <= z[i]*PMAX/SW)
        cost = (-V.cT*(EMAX - E[T]) + gp.quicksum(float(V.cE[i])*E[i] for i in range(T)) + gp.quicksum(float(V.cP[i])*Pe[i] + float(V.cZ[i])*z[i] for i in range(T)))
        Tn = T
    else:
        VEH = C["veh"]; Tn, S = VEH["T"], VEH["S_MODES"]; AL = np.array(VEH["alpha"]); BE = np.array(VEH["beta"])
        m = gp.Model(env=S1.H.ENV); m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
        E = m.addVars(Tn+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(Tn, lb=0, ub=VEH["P_max"]); Pb = m.addVars(Tn, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
        z = m.addVars(Tn, lb=0, ub=S, vtype=GRB.INTEGER); m.addConstr(E[0] == VEH["E_init"])
        for t in range(Tn):
            m.addQConstr(E[t+1] - E[t] + VEH["tau"]*Pb[t] + VEH["gamma"]*Pb[t]*Pb[t] <= 0)
            m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t] + Pe[t] >= D[t])
        cost = gp.quicksum(AL[t]*Pe[t]*Pe[t] + BE[t]*z[t] for t in range(Tn)) + VEH["eta"]*(VEH["E_max"] - E[Tn])
    if lb is not None:
        for t in range(Tn): z[t].lb, z[t].ub = float(lb[t]), float(ub[t])
    if z0 is not None:
        for t in range(Tn): z[t].Start = float(z0[t])
    m.setObjective(cost, GRB.MINIMIZE); t0 = time.time(); m.optimize()
    if m.SolCount == 0: return None
    return float(cost.getValue()), np.array([round(z[t].X) for t in range(Tn)], float), time.time() - t0


def price(C, D, z):
    p = C["solve"](D, 1e6, relax=True, bnd=(z, z))
    return None if p is None else (p[0], np.asarray(p[2], float))


def scored(C, D, z, secs, fb=False):
    if z is None: return None
    t0 = time.time(); p = price(C, D, z)
    if p is None: return None
    return dict(cost=p[0], z=np.asarray(z, float), Pe=p[1], secs=secs + time.time() - t0, fallback=fb)


def case_job(name):
    torch.set_num_threads(1)
    F = import_module("16_vehicle_final"); S1 = import_module("12_fixcuts_stage1"); L = import_module("18_lf_teacher")
    C = F.load_case(name); T = C["D"].shape[1]; SW = C["SW"]
    te = C["te"]; tid = [i for i in te if i in C["REF"] and C["REF"][i]["ok"]]; exc = [i for i in te if i in C["REF"] and not C["REF"][i]["ok"]]
    allt = tid + exc
    out = dict(name=name, tid=tid, exc=exc, ref={i: C["REF"][i] for i in allt}, arms={}, info={})
    X = torch.tensor(C["D"], dtype=torch.float32)
    t0 = time.time()
    # ---------- seed-independent baselines
    for W in (0.01, 0.02, 0.05, 0.2):
        R = {}
        for i in allt:
            r = solve_ws(C, C["D"][i], W)
            R[i] = None if r is None else scored(C, C["D"][i], r[1], r[2])
        out["arms"][f"nocut W={W}"] = {0: R}
    R = {}
    for i in allt:
        r = L.lf_target(C, C["D"][i]); R[i] = None if r is None else dict(cost=r["cost"], z=np.array(r["z"], float), Pe=np.array(r["Pe"]), secs=r["secs"], fallback=False)
    out["arms"]["LF teacher procedure (no learning)"] = {0: R}
    # nearest-neighbour lookup (in D space) of training targets
    trk = [i for i in C["tr"] + C["ex"] if C["TG"].get(i) is not None]
    trl = [i for i in C["tr"] if i in C["REF"] and C["REF"][i]["ok"]]
    for lab, pool_, Z in (("NN lookup of our label-free targets (400/444)", trk, lambda j: np.array(C["TG"][j]["z"], float)),
                          ("NN lookup of the proxy's reference labels", trl, lambda j: C["REF"][j]["z"])):
        Dp = C["D"][pool_]; R = {}
        for i in allt:
            ts = time.time(); j = pool_[int(np.argmin(((Dp - C["D"][i])**2).sum(1)))]
            R[i] = scored(C, C["D"][i], Z(j), time.time() - ts)
        out["arms"][lab] = {0: R}
    out["info"]["n_trk"] = len(trk); out["info"]["n_trl"] = len(trl)
    print(f"[{name}] seed-independent arms done [{time.time()-t0:.0f}s]", flush=True)
    # ---------- per-seed arms
    for s in SEEDS:
        B = torch.load(os.path.join(F.OUT, f"{name}_s{s}.pt"), weights_only=False); Xn = (X - B["mu"])/B["sg"]
        M = F.Method(C, Xn, B["cut_state"], B["c"], B["gen_state"], s)
        out["arms"].setdefault("OURS (deployed)", {})[s] = {i: M.deploy(i) for i in allt}
        Mi = F.Method(C, Xn, B["cut_state"], B["c"], None, s)
        out["arms"].setdefault("ours: layer-trained integer cuts only (rows off)", {})[s] = {i: Mi.deploy(i) for i in allt}
        # imitation warm start (as train_job: tr + extra with label-free targets, seed s)
        warm = S1.cutnet(Xn[trk], np.stack([np.array(C["TG"][i]["z"], float) for i in trk]), seed=s)
        Mw = F.Method(C, Xn, warm.state_dict(), B["c"], None, s)
        va = C["va"]; SCv = {i: abs(C["solve"](C["D"][i], 1e6, relax=True)[0]) for i in va}
        Rv = {i: Mw.deploy(i) for i in va}; v0 = float(np.mean([Rv[i]["cost"]/SCv[i] if Rv[i] else 10.0 for i in va]))
        out["info"][f"warm_check_s{s}"] = (v0, B["val_scores"]["imitation"])
        out["arms"].setdefault("ours: imitation-only integer cuts (no layer, no rows)", {})[s] = {i: Mw.deploy(i) for i in allt}
        # imitation cut-net on the ORIGINAL training profiles only (data-matched to the proxy's split), label-free targets, c re-selected label-free
        trk0 = [i for i in C["tr"] if C["TG"].get(i) is not None]
        w0 = S1.cutnet(Xn[trk0], np.stack([np.array(C["TG"][i]["z"], float) for i in trk0]), seed=s)
        M0 = F.Method(C, Xn, w0.state_dict(), F.CS[0], None, s); sc = {}
        for c in F.CS:
            M0.c = c; Rv = {i: M0.deploy(i) for i in va}; sc[c] = float(np.mean([Rv[i]["cost"]/SCv[i] if Rv[i] else 10.0 for i in va]))
        M0.c = min(F.CS, key=lambda c: sc[c])
        out["arms"].setdefault("ours-imitation on original train profiles only (100/144, label-free)", {})[s] = {i: M0.deploy(i) for i in allt}
        # SUPERVISED cut-net: same architecture/deployment, trained on the proxy's REFERENCE labels (tr with converged refs), c label-free
        ws = S1.cutnet(Xn[trl], np.stack([C["REF"][i]["z"] for i in trl]), seed=s)
        Ms = F.Method(C, Xn, ws.state_dict(), F.CS[0], None, s); sc = {}
        for c in F.CS:
            Ms.c = c; Rv = {i: Ms.deploy(i) for i in va}; sc[c] = float(np.mean([Rv[i]["cost"]/SCv[i] if Rv[i] else 10.0 for i in va]))
        Ms.c = min(F.CS, key=lambda c: sc[c]); out["info"][f"sup_c_s{s}"] = Ms.c
        out["arms"].setdefault("SUPERVISED cut-net (reference labels) + same short solve", {})[s] = {i: Ms.deploy(i) for i in allt}
        # proxy (reproduce) + proxy-guided short solves
        pm = F.proxy_net(T, B["proxy_cfg"], B["proxy_state"])
        out["arms"].setdefault("PROXY (deployed)", {})[s] = {i: F.proxy_deploy(C, pm, Xn, i) for i in allt}
        Rws, Rbd = {}, {}
        for i in allt:
            ts = time.time()
            with torch.no_grad(): zf = np.clip(pm(Xn[[i]])[0].numpy(), 0, SW)
            z0 = np.round(zf); r = solve_ws(C, C["D"][i], 0.01, z0=z0)
            Rws[i] = None if r is None else scored(C, C["D"][i], r[1], time.time() - ts)
            ts = time.time(); lo, hi = np.floor(zf), np.ceil(zf); r = solve_ws(C, C["D"][i], 0.01, lb=lo, ub=hi)
            fb = r is None
            if fb: r = solve_ws(C, C["D"][i], 0.01)
            Rbd[i] = None if r is None else scored(C, C["D"][i], r[1], time.time() - ts, fb)
        out["arms"].setdefault("proxy as warm start + W=0.01 solve", {})[s] = Rws
        out["arms"].setdefault("proxy band [floor,ceil] + W=0.01 solve", {})[s] = Rbd
        print(f"[{name}] seed {s} done [{time.time()-t0:.0f}s] warm check {out['info'][f'warm_check_s{s}']}", flush=True)
    pickle.dump(out, open(os.path.join(OUTD, f"bl_{name}.pkl"), "wb"))
    return name


if __name__ == "__main__":
    which = CASES if sys.argv[1] == "all" else sys.argv[1].split(",")
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    with mp.get_context("spawn").Pool(nproc) as pool:
        for r in pool.imap_unordered(case_job, which): print("[done]", r, flush=True)

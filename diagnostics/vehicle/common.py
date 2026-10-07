"""Shared setup for the vehicle diagnostics. Read-only on the repo.
Mirrors vehicle/02_train.py exactly (pool, splits, scaling, anchor, qz_restore)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
import numpy as np, torch
torch.set_num_threads(1)
from src_hv import load_pool, splits, HVRelax, deploy, round_z
from cutnet_hv import CutNetHV, rows_torch
from diffhv import DiffHV, soft_round_int
import gurobipy as gp
from gurobipy import GRB

ENV = gp.Env(empty=True); ENV.setParam("OutputFlag", 0); ENV.setParam("Threads", 1); ENV.start()
VEH, D, REF = load_pool(os.path.join(REPO, "vehicle/results/pool/pool_*.json"))
T, S = VEH["T"], VEH["S_MODES"]
te, va, tr = splits(len(REF))
te = [i for i in te if REF[i]]; va = [i for i in va if REF[i]]; tr = [i for i in tr if REF[i]]
X = torch.tensor(D/D.mean(), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg


def make_relax(kcuts=2):
    return HVRelax(VEH, n_cuts=2*kcuts)


def anchors(R, ids):
    W, SCALE, RX = {}, {}, {}
    for i in ids:
        r = R.solve(D[i])
        W[i] = np.concatenate([r["E"], r["Peng"], r["Pbatt"], r["z"]])
        SCALE[i] = float(r["cost"]); RX[i] = r
    return W, SCALE, RX


def qz_restore(Dp, z_frac, tl=60.):
    """Copy of vehicle/02_train.py:qz_restore (Threads=1)."""
    m = gp.Model(env=ENV); m.Params.TimeLimit = tl
    E = m.addVars(T+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(T, lb=0, ub=VEH["P_max"])
    Pb = m.addVars(T, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(T, lb=0, ub=S, vtype=GRB.INTEGER); w = m.addVars(T, lb=0)
    m.addConstr(E[0] == VEH["E_init"])
    for t in range(T):
        m.addQConstr(E[t+1]-E[t]+VEH["tau"]*Pb[t]+VEH["gamma"]*Pb[t]*Pb[t] <= 0)
        m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S))
        m.addConstr(Pb[t] + Pe[t] >= Dp[t])
        m.addConstr(w[t] >= z[t] - float(z_frac[t]))
        m.addConstr(w[t] >= float(z_frac[t]) - z[t])
    m.setObjective(gp.quicksum(w[t] for t in range(T)), GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0: return None
    return np.array([round(z[t].X) for t in range(T)], float)


def net_rows(net, ids, W):
    WR = torch.tensor(np.stack([W[i] for i in ids]), dtype=torch.float32)
    aE, ape, apb, az, az2, d, d2, th = net(Xn[ids])
    A, b = rows_torch(aE, ape, apb, az, az2, d, d2, WR)
    return A, b, d, d2


def wvec(r):
    return np.concatenate([r["E"], r["Peng"], r["Pbatt"], r["z"]])


def evaluate(R, ids, W, SCALE, A=None, b=None, DD0=None, DD5=None):
    """Per-instance diagnostics for one set of cut rows (None = uncut).
    Labels (REF) are used ONLY to score, never to choose anything."""
    rows = []
    for k, i in enumerate(ids):
        Ai = None if A is None else A[k]; bi = None if b is None else b[k]
        r1 = R.solve(D[i], A=Ai, b=bi, cut_cap=0.0)
        rec = dict(i=i)
        if r1 is None:
            rec["fail"] = "layer1"; rows.append(rec); continue
        zf = r1["z"]; w1 = wvec(r1)
        rec["relax_cost"] = r1["cost"]/SCALE[i]
        rec["nfrac"] = int((np.abs(zf-np.round(zf)) > 1e-4).sum())
        rec["fracdist"] = float(np.abs(zf-np.round(zf)).mean())
        rec["int_pen"] = float((np.sin(np.pi*zf)**2).sum())
        if Ai is not None:
            sl = bi - Ai @ w1
            rec["slack"] = sl.tolist()
        zq = qz_restore(D[i], zf)
        pr = R.solve(D[i], z_lo=zq, z_hi=zq) if zq is not None else None
        if pr is None:
            rec["fail"] = "deploy"; rows.append(rec); continue
        zr = np.array(REF[i]["z"], float)
        rec["dep_cost"] = pr["cost"]/SCALE[i]
        rec["gap"] = 100*(pr["cost"]-REF[i]["cost"])/abs(REF[i]["cost"])
        rec["ham"] = int((zq != zr).sum()); rec["l1"] = float(np.abs(zq-zr).sum())
        rec["zq"] = zq.tolist(); rec["zf"] = zf.tolist()
        if DD0 is not None and Ai is not None:
            L0, gA0, gb0, inf0 = DD0(D[i], Ai, bi, SCALE[i])
            L5, gA5, gb5, inf5 = DD5(D[i], Ai, bi, SCALE[i])
            if gA0 is not None and gA5 is not None:
                rec["L_cost"] = L0; rec["L_tot"] = L5
                rec["drop"] = inf5["drop"]
                rec["g_cost"] = float(np.linalg.norm(np.r_[gA0.ravel(), gb0]))
                rec["g_int"] = float(np.linalg.norm(np.r_[(gA5-gA0).ravel(), gb5-gb0]))
                rec["g_cos"] = float(np.dot(np.r_[gA0.ravel(), gb0], np.r_[(gA5-gA0).ravel(), gb5-gb0])
                                     / (rec["g_cost"]*rec["g_int"] + 1e-30))
                # soft-rounded z the surrogate trains on vs the z actually deployed
                th = np.clip(0.5 - inf5["drop"], 0.02, 0.98)
                zs, s = soft_round_int(zf, th, DD5.tau, S, DD5.zc)
                rec["soft_vs_qz_l1"] = float(np.abs(zs - zq).sum())
                rec["soft_vs_round_l1"] = float(np.abs(np.round(zs) - zq).sum())
                rec["unsat"] = int((s*(1-s) > 1e-3).sum())
        rows.append(rec)
    return rows


def summarise(rows, tag=""):
    ok = [r for r in rows if "fail" not in r]
    f = lambda k: np.array([r[k] for r in ok if k in r], float)
    out = dict(n=len(rows), fails=len(rows)-len(ok))
    for k in ("relax_cost", "dep_cost", "gap", "ham", "l1", "nfrac", "fracdist", "int_pen",
              "L_cost", "L_tot", "g_cost", "g_int", "g_cos", "soft_vs_qz_l1", "soft_vs_round_l1", "unsat"):
        v = f(k)
        if len(v): out[k] = float(v.mean())
    if ok and "gap" in ok[0]:
        out["absgap"] = float(np.abs(f("gap")).mean())
        out["exact_match"] = int((f("ham") == 0).sum())
    sl = [s for r in ok if "slack" in r for s in r["slack"]]
    if sl:
        sl = np.array(sl)
        out["rows_active_frac"] = float((sl < 1e-5).mean()); out["slack_mean"] = float(sl.mean())
        out["slack_max"] = float(sl.max())
    if tag:
        print(f"{tag:22s} " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in out.items()), flush=True)
    return out

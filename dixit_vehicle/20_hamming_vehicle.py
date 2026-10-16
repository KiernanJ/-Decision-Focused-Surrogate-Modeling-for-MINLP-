# %% HAMMING-BALL INTEGER CUT for the HYBRID VEHICLE (2026-10-16) -- VALIDATION ONLY, one seed, exploratory ("to get the idea"). NO per-step integer bounds, NO general rows.
#    The only learned cut is ONE general integer row family: the L1 ball  sum_t |z_t - zhat_t| <= r  (zhat = learned centre in [0, SW], r chosen label-free), written with auxiliary
#    variables w_t >= |z_t - zhat_t|; SOFT (slack priced 1e3, as the final vehicle rows). Same label-free targets (VEH_LF=1: own W=0.01 solves + 300 extra profiles), same validation profiles,
#    same deployment (one short solve W=0.01 with the cut, fallback = same solve without the cut, scored by the fixed-z re-solve) as the final vehicle method.
#    Centre net: MLP T -> 128 -> 128 -> T, zhat = SW*sigmoid(out); warm start = MSE to the own targets; then fine-tuned THROUGH THE EXACT LAYER (tracking ||z_relax - z_own||^2/T + 10*validity
#    relu(L1(z_own, zhat) - r)^2), FD gate first; r from RGRID and the epoch among TRAINED epochs by the LABEL-FREE score (deployed cost / relaxation cost on ALL 24 validation profiles).
#    Report only: four metrics vs the exact reference on the validation profiles that have one, next to the CURRENT final method's own bundle (results/final_lf/<case>_s0.pt, same profiles).
#    Usage: python 20_hamming_vehicle.py milp_v0     |     python 20_hamming_vehicle.py minlp      (env SEED, default 0)
import os, sys, time
os.environ["VEH_LF"] = "1"                                              # label-free targets + extra profiles, as the final vehicle method
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src")); sys.path.insert(0, HERE)
import numpy as np, torch, torch.nn as nn, cvxpy as cp, gurobipy as gp
from gurobipy import GRB
from importlib import import_module
NAME = sys.argv[1]; SEED = int(os.environ.get("SEED", 0))
W, EPOCHS, BATCH, LR, LAM, PRICE, RGRID = 0.01, 10, 24, 1e-4, 10.0, 1e3, (1, 2, 3, 5, 8, 12)
F = import_module("16_vehicle_final"); S1 = import_module("12_fixcuts_stage1"); H = S1.H
from conelayer import ConeLayer


# %% the L1-ball layer (cvxpy relaxation + implicit differentiation w.r.t. the centre)
class HamLayer:
    def __init__(self, kind, veh, gamma=1.0):
        E, Pe, Pb, z = None, None, None, None
        if kind == "milp":
            from dixit_vehicle import T, SW, TAU, E0, EMAX, PMAX
            E = cp.Variable(T+1); Pe = cp.Variable(T); Pb = cp.Variable(T); z = cp.Variable(T); self.D = cp.Parameter(T)
            C = [E[0] == E0, E >= 0, E <= EMAX, Pe >= 0, Pe <= PMAX, z >= 0, z <= SW, E[1:] == E[:-1] - TAU*Pb, Pb + Pe >= self.D, Pe <= z*PMAX/SW]
            cost = (-veh.cT*(EMAX - E[T]) + veh.cE @ E[:T] + veh.cP @ Pe + veh.cZ @ z)
        else:
            T, S = veh["T"], veh["S_MODES"]; SW = S; AL = np.array(veh["alpha"]); BE = np.array(veh["beta"])
            E = cp.Variable(T+1); Pe = cp.Variable(T); Pb = cp.Variable(T); z = cp.Variable(T); self.D = cp.Parameter(T)
            C = [E[0] == veh["E_init"], E >= 0, E <= veh["E_max"], Pe >= 0, Pe <= veh["P_max"], Pb >= -veh["P_batt_max"], Pb <= veh["P_batt_max"], z >= 0, z <= S,
                 E[1:] - E[:-1] + veh["tau"]*Pb + veh["gamma"]*cp.square(Pb) <= 0, Pe <= z*(veh["P_max"]/S), Pb + Pe >= self.D]
            cost = cp.sum(cp.multiply(AL, cp.square(Pe))) + BE @ z + veh["eta"]*(veh["E_max"] - E[T])
        self.T, self.SW = T, SW
        self.cen = cp.Parameter(T); self.rad = cp.Parameter(nonneg=True); w = cp.Variable(T); s = cp.Variable(nonneg=True)
        C += [w >= z - self.cen, w >= self.cen - z, cp.sum(w) <= self.rad + s]                 # L1 ball, soft
        self.prob = cp.Problem(cp.Minimize(cost + PRICE*s + gamma*cp.sum_squares(z - SW/2)), C)
        self.z, self.Pe = z, Pe; self.L = ConeLayer(self.prob, [self.cen], [self.Pe, self.z], reg=1e-10); self.built = False

    def track(self, D, cen, rad, z_own, want_grad=True):
        self.D.value = np.asarray(D, float); self.cen.value = np.asarray(cen, float); self.rad.value = float(rad)
        try: self.prob.solve(solver=cp.CLARABEL, warm_start=False)
        except Exception: return None
        if self.z.value is None: return None
        loss = float(((self.z.value - z_own)**2).sum()/self.T)
        if not want_grad: return (loss,)
        vals = [np.asarray(cen, float)]
        if not self.built: self.L.build_structure(vals); self.sl = self.L.var_slices(); self.built = True
        x, st = self.L.forward(vals, tol=1e-10)
        if x is None: return None
        zz = x[self.sl[self.z.id]]; dx = np.zeros(x.size); dx[self.sl[self.z.id]] = 2.0*(zz - z_own)/self.T
        try: gr = self.L.vjp(st, dx)
        except RuntimeError: return None
        return (float(((zz - z_own)**2).sum()/self.T), np.asarray(gr[0]).reshape(self.T))


# %% deployed solves with the L1-ball cut (copies of 12_fixcuts_stage1.milp / minlp + the ball)
def milp_h(V, D, work, zhat, r):
    from dixit_vehicle import T, SW, TAU, E0, EMAX, PMAX
    m = gp.Model(env=H.ENV); m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
    Pb = m.addVars(T, lb=-GRB.INFINITY); Pe = m.addVars(T, lb=0, ub=PMAX); E = m.addVars(T+1, lb=0, ub=EMAX); z = m.addVars(T, lb=0, ub=SW, vtype=GRB.INTEGER); m.addConstr(E[0] == E0)
    for i in range(T): m.addConstr(E[i+1] == E[i] - TAU*Pb[i]); m.addConstr(Pb[i] + Pe[i] >= D[i]); m.addConstr(Pe[i] <= z[i]*PMAX/SW)
    w = m.addVars(T, lb=0); s = m.addVar(lb=0)
    for i in range(T): m.addConstr(w[i] >= z[i] - float(zhat[i])); m.addConstr(w[i] >= float(zhat[i]) - z[i])
    m.addConstr(w.sum() <= float(r) + s)
    cost = (-V.cT*(EMAX - E[T]) + gp.quicksum(float(V.cE[i])*E[i] for i in range(T)) + gp.quicksum(float(V.cP[i])*Pe[i] + float(V.cZ[i])*z[i] for i in range(T)))
    m.setObjective(cost + PRICE*s, GRB.MINIMIZE); m.optimize()
    if m.SolCount == 0: return None
    return float(cost.getValue()), np.array([round(z[i].X) for i in range(T)]), np.array([Pe[i].X for i in range(T)])


def minlp_h(VEH, Dp, work, zhat, r):
    Tn, S = VEH["T"], VEH["S_MODES"]; AL = np.array(VEH["alpha"]); BE = np.array(VEH["beta"])
    m = gp.Model(env=H.ENV); m.Params.TimeLimit = 1e6; m.Params.WorkLimit = work; m.Params.MIPGap = 1e-4
    E = m.addVars(Tn+1, lb=0, ub=VEH["E_max"]); Pe = m.addVars(Tn, lb=0, ub=VEH["P_max"]); Pb = m.addVars(Tn, lb=-VEH["P_batt_max"], ub=VEH["P_batt_max"])
    z = m.addVars(Tn, lb=0, ub=S, vtype=GRB.INTEGER); m.addConstr(E[0] == VEH["E_init"])
    for t in range(Tn):
        m.addQConstr(E[t+1] - E[t] + VEH["tau"]*Pb[t] + VEH["gamma"]*Pb[t]*Pb[t] <= 0); m.addConstr(Pe[t] <= z[t]*(VEH["P_max"]/S)); m.addConstr(Pb[t] + Pe[t] >= Dp[t])
    w = m.addVars(Tn, lb=0); s = m.addVar(lb=0)
    for t in range(Tn): m.addConstr(w[t] >= z[t] - float(zhat[t])); m.addConstr(w[t] >= float(zhat[t]) - z[t])
    m.addConstr(w.sum() <= float(r) + s)
    obj = gp.quicksum(AL[t]*Pe[t]*Pe[t] + BE[t]*z[t] for t in range(Tn)) + VEH["eta"]*(VEH["E_max"] - E[Tn])
    m.setObjective(obj + PRICE*s, GRB.MINIMIZE); m.optimize()
    if m.SolCount == 0: return None
    return float(obj.getValue()), np.array([round(z[t].X) for t in range(Tn)]), np.array([Pe[t].X for t in range(Tn)])


def ham_deploy(C, i, zhat, r):
    """ONE short solve with the L1-ball cut (fallback: the same solve without it), scored by the fixed-z re-solve = Method.deploy of the final vehicle method."""
    t0 = time.time(); D = C["D"][i]
    res = (milp_h if C["kind"] == "milp" else minlp_h)(C["veh"], D, W, zhat, r); fb = res is None
    if fb: res = C["solve"](D, W)
    if res is None: return None
    z = np.asarray(res[1], float); p = C["solve"](D, 1e6, relax=True, bnd=(z, z))
    if p is None: return None
    return dict(cost=p[0], z=z, Pe=p[2], secs=time.time() - t0, fallback=fb)


# %% main
def main():
    t0 = time.time(); torch.set_num_threads(1)
    C = F.load_case(NAME); Dall = C["D"]; T = Dall.shape[1]; tr, va, SW = C["tr"], C["va"], C["SW"]
    X = torch.tensor(Dall, dtype=torch.float32); mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    trk = [i for i in tr + C.get("ex", []) if C["TG"].get(i) is not None]; ZO = {i: np.array(C["TG"][i]["z"], float) for i in trk}
    assert not set(trk) & set(va) and not set(trk) & set(C["te"]), "training profiles must be disjoint from validation and test"
    SCv = {i: abs(C["solve"](Dall[i], 1e6, relax=True)[0]) for i in va}              # label-free normaliser (relaxation), ALL validation profiles
    print(f"[config] {NAME} SEED={SEED}: {len(trk)} training targets (label-free), {len(va)} validation profiles, W={W}, soft L1 ball (price {PRICE:.0e}), NO per-step bounds, NO general rows", flush=True)
    torch.manual_seed(SEED)
    net = nn.Sequential(nn.Linear(T, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, T))
    cen = lambda idx, grad=False: (SW*torch.sigmoid(net(Xn[idx]))) if grad else SW*torch.sigmoid(net(Xn[idx])).detach()
    Zt = torch.tensor(np.stack([ZO[i] for i in trk]), dtype=torch.float32); o = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(1500): o.zero_grad(); l = ((SW*torch.sigmoid(net(Xn[trk])) - Zt)**2).mean(); l.backward(); o.step()      # imitation warm start of the CENTRE (MSE to own targets)
    def score(r):
        with torch.no_grad(): Z = (SW*torch.sigmoid(net(Xn[va]))).numpy().astype(float)
        R = {i: ham_deploy(C, i, Z[k], r) for k, i in enumerate(va)}
        return float(np.mean([R[i]["cost"]/SCv[i] if R[i] else 10.0 for i in va])), R, sum(bool(R[i]["fallback"]) for i in va if R[i])
    rs = {}
    for r_ in RGRID: s_ = score(r_); rs[r_] = s_[0]; print(f"[r] {r_}: label-free val {s_[0]:.6f} (fallbacks {s_[2]})  [{time.time()-t0:.0f}s]", flush=True)
    r = float(min(RGRID, key=lambda k: rs[k])); print(f"[r] selected r = {r:.0f} (label-free)", flush=True)
    # ---- FD gate of the layer gradient w.r.t. the centre, at the warm start
    L = HamLayer(C["kind"], C["veh"]); rng = np.random.default_rng(SEED); gate = []
    for i in trk[:3]:
        c0 = cen([i]).numpy()[0].astype(float); R0 = L.track(Dall[i], c0, r, ZO[i], True)
        if R0 is None: print(f"[gate] inst {i}: failed", flush=True); continue
        fd, an = [], []
        for _ in range(8):
            d = rng.standard_normal(T); d /= np.linalg.norm(d); h = 1e-2
            a, b = L.track(Dall[i], c0 + h*d, r, ZO[i], False), L.track(Dall[i], c0 - h*d, r, ZO[i], False)
            if a is None or b is None: continue
            fd.append((a[0] - b[0])/(2*h)); an.append(float(R0[1] @ d))
        ca = np.corrcoef(fd, an)[0, 1] if len(fd) > 2 else np.nan; gate.append(ca); print(f"[gate] inst {i}: {len(fd)}/8 directions | analytic vs FD corr {ca:.4f} | |FD| {np.abs(fd).mean():.2e}", flush=True)
    ok_gate = bool(gate) and all(np.isfinite(g) and g >= 0.95 for g in gate)
    print(f"[gate] {'PASSED -> train' if ok_gate else 'FAILED -> NO TRAINING (diagnose; not deployable as learned)'}", flush=True)
    hist, states, trained = [], {}, False
    s_init = score(r); print(f"[val] label-free: initial centre {s_init[0]:.6f} (fallbacks {s_init[2]})", flush=True)
    if ok_gate:
        opt = torch.optim.Adam(net.parameters(), lr=LR); trained = True
        for ep in range(EPOCHS):
            order = rng.permutation(trk); Lt, Lv, nf = [], [], 0
            for s0 in range(0, len(order), BATCH):
                b = [int(i) for i in order[s0:s0 + BATCH]]; Z_t = SW*torch.sigmoid(net(Xn[b])); Zn = Z_t.detach().numpy().astype(float); GZ, ok = np.zeros_like(Zn), 0
                for k, i in enumerate(b):
                    rr = L.track(Dall[i], Zn[k], r, ZO[i], True)
                    if rr is None or len(rr) < 2: nf += 1; continue
                    Lt.append(rr[0]); GZ[k] = rr[1]; ok += 1
                if ok == 0: continue
                zo = torch.tensor(np.stack([ZO[i] for i in b]), dtype=torch.float32)
                lval = LAM*(torch.relu((Z_t - zo).abs().sum(1) - r)**2).mean(); Lv.append(float(lval))        # the own target must lie inside the ball
                opt.zero_grad(); (((Z_t*torch.tensor(GZ, dtype=torch.float32)).sum()).div(ok) + lval).backward(); opt.step()
            states[ep] = {k: t.clone() for k, t in net.state_dict().items()}; se = score(r); hist.append((ep, se[0], se[2]))
            print(f"[ep {ep}] tracking {np.mean(Lt):.5f} validity {np.mean(Lv):.5f} (failed {nf}) | label-free val {se[0]:.6f} (fallbacks {se[2]})  [{time.time()-t0:.0f}s]", flush=True)
        pick = min(hist, key=lambda t: t[1]); net.load_state_dict(states[pick[0]]); print(f"[select] label-free: epoch {pick[0]} (score {pick[1]:.6f})", flush=True)
    else: print("[select] NOT TRAINED -> reported with the warm-start centre only", flush=True)
    # ---- REPORT ONLY: four metrics vs the exact reference on the validation profiles with a reference; the current final method's own bundle on the SAME profiles
    B = torch.load(os.path.join(HERE, "results", "final_lf", f"{NAME}_s0.pt"), weights_only=False); vr = B["val_ids"]
    fin = score(r); f = lambda v: f"{np.nanmean(v[:,0]):.4f}% | {np.nanmedian(v[:,0]):.4f}% | {np.nanmean(v[:,1]):.2f}% | {np.nanmean(v[:,2]):.2f}% | {1000*np.nanmedian(v[:,3]):.1f} ms | fallbacks {int(np.nansum(v[:,4]))}"
    ours = np.array([F.metrics(fin[1].get(i), C["REF"][i]) for i in vr])
    print(f"[REPORT] {NAME} VALIDATION ({len(vr)} profiles with references) gap mean | median | disc | cont | time | fallbacks:", flush=True)
    print(f"   Hamming-ball row (r={r:.0f}, {'trained' if trained else 'warm start only'}, label-free {fin[0]:.6f}): {f(ours)}", flush=True)
    print(f"   current final method (per-step integer cuts + 2 general rows; label-free {B['val_scores']['gen_best']:.6f}): {f(B['val_ours'])}", flush=True)
    print(f"   supervised NN proxy (same profiles): {f(B['val_proxy'])}   [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

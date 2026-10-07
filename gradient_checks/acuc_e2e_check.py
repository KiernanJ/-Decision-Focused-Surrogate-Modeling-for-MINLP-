# %% End-to-end gradient check in the TRAINING regime (audit 2026-10-06):
#    loss = DiffDeploy (layer 1 -> soft rounding -> layer 2) at rho = 1e6, at the
#    trained network's (V, A, b) on validation instances. Exact directional
#    derivatives vs central finite differences whose convergence is checked
#    (h = 1e-3 vs 3e-4). Separately for the cut (A, b jointly) and for V.
#    Usage: python acuc_e2e_check.py case118   (writes out/e2e_<case>.json)
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
CASE = sys.argv[1]; os.environ["QCAC_CASE"] = CASE; os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
import numpy as np, torch
from pipeline import grid, RESERVE, set_cuts
from socp import Socp
from cutnet_k import CutNetK, rows_torch
from framework import DiffDeploy
import data2
HERE = os.path.dirname(os.path.abspath(__file__))
set_cuts(4); g, nl = grid(CASE); S = Socp(g, nl, n_cuts=4)
INST = data2.instances(g, CASE); te, va, tr = data2.split()
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
net = CutNetK(g.n_bus, g.n_gen, k_rows=2, v_scale=0.5, hidden=256, seed=0, learn_thr=False)
net.load_state_dict(torch.load(f"{HERE}/../acuc/checkpoints2/{CASE}_s0.pt", weights_only=False)["net"])
DD = DiffDeploy(S, g, nl, reserve=RESERVE, tau=0.08, thr=0.5, rho=1e6, reg=1e-10, reg2=1e-10)
rng = np.random.default_rng(0)
rows = []
for j in va[:6]:
    pd_, qd_ = INST[j]
    with torch.no_grad():
        Vr_t, Vi_t, at, alt, a2t, dt, d2t, _ = net(Xn[[j]])
    Vr = Vr_t[0].numpy().astype(float); Vi = Vi_t[0].numpy().astype(float)
    r0 = S.solve(pd_, qd_, Vr, Vi, rho=1e6)
    sc = float(S.solve(pd_, qd_, np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e6)["cost"])
    with torch.no_grad():
        A_t, b_t = rows_torch(at, alt, a2t, dt, d2t,
                              torch.tensor(np.concatenate([r0["pg"], r0["u"]])[None], dtype=torch.float32))
    A = A_t[0].numpy().astype(float); b = b_t[0].numpy().astype(float)
    L, gA, gb, info = DD(pd_, qd_, Vr, Vi, A, b, sc)
    if gA is None:
        rows.append(dict(j=j, fail=info.get("fail"))); continue
    f = lambda Vr_, Vi_, A_, b_: DD(pd_, qd_, Vr_, Vi_, A_, b_, sc, want_grad=False)[0]
    out = dict(j=j)
    for name in ("cut", "V"):
        errs, conv = [], []
        for _ in range(3):
            if name == "cut":
                eA = rng.normal(size=A.shape); eb = rng.normal(size=b.shape)
                nrm = np.sqrt((eA**2).sum() + (eb**2).sum()); eA /= nrm; eb /= nrm
                ex = float((gA*eA).sum() + (gb*eb).sum())
                fd = {h: (f(Vr, Vi, A+h*eA, b+h*eb) - f(Vr, Vi, A-h*eA, b-h*eb))/(2*h) for h in (1e-3, 3e-4)}
            else:
                er = rng.normal(size=Vr.shape); ei = rng.normal(size=Vi.shape)
                nrm = np.sqrt((er**2).sum() + (ei**2).sum()); er /= nrm; ei /= nrm
                ex = float(info["dVr"]@er + info["dVi"]@ei)
                fd = {h: (f(Vr+h*er, Vi+h*ei, A, b) - f(Vr-h*er, Vi-h*ei, A, b))/(2*h) for h in (1e-3, 3e-4)}
            ref = 0.5*(fd[1e-3] + fd[3e-4]); sc_ = max(abs(ref), 1e-12)
            conv.append(abs(fd[1e-3]-fd[3e-4])/sc_ < 0.03)
            errs.append(abs(ex-ref)/sc_)
        out[name] = dict(rel_err=[float(e) for e in errs], ref_converged=[bool(c) for c in conv])
    rows.append(out)
    print(j, {k: (np.round(v["rel_err"], 4).tolist(), v["ref_converged"]) for k, v in out.items() if k != "j"}, flush=True)
for name in ("cut", "V"):
    e = [x for r in rows if name in r for x, c in zip(r[name]["rel_err"], r[name]["ref_converged"]) if c]
    print(f"{CASE} {name}: {len(e)} converged directions, median rel err {np.median(e):.3%}, "
          f"within 10%: {np.mean(np.array(e) < 0.1):.0%}" if e else f"{CASE} {name}: no converged reference")
json.dump(rows, open(f"{HERE}/out/e2e_{CASE}.json", "w"), indent=1)

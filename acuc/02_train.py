# %% Self-supervised training of the learned linearisation point V and the
#    learned cuts for AC unit commitment, by EXACT gradients through the convex
#    relaxation (ConeLayer, regularised sparse LU on the KKT).
#
#    Audit version (2026-10-06). Changes from the release:
#      * ANCHOR AT THE PREDICTED V. b = <A, w_rlx(V_pred)> - depth, with w_rlx the
#        uncut relaxation solved at the network's own V (stop-gradient). The
#        release anchored at FLAT V while layer 1 is solved at the predicted V,
#        so most rows ended up slack (zero gradient) on case118.
#      * the validation-selected model is what gets SAVED (ck["net"]); the last
#        epoch is kept separately as ck["last"]. The release saved the last
#        epoch under "net" and every evaluation script loaded it.
#      * no test split, no reference costs, no labels anywhere in this script.
#      * training rounding threshold 0.5 for both cases (the 0.8 used before
#        came from a sweep over the test split).
#
#    Loss = deployed cost after soft rounding and the priced re-solve, divided
#    by the uncut relaxation cost at flat V (label-free normaliser).
#    Selection = lowest mean validation loss (24 instances, label-free).
import os, sys, time, json, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="case118")
ap.add_argument("--kcuts", type=int, default=2)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--tau", type=float, default=0.08)
ap.add_argument("--init-depth", type=float, default=0.25)
ap.add_argument("--max-depth", type=float, default=2.0)
ap.add_argument("--rho", type=float, default=1e6)
ap.add_argument("--dthr", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--v-scale", type=float, default=0.5)
ap.add_argument("--hidden", type=int, default=256)
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case; os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from pipeline import grid, RESERVE, set_cuts
from socp import Socp
from cutnet_k import CutNetK, rows_torch
from framework import DiffDeploy
import data2

# %% data (instances only -- no references)
REG1 = REG2 = 1e-10
K = a.kcuts; ROWS = 2*K
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
set_cuts(ROWS)
g, nl = grid(a.case); NG, NB = g.n_gen, g.n_bus
S = Socp(g, nl, n_cuts=ROWS)
V1, V0 = np.ones(NB), np.zeros(NB)
INST = data2.instances(g, a.case)
te, va, tr = data2.split()
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
SCALE = {}
for j in tr + va:
    r = S.solve(INST[j][0], INST[j][1], V1, V0, rho=a.rho)
    if r is not None:
        SCALE[j] = float(r["cost"])
nbad = len([j for j in tr + va if j not in SCALE])
tr = [j for j in tr if j in SCALE]; va = [j for j in va if j in SCALE]
steps = a.epochs*max(1, int(np.ceil(len(tr)/a.batch)))
print(f"[{a.case} s{a.seed}] {len(tr)} train / {len(va)} val ({nbad} uncut relaxations failed) | "
      f"{ROWS} cut rows | {steps} steps | lr {a.lr} | dthr {a.dthr}", flush=True)

net = CutNetK(NB, NG, k_rows=K, v_scale=a.v_scale, hidden=a.hidden, seed=a.seed,
              init_depth=a.init_depth, max_depth=a.max_depth, learn_thr=False)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
DD = DiffDeploy(S, g, nl, reserve=RESERVE, tau=a.tau, thr=a.dthr, rho=a.rho, reg=REG1, reg2=REG2)
print(f"[net  ] {net.n_params():,} params", flush=True)


def anchor(ids, Vr, Vi):
    """Uncut relaxation at the PREDICTED V -> w_rlx (constant, no gradient)."""
    W, ok = [], []
    for k, j in enumerate(ids):
        r = S.solve(INST[j][0], INST[j][1], Vr[k], Vi[k], rho=a.rho)
        if r is not None:
            W.append(np.concatenate([r["pg"], r["u"]])); ok.append(k)
    return ok, (torch.tensor(np.stack(W), dtype=torch.float32) if W else None)


def batch_rows(ids, grad=True):
    with torch.set_grad_enabled(grad):
        Vr_t, Vi_t, at, alt, a2t, dt, d2t, _ = net(Xn[ids])
    Vrn = Vr_t.detach().numpy().astype(float); Vin = Vi_t.detach().numpy().astype(float)
    ok, WR = anchor(ids, Vrn, Vin)
    if not ok:
        return None
    with torch.set_grad_enabled(grad):
        A_t, b_t = rows_torch(at[ok], alt[ok], a2t[ok], dt[ok], d2t[ok], WR)
    return ok, Vr_t[ok], Vi_t[ok], A_t, b_t


def val_loss():
    with torch.no_grad():
        out = batch_rows(va, grad=False)
    if out is None:
        return np.nan, 0
    ok, Vr_t, Vi_t, A_t, b_t = out
    L = []
    for k, idx in enumerate(ok):
        j = va[idx]
        l, _, _, _ = DD(INST[j][0], INST[j][1], Vr_t[k].numpy().astype(float),
                        Vi_t[k].numpy().astype(float), A_t[k].numpy().astype(float),
                        b_t[k].numpy().astype(float), SCALE[j], want_grad=False)
        if l is not None:
            L.append(l)
    # failures count as the worst loss seen, never silently dropped
    n_fail = len(va) - len(L)
    if n_fail and L:
        L += [max(L)]*n_fail
    return (float(np.mean(L)) if L else np.nan), n_fail


# %% training loop
t0all = time.time(); hist = []; best = None
for ep in range(a.epochs):
    t0 = time.time(); order = rng.permutation(tr)
    Ls, nok, fails, gnorms = [], 0, {}, []
    for s0 in range(0, len(order), a.batch):
        bidx = [int(j) for j in order[s0:s0+a.batch]]
        out = batch_rows(bidx)
        if out is None:
            continue
        ok, Vr_t, Vi_t, A_t, b_t = out
        An = A_t.detach().numpy().astype(float); bn = b_t.detach().numpy().astype(float)
        Vrn = Vr_t.detach().numpy().astype(float); Vin = Vi_t.detach().numpy().astype(float)
        GA = np.zeros_like(An); GB = np.zeros_like(bn)
        GVr = np.zeros_like(Vrn); GVi = np.zeros_like(Vin)
        got = 0
        fails["anchor"] = fails.get("anchor", 0) + len(bidx) - len(ok)
        for k, idx in enumerate(ok):
            j = bidx[idx]
            L, gA, gb, info = DD(INST[j][0], INST[j][1], Vrn[k], Vin[k], An[k], bn[k], SCALE[j])
            if L is None or gA is None:
                f = info.get("fail", "?"); fails[f] = fails.get(f, 0)+1
                continue
            GA[k], GB[k] = gA, gb
            GVr[k], GVi[k] = info["dVr"], info["dVi"]
            Ls.append(L); got += 1
        if got == 0:
            continue
        nok += got
        opt.zero_grad()
        loss = ((A_t*torch.tensor(GA, dtype=torch.float32)).sum()
                + (b_t*torch.tensor(GB, dtype=torch.float32)).sum()
                + (Vr_t*torch.tensor(GVr, dtype=torch.float32)).sum()
                + (Vi_t*torch.tensor(GVi, dtype=torch.float32)).sum())/got
        loss.backward()
        gnorms.append(float(sum((p.grad**2).sum() for p in net.parameters()
                                if p.grad is not None)**0.5))
        opt.step(); sched.step()
    vl, vf = val_loss()
    if best is None or (np.isfinite(vl) and vl < best[0]):
        best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
    hist.append(dict(ep=ep, loss=float(np.mean(Ls)) if Ls else np.nan, val=vl, val_fail=vf,
                     nok=nok, fails=fails, secs=time.time()-t0))
    print(f"[ep {ep:3d}] loss {np.mean(Ls) if Ls else np.nan:+.5f}  ok {nok}/{len(tr)}  "
          f"val {vl:+.5f} (fail {vf})  |g| {np.mean(gnorms) if gnorms else 0:.2e}  "
          f"{time.time()-t0:6.1f}s  {fails}", flush=True)
    if nok == 0:
        raise SystemExit(f"FATAL: epoch {ep} produced no gradient; failures {fails}")
    os.makedirs("checkpoints2", exist_ok=True)
    torch.save(dict(net=best[1], last=net.state_dict(), best_epoch=best[2], best_val=best[0],
                    hist=hist, args=vars(a)), f"checkpoints2/{a.case}_s{a.seed}.pt")
train_secs = time.time()-t0all
os.makedirs("results/train2", exist_ok=True)
json.dump(dict(case=a.case, seed=a.seed, best_epoch=best[2], best_val=best[0],
               train_secs=train_secs, n_params=net.n_params(), hist=hist, args=vars(a)),
          open(f"results/train2/{a.case}_s{a.seed}.json", "w"), indent=1)
print(f"[done] best epoch {best[2]} val {best[0]:+.5f}  train {train_secs:.0f}s", flush=True)

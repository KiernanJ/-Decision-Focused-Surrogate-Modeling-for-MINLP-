# %% FINITE-DIFFERENCE GATE on dL/dA, dL/db, dL/dthr before training on them.
#    On AC-UC, dL/dthr FAILED this check (9-18% error on four instances,
#    100-160% on two) and had to be abandoned; three wrong diagnoses in that
#    project came from trusting a gradient that merely looked plausible.
#    Directional derivatives with a CONVERGENCE check over several h -- not a
#    single step size, which read pure noise as signal earlier.
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, json
from src_hv import load_pool, splits, HVRelax
from cutnet_hv import CutNetHV, rows_torch
from diffhv import DiffHV
VEH, D, REF = load_pool(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "vehicle", "results", "pool", "pool_*.json")); T = VEH["T"]
R = HVRelax(VEH, n_cuts=4); net = CutNetHV(T, k_rows=2, hidden=16, seed=0)
te, va, tr = splits(len(REF))
X = torch.tensor(D/D.mean(), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
DD = DiffHV(R, VEH, tau=0.08)
rng = np.random.default_rng(0)

def fd_conv(fn, x0, e, hs=(3e-3, 1e-3, 3e-4, 1e-4)):
    prev, best = None, None
    for h in hs:
        lp, lm = fn(x0 + h*e), fn(x0 - h*e)
        if lp is None or lm is None: continue
        fd = (lp - lm)/(2*h)
        if prev is not None and abs(fd-prev) < 0.03*max(abs(fd), 1e-30): best = fd
        prev = fd
    return best if best is not None else prev

rows = []
for i in te[:10]:
    r = R.solve(D[i])
    W = torch.tensor(np.concatenate([r["E"], r["Peng"], r["Pbatt"], r["z"]]), dtype=torch.float32)[None]
    with torch.no_grad():
        hh = net(Xn[[i]]); A, b = rows_torch(*hh[:7], W); th = hh[7]
    A0 = A[0].numpy().astype(float); b0 = b[0].numpy().astype(float)
    th0 = th[0].numpy().astype(float); sc = float(r["cost"])
    L, gA, gb, info = DD(D[i], A0, b0, sc, thr=th0)
    if gA is None: continue
    out = {}
    for name, g, x0, setter in (
        ("dL/dA",   gA.ravel(), A0.ravel(),
         lambda v: DD(D[i], v.reshape(A0.shape), b0, sc, thr=th0, want_grad=False)[0]),
        ("dL/db",   gb,  b0,  lambda v: DD(D[i], A0, v, sc, thr=th0, want_grad=False)[0]),
        ("dL/dthr", info["dthr"], th0,
         lambda v: DD(D[i], A0, b0, sc, thr=v, want_grad=False)[0])):
        e = rng.normal(size=x0.size); e /= np.linalg.norm(e)
        ex = float(g @ e); fd = fd_conv(setter, x0, e)
        out[name] = (abs(fd-ex)/max(abs(ex), 1e-30)) if fd is not None else np.nan
    rows.append(out)
    print(f"  inst {i}: " + "  ".join(f"{k} {v:7.2%}" for k, v in out.items()), flush=True)
print("\n=== finite-difference gate ===")
for k in ("dL/dA", "dL/db", "dL/dthr"):
    v = np.array([r[k] for r in rows]); v = v[np.isfinite(v)]
    print(f"  {k:8s} median {np.median(v):7.2%}  mean {v.mean():7.2%}  "
          f"within 10%: {int((v<0.10).sum())}/{len(v)}")
json.dump([{k: float(x) for k, x in r.items()} for r in rows],
          open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "vehicle_grad_check.json"), "w"), indent=1)

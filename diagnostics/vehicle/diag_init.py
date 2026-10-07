"""Diagnostics at initialisation (no training):
  1. headroom: how good is uncut relaxation + qz, and what is the integrality gap
  2. sanity: qz(z_ref) returns z_ref at the reference cost
  3. init net (paper config): cut activity, loss decomposition, soft-vs-deployed z
  4. finite-difference check of the surrogate gradient with w_int=5 (paper config)
"""
import json, time
from common import *

R = make_relax(2)
ids = va + tr
W, SCALE, RX = anchors(R, ids)
print(f"{len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

# 1. headroom on validation and train
res = {}
for name, sub in (("val", va), ("train", tr[:48])):
    rows = evaluate(R, sub, W, SCALE)
    res[f"uncut_{name}"] = summarise(rows, f"uncut+qz {name}")
    relgap = [100*(REF[i]["cost"] - SCALE[i])/abs(REF[i]["cost"]) for i in sub]
    res[f"intgap_{name}"] = float(np.mean(relgap))
    print(f"   integrality gap (ref - relax)/ref on {name}: {np.mean(relgap):.3f}%  "
          f"[min {np.min(relgap):.3f}, max {np.max(relgap):.3f}]", flush=True)
    # round+repair restoration, for contrast
    g = []
    for i in sub:
        c, inf = deploy(R, VEH, D[i], zf=RX[i]["z"])
        g.append(100*(c-REF[i]["cost"])/abs(REF[i]["cost"]))
    print(f"   uncut + round/repair deploy: |gap| {np.mean(np.abs(g)):.4f}%", flush=True)
    res[f"uncut_roundrepair_{name}"] = float(np.mean(np.abs(g)))
# per-instance structure of the remaining error on val
rows = evaluate(R, va, W, SCALE)
print("   val per-instance (gap%, hamming, nfrac relax):",
      [(round(r["gap"], 3), r["ham"], r["nfrac"]) for r in rows], flush=True)
res["uncut_val_rows"] = rows

# 2. sanity: projection of the reference schedule
bad = 0
for i in va:
    zr = np.array(REF[i]["z"], float); zq = qz_restore(D[i], zr)
    pr = R.solve(D[i], z_lo=zq, z_hi=zq)
    if np.any(zq != zr) or abs(pr["cost"]-REF[i]["cost"]) > 1e-4*abs(REF[i]["cost"]): bad += 1
print(f"sanity: qz(z_ref) != z_ref or cost mismatch on {bad}/{len(va)} val", flush=True)
res["sanity_bad"] = bad

# 3. init net, paper configuration
DD0 = DiffHV(R, VEH, tau=0.08, w_int=0.0)
DD5 = DiffHV(R, VEH, tau=0.08, w_int=5.0)
for seed in range(2):
    torch.manual_seed(seed)
    net = CutNetHV(T, k_rows=2, hidden=16, seed=seed, init_depth=0.25, max_depth=2.0, learn_thr=False)
    with torch.no_grad():
        A, b, d, d2 = net_rows(net, va, W)
    rows = evaluate(R, va, W, SCALE, A.numpy().astype(float), b.numpy().astype(float), DD0, DD5)
    res[f"init_s{seed}"] = summarise(rows, f"init net s{seed} val")
    res[f"init_s{seed}_rows"] = rows

# 4. FD check of dL/db (w_int=5, as trained) on 4 val instances
torch.manual_seed(0)
net = CutNetHV(T, k_rows=2, hidden=16, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False)
with torch.no_grad():
    A, b, d, d2 = net_rows(net, va[:4], W)
A = A.numpy().astype(float); b = b.numpy().astype(float)
fd = []
rng = np.random.default_rng(0)
for k, i in enumerate(va[:4]):
    for DD, nm in ((DD0, "w0"), (DD5, "w5")):
        L, gA, gb, inf = DD(D[i], A[k], b[k], SCALE[i])
        for h in (1e-4, 1e-5):
            num = np.zeros(len(b[k]))
            for j in range(len(b[k])):
                bp = b[k].copy(); bp[j] += h; bm = b[k].copy(); bm[j] -= h
                Lp = DD(D[i], A[k], bp, SCALE[i], want_grad=False)[0]
                Lm = DD(D[i], A[k], bm, SCALE[i], want_grad=False)[0]
                num[j] = (Lp-Lm)/(2*h)
            # random direction in A
            dA = rng.standard_normal(A[k].shape); dA /= np.linalg.norm(dA)
            Lp = DD(D[i], A[k]+h*dA, b[k], SCALE[i], want_grad=False)[0]
            Lm = DD(D[i], A[k]-h*dA, b[k], SCALE[i], want_grad=False)[0]
            numA = (Lp-Lm)/(2*h); anaA = float((gA*dA).sum())
            cos = float(num@gb/(np.linalg.norm(num)*np.linalg.norm(gb)+1e-30))
            rec = dict(i=i, w=nm, h=h, gb=gb.tolist(), fd_b=num.tolist(), cos_b=cos,
                       relerr_b=float(np.linalg.norm(num-gb)/(np.linalg.norm(num)+1e-30)),
                       dirA_ana=anaA, dirA_fd=numA, drop=inf["drop"])
            fd.append(rec)
            print(f"FD i={i} {nm} h={h:g} drop={inf['drop']}: cos_b={cos:+.4f} relerr_b={rec['relerr_b']:.3g} "
                  f"|gb|={np.linalg.norm(gb):.3g} |fd|={np.linalg.norm(num):.3g}  dirA ana={anaA:+.4g} fd={numA:+.4g}",
                  flush=True)
res["fd"] = fd
json.dump(res, open(f"{OUT}/diag_init.json", "w"), indent=1)
print("saved", f"{OUT}/diag_init.json")

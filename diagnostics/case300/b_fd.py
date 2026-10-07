"""B. Finite-difference check of the training gradient on case300 (the training
config: REG 1e-10, rho 1e6, thr 0.8). Directional derivatives of the surrogate
loss along each b row, a random A direction, and random Vr / Vi directions.
Usage: python b_fd.py SEED WHICH  (WHICH in init|best|net)"""
import json, sys
from c300common import *

seed, which = int(sys.argv[1]), sys.argv[2]
torch.manual_seed(seed)
net = load_net("case300", seed, which, g)
ids = E["va"][:4]
with torch.no_grad():
    Vr_t, Vi_t, A_t, b_t = forward(net, ids)
Vr, Vi, A, b = npf(Vr_t), npf(Vi_t), npf(A_t), npf(b_t)
rng = np.random.default_rng(0)
out = []
for k, j in enumerate(ids):
    L, gA, gb, info = surrogate(j, Vr[k], Vi[k], A[k], b[k])
    if gA is None:
        print("fail", j, info.get("fail")); continue
    dirs = []
    for r in range(len(b[k])):
        e = np.zeros_like(b[k]); e[r] = 1.0
        dirs.append((f"b[{r}]", dict(b=e), float(gb[r])))
    dA = rng.standard_normal(A[k].shape); dA /= np.linalg.norm(dA)
    dirs.append(("A rand", dict(A=dA), float((gA*dA).sum())))
    for nm, key, gv in (("Vr rand", "Vr", info["dVr"]), ("Vi rand", "Vi", info["dVi"])):
        dv = rng.standard_normal(NB); dv /= np.linalg.norm(dv)
        if key == "Vi":
            dv[int(g.ref)] = 0.0
        dirs.append((nm, {key: dv}, float(gv @ dv)))
    for nm, dd, ana in dirs:
        line = f"s{seed} {which} j{j:3d} {nm:8s} ana {ana:+.4e}"
        rec = dict(j=j, dir=nm, ana=ana, fd={}, meta=dict(margin=info["margin"], binding=bool(info["binding"])))
        hs = (1e-3, 1e-4) if nm.startswith(("b", "A")) else (1e-4, 1e-5, 1e-6)
        for h in hs:
            def Lat(s):
                return surrogate(j, Vr[k]+s*dd.get("Vr", 0), Vi[k]+s*dd.get("Vi", 0),
                                 A[k]+s*dd.get("A", 0), b[k]+s*dd.get("b", 0), want_grad=False)
            lp, _, _, ip = Lat(h); lm, _, _, im = Lat(-h)
            fd = (lp-lm)/(2*h) if lp is not None and lm is not None else np.nan
            rec["fd"][h] = fd
            same = ip.get("margin") == im.get("margin") == info["margin"]
            line += f" | h={h:g} fd {fd:+.4e}{'' if same else ' (margin changed)'}"
        print(line, flush=True)
        out.append(rec)
json.dump(out, open(f"{OUT}/b_fd_s{seed}_{which}.json", "w"), indent=1)

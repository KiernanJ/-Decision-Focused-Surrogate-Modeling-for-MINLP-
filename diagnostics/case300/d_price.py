"""D. Does the training loss price commitments honestly at the predicted V?
For the surrogate's commitment z (soft-rounded, as trained), compare
  layer-2 cost at the predicted V (what the loss sees, slack xi unpenalised in L)
  layer-2 cost at flat V
  S.price(z): QCAC re-linearisation from flat V until xi ~ 0 (what deployment uses)
Usage: python d_price.py SEED"""
import json, sys
from c300common import *
import c300common as C

seed = int(sys.argv[1]); va = E["va"]; res = {}
for which in ("init", "best", "net"):
    torch.manual_seed(seed)
    net = load_net("case300", seed, which, g)
    with torch.no_grad():
        Vr_t, Vi_t, A_t, b_t = forward(net, va)
    Vr, Vi, A, b = npf(Vr_t), npf(Vi_t), npf(A_t), npf(b_t)
    rows = []
    for k, j in enumerate(va):
        pd_, qd_ = POOL[j][0], POOL[j][1]
        L, _, _, info = surrogate(j, Vr[k], Vi[k], A[k], b[k], want_grad=False)
        if L is None: continue
        rc = S.solve(pd_, qd_, Vr[k], Vi[k], rho=RHO, A=A[k], b=b[k])
        z = soft_round(rc["u"], info["thr"], 0.08)
        rp = S.solve(pd_, qd_, Vr[k], Vi[k], rho=RHO, u_lo=z, u_hi=z)
        rf = S.solve(pd_, qd_, V1, V0, rho=RHO, u_lo=z, u_hi=z)
        pr = S.price(pd_, qd_, z)
        sc = C._SC[j]
        rows.append(dict(j=j, L=L, at_pred=rp["cost"]/sc, xi_pred=rp["slack"],
                         at_flat=rf["cost"]/sc if rf else None, xi_flat=rf["slack"] if rf else None,
                         price=pr["cost"]/sc if pr else None, xi_price=pr["slack"] if pr else None,
                         price_iters=pr["iters"] if pr else None, nz=float(z.sum())))
    f = lambda key: float(np.mean([r[key] for r in rows if r[key] is not None]))
    ok = [r for r in rows if r["price"] is not None and r["xi_price"] < 1e-4]
    under = np.array([r["price"] - r["at_pred"] for r in ok])
    print(f"s{seed} {which:4s} n={len(rows)} priced-converged={len(ok)}  loss(at pred V)={f('at_pred'):.4f} xi={f('xi_pred'):.2f} | "
          f"at flat V={f('at_flat'):.4f} xi={f('xi_flat'):.2f} | converged price={np.mean([r['price'] for r in ok]):.4f} | "
          f"price - loss: mean {under.mean():+.4f}, >0 on {(under > 0).sum()}/{len(ok)} | committed {f('nz'):.1f}", flush=True)
    res[which] = rows
json.dump(res, open(f"{OUT}/d_price_s{seed}.json", "w"), indent=1)

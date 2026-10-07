"""A. State of the trained case300 models on the 24 validation instances.
For init / best (selected epoch) / net (last epoch, what the paper evaluated):
  V statistics and tanh saturation; relaxation slack xi at flat vs predicted V;
  cut-row slack at predicted V; what layer 2 does at predicted V; gradient split
  between the V head and the cut heads (per network head, via autograd).
Usage: python a_state.py SEED"""
import json, sys, time
from c300common import *
import c300common as C

seed = int(sys.argv[1])
va = E["va"]
res = {}
for which in ("init", "best", "net"):
    torch.manual_seed(seed)
    net = load_net("case300", seed, which, g)
    net.train()
    for p in net.parameters():
        p.grad = None
    Vr_t, Vi_t, A_t, b_t = forward(net, va)
    Vr, Vi, A, b = npf(Vr_t), npf(Vi_t), npf(A_t), npf(b_t)
    rows = []; GA = np.zeros_like(A); GB = np.zeros_like(b); GVr = np.zeros_like(Vr); GVi = np.zeros_like(Vi)
    t0 = time.time()
    for k, j in enumerate(va):
        pd_, qd_ = POOL[j][0], POOL[j][1]
        w0, r0 = anchor(j)
        ru = S.solve(pd_, qd_, Vr[k], Vi[k], rho=RHO)                      # predicted V, no cut
        rc = S.solve(pd_, qd_, Vr[k], Vi[k], rho=RHO, A=A[k], b=b[k])      # predicted V, cut (layer 1)
        rf = S.solve(pd_, qd_, V1, V0, rho=RHO, A=A[k], b=b[k])            # flat V, cut
        vm = np.hypot(Vr[k], Vi[k])
        rec = dict(j=j, vmin=float(vm.min()), vmax=float(vm.max()), vdev=float(np.abs(vm-1).mean()),
                   sat_r=float((np.abs(Vr[k]-1) > 0.475).mean()), sat_i=float((np.abs(Vi[k]) > 0.475).mean()),
                   vi_max=float(np.abs(Vi[k]).max()),
                   xi_flat=r0["slack"], xi_pred=ru["slack"], xi_pred_cut=rc["slack"],
                   relax_flat=r0["cost"]/C._SC[j], relax_pred=ru["cost"]/C._SC[j], relax_pred_cut=rc["cost"]/C._SC[j],
                   rowslack_pred=(b[k]-A[k]@np.r_[rc["pg"], rc["u"]]).tolist(),
                   rowslack_flat=(b[k]-A[k]@np.r_[rf["pg"], rf["u"]]).tolist(),
                   soft_sc=rc["cut_slack"],
                   u_diff_cut=float(np.abs(rc["u"]-ru["u"]).sum()),     # does the cut change uf at all?
                   u_sum=float(rc["u"].sum()), nfrac=int(((rc["u"] > 1e-3) & (rc["u"] < 1-1e-3)).sum()),
                   n_above_thr=int((rc["u"] > 0.8).sum()))
        L, gA, gb, info = surrogate(j, Vr[k], Vi[k], A[k], b[k])
        if L is not None and gA is not None:
            rec.update(L=L, cost2=info["cost"]/C._SC[j], binding=bool(info["binding"]), margin=info["margin"],
                       gA=float(np.linalg.norm(gA)), gb=float(np.linalg.norm(gb)),
                       gV=float(np.linalg.norm(np.r_[info["dVr"], info["dVi"]])))
            # layer-2 slack: does the priced re-solve at predicted V lean on xi?
            z = soft_round(rc["u"], info["thr"], 0.08)
            r2 = S.solve(pd_, qd_, Vr[k], Vi[k], rho=RHO, u_lo=z, u_hi=z)
            rec["xi_layer2"] = r2["slack"] if r2 else None
            GA[k], GB[k], GVr[k], GVi[k] = gA, gb, info["dVr"], info["dVi"]
        else:
            rec["fail"] = info.get("fail")
        rows.append(rec)
    # per-head parameter gradients (the trainer's surrogate backward)
    n = max(1, sum("L" in r for r in rows))
    obj = ((A_t*torch.tensor(GA, dtype=torch.float32)).sum() + (b_t*torch.tensor(GB, dtype=torch.float32)).sum()
           + (Vr_t*torch.tensor(GVr, dtype=torch.float32)).sum() + (Vi_t*torch.tensor(GVi, dtype=torch.float32)).sum())/n
    obj.backward()
    heads = {}
    for name, p in net.named_parameters():
        h = name.split(".")[0]
        if p.grad is not None:
            heads[h] = heads.get(h, 0.0) + float((p.grad**2).sum())
    heads = {h: v**0.5 for h, v in heads.items()}
    ok = [r for r in rows if "L" in r]
    f = lambda key: float(np.mean([r[key] for r in ok if r.get(key) is not None])) if ok else None
    rs = np.array([s for r in rows for s in r["rowslack_pred"]])
    rsf = np.array([s for r in rows for s in r["rowslack_flat"]])
    summ = dict(n=len(rows), fails=len(rows)-len(ok),
                vmin=float(np.min([r["vmin"] for r in rows])), vmax=float(np.max([r["vmax"] for r in rows])),
                vdev=f("vdev"), sat_r=f("sat_r"), sat_i=f("sat_i"), vi_max=f("vi_max"),
                xi_flat=f("xi_flat"), xi_pred=f("xi_pred"), xi_layer2=f("xi_layer2"),
                relax_flat=f("relax_flat"), relax_pred=f("relax_pred"), relax_pred_cut=f("relax_pred_cut"),
                rows_active_pred=float((rs < 1e-4).mean()), rows_active_flat=float((rsf < 1e-4).mean()),
                rowslack_pred_med=float(np.median(rs)), u_diff_cut=f("u_diff_cut"),
                u_sum=f("u_sum"), nfrac=f("nfrac"), n_above_thr=f("n_above_thr"),
                L=f("L"), cost2=f("cost2"), binding=f("binding"), margin=f("margin"),
                gA=f("gA"), gb=f("gb"), gV=f("gV"), head_grad=heads, secs=time.time()-t0)
    print(f"s{seed} {which:4s} " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in summ.items() if k != "head_grad"), flush=True)
    print(f"        head grad norms: " + "  ".join(f"{h}={v:.3g}" for h, v in heads.items()), flush=True)
    res[which] = dict(summary=summ, rows=rows)
json.dump(res, open(f"{OUT}/a_state_s{seed}.json", "w"), indent=1)

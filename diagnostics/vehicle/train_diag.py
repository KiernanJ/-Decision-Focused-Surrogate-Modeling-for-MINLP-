"""Copy of vehicle/02_train.py's training loop (paper defaults) with per-epoch
validation diagnostics. Writes only to ./out. Test split is never touched."""
import argparse, json, time
from common import *
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--kcuts", type=int, default=2)
ap.add_argument("--tau", type=float, default=0.08)
ap.add_argument("--init-depth", type=float, default=0.25)
ap.add_argument("--hidden", type=int, default=16)
ap.add_argument("--w-int", type=float, default=5.0)
ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
R = make_relax(a.kcuts)
W, SCALE, _ = anchors(R, tr + va)
net = CutNetHV(T, k_rows=a.kcuts, hidden=a.hidden, seed=a.seed, init_depth=a.init_depth,
               max_depth=2.0, learn_thr=False)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
steps = a.epochs*int(np.ceil(len(tr)/a.batch))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
DD = DiffHV(R, VEH, tau=a.tau, w_int=a.w_int)
DD0 = DiffHV(R, VEH, tau=a.tau, w_int=0.0)
DD5 = DiffHV(R, VEH, tau=a.tau, w_int=5.0)
name = f"w{a.w_int:g}_d{a.init_depth:g}_s{a.seed}{a.tag}"
log = []


def val_diag(ep):
    with torch.no_grad():
        A, b, d, d2 = net_rows(net, va, W)
    rows = evaluate(R, va, W, SCALE, A.numpy().astype(float), b.numpy().astype(float), DD0, DD5)
    s = summarise(rows, f"[{name} ep{ep}]")
    ok = [r for r in rows if "fail" not in r and "L_tot" in r]
    if len(ok) > 3:
        s["rho_Ltot_dep"] = float(spearmanr([r["L_tot"] for r in ok], [r["dep_cost"] for r in ok])[0])
        s["rho_Lcost_dep"] = float(spearmanr([r["L_cost"] for r in ok], [r["dep_cost"] for r in ok])[0])
    s["fail_kinds"] = {k: sum(r.get("fail") == k for r in rows) for k in ("layer1", "deploy")}
    s["depth"] = torch.cat([d, d2], 1).mean(0).tolist()
    # val loss under the objective actually trained (label-free, as a selector would see it)
    s["val_Ltrain"] = float(np.mean([r["L_tot" if a.w_int > 0 else "L_cost"] for r in ok])) if ok else None
    return s, rows


s, rows = val_diag(-1); s["ep"] = -1; log.append(s)
t0 = time.time()
for ep in range(a.epochs):
    order = rng.permutation(tr); Lc, Lt, nok, fails = [], [], 0, {}
    for s0 in range(0, len(order), a.batch):
        bid = [int(i) for i in order[s0:s0+a.batch]]
        WR = torch.tensor(np.stack([W[i] for i in bid]), dtype=torch.float32)
        aE, ape, apb, az, az2, d, d2, th = net(Xn[bid])
        A, b = rows_torch(aE, ape, apb, az, az2, d, d2, WR)
        An = A.detach().numpy().astype(float); bn = b.detach().numpy().astype(float)
        GA = np.zeros_like(An); GB = np.zeros_like(bn); got = 0
        for k, i in enumerate(bid):
            L, gA, gb, info = DD(D[i], An[k], bn[k], SCALE[i])
            if L is None or gA is None:
                f = info.get("fail", "?"); fails[f] = fails.get(f, 0)+1; continue
            GA[k], GB[k] = gA, gb; got += 1
            Lt.append(L); Lc.append(info["cost"]/SCALE[i])
        if got == 0: continue
        nok += got
        obj = (A*torch.tensor(GA, dtype=torch.float32)).sum() + (b*torch.tensor(GB, dtype=torch.float32)).sum()
        opt.zero_grad(); (obj/got).backward(); opt.step(); sched.step()
    s, rows = val_diag(ep)
    s.update(ep=ep, train_Ltot=float(np.mean(Lt)), train_Lcost=float(np.mean(Lc)), nok=nok,
             train_fails=fails, secs=time.time()-t0)
    print(f"   train Ltot {s['train_Ltot']:.4f} Lcost {s['train_Lcost']:.4f} ok {nok}/{len(tr)} {fails} "
          f"{s['secs']:.0f}s", flush=True)
    log.append(s)
    torch.save(net.state_dict(), f"{OUT}/ck_{name}_ep{ep}.pt")
    json.dump(dict(args=vars(a), log=log, last_rows=rows), open(f"{OUT}/train_{name}.json", "w"), indent=1)

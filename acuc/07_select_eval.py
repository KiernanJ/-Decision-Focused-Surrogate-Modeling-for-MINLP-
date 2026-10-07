# %% Label-free deployment selection on VALIDATION, then one test evaluation.
#    Audit version (2026-10-06). One process per (case, seed, method).
#
#    method  ours       trained net (validation-selected weights), cuts anchored at predicted V
#            nocut      same trained net and V, cuts removed            (control, audit item 5)
#            untrained  same architecture at initialisation              (control, audit item 3)
#            proxy      supervised MLP on u* (converged train labels), hyperparameters on val
#
#    Deployment settings: repair ORDER {confidence, merit} x THRESHOLD {0.5, 0.65, 0.8}
#    chosen per (case, seed) on the 24 VALIDATION instances by mean deployed cost / J_R
#    (J_R = uncut relaxation cost at flat V; no labels). NDOWN is a compute budget,
#    not tuned: 25 (case118, where it saturates) and 40 (case300). Controls reuse the
#    setting selected for `ours` (one variable differs). The proxy gets the SAME
#    selection procedure on its own output.
#
#    Test scoring: only instances whose reference CONVERGED; J_ref = price(u_ref) with the
#    same evaluator as the methods. Deployment failures are counted, never dropped.
#    Time per instance = forward + every conic solve the method needs (incl. the anchor
#    solve) + restoration, single thread, no cache.
import os, sys, json, time, argparse, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--method", required=True, choices=["ours", "nocut", "untrained", "proxy"])
a = ap.parse_args()
os.environ["QCAC_CASE"] = a.case; os.environ["OMP_NUM_THREADS"] = "1"
from pipeline import grid, RESERVE, set_cuts
from socp import Socp
from cutnet_k import CutNetK, rows_torch
import data2

# %% setup
RHO, K = 1e6, 2
NDOWN = {"case118": 25, "case300": 40}[a.case]
GRID = [(o, t) for o in ("conf", "merit") for t in (0.5, 0.65, 0.8)]
set_cuts(2*K)
g, nl = grid(a.case); NG, NB = g.n_gen, g.n_bus
S = Socp(g, nl, n_cuts=2*K)
INST = data2.instances(g, a.case)
te, va, tr = data2.split()
REF = data2.refs(a.case)
V1, V0 = np.ones(NB), np.zeros(NB)
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0)+1e-6; Xn = (X-mu)/sg
MERIT = np.argsort(g.c1 + g.c2*g.pmax + nl/np.maximum(g.pmax, 1e-9), kind="stable")
OUT = "results/eval2"
if os.environ.get("SMOKE"):                 # tiny dry run, separate output dir
    va, te, OUT = va[:3], te[:8], "results/eval2_smoke"
os.makedirs(OUT, exist_ok=True)
SCALE = {}
for j in va:
    r = S.solve(INST[j][0], INST[j][1], V1, V0, rho=RHO)
    SCALE[j] = float(r["cost"]) if r is not None else np.nan

CACHE = {}
def price(j, z, cache=True):
    key = (j, np.asarray(z).astype(np.int8).tobytes())
    if cache and key in CACHE:
        return CACHE[key]
    pr = S.price(INST[j][0], INST[j][1], np.asarray(z, float))
    pr = None if pr is None else dict(cost=float(pr["cost"]), slack=float(pr["slack"]), pg=pr["pg"].copy())
    if cache:
        CACHE[key] = pr
    return pr


def restore(j, uf, thr, oname, cache=True):
    """threshold -> reserve top-up in order -> upward repair -> NDOWN downward trials
    in the mirrored order. Identical for every method."""
    pd_ = INST[j][0]
    uf = np.asarray(uf, float)
    z = (uf > thr).astype(float)
    order = MERIT if oname == "merit" else np.argsort(-uf, kind="stable")
    need = (1.0 + RESERVE)*float(pd_.sum())
    for k in order:
        if float(g.pmax @ z) >= need: break
        z[k] = 1.0
    best = None
    for _ in range(NG + 1):
        pr = price(j, z, cache)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"], pr["pg"]); break
        off = [k for k in order if z[k] < 0.5]
        if not off: return None
        z[off[0]] = 1.0
    if best is None: return None
    z, cost, pg = best
    for k in [t for t in order[::-1] if z[t] > 0.5][:NDOWN]:
        w = z.copy(); w[k] = 0.0
        if float(g.pmax @ w) < need: continue
        pr = price(j, w, cache)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w, pr["cost"], pr["pg"]
    return z.astype(int), cost, pg


# %% relaxed commitments uf for validation + test, with their solve time
def net_uf(net, ids, cuts=True):
    UF, T = {}, {}
    for j in ids:
        t0 = time.time()
        with torch.no_grad():
            Vr_t, Vi_t, at, alt, a2t, dt, d2t, _ = net(Xn[[j]])
        Vr = Vr_t[0].numpy().astype(float); Vi = Vi_t[0].numpy().astype(float)
        r0 = S.solve(INST[j][0], INST[j][1], Vr, Vi, rho=RHO)          # anchor / no-cut solve
        if r0 is None:
            UF[j] = None; T[j] = time.time()-t0; continue
        if not cuts:
            UF[j] = r0["u"]; T[j] = time.time()-t0; continue
        WR = torch.tensor(np.concatenate([r0["pg"], r0["u"]])[None], dtype=torch.float32)
        with torch.no_grad():
            A_t, b_t = rows_torch(at, alt, a2t, dt, d2t, WR)
        r1 = S.solve(INST[j][0], INST[j][1], Vr, Vi, rho=RHO,
                     A=A_t[0].numpy().astype(float), b=b_t[0].numpy().astype(float))
        UF[j] = None if r1 is None else r1["u"]; T[j] = time.time()-t0
    return UF, T


def make_net():
    return CutNetK(NB, NG, k_rows=K, v_scale=0.5, hidden=256, seed=a.seed, learn_thr=False)


info = {}
if a.method in ("ours", "nocut", "untrained"):
    net = make_net()
    if a.method != "untrained":
        ck = torch.load(f"checkpoints2/{a.case}_s{a.seed}.pt", weights_only=False)
        net.load_state_dict(ck["net"]); info["best_epoch"] = ck["best_epoch"]
    net.eval()
    UF, TUF = net_uf(net, va + te, cuts=(a.method != "nocut"))
else:
    # supervised proxy: labels = converged reference commitments. Wait until every
    # training/validation reference has been attempted (they are solved after test).
    while not set(tr + va) <= set(data2.refs(a.case).keys()):
        time.sleep(60)
    REF = data2.refs(a.case)
    ytr = [j for j in tr if data2.converged(REF.get(j))]
    yva = [j for j in va if data2.converged(REF.get(j))]
    Y = {j: torch.tensor(REF[j]["u"], dtype=torch.float32) for j in ytr + yva}
    best = None
    for lr in (1e-3, 3e-3):
        for ep in (500, 1500):
            for wd in (0.0, 1e-4):
                torch.manual_seed(a.seed)
                m = nn.Sequential(nn.Linear(2*NB, 256), nn.SiLU(), nn.Linear(256, 256),
                                  nn.SiLU(), nn.Linear(256, NG))
                o = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
                Xt = Xn[ytr]; Yt = torch.stack([Y[j] for j in ytr])
                for _ in range(ep):
                    o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(m(Xt), Yt)
                    l.backward(); o.step()
                with torch.no_grad():
                    vl = float(nn.functional.binary_cross_entropy_with_logits(
                        m(Xn[yva]), torch.stack([Y[j] for j in yva])))
                if best is None or vl < best[0]:
                    best = (vl, m, dict(lr=lr, epochs=ep, wd=wd))
    info.update(proxy_cfg=best[2], proxy_val_bce=best[0], n_train_labels=len(ytr), n_val_labels=len(yva))
    pm = best[1].eval()
    torch.save(dict(state=pm.state_dict(), cfg=best[2]), f"{OUT}/proxy_{a.case}_s{a.seed}.pt")
    UF, TUF = {}, {}
    for j in va + te:
        t0 = time.time()
        with torch.no_grad():
            UF[j] = torch.sigmoid(pm(Xn[[j]]))[0].numpy().astype(float)
        TUF[j] = time.time()-t0

# %% selection on validation (label-free)
sel_file = f"{OUT}/sel_{a.case}_s{a.seed}_ours.json"
if a.method in ("ours", "proxy"):
    table = {}
    for o, t in GRID:
        sc = []
        for j in va:
            r = restore(j, UF[j], t, o) if UF[j] is not None else None
            sc.append(10.0 if (r is None or not np.isfinite(SCALE[j])) else r[1]/SCALE[j])
        table[f"{o}|{t}"] = float(np.mean(sc))
        print(f"[val] {a.method} {o:5s} thr {t:.2f}  score {table[f'{o}|{t}']:.6f}", flush=True)
    choice = min(table, key=table.get)
    if a.method == "ours":
        json.dump(dict(table=table, choice=choice), open(sel_file, "w"))
else:
    while not os.path.exists(sel_file):
        time.sleep(30)
    table = None; choice = json.load(open(sel_file))["choice"]
o_sel, t_sel = choice.split("|")[0], float(choice.split("|")[1])
print(f"[sel] {a.method}: order {o_sel}, threshold {t_sel}, NDOWN {NDOWN}", flush=True)

# %% test, converged references only, J_ref = price(u_ref)
while not set(te) <= set(data2.refs(a.case).keys()):      # wait for every test reference
    time.sleep(60)
REF = data2.refs(a.case)
rows, excluded = [], {"no_ref": 0, "not_converged": 0, "ref_unpriceable": 0}
for pos, j in enumerate(te):
    r = REF.get(j)
    if r is None: excluded["no_ref"] += 1; continue
    if not data2.converged(r): excluded["not_converged"] += 1; continue
    pr_ref = price(j, r["u"])
    if pr_ref is None or pr_ref["slack"] >= 1e-4: excluded["ref_unpriceable"] += 1; continue
    t0 = time.time()
    d = restore(j, UF[j], t_sel, o_sel, cache=False) if UF[j] is not None else None
    secs = TUF[j] + time.time() - t0
    if d is None:
        rows.append(dict(pos=pos, idx=j, ok=False, secs=secs)); continue
    z, c, pg = d
    rows.append(dict(pos=pos, idx=j, ok=True, secs=secs,
                     gap=100*(c - pr_ref["cost"])/pr_ref["cost"],
                     disc=100*float((z != np.asarray(r["u"]).astype(int)).mean()),
                     cont=100*float(np.abs(pg - pr_ref["pg"]).sum()/pr_ref["pg"].sum()),
                     ref_secs=float(r["secs"])))
ok = [x for x in rows if x["ok"]]
G = np.array([x["gap"] for x in ok])
print(f"[TEST] {a.case} s{a.seed} {a.method}: scored {len(rows)} (failed {len(rows)-len(ok)}), "
      f"excluded {excluded} | |gap| {np.abs(G).mean():.4f}%  signed {G.mean():+.4f}%  "
      f"disc {np.mean([x['disc'] for x in ok]):.2f}%  cont {np.mean([x['cont'] for x in ok]):.2f}%  "
      f"median time {np.median([x['secs'] for x in rows]):.2f}s", flush=True)
json.dump(dict(case=a.case, seed=a.seed, method=a.method, ndown=NDOWN, choice=choice,
               val_table=table, excluded=excluded, rows=rows, **info),
          open(f"{OUT}/{a.case}_s{a.seed}_{a.method}.json", "w"), indent=1)

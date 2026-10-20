# B + E: are the deployed general rows active (anchored relaxation / deploy-like relaxation / deployed MIQCQP)? does the gamma=1000 smoothing
# and the fractional unit-row anchor distort the anchor point relative to what deployment solves?
import sys, numpy as np, torch, os, multiprocessing as mp, pickle
from common import *
case = sys.argv[1]; NMILP = int(sys.argv[2]); RHO_DEP = float(sys.argv[3]); W = float(sys.argv[4])
D = setup(case); n, G = D["n"], D["G"]; data2 = D["data2"]
from cutnet_k import CutNetK, rows_torch
import cuttrain
GR = torch.load(os.path.join(RES, f"general_rows_{case}_lfc.pt"), weights_only=False); K = GR["K"]
OUT, RHO, SEL = vnet(D, case); V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
cd, UC = deployed_cutnet(D, case); c = UC["c"]
with torch.no_grad(): M = torch.sigmoid(cd(D["Xn"])).numpy()
LO, HI = lohi(M, c)
gnet = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False); gnet.load_state_dict(GR["selected"]); gnet.eval()
ids = list(D["va"])
if __name__ == "__main__":
    pool = mp.get_context("spawn").Pool(2, initializer=__import__("brow").init, initargs=(case, K))
    import brow
    # anchors first (in the pool, as deployment), then rows from the deployed row-net
    AN = dict(pool.imap_unordered(cuttrain.anchor6, [(i, *V(i), LO[i], HI[i]) for i in ids]))
    jobs = []
    for t, i in enumerate(ids):
        if AN.get(i) is None: jobs.append((i, V(i), LO[i], HI[i], None, None, RHO_DEP, W, t < NMILP)); continue
        with torch.no_grad():
            _, _, a, al, a2, d, d2, _ = gnet(D["Xn"][[i]]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.tensor(AN[i], dtype=torch.float32)[None])
        jobs.append((i, V(i), LO[i], HI[i], A_[0, :K].numpy().astype(float), b_[0, :K].numpy().astype(float), RHO_DEP, W, t < NMILP))
    R = list(pool.imap_unordered(brow.work, jobs)); pool.close()
    pickle.dump(R, open(f"b_rows_{case}.pkl", "wb"))
    # also: row coefficients summary
    A0 = jobs[0][4]
    if A0 is not None:
        print(f"[{case}] row 0 coefficients: alpha(pg) |.|mean {np.abs(A0[0,:G]).mean():.3f} max {np.abs(A0[0,:G]).max():.3f}; a(u) mean {A0[0,G:].mean():.3f} range [{A0[0,G:].min():.3f}, {A0[0,G:].max():.3f}]")
    dep = np.array([r["depth"] for r in R if "depth" in r])
    print(f"[{case}] rows anchored on {len(dep)}/{len(R)} validation draws; depth at anchor (A w_anchor - b) mean {dep.mean(0)} min {dep.min():.3f} max {dep.max():.3f}")
    rd = np.array([r["row_at_deployrelax"] for r in R if "row_at_deployrelax" in r])
    print(f"[{case}] row value at the DEPLOY-LIKE relaxation (no rows; >0 = row would cut it off): per row mean {rd.mean(0)}, violated fraction {(rd > 1e-6).mean(0)}")
    for key, lab in (("du_smooth", "L1 |u_anchor(gamma=1000) - u(gamma=0)| (same fractional rows, rho 1e6)"), ("du_anchor_vs_deploy", "L1 |u_anchor - u_deploy-like relaxation (hard fixings, deploy rho, gamma 0)|"),
                     ("n_half_smooth", "# units whose u crosses 0.5 differently because of smoothing"), ("frac_s", "# fractional u in the smoothed anchor"), ("frac_f0", "# fractional u unsmoothed")):
        v = np.array([r[key] for r in R if key in r]); print(f"[{case}] {lab}: mean {v.mean():.3f} median {np.median(v):.3f} max {v.max():.3f}")
    MI = [r for r in R if "u_changed" in r]
    if MI:
        uc = np.array([r["u_changed"] for r in MI]); print(f"[{case}] deployed MIQCQP (W={W}, rho {RHO_DEP:.0e}) on {len(MI)} draws: commitment changed by rows on {(uc > 0).sum()}/{len(MI)} draws (units changed: {uc.tolist()})")
        ro = np.array([r["row_on"] for r in MI if r.get("row_on") is not None]); rf = np.array([r["row_off"] for r in MI if r.get("row_off") is not None])
        if len(ro):
            print(f"[{case}]   row residual A x - b at the rows-ON MIQCQP solution: binding(|r|<=1e-6) {(np.abs(ro) <= 1e-6).mean():.2f}, slack(r<-1e-6) {(ro < -1e-6).mean():.2f}, VIOLATED(r>1e-6, paid at 1e6) {(ro > 1e-6).mean():.2f}; mean slack {ro.mean(0)}")
            print(f"[{case}]   row residual at the rows-OFF MIQCQP solution: violated (row would cut it off) {(rf > 1e-6).mean():.2f}; per row mean {rf.mean(0)}")
            dob = np.array([(r["obj_on"] - r["obj_off"])/r["obj_off"] for r in MI if "obj_on" in r]); print(f"[{case}]   MIQCQP cost on vs off (relative, linearised QCAC cost): mean {100*dob.mean():+.4f}% max |.| {100*np.abs(dob).max():.4f}%")

# C: what does the deployed V-net (vnet_layer_<case>_env_nc.pt) predict? distance from flat, vs AC voltages of strict-priced reference commitments
import sys, numpy as np, torch, pickle, os
from common import *
case = sys.argv[1]
D = setup(case); n = D["n"]; data2 = D["data2"]
OUT, RHO, SEL = vnet(D, case)
SR = pickle.load(open(os.path.join(RES, f"strict_refs_{case}.pkl"), "rb"))
ok = lambda i: SR.get(i) is not None and SR[i][4] == 0
print(f"[{case}] V-net sel={SEL} rho={RHO:.0e}; strict refs available: {sum(ok(i) for i in SR)} (tr {sum(ok(i) for i in D['tr'])}, va {sum(ok(i) for i in D['va'])}, te {sum(ok(i) for i in D['te'])})")
Vp = lambda i: (1.0 + OUT[i, :n]) + 1j*OUT[i, n:]
Vr = lambda i: SR[i][2] + 1j*SR[i][3]
trr = [i for i in D["tr"] if ok(i)]
Vbar = np.mean([Vr(i) for i in trr], 0)       # constant predictor: mean reference voltage over TRAINING instances (uses labels; diagnostic only)
allp = np.stack([Vp(i) for i in range(len(D["INST"]))])
print(f"  V-net output over all {len(allp)} draws: |V-1| max {np.abs(allp-1).max():.4f}, mean over buses&draws {np.abs(allp-1).mean():.4f}; |V| range [{np.abs(allp).min():.4f}, {np.abs(allp).max():.4f}]; angle range [{np.angle(allp).min():.4f}, {np.angle(allp).max():.4f}] rad")
print(f"  input dependence: mean over buses of std across draws: V-net |V| {np.abs(allp).std(0).mean():.5f}, angle {np.angle(allp).std(0).mean():.5f}")
for nm, S in (("train", trr), ("validation", [i for i in D["va"] if ok(i)]), ("test(orig)", [i for i in D["te"] if ok(i)])):
    if not S: continue
    ref = np.stack([Vr(i) for i in S]); pr = np.stack([Vp(i) for i in S])
    e_net = np.abs(pr - ref); e_flat = np.abs(1 - ref); e_bar = np.abs(Vbar[None] - ref)
    em_net = np.abs(np.abs(pr) - np.abs(ref)); em_flat = np.abs(1 - np.abs(ref))
    ea_net = np.abs(np.angle(pr) - np.angle(ref)); ea_flat = np.abs(np.angle(ref))
    # per-bus correlation across instances of predicted vs reference angle (load-dependence tracking)
    cors = [np.corrcoef(np.angle(pr[:, b]), np.angle(ref[:, b]))[0, 1] for b in range(n) if np.angle(ref[:, b]).std() > 1e-6 and np.angle(pr[:, b]).std() > 1e-9]
    print(f"  {nm:11s} n={len(S):3d}: complex |V-Vref| mean/maxbus: V-net {e_net.mean():.4f}/{e_net.max(1).mean():.4f}  flat {e_flat.mean():.4f}/{e_flat.max(1).mean():.4f}  mean-ref const {e_bar.mean():.4f}/{e_bar.max(1).mean():.4f} | "
          f"|V| err V-net {em_net.mean():.4f} flat {em_flat.mean():.4f} | angle err V-net {ea_net.mean():.4f} flat {ea_flat.mean():.4f} rad | per-bus angle corr (pred vs ref across inst) median {np.median(cors) if cors else np.nan:.3f}")

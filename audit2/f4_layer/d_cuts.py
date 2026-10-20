# D2: the per-unit cut training gradient (cuttrain.track_soft, tracking loss through the smoothed soft-row layer, KKT VJP) at the DEPLOYED
# cut-net's (l,h) on training instances: KKT solve mode ('lu' exact vs 'lu+reg' regularised), forward status, FD agreement, and how close the
# exact-layer gradient is to the TRIVIAL imitation gradient 2(u_k - u_own_k)/G on binding rows (i.e. what BCE-like imitation would give).
import sys, numpy as np, torch, os, json
from common import *
case = sys.argv[1]; NI = int(sys.argv[2]); NFD = int(sys.argv[3])
D = setup(case); n, G = D["n"], D["G"]; data2 = D["data2"]
import cuttrain, conelayer
STAT = []
_fw = conelayer.ConeLayer.forward
def fw(self, vals, tol=None):
    x, st = _fw(self, vals, tol); STAT.append(st["status"]); return x, st
conelayer.ConeLayer.forward = fw
cuttrain.init4(case, 1000.0); W = cuttrain._W; L4 = W["L4"]; S1 = W["S1"]
OUT, RHO, SEL = vnet(D, case); V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
cd, UC = deployed_cutnet(D, case); c = UC["c"]
TG = {int(k): np.array(v, float) for k, v in json.load(open(os.path.join(RES, f"selfteacher_{case}_bestof_extra_nc.json"))).items()}
trn = [i for i in D["tr"] + data2.extra_ids() if i in TG]
with torch.no_grad(): M = torch.sigmoid(cd(D["Xn"])).numpy()
LO, HI = lohi(M, c)
rng = np.random.default_rng(0); sample = list(rng.choice(trn, NI, replace=False))
modes, cos_triv, nact, fdres = [], [], [], []
for t, i in enumerate(sample):
    lo, hi = LO[i].astype(float), HI[i].astype(float); uo = TG[i]
    r = cuttrain.track_soft((i, *V(i), lo, hi, uo, True))[1]
    if r is None: modes.append("FAIL"); continue
    L0, gl, gh = r; modes.append((L4.last_mode, L4.last_residual))
    # trivial imitation gradient on binding rows (u_k = l_k or u_k = h_k in the smoothed soft-row relaxation)
    rr = S1.solve(*W["inst"][i], *V(i), rho=1e6, A=cuttrain.unit_rows(lo, hi, G)[0], b=cuttrain.unit_rows(lo, hi, G)[1], cut_cap=1e6, soft_fallback=False)
    u = rr["u"]; tl = np.where((lo > 0) & (np.abs(u - lo) < 1e-4), 2*(u - uo)/G, 0.0); th = np.where((hi < 1) & (np.abs(u - hi) < 1e-4), 2*(u - uo)/G, 0.0)
    ga, gt = np.r_[gl, gh], np.r_[tl, th]
    cs = ga @ gt/(np.linalg.norm(ga)*np.linalg.norm(gt)) if np.linalg.norm(gt) > 0 and np.linalg.norm(ga) > 0 else np.nan
    cos_triv.append(cs); nact.append(int(((lo > 0) | (hi < 1)).sum()))
    if t < NFD:
        for h in (1e-2, 1e-3):
            rows = []
            for _ in range(8):
                dl = rng.standard_normal(G); dh = rng.standard_normal(G)
                dl[(lo + 2*h > np.minimum(hi, 1.0))] = 0.0; dh[(hi - 2*h < np.maximum(lo, 0.0))] = 0.0
                dl[lo <= 0] = 0.0; dh[hi >= 1] = 0.0          # only coordinates whose row exists (others have zero gradient by construction)
                nr = np.sqrt((dl**2).sum() + (dh**2).sum())
                if nr == 0: continue
                dl /= nr; dh /= nr
                Lp = cuttrain.track_soft((i, *V(i), lo + h*dl, hi + h*dh, uo, False))[1]; Lm = cuttrain.track_soft((i, *V(i), lo - h*dl, hi - h*dh, uo, False))[1]
                if Lp is None or Lm is None: continue
                rows.append(((Lp[0] - Lm[0])/(2*h), (Lp[0] - L0)/h, (L0 - Lm[0])/h, dl @ gl + dh @ gh))
            if len(rows) >= 3:
                R = np.array(rows); fd, an = R[:, 0], R[:, 3]
                cfd = np.corrcoef(R[:, 1], R[:, 2])[0, 1]; corr = np.corrcoef(fd, an)[0, 1]; cos = fd @ an/(np.linalg.norm(fd)*np.linalg.norm(an) + 1e-300); rel = np.linalg.norm(an - fd)/(np.linalg.norm(fd) + 1e-300)
                fdres.append((i, h, L4.last_mode if False else modes[-1][0], cfd, corr, cos, rel, len(rows)))
                print(f"[{case}] inst {i} h {h:.0e} mode {modes[-1][0]} (resid {modes[-1][1]:.1e}) interior dirs {len(rows)} | FD self-cons {cfd:+.3f} | corr {corr:+.4f} cos {cos:+.4f} rel.err {rel:.3f} | cos(VJP, trivial imitation) {cs:+.4f}", flush=True)
from collections import Counter
mm = Counter(m if isinstance(m, str) else m[0] for m in modes)
print(f"[{case}] track_soft over {len(sample)} training instances: KKT modes {dict(mm)}; residual of regularised solves: {[f'{m[1]:.1e}' for m in modes if not isinstance(m, str) and m[0] != 'lu'][:12]}")
print(f"[{case}] active cut coordinates per instance (l>0 or h<1): mean {np.mean(nact):.1f}/{G}; cos(VJP gradient, trivial imitation gradient 2(u-u_own)/G on binding rows): median {np.nanmedian(cos_triv):.4f}, min {np.nanmin(cos_triv):.4f}")
print(f"[{case}] forward statuses: {dict(Counter(STAT))}")
ct = np.array(cos_triv); print(f"[{case}] cos(VJP, trivial) all: {np.round(ct, 3).tolist()}; fraction >= 0.9: {np.nanmean(ct >= 0.9):.2f}; per mode: " + str({mo: np.round(np.nanmedian([c_ for c_, m_ in zip(cos_triv, [m for m in modes if not isinstance(m, str)]) if m_[0] == mo]), 3) for mo in ("lu", "lu+reg")}))

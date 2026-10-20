# D3: the general-row training gradient (cuttrain.track_gen, tracking loss through the smoothed layer with frozen fractional unit rows + K
# general rows, KKT VJP) at the DEPLOYED rows: KKT mode stats on training instances and FD agreement (gate-style, h retry).
import sys, numpy as np, torch, os, json
from common import *
case = sys.argv[1]; NI = int(sys.argv[2]); NFD = int(sys.argv[3])
D = setup(case); n, G = D["n"], D["G"]; data2 = D["data2"]
import cuttrain, conelayer
from cutnet_k import CutNetK, rows_torch
STAT = []
_fw = conelayer.ConeLayer.forward
def fw(self, vals, tol=None):
    x, st = _fw(self, vals, tol); STAT.append(st["status"]); return x, st
conelayer.ConeLayer.forward = fw
GR = torch.load(os.path.join(RES, f"general_rows_{case}_lfc.pt"), weights_only=False); K = GR["K"]
cuttrain.init6(case, 1000.0, K); W = cuttrain._W; L6 = W["L6"]
OUT, RHO, SEL = vnet(D, case); V = lambda i: (1.0 + OUT[i, :n], OUT[i, n:])
cd, UC = deployed_cutnet(D, case); c = UC["c"]
with torch.no_grad(): M = torch.sigmoid(cd(D["Xn"])).numpy()
LO, HI = lohi(M, c)
gnet = CutNetK(n, G, k_rows=K, v_scale=0.5, hidden=256, seed=0, init_depth=0.25, max_depth=2.0, learn_thr=False); gnet.load_state_dict(GR["selected"]); gnet.eval()
TG = {int(k): np.array(v, float) for k, v in json.load(open(os.path.join(RES, f"selfteacher_{case}_bestof_extra_nc.json"))).items()}
trn = [i for i in D["tr"] + data2.extra_ids() if i in TG]
rng = np.random.default_rng(0); sample = list(rng.choice(trn, NI, replace=False)); modes = []
for t, i in enumerate(sample):
    w = cuttrain.anchor6((i, *V(i), LO[i], HI[i]))[1]
    if w is None: modes.append("anchor fail"); continue
    with torch.no_grad():
        _, _, a, al, a2, d, d2, _ = gnet(D["Xn"][[i]]); A_, b_ = rows_torch(a, al, a2, d, d2, torch.tensor(w, dtype=torch.float32)[None])
    Ag, bg = A_[0, :K].numpy().astype(float), b_[0, :K].numpy().astype(float)
    r = cuttrain.track_gen((i, *V(i), LO[i], HI[i], Ag, bg, TG[i], True))[1]
    if r is None: modes.append("FAIL"); continue
    L0, gA, gb = r; modes.append((L6.last_mode, L6.last_residual))
    if t < NFD:
        dirs = []
        for _ in range(8):
            dA = rng.standard_normal(Ag.shape); db = rng.standard_normal(bg.shape); nr = np.sqrt((dA**2).sum() + (db**2).sum()); dirs.append((dA/nr, db/nr))
        for h in (1e-2, 3e-3, 1e-3):
            R = []
            for dA, db in dirs:
                Lp = cuttrain.track_gen((i, *V(i), LO[i], HI[i], Ag + h*dA, bg + h*db, TG[i], False))[1]; Lm = cuttrain.track_gen((i, *V(i), LO[i], HI[i], Ag - h*dA, bg - h*db, TG[i], False))[1]
                if Lp is None or Lm is None: continue
                R.append(((Lp[0] - Lm[0])/(2*h), (Lp[0] - L0)/h, (L0 - Lm[0])/h, (gA*dA).sum() + gb @ db))
            R = np.array(R); fd, an = R[:, 0], R[:, 3]
            cfd = np.corrcoef(R[:, 1], R[:, 2])[0, 1]; corr = np.corrcoef(fd, an)[0, 1]; cos = fd @ an/(np.linalg.norm(fd)*np.linalg.norm(an) + 1e-300); rel = np.linalg.norm(an - fd)/(np.linalg.norm(fd) + 1e-300)
            print(f"[{case}] inst {i} h {h:.0e} mode {modes[-1][0]} (resid {modes[-1][1]:.1e}) | FD self-cons {cfd:+.3f} | corr {corr:+.4f} cos {cos:+.4f} rel.err {rel:.3f} | |FD| {np.abs(fd).mean():.2e} | loss {L0:.4e}", flush=True)
from collections import Counter
print(f"[{case}] track_gen over {len(sample)} training instances: KKT modes {dict(Counter(m if isinstance(m, str) else m[0] for m in modes))}")
print(f"[{case}] forward statuses: {dict(Counter(STAT))}")

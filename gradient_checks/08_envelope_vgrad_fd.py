# %% FD GATE for the V-net training gradient (2026-10-10). combopt_ac.vgrad_env (envelope gradient of the
#    cut-constrained relaxed QCAC optimal value) vs central finite differences, at flat V AND at the trained
#    V-net's V, rho in {1e4, 1e5}, 3 TRAINING instances. Pass: corr >= 0.99 and sign agreement >= 7/8 everywhere.
#    The old VJP (combopt_ac.vgrad) is printed alongside for the record.
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.environ["QCAC_CASE"] = "case118"; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, torch, torch.nn as nn
import combopt_ac, data2
from pipeline import grid
CASE = "case118"; combopt_ac.init(CASE, True); W_ = combopt_ac._W; S = W_["S"]
g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = W_["inst"]; te, va, tr = data2.split()
base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
TG = {int(k): np.array(v) for k, v in json.load(open(os.path.join(HERE, "..", "acuc", "results", "teacher_case118_w120.json"))).items()}
ids = [i for i in tr if i in TG]; Y = torch.tensor(np.stack([TG[i] for i in ids]), dtype=torch.float32)
torch.manual_seed(0)
cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
o = torch.optim.Adam(cm.parameters(), lr=1e-3, weight_decay=1e-4)
for _ in range(1500):
    o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(cm(Xn[ids]), Y); l.backward(); o.step()
with torch.no_grad(): Pr = torch.sigmoid(cm.eval()(Xn)).numpy()
fx = lambda i: ([k for k in range(G) if Pr[i, k] >= 0.9], [k for k in range(G) if Pr[i, k] <= 0.1])
VS = torch.load(os.path.join(HERE, "..", "acuc", "results", "vnet_layer_case118.pt"), weights_only=False)
vm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2*n)); vm.load_state_dict(VS["state"]); vm.eval()
with torch.no_grad(): OUT = vm(Xn).numpy().astype(float)


def loss(i, Vr, Vi, rho, cuts):
    lo, hi = np.zeros(G), np.ones(G); on, off = cuts; lo[on] = 1.0; hi[off] = 0.0
    r = S.solve(*INST[i], Vr, Vi, rho=rho, u_lo=lo, u_hi=hi); return r["cost"] + rho*r["slack"]


rng = np.random.default_rng(0); ok_all = True
for i in tr[:3]:
    cuts = fx(i)
    for where, (Vr, Vi) in (("flat V", (np.ones(n), np.zeros(n))), ("trained V", (1.0 + OUT[i, :n], OUT[i, n:]))):
        for rho in (1e4, 1e5):
            _, e = combopt_ac.vgrad_env((i, Vr, Vi, rho, cuts)); _, v = combopt_ac.vgrad((i, Vr, Vi, rho, cuts))
            ge, gv = np.r_[e[1], e[2]], np.r_[v[1], v[2]]; rows = []
            for _ in range(8):
                d = rng.standard_normal(2*n); d[n + int(g.ref)] = 0; d /= np.linalg.norm(d); h = 1e-4
                fd = (loss(i, Vr + h*d[:n], Vi + h*d[n:], rho, cuts) - loss(i, Vr - h*d[:n], Vi - h*d[n:], rho, cuts))/(2*h)
                rows.append((fd, ge @ d, gv @ d))
            R = np.array(rows); ce = np.corrcoef(R[:, 0], R[:, 1])[0, 1]; cv = np.corrcoef(R[:, 0], R[:, 2])[0, 1]
            se = int((np.sign(R[:, 0]) == np.sign(R[:, 1])).sum()); sv = int((np.sign(R[:, 0]) == np.sign(R[:, 2])).sum())
            re = np.linalg.norm(R[:, 1] - R[:, 0])/np.linalg.norm(R[:, 0]); passed = ce >= 0.99 and se >= 7; ok_all &= passed
            print(f"inst {i:3d} {where:9s} rho {rho:.0e} | ENVELOPE corr {ce:.4f} signs {se}/8 rel.err {re:.4f} {'PASS' if passed else 'FAIL'} | old VJP corr {cv:.3f} signs {sv}/8", flush=True)
print("GATE", "PASSED" if ok_all else "FAILED")

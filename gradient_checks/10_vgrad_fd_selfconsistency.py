# %% DIAGNOSTIC (2026-10-11): 09 FAILED on hard case118 at the trained V-nets. Is the FD reference reliable there? Same points, same
#    directions; adds forward/backward FD self-consistency and steps h in {1e-3, 1e-4, 1e-5}. Envelope gradient judged only where FD is
#    self-consistent (corr(fw, bw) >= 0.99). No training, no results change.
#    Usage: python 09_envelope_vgrad_fd_final.py <case> <seeds e.g. 0,1,2>  -> gradient_checks/out/vgrad_fd_<case>.txt
#    Same check as 08 (FD gate for the V-net training gradient, 2026-10-10). combopt_ac.vgrad_env (envelope gradient of the
#    cut-constrained relaxed QCAC optimal value) vs central finite differences, at flat V AND at the trained
#    V-net's V, rho in {1e4, 1e5}, 3 TRAINING instances. Pass: corr >= 0.99 and sign agreement >= 7/8 everywhere.
#    The old VJP (combopt_ac.vgrad) is printed alongside for the record.
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
CASE = sys.argv[1]; SEEDS = [int(x) for x in sys.argv[2].split(",")]
os.environ["QCAC_CASE"] = CASE; os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, torch, torch.nn as nn
import combopt_ac, data2
from pipeline import grid
combopt_ac.init(CASE, True); W_ = combopt_ac._W; S = W_["S"]
g, nl = grid(CASE); n, G = g.n_bus, g.n_gen; INST = W_["inst"]; te, va, tr = data2.split()
base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
TG = {int(k): np.array(v) for k, v in json.load(open(os.path.join(HERE, "..", "acuc", "results", f"teacher_{CASE}_w{ {'case300': 180}.get(CASE, 120) }.json"))).items()}
ids = [i for i in tr if i in TG]; Y = torch.tensor(np.stack([TG[i] for i in ids]), dtype=torch.float32)
torch.manual_seed(0)
cm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
o = torch.optim.Adam(cm.parameters(), lr=1e-3, weight_decay=1e-4)
for _ in range(1500):
    o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(cm(Xn[ids]), Y); l.backward(); o.step()
with torch.no_grad(): Pr = torch.sigmoid(cm.eval()(Xn)).numpy()
fx = lambda i: ([k for k in range(G) if Pr[i, k] >= 0.9], [k for k in range(G) if Pr[i, k] <= 0.1])
OUTS = {}
for sd in SEEDS:
    VS = torch.load(os.path.join(HERE, "..", "acuc", "results", f"vnet_layer_{CASE}{'' if sd == 0 else f'_s{sd}'}_env.pt"), weights_only=False)
    vm = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2*n)); vm.load_state_dict(VS["state"]); vm.eval()
    with torch.no_grad(): OUTS[sd] = (vm(Xn).numpy().astype(float), VS["rho"])


def loss(i, Vr, Vi, rho, cuts):
    lo, hi = np.zeros(G), np.ones(G); on, off = cuts; lo[on] = 1.0; hi[off] = 0.0
    r = S.solve(*INST[i], Vr, Vi, rho=rho, u_lo=lo, u_hi=hi); return r["cost"] + rho*r["slack"]


rng = np.random.default_rng(0); ok_all = True
for i in tr[:3]:
    cuts = fx(i)
    pts = [("flat V", np.ones(n), np.zeros(n), 1e4)] + [(f"V-net s{sd}", 1.0 + O[i, :n], O[i, n:], rho_) for sd, (O, rho_) in OUTS.items()]
    for where, Vr, Vi, rho in pts:
        if True:
            _, e = combopt_ac.vgrad_env((i, Vr, Vi, rho, cuts)); _, v = combopt_ac.vgrad((i, Vr, Vi, rho, cuts))
            ge = np.r_[e[1], e[2]]; L0 = loss(i, Vr, Vi, rho, cuts)
            D = []
            for _ in range(8):
                d = rng.standard_normal(2*n); d[n + int(g.ref)] = 0; d /= np.linalg.norm(d); D.append(d)
            for h in (1e-3, 1e-4, 1e-5):
                rows = []
                for d in D:
                    Lp = loss(i, Vr + h*d[:n], Vi + h*d[n:], rho, cuts); Lm = loss(i, Vr - h*d[:n], Vi - h*d[n:], rho, cuts)
                    rows.append(((Lp - Lm)/(2*h), (Lp - L0)/h, (L0 - Lm)/h, ge @ d))
                R = np.array(rows); cfd = np.corrcoef(R[:, 1], R[:, 2])[0, 1]; ce = np.corrcoef(R[:, 0], R[:, 3])[0, 1]
                se = int((np.sign(R[:, 0]) == np.sign(R[:, 3])).sum()); re = np.linalg.norm(R[:, 3] - R[:, 0])/np.linalg.norm(R[:, 0])
                kink = np.median(np.abs(R[:, 1] - R[:, 2])/(np.abs(R[:, 0]) + 1e-12))
                verdict = ("PASS" if (ce >= 0.99 and se >= 7) else "FAIL") if cfd >= 0.99 else "FD unreliable"
                print(f"inst {i:3d} {where:9s} rho {rho:.0e} h {h:.0e} | FD self-cons {cfd:.4f} (|fw-bw|/|fd| median {kink:.3f}) | ENVELOPE corr {ce:.4f} signs {se}/8 rel.err {re:.4f} | {verdict}", flush=True)
print("done")

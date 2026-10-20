# Provenance check (no solver): does the committed cut-net look like a fine-tune of the BCE warm start on the LABEL-FREE targets
# (as 46 does), rather than of a cut-net fitted on reference labels? Compare fixing decisions (c from the file) on held-out blocks.
import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src")
import torch, torch.nn as nn, data2
from pipeline import grid
torch.set_num_threads(1)
te, va, tr = data2.split()
for case in ("case118", "case300", "pglib_opf_case118_ieee__api"):
    g, nl = grid(case); G = g.n_gen; INST = data2.all_instances(g, case)
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    UC = torch.load(f"{R}/acuc/results/unit_cuts_layer_{case}_soft_lfc.pt", weights_only=False); c = UC["c"]
    def net(state=None):
        m = nn.Sequential(nn.Linear(X.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, G))
        if state is not None: m.load_state_dict(state)
        return m
    def fit(ids, Y):
        torch.manual_seed(0); m = net(); o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(1500):
            o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(m(Xn[ids]), Y); l.backward(); o.step()
        return m.eval()
    TG = {int(k): np.array(v, float) for k, v in json.load(open(f"{R}/acuc/results/selfteacher_{case}_bestof_extra_nc.json")).items()}
    ids = [i for i in tr + data2.extra_ids() if i in TG]; m_lf = fit(ids, torch.tensor(np.stack([TG[i] for i in ids]), dtype=torch.float32))
    RP = data2.refs(case, R + "/acuc/results/pool2"); lab = [i for i in tr + va if data2.converged(RP.get(i))]
    m_ref = fit(lab, torch.tensor(np.stack([RP[i]["u"] for i in lab]), dtype=torch.float32))
    m_c = net(UC["state"]).eval(); m_last = net(UC["last"]).eval()
    ev = va + te + data2.fresh5_ids() + data2.fresh6_ids()
    def fx(m):
        with torch.no_grad(): P = torch.sigmoid(m(Xn[ev])).numpy()
        return np.where(P > c, 1, np.where(P < 1 - c, -1, 0)), P
    Fc, Pc = fx(m_c); Fl, Pl = fx(m_lf); Fr, Pr = fx(m_ref); Fz, Pz = fx(m_last)
    agree = lambda A, B: 100*np.mean(A == B)
    print(f"{case}: c={c}; fixing decisions (on/off/free per unit) on {len(ev)} held-out draws: committed vs label-free BCE warm start {agree(Fc, Fl):.2f}% "
          f"(mean |dP| {np.abs(Pc-Pl).mean():.4f}); committed vs reference-label BCE {agree(Fc, Fr):.2f}% (mean |dP| {np.abs(Pc-Pr).mean():.4f}); "
          f"label-free vs reference BCE {agree(Fl, Fr):.2f}%; units fixed per draw: committed {np.mean((Fc != 0).sum(1)):.1f}/{G}")

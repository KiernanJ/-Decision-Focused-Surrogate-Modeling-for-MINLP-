import os, sys, json
import os; REPO = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
RES = os.path.join(REPO, "acuc", "results")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)

def setup(case):
    os.environ["QCAC_CASE"] = case
    import data2
    from pipeline import grid
    g, nl = grid(case); n, G = g.n_bus, g.n_gen
    INST = data2.all_instances(g, case); te, va, tr = data2.split()
    base = float(g.pd.sum()); X = torch.tensor(np.stack([np.r_[p, q]/base for p, q in INST]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    return dict(g=g, nl=nl, n=n, G=G, INST=INST, te=te, va=va, tr=tr, X=X, Xn=Xn, data2=data2)

def mlp(din, dout):
    return nn.Sequential(nn.Linear(din, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, dout))

def warm_cutnet(D, case):
    """exact re-creation of 46_unit_cuts_layer.py's BCE warm start (extra draws, _nc teacher)."""
    data2 = D["data2"]
    TG = {int(k): np.array(v, float) for k, v in json.load(open(os.path.join(RES, f"selfteacher_{case}_bestof_extra_nc.json"))).items()}
    TRN = D["tr"] + data2.extra_ids()
    ids = [i for i in TRN if i in TG]; Y = torch.tensor(np.stack([TG[i] for i in ids]), dtype=torch.float32)
    torch.manual_seed(0)
    cm = mlp(D["X"].shape[1], D["G"])
    o = torch.optim.Adam(cm.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(1500):
        o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(cm(D["Xn"][ids]), Y); l.backward(); o.step()
    return cm, TG, ids

def deployed_cutnet(D, case):
    UC = torch.load(os.path.join(RES, f"unit_cuts_layer_{case}_soft_lfc.pt"), weights_only=False)
    cm = mlp(D["X"].shape[1], D["G"]); cm.load_state_dict(UC["state"]); cm.eval()
    return cm, UC

def lohi(M, c):
    LO = np.maximum((M - c)/(1 - c), 0); HI = 1 - np.maximum(((1 - c) - M)/(1 - c), 0)
    return LO, HI

def vnet(D, case):
    from vnetio import vnet_out
    OUT, RHO, SEL = vnet_out(os.path.join(RES, f"vnet_layer_{case}_env_nc.pt"), D["Xn"], D["n"])
    return OUT, RHO, SEL

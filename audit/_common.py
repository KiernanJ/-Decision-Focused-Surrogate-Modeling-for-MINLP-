"""Shared helpers for the audit scripts. Everything here only READS the repo.

Run every audit script from the repo root, e.g.  python audit/c2_checkpoint_epochs.py
Outputs go to audit/out/.
"""
import glob
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "audit", "out")
sys.path.insert(0, os.path.join(REPO, "src"))
os.makedirs(OUT, exist_ok=True)

import numpy as np

NT, NV, NTR = 48, 24, 144          # test / validation / train, as in acuc/*.py
RHO = 1e6


def pool_raw(case):
    """{global key: (inst, res)} from acuc/results/wide, exactly as poolload merges it."""
    merged = {}
    for f in sorted(glob.glob(f"{REPO}/acuc/results/wide/{case}_W_*.pkl")):
        d = pickle.load(open(f, "rb"))
        for i, r in d["res"].items():
            if r is not None:
                merged[i] = (d["inst"][i], r)
    return merged


def ref_slack(case):
    """Stored QCAC-iterative slack per pool position (same order as poolload)."""
    m = pool_raw(case)
    return np.array([float(m[k][1].get("slack", np.nan)) for k in sorted(m)])


def splits(n):
    """The split every acuc script uses: test first, then val, then train."""
    perm = np.random.default_rng(0).permutation(n)
    te = [int(i) for i in perm[:NT]]
    rest = [int(i) for i in perm[NT:NT + NTR + NV]]
    return te, rest[:NV], rest[NV:]


def setup(case):
    """Grid, SOCP, pool and normalised inputs, mirroring acuc/04_eval_*.py."""
    os.environ["QCAC_CASE"] = case
    import torch
    torch.set_num_threads(1)
    from pipeline import grid, set_cuts
    from socp import Socp
    import poolload
    set_cuts(4)
    g, nl = grid(case)
    S = Socp(g, nl, n_cuts=4)
    poolload.set_grid(g)
    POOL = poolload.load_pool(case, "wide", None,
                              wide_dir=f"{REPO}/acuc/results/wide")
    te, va, tr = splits(len(POOL))
    base = float(g.pd.sum())
    X = torch.tensor(np.stack([np.r_[p, q] / base for p, q, _, _ in POOL]),
                     dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
    return dict(g=g, nl=nl, S=S, POOL=POOL, te=te, va=va, tr=tr, Xn=(X - mu) / sg)


def load_net(case, seed, which, g):
    """which = 'net' (what the eval scripts load: last epoch),
               'best' (the epoch selected on validation), or
               'init' (untrained network, same seed)."""
    import torch
    from cutnet_k import CutNetK
    torch.manual_seed(seed)
    net = CutNetK(g.n_bus, g.n_gen, k_rows=2, v_scale=0.5, seed=seed)
    if which != "init":
        ck = torch.load(f"{REPO}/acuc/checkpoints/{case}_s{seed}.pt",
                        weights_only=False, map_location="cpu")
        net.load_state_dict(ck["net"] if which == "net" else ck["best"][1], strict=False)
    net.eval()
    return net


def predict(E, net, j):
    """Uncut flat-V relaxation (the anchor), then the net's V and anchored cut rows."""
    import torch
    from cutnet_k import rows_torch
    g, S, POOL = E["g"], E["S"], E["POOL"]
    r0 = S.solve(POOL[j][0], POOL[j][1], np.ones(g.n_bus), np.zeros(g.n_bus), rho=RHO)
    w = torch.tensor(np.r_[r0["pg"], r0["u"]][None], dtype=torch.float32)
    with torch.no_grad():
        Vr, Vi, a, al, a2, d, d2, _ = net(E["Xn"][[j]])
        A, b = rows_torch(a, al, a2, d, d2, w)
    f = lambda t: t[0].numpy().astype(float)
    return f(Vr), f(Vi), f(A), f(b)


def merit(g, nl):
    return np.argsort(g.c1 + g.c2 * g.pmax + nl / np.maximum(g.pmax, 1e-9))

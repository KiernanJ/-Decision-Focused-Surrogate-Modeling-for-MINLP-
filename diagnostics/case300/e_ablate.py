"""E. Which component triggers layer-2 failure / reserve-margin escalation?
(pred V, cut) vs (pred V, no cut) vs (flat V, cut) vs (flat V, no cut), surrogate loss only."""
import json, sys
from c300common import *
seed = int(sys.argv[1]); va = E["va"]; res = {}
A0 = np.zeros((4, 2*NG)); b0 = np.ones(4)
for which in ("best", "net"):
    torch.manual_seed(seed); net = load_net("case300", seed, which, g)
    with torch.no_grad():
        Vr_t, Vi_t, A_t, b_t = forward(net, va)
    Vr, Vi, A, b = npf(Vr_t), npf(Vi_t), npf(A_t), npf(b_t)
    for arm in ("predV+cut", "predV nocut", "flatV+cut", "flatV nocut"):
        M, L, U = [], [], []
        for k, j in enumerate(va):
            vr, vi = (Vr[k], Vi[k]) if arm.startswith("pred") else (V1, V0)
            a_, b_ = (A[k], b[k]) if arm.endswith("+cut") else (A0, b0)
            l, _, _, info = surrogate(j, vr, vi, a_, b_, want_grad=False)
            if l is None: continue
            M.append(info["margin"]); L.append(l)
        M = np.array(M); L = np.array(L)
        print(f"s{seed} {which:4s} {arm:12s} n={len(L)} loss {L.mean():.4f}  needs escalation {np.mean(M > 0.02):.0%}  mean margin {M.mean():.3f}", flush=True)
        res[f"{which}|{arm}"] = dict(L=L.tolist(), M=M.tolist())
json.dump(res, open(f"{OUT}/e_ablate_s{seed}.json", "w"))

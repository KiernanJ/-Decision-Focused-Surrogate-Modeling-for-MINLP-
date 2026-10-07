"""F. Split dL/dV into its two paths: layer 1 (V changes the relaxation -> the
commitment decision) and layer 2 (V changes the price of a FIXED commitment).
Deployment re-prices from flat V, so only the layer-1 path affects deployed decisions."""
import sys
from c300common import *
import framework
calls = []
_orig = framework.v_chain
def rec(*a, **k):
    out = _orig(*a, **k); calls.append(np.linalg.norm(np.r_[out[0], out[1]])); return out
framework.v_chain = rec
seed = int(sys.argv[1]); va = E["va"]
for which in ("init", "best"):
    torch.manual_seed(seed); net = load_net("case300", seed, which, g)
    with torch.no_grad():
        Vr_t, Vi_t, A_t, b_t = forward(net, va)
    Vr, Vi, A, b = npf(Vr_t), npf(Vi_t), npf(A_t), npf(b_t)
    l2, l1, gc = [], [], []
    for k, j in enumerate(va):
        calls.clear()
        L, gA, gb, info = surrogate(j, Vr[k], Vi[k], A[k], b[k])
        if gA is None or len(calls) != 2: continue
        l2.append(calls[0]); l1.append(calls[1]); gc.append(np.linalg.norm(np.r_[gA.ravel(), gb]))
    l1, l2, gc = map(np.array, (l1, l2, gc))
    print(f"s{seed} {which:4s} |dL/dV| via layer 2 (pricing) {l2.mean():.3e}   via layer 1 (decision) {l1.mean():.3e}   "
          f"ratio {np.median(l2/l1):.1f}x (median)   |dL/d(A,b)| {gc.mean():.3e}", flush=True)

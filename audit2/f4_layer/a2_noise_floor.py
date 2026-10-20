# A2: (i) is the BCE warm-start reconstruction deterministic here? (ii) noise floor: how many fixings differ between two BCE
# trainings that differ only by numerics (thread count / 1e-6 init jitter)? (iii) deployed (selected epoch) vs last epoch.
import sys, numpy as np, torch, json, os
from common import *
case = sys.argv[1]
D = setup(case); data2 = D["data2"]; G = D["G"]
TG = {int(k): np.array(v, float) for k, v in json.load(open(os.path.join(RES, f"selfteacher_{case}_bestof_extra_nc.json"))).items()}
ids = [i for i in D["tr"] + data2.extra_ids() if i in TG]; Y = torch.tensor(np.stack([TG[i] for i in ids]), dtype=torch.float32)
def bce(jitter=0.0, threads=1):
    torch.set_num_threads(threads); torch.manual_seed(0)
    cm = mlp(D["X"].shape[1], G)
    if jitter:
        g_ = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for p in cm.parameters(): p.add_(jitter*torch.randn(p.shape, generator=g_))
    o = torch.optim.Adam(cm.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(1500):
        o.zero_grad(); l = nn.functional.binary_cross_entropy_with_logits(cm(D["Xn"][ids]), Y); l.backward(); o.step()
    torch.set_num_threads(1); cm.eval(); return cm
_, UC = deployed_cutnet(D, case); c = UC["c"]
cd = mlp(D["X"].shape[1], G); cd.load_state_dict(UC["state"]); cd.eval()
cl = mlp(D["X"].shape[1], G); cl.load_state_dict(UC["last"]); cl.eval()
nets = {"warmA": bce(), "warmA_again": bce(), "warm_threads4": bce(threads=4), "warm_jit1e-6": bce(jitter=1e-6), "deployed": cd, "last(ep14)": cl}
S = list(D["va"]) + data2.fresh5_ids() + (data2.fresh6_ids() if case == "case118" else [])
with torch.no_grad(): M = {k: torch.sigmoid(v(D["Xn"])).numpy() for k, v in nets.items()}
fx = lambda m: (m[S] > c, m[S] < 1 - c)
def cmp(a, b):
    sa, sb = nets[a].state_dict(), nets[b].state_dict(); mx = max(float((sa[k]-sb[k]).abs().max()) for k in sa)
    (ona, offa), (onb, offb) = fx(M[a]), fx(M[b]); ch = (ona != onb) | (offa != offb)
    print(f"  {a:14s} vs {b:14s}: max|dtheta| {mx:.5f} | fixings changed {ch.sum():4d}/{ch.size} ({100*ch.mean():.2f}%) on {ch.any(1).sum()}/{len(S)} instances")
print(f"[{case}] val+fresh draws n={len(S)}, c={c}, deployed epoch {UC['epoch']}")
for a, b in [("warmA", "warmA_again"), ("warmA", "warm_threads4"), ("warmA", "warm_jit1e-6"), ("warm_threads4", "warm_jit1e-6"), ("warmA", "deployed"), ("deployed", "last(ep14)"), ("warmA", "last(ep14)")]:
    cmp(a, b)

"""C. Is the training gradient a descent direction for the training loss?
On a fixed batch of 16 train instances: compute the trainer's parameter gradient,
split into V-head and cut-head parts, then evaluate the batch loss after small
steps along -grad (normalised) and along an Adam-like first step (-sign(grad)).
Usage: python c_descent.py SEED WHICH"""
import json, sys, copy
from c300common import *

seed, which = int(sys.argv[1]), sys.argv[2]
torch.manual_seed(seed)
net0 = load_net("case300", seed, which, g); net0.train()
ids = list(np.random.default_rng(1).choice(E["tr"], 16, replace=False))
VHEAD = ("head_v",)
CUT = ("head_a", "head_al", "head_d", "head_a2", "head_d2")


def batch_loss(net, grad=False):
    Vr_t, Vi_t, A_t, b_t = forward(net, ids)
    Vr, Vi, A, b = npf(Vr_t), npf(Vi_t), npf(A_t), npf(b_t)
    Ls = []; GA = np.zeros_like(A); GB = np.zeros_like(b); GVr = np.zeros_like(Vr); GVi = np.zeros_like(Vi)
    for k, j in enumerate(ids):
        L, gA, gb, info = surrogate(j, Vr[k], Vi[k], A[k], b[k], want_grad=grad)
        Ls.append(np.nan if L is None else L)
        if grad and gA is not None:
            GA[k], GB[k], GVr[k], GVi[k] = gA, gb, info["dVr"], info["dVi"]
    if grad:
        obj = ((A_t*torch.tensor(GA, dtype=torch.float32)).sum() + (b_t*torch.tensor(GB, dtype=torch.float32)).sum()
               + (Vr_t*torch.tensor(GVr, dtype=torch.float32)).sum()
               + (Vi_t*torch.tensor(GVi, dtype=torch.float32)).sum())/len(ids)
        for p in net.parameters(): p.grad = None
        obj.backward()
    return float(np.nanmean(Ls)), Ls


L0, _ = batch_loss(net0, grad=True)
G = {n: p.grad.detach().clone() for n, p in net0.named_parameters() if p.grad is not None}
gnorm = lambda keys: float(sum((G[n]**2).sum() for n in G if n.split(".")[0] in keys)**0.5)
print(f"s{seed} {which} base batch loss {L0:.6f}  |grad| V-head {gnorm(VHEAD):.3e}  cut heads {gnorm(CUT):.3e}  "
      f"trunk {gnorm(('trunk',)):.3e}", flush=True)
res = dict(L0=L0, grad=dict(v=gnorm(VHEAD), cut=gnorm(CUT), trunk=gnorm(("trunk",))), steps=[])
for part, keys in (("all", None), ("V-head only", VHEAD), ("cut heads only", CUT)):
    sel = {n: g_ for n, g_ in G.items() if keys is None or n.split(".")[0] in keys}
    nrm = float(sum((v**2).sum() for v in sel.values())**0.5) + 1e-30
    for mode in ("grad", "sign"):
        for eta in ((1e-3, 1e-2, 1e-1, 1.0) if mode == "grad" else (1e-5, 5e-5, 5e-4)):
            net = copy.deepcopy(net0)
            with torch.no_grad():
                for n, p in net.named_parameters():
                    if n in sel:
                        p -= eta*(sel[n]/nrm if mode == "grad" else torch.sign(sel[n]))
            L1, _ = batch_loss(net)
            pred = -eta*nrm if mode == "grad" else -eta*float(sum(v.abs().sum() for v in sel.values()))
            print(f"   {part:15s} {mode:4s} eta {eta:<7g} dL {L1-L0:+.3e}   first-order prediction {pred:+.3e}", flush=True)
            res["steps"].append(dict(part=part, mode=mode, eta=eta, dL=L1-L0, pred=pred))
json.dump(res, open(f"{OUT}/c_descent_s{seed}_{which}.json", "w"), indent=1)

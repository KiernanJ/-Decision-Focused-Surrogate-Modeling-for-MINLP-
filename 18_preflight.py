"""Preflight: the checks that catch the bugs which produce PLAUSIBLE wrong answers.

Every serious mistake in this project so far belonged to one of five classes,
and none announced itself -- each produced a believable null result or a
believable number. These are the cheap mechanical checks for each.

  1 UNREACHABLE TARGET. A head clamped so it cannot represent the answer.
    v_scale=0.15 while 4.0% of V* components need up to 0.4038; the cardinality
    band anchored at the relaxation sum (~27) with max_off=6 while n*~14;
    max_off=15 on case300 which needs +28.4. Three separate occurrences, each
    costing a full training run that looked like "the method does not work".

  2 NO GRADIENT. An output that cannot influence the loss. sigma=0.03 on a
    quantity measured in generators (never changes a rounded commitment); a cut
    band initialised NON-binding (an inactive constraint has zero derivative
    w.r.t. its own coefficients). Both ran 50 epochs and moved nothing.

  3 SILENT DROPS. Failures skipped with `continue`, so arms get averaged over
    different instance counts and the means are not comparable.

  4 SPLIT / NORMALISATION MISMATCH. Evaluating a net against a pool it was not
    trained on: input standardisation is computed from the train split, so a
    grown pool silently corrupts every prediction.

  5 LABEL LEAK. Reference costs entering training or checkpoint selection in a
    method that claims to use none.

Run before trusting any result.
"""
# %% setup
import os, sys, re, subprocess
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp
exec(open(f"{HERE}/11_vcard.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

NTEST = int(os.environ.get("NTEST", "48"))
CKPT = os.environ.get("NET_VC", "net_c118_big_vcard.pt")
set_cuts(8); set_down(6)
g, nl = grid(); NG = g.n_gen; S = Socp(g, nl, n_cuts=8)
PASS, FAIL = [], []


def check(name, ok, detail):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {detail}", flush=True)


POOL = []
for f in sorted(os.listdir(f"{HERE}/results")):
    m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
    if not m:
        continue
    REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
    inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
    POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
             for i in range(int(m.group(2))) if REF[i] is not None]
rng = np.random.default_rng(0); perm = rng.permutation(len(POOL))
te = [int(i) for i in perm[:NTEST]]; tr = [int(i) for i in perm[NTEST:]]
print(f"\n{CASE}: {len(POOL)} instances, {len(tr)} train / {len(te)} test, net {CKPT}\n")

# ---- 1 UNREACHABLE TARGET --------------------------------------------------
print("1  target reachability")
sub = te[:12]
VS = np.stack([np.r_[(lambda r: (r['vr'], r['vi']))(
    S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float)))[0],
    (lambda r: (r['vr'], r['vi']))(
    S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float)))[1]] for j in sub])
V0n = np.r_[np.ones(g.n_bus), np.zeros(g.n_bus)]
dmax = float(np.abs(VS - V0n).max())
net = VCardNet(g.n_bus, NG)
check("head_v can reach V*", dmax <= net.v_scale,
      f"max |dV_i| needed {dmax:.4f}  vs v_scale {net.v_scale}")
order = np.argsort(-g.pmax); cum = np.cumsum(g.pmax[order])
nmin = np.array([float(np.searchsorted(cum, (1.0+RESERVE)*float(POOL[j][0].sum()))+1)
                 for j in sub])
nstar = np.array([float(POOL[j][2].sum()) for j in sub])
check("head_card can reach n*", bool((nstar <= NG).all() and (nstar >= nmin).all()),
      f"n* {nstar.min():.0f}-{nstar.max():.0f}, n_min {nmin.mean():.1f}, "
      f"span reaches {NG}")

# ---- 4 SPLIT / NORMALISATION ----------------------------------------------
print("\n4  split / normalisation integrity")
pth = f"{HERE}/results/{CKPT}"
sd = torch.load(pth) if os.path.exists(pth) else None
check("checkpoint exists", sd is not None, CKPT)
if sd is not None:
    net.load_state_dict(sd); net.eval()
    in_dim = sd["trunk.0.weight"].shape[1]
    check("net input dim matches grid", in_dim == 2*g.n_bus,
          f"net expects {in_dim}, grid gives {2*g.n_bus}")
    out_dim = sd["head_v.weight"].shape[0]
    check("net output dim matches grid", out_dim == 2*g.n_bus,
          f"net emits {out_dim}, grid needs {2*g.n_bus}")

base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
SRt = torch.tensor([float(np.searchsorted(cum, (1.0+RESERVE)*float(p.sum()))+1)
                    for p, _, _, _ in POOL], dtype=torch.float32)

# ---- 2 NO GRADIENT ---------------------------------------------------------
print("\n2  every output can actually move the loss")
with torch.no_grad():
    vr, vi, lo, hi = net(Xn[sub], SRt[sub])
vr, vi, lo, hi = [t.numpy().astype(float) for t in (vr, vi, lo, hi)]
band_binds = 0
for k, j in enumerate(sub):
    A, b = card_rows(lo[k], hi[k], NG)
    r = S.solve(POOL[j][0], POOL[j][1], vr[k], vi[k], rho=1e6, A=A, b=b)
    if r is None:
        continue
    su = float(r["u"].sum())
    if su >= hi[k] - 0.05 or su <= lo[k] + 0.05:
        band_binds += 1
check("cardinality band BINDS", band_binds >= len(sub)//2,
      f"active on {band_binds}/{len(sub)} instances "
      f"(inactive cut => zero gradient on its own rows)")


def dep(j, Vr, Vi, A, b):
    r = S.solve(POOL[j][0], POOL[j][1], Vr, Vi, rho=1e6, A=A, b=b)
    if r is None:
        return None
    uf = r["u"]; z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0+RESERVE)*float(POOL[j][0].sum())
    for t_ in o:
        if float(g.pmax @ z) >= need:
            break
        z[t_] = 1.0
    pr = S.price(POOL[j][0], POOL[j][1], z)
    return None if pr is None else pr["cost"]


for gname, sig in [("V outputs", 0.03), ("cardinality outputs", 1.0)]:
    moved = 0
    for k, j in enumerate(sub[:6]):
        A, b = card_rows(lo[k], hi[k], NG)
        c0 = dep(j, vr[k], vi[k], A, b)
        if gname == "V outputs":
            e = rng.standard_normal(2*g.n_bus)*sig
            c1 = dep(j, vr[k]+e[:g.n_bus], vi[k]+e[g.n_bus:], A, b)
        else:
            A2, b2 = card_rows(lo[k]-sig, hi[k]-sig, NG)
            c1 = dep(j, vr[k], vi[k], A2, b2)
        if c0 is not None and c1 is not None and abs(c1-c0) > 1e-6:
            moved += 1
    check(f"perturbing {gname} changes cost", moved >= 3,
          f"{moved}/6 instances responded at sigma={sig}")

# ---- 3 SILENT DROPS --------------------------------------------------------
print("\n3  failure accounting")
ok = sum(1 for k, j in enumerate(sub) if dep(j, vr[k], vi[k],
         *card_rows(lo[k], hi[k], NG)) is not None)
check("no silent deployment drops", ok == len(sub), f"{ok}/{len(sub)} deployed")

# ---- 5 LABEL LEAK ----------------------------------------------------------
print("\n5  label leak in the training path")
src = open(f"{HERE}/11_vcard.py").read()
train_blk = src[src.index("# %% train"):src.index("net.load_state_dict(best[1])")]
leaks = [l.strip() for l in train_blk.splitlines()
         if "POOL[j][3]" in l and "MONITORING" not in l and "monitoring" not in l
         and not l.strip().startswith("#")]
check("gradient scale is label-free", "SCALE[bidx]" in train_blk,
      "uses flat-V deployment cost, not the reference")
check("checkpoint selection is label-free", "sel < best[0]" in train_blk,
      "selects on raw deployment cost")
print(f"     (reference touched on {len(leaks)} training line(s); "
      f"expected 1, the monitoring-only gap print)")

print(f"\n{'='*60}\n  {len(PASS)} passed, {len(FAIL)} FAILED")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("="*60)
sys.exit(1 if FAIL else 0)

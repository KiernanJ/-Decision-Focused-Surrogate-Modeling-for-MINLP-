"""Figure: learned tiered cuts against the controls that could refute them.

Three error measures, because the gap alone hides two different failures:
  DISCRETE    % of generators committed wrongly
  CONTINUOUS  ||pg - pg*||_1 / sum(pg*) on the RECOVERED dispatch
  GAP         true AC cost vs the QCAC-iterative reference

The fourth panel is the claim itself: the learned tier counts PER INSTANCE
against the constant the mean-band control uses. If those two collapsed onto
each other the cut would be decorative -- which is exactly what happened to the
previous cardinality cut, where a dummy band reproduced the learned one bit for
bit.

Colour encodes ROLE, not arm identity -- baseline / method / control -- so only
three hues are in play, taken unchanged from the documented validated
categorical palette (slot 1 blue, slot 8 red) plus a neutral for the reference.
Arms are told apart by position and label, never by colour alone.
"""
# %% setup
import os, sys, re, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down
from socp import Socp
exec(open(f"{HERE}/21_tiered_train.py").read().split("# %% main")[0].replace(
    "if __name__", "if False and __name__"), globals())

K = 3; K_ROWS = 2*K; NTEST = 48; NDOWN = 6
set_cuts(K_ROWS); set_down(NDOWN)
g, nl = grid(); NG = g.n_gen; S = Socp(g, nl, n_cuts=K_ROWS)
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
avg = (g.c2*g.pmax**2 + g.c1*g.pmax + nl)/np.maximum(g.pmax, 1e-6)
MERIT = np.argsort(avg); TIERS = [MERIT[i::K] for i in range(K)]
base = float(g.pd.sum())
X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
net = TieredNet(g.n_bus, NG, [len(T) for T in TIERS])
net.load_state_dict(torch.load(f"{HERE}/results/net_tier_k3full.pt")); net.eval()
with torch.no_grad():
    VR, VI, CNT = net(Xn[te])
VR, VI, CNT = VR.numpy().astype(float), VI.numpy().astype(float), CNT.numpy().astype(float)
PGS = {j: S.price(POOL[j][0], POOL[j][1], POOL[j][2].astype(float))["pg"] for j in te}


def deploy(j, Vr, Vi, cnt):
    pd_, qd_ = POOL[j][0], POOL[j][1]
    A, b = (tier_rows(cnt, TIERS, NG, K_ROWS) if cnt is not None else (None, None))
    r = S.solve(pd_, qd_, Vr, Vi, rho=1e6, A=A, b=b, cut_cap=0.0)
    if r is None:
        return None
    uf = r["u"]; z = (uf > 0.5).astype(float); o = np.argsort(-uf)
    need = (1.0+RESERVE)*float(pd_.sum())
    for t_ in o:
        if float(g.pmax @ z) >= need:
            break
        z[t_] = 1.0
    best = None
    for _ in range(NG+1):
        pr = S.price(pd_, qd_, z)
        if pr is not None and pr["slack"] < 1e-4:
            best = (z.copy(), pr["cost"], pr["pg"]); break
        off = [t_ for t_ in o if z[t_] < 0.5]
        if not off:
            return None
        z[off[0]] = 1.0
    if best is None:
        return None
    z, cost, pg = best
    for t_ in [t2 for t2 in np.argsort(uf) if z[t2] > 0.5][:NDOWN]:
        w = z.copy(); w[t_] = 0.0
        if float(g.pmax @ w) < need:
            continue
        pr = S.price(pd_, qd_, w)
        if pr is not None and pr["slack"] < 1e-4 and pr["cost"] < cost:
            z, cost, pg = w, pr["cost"], pr["pg"]
    return dict(disc=100*float((z.astype(int) != POOL[j][2]).mean()),
                cont=100*float(np.abs(pg-PGS[j]).sum()/PGS[j].sum()),
                gap=100*(cost-POOL[j][3])/POOL[j][3])


ARMS = [
    ("no cut",   "baseline", lambda k, j: (np.ones(g.n_bus), np.zeros(g.n_bus), None)),
    ("LEARNED",  "method",   lambda k, j: (VR[k], VI[k], CNT[k])),
    ("dummy",    "control",  lambda k, j: (VR[k], VI[k], np.zeros(K))),
    ("random",   "control",  lambda k, j: (VR[k], VI[k], rng.uniform(0, [len(T) for T in TIERS]))),
    ("mean",     "control",  lambda k, j: (VR[k], VI[k], CNT.mean(0))),
]
DATA = []
for name, role, fn in ARMS:
    rows = []
    for k, j in enumerate(te):
        vr_, vi_, c_ = fn(k, j)
        r = deploy(j, vr_, vi_, c_)
        if r:
            rows.append(r)
    DATA.append((name, role, rows))
    print(f"{name.replace(chr(10),' '):28s} n={len(rows):>3d}  "
          f"disc {np.mean([x['disc'] for x in rows]):6.2f}%  "
          f"cont {np.mean([x['cont'] for x in rows]):6.2f}%  "
          f"gap {np.mean([x['gap'] for x in rows]):+7.3f}%", flush=True)

# %% figure -- colour encodes ROLE; hues unchanged from the validated palette
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
ROLE = {"baseline": MUTED, "method": "#2a78d6", "control": "#e34948"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#d7d7d2",
                     "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})
fig, ax = plt.subplots(1, 4, figsize=(15, 4.1))
for a_, (key, title, unit) in zip(ax[:3], [
        ("disc", "Discrete decision error", "% of generators"),
        ("cont", "Continuous decision error", "% of generation"),
        ("gap",  "Optimality gap", "%")]):
    a_.set_axisbelow(True); a_.grid(axis="y", color="#e8e8e3", lw=1)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    for i, (name, role, rows) in enumerate(DATA):
        v = np.array([x[key] for x in rows]); c = ROLE[role]
        bp = a_.boxplot([v], positions=[i], widths=.52, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=2),
                        whiskerprops=dict(color=c, lw=2),
                        capprops=dict(color=c, lw=2))
        bp["boxes"][0].set(facecolor=c, alpha=.28, edgecolor=c, lw=2)
        a_.scatter(np.full(len(v), i)+rng.uniform(-.11, .11, len(v)), v,
                   s=11, color=c, alpha=.55, linewidths=.5, edgecolors="#fcfcfb",
                   zorder=3)
        a_.text(i, a_.get_ylim()[1], "", ha="center")
        a_.annotate(f"{v.mean():.2f}", (i, v.mean()), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK, fontweight="bold",
                    bbox=dict(fc="#fcfcfb", ec="none", pad=1.2))
    a_.set_xticks(range(len(DATA)))
    a_.set_xticklabels([n for n, _, _ in DATA], fontsize=8, color=INK2)
    a_.set_title(title, fontsize=9.5, color=INK, pad=9, loc="left")
    a_.set_ylabel(unit, fontsize=8)

# panel 4 -- the claim: the band varies per instance
a_ = ax[3]
a_.set_axisbelow(True); a_.grid(axis="y", color="#e8e8e3", lw=1)
for sp in ("top", "right"):
    a_.spines[sp].set_visible(False)
for t in range(K):
    xs = np.full(NTEST, t) + rng.uniform(-.16, .16, NTEST)
    a_.scatter(xs, CNT[:, t], s=13, color=ROLE["method"], alpha=.6,
               linewidths=.5, edgecolors="#fcfcfb", zorder=3,
               label="learned, per instance" if t == 0 else None)
    a_.hlines(CNT[:, t].mean(), t-.3, t+.3, color=ROLE["control"], lw=2.5,
              zorder=4, label="mean band (control)" if t == 0 else None)
    a_.annotate(f"sd {CNT[:, t].std():.2f}", (t, CNT[:, t].max()),
                xytext=(0, 5), textcoords="offset points", ha="center",
                fontsize=8, color=INK)
a_.set_xticks(range(K))
a_.set_xticklabels([f"tier {t+1}\n({'cheap' if t==0 else 'mid' if t==1 else 'costly'})"
                    for t in range(K)], fontsize=7.5, color=INK2)
a_.set_ylabel("generators committed in tier", fontsize=8)
a_.set_title("The learned cut is instance-specific", fontsize=9.5, color=INK,
             pad=9, loc="left")
a_.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.legend(handles=[Patch(fc=ROLE["baseline"], alpha=.28, ec=ROLE["baseline"], label="baseline"),
                    Patch(fc=ROLE["method"], alpha=.28, ec=ROLE["method"], label="proposed method"),
                    Patch(fc=ROLE["control"], alpha=.28, ec=ROLE["control"], label="control (refutation test)")],
           ncol=3, frameon=False, fontsize=8.5, loc="upper right",
           bbox_to_anchor=(.995, 1.045))
fig.text(.008, .955,
         "no cut = flat V   |   LEARNED = V + learned tier cuts   |   "
         "dummy = band [0,0]   |   random = random band   |   "
         "mean = one band for every instance",
         fontsize=8, color=INK2, ha="left")
fig.suptitle(f"{CASE}: learned tiered cuts vs the controls that would refute them "
             f"-- {NTEST} held-out instances, ONE continuous solve",
             fontsize=10.5, color=INK, x=.008, ha="left", y=1.045)
fig.tight_layout(rect=[0, 0, 1, .90])
fig.savefig(f"{HERE}/results/tiered_cuts_{CASE}.png", dpi=170, bbox_inches="tight")
print(f"\nwrote results/tiered_cuts_{CASE}.png")

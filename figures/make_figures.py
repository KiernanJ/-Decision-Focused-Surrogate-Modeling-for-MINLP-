# %% Four-panel figure per case study (layout of Dixit, Gupta & Zhang):
#      box plots of discrete error, continuous error, |cost gap|
#      + a CDF of per-instance solution time on a log axis.
#    Arms: proposed method (self-supervised learned cut) vs the SUPERVISED
#    NN proxy, both followed by the SAME restoration. Time panel also shows
#    the reference solver (QCAC iterative for AC-UC, Gurobi MINLP for vehicle).
#    Run from the release root:  python figures/make_figures.py
import json, glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AC, VH, OUT = f"{ROOT}/acuc/results", f"{ROOT}/vehicle/results", f"{ROOT}/figures"
C_PX, C_OURS, C_EXACT = "#6E7F8D", "#2E7D6E", "#B23A3A"


def rows(pat, key=None):
    out = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        src = d["out"][key] if key else d["rows"]
        out += [[r[0], r[1], r[2]] if isinstance(r, list) else
                [r["gap"], r["disc"], r["cont"]] for r in src]
    return np.array(out)


def secs(pat):
    return np.array([r["secs"] for f in sorted(glob.glob(pat))
                     for r in json.load(open(f))["rows"]])


CASES = {
 "case118": dict(
    ours=lambda: rows(f"{AC}/thrlf_eval_case118_s*_d25.json", "fitted"),
    proxy=lambda: rows(f"{AC}/proxyfair_case118_s*.json"),
    proxy_t=lambda: secs(f"{AC}/proxyfair_case118_s*.json"),
    tim=f"{AC}/timingF_case118.json", ngen=54, exact=False,
    ref="QCAC iterative (Constante-Flores & Li)"),
 "case300": dict(
    ours=lambda: np.array([r[:3] for f in sorted(glob.glob(f"{AC}/c300pipe_s*_merit0.80_dt.json"))
                           for rr in json.load(open(f)).values() for r in rr]),
    proxy=lambda: rows(f"{AC}/proxyfair_case300_s*.json"),
    proxy_t=lambda: secs(f"{AC}/proxyfair_case300_s*.json"),
    tim=f"{AC}/timingF_case300.json", ngen=69, exact=False,
    ref="QCAC iterative (Constante-Flores & Li)"),
 "hybrid_vehicle": dict(
    ours=lambda: rows(f"{VH}/train_s?.json"),
    proxy=lambda: rows(f"{VH}/proxy_s*.json"),
    proxy_t=lambda: secs(f"{VH}/proxy_s*.json"),
    tim=f"{VH}/timing_hv.json", ngen=30, exact=True,
    ref="exact MINLP (Gurobi)"),
}

summary = {}
for case, cf in CASES.items():
    ours, px = cf["ours"](), cf["proxy"]()
    T = json.load(open(cf["tim"]))
    dep, ref, pxt = np.array(T["deploy"]), np.array(T["reference"]), cf["proxy_t"]()
    EX = cf["exact"]
    ARMS = [("NN proxy\n(supervised)", C_PX, px), ("proposed\n(learned cut)", C_OURS, ours)]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                         "grid.linestyle": "--", "figure.facecolor": "white"})
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.0))
    rn = "MINLP optimum" if EX else "QCAC reference"
    panels = [(1, "Discrete decision error (%)", f"% integer decisions differing from {rn}"),
              (2, "Continuous decision error (%)", f"relative L1 error vs {rn} (%)"),
              (0, "Optimality gap (%)" if EX else "Cost deviation from QCAC (%)",
               r"$|J-J^\star|/|J^\star|$ (%)" if EX else r"$|J-J^{\rm QCAC}|/|J^{\rm QCAC}|$ (%)")]
    for a_, (col, title, ylab) in zip(ax[:3], panels):
        data = [np.abs(A[:, col]) if col == 0 else A[:, col] for _, _, A in ARMS]
        data = [x[np.isfinite(x)] for x in data]
        bp = a_.boxplot(data, patch_artist=True, widths=.55, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.2))
        for p, (_, c, _) in zip(bp["boxes"], ARMS):
            p.set_facecolor(c); p.set_edgecolor("black"); p.set_linewidth(.8)
        a_.set_xticks([1, 2]); a_.set_xticklabels([n for n, _, _ in ARMS], fontsize=8.5)
        a_.set_title(title, fontsize=10.5); a_.set_ylabel(ylab, fontsize=8.5)
        hi = max(float(np.percentile(x, 75) + 1.5*(np.percentile(x, 75)-np.percentile(x, 25)))
                 for x in data)
        hi = max(hi, max(float(np.median(x)) for x in data)) or 1e-3
        a_.set_ylim(0, hi*1.25)
        for i, x in enumerate(data, start=1):
            a_.text(i, hi*1.22, f"mean {np.mean(x):.3g}", ha="center", va="top", fontsize=8)
    a_ = ax[3]
    def cdf(v, c, lab, lw=2.0):
        x = np.sort(np.asarray(v, float)); x = x[np.isfinite(x) & (x > 0)]
        y = 100.0*np.arange(1, x.size+1)/x.size
        a_.step(np.r_[x[0]*.7, x], np.r_[0., y], where="post", color=c,
                label=f"{lab} (median {np.median(x):.3g}s)", linewidth=lw)
    cdf(ref, C_EXACT, cf["ref"], lw=2.4)
    cdf(pxt, C_PX, "NN proxy + restoration")
    cdf(dep, C_OURS, "proposed + restoration")
    a_.set_xscale("log"); a_.set_xlabel("solution time per instance (s, log)")
    a_.set_ylabel("% of instances solved"); a_.set_title("Solution time CDF", fontsize=10.5)
    a_.set_ylim(-2, 102); a_.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/QUAD_{case}.{ext}", dpi=185, bbox_inches="tight")
    plt.close(fig)
    summary[case] = {}
    for n, A, t in (("proxy", px, pxt), ("ours", ours, dep)):
        summary[case][n] = dict(abs_gap=float(np.abs(A[:, 0]).mean()), signed_gap=float(A[:, 0].mean()),
                                median_abs_gap=float(np.median(np.abs(A[:, 0]))),
                                disc=float(A[:, 1].mean()), cont=float(A[:, 2].mean()),
                                n=int(len(A)), median_secs=float(np.median(t)))
    summary[case]["reference_median_secs"] = float(np.median(ref))
    print(case, json.dumps(summary[case], indent=1))
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)

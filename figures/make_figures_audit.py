# %% Four-panel figures from the AUDIT evaluation (2026-10-06):
#    discrete error, continuous error, |cost gap| box plots (proposed vs supervised
#    NN proxy, both with the same validation-selected restoration procedure) +
#    CDF of per-instance solution time incl. the reference solver.
#    AC-UC: test instances with CONVERGED references only, J_ref = price(u_ref).
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C_PX, C_OURS, C_EXACT = "#6E7F8D", "#2E7D6E", "#B23A3A"


def acuc(case, meth):
    rows = [r for f in sorted(glob.glob(f"{ROOT}/acuc/results/eval2/{case}_s?_{meth}.json"))
            for r in json.load(open(f))["rows"]]
    ok = [r for r in rows if r["ok"]]
    return (np.array([[r["gap"], r["disc"], r["cont"]] for r in ok]),
            np.array([r["secs"] for r in rows]))


def acuc_ref_secs(case):
    f = f"{ROOT}/acuc/results/eval2/timing_ref_{case}.json"
    if os.path.exists(f):
        return np.array(json.load(open(f))["secs"])
    d = json.load(open(sorted(glob.glob(f"{ROOT}/acuc/results/eval2/{case}_s?_ours.json"))[0]))
    return np.array([r["ref_secs"] for r in d["rows"] if "ref_secs" in r])


V = json.load(open(f"{ROOT}/vehicle/results/audit_vehicle.json"))
def veh(key):
    rows = [r for s in V[key].values() for r in s["rows"]]
    ok = [r for r in rows if r["ok"]]
    return np.array([[r["gap"], r["disc"], r["cont"]] for r in ok]), np.array([r["secs"] for r in rows])

CASES = {"case118": ("QCAC iterative", False), "case300": ("QCAC iterative", False),
         "hybrid_vehicle": ("exact MINLP (Gurobi)", True)}
for case, (refname, EX) in CASES.items():
    if case == "hybrid_vehicle":
        (ours, t_o), (px, t_p) = veh("ours"), veh("proxy"); t_ref = np.array(V["exact"]["secs"])
    else:
        if not glob.glob(f"{ROOT}/acuc/results/eval2/{case}_s?_proxy.json"):
            print("skip", case, "(proxy not finished)"); continue
        (ours, t_o), (px, t_p) = acuc(case, "ours"), acuc(case, "proxy"); t_ref = acuc_ref_secs(case)
    ARMS = [("NN proxy\n(supervised)", C_PX, px), ("proposed\n(learned cut)", C_OURS, ours)]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "grid.linestyle": "--"})
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.0))
    rn = "MINLP optimum" if EX else "QCAC reference"
    for a_, (col, title, ylab) in zip(ax[:3], [
            (1, "Discrete decision error (%)", f"% integer decisions differing from {rn}"),
            (2, "Continuous decision error (%)", f"relative L1 error vs {rn} (%)"),
            (0, "Optimality gap (%)" if EX else "Cost deviation from QCAC (%)",
             r"$|J-J^\star|/|J^\star|$ (%)" if EX else r"$|J-J^{\rm QCAC}|/|J^{\rm QCAC}|$ (%)")]):
        data = [np.abs(A[:, col]) if col == 0 else A[:, col] for _, _, A in ARMS]
        bp = a_.boxplot(data, patch_artist=True, widths=.55, showfliers=False,
                        medianprops=dict(color="black", linewidth=1.2))
        for p, (_, c, _) in zip(bp["boxes"], ARMS):
            p.set_facecolor(c); p.set_edgecolor("black")
        a_.set_xticks([1, 2]); a_.set_xticklabels([n for n, _, _ in ARMS], fontsize=8.5)
        a_.set_title(title, fontsize=10.5); a_.set_ylabel(ylab, fontsize=8.5)
        hi = max(float(np.percentile(x, 75) + 1.5*(np.percentile(x, 75)-np.percentile(x, 25))) for x in data)
        hi = max(hi, max(float(np.median(x)) for x in data), 1e-3)
        a_.set_ylim(0, hi*1.25)
        for i, x in enumerate(data, start=1):
            a_.text(i, hi*1.22, f"mean {np.mean(x):.3g}", ha="center", va="top", fontsize=8)
    a_ = ax[3]
    for v, c, lab, lw in ((t_ref, C_EXACT, refname, 2.4), (t_p, C_PX, "NN proxy + restoration", 2),
                          (t_o, C_OURS, "proposed + restoration", 2)):
        x = np.sort(v[np.isfinite(v) & (v > 0)]); y = 100*np.arange(1, x.size+1)/x.size
        a_.step(np.r_[x[0]*.7, x], np.r_[0, y], where="post", color=c, lw=lw,
                label=f"{lab} (median {np.median(x):.3g}s)")
    a_.set_xscale("log"); a_.set_xlabel("solution time per instance (s, log)")
    a_.set_ylabel("% of instances solved"); a_.set_title("Solution time CDF", fontsize=10.5)
    a_.set_ylim(-2, 102); a_.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{ROOT}/figures/QUAD_{case}.{ext}", dpi=185, bbox_inches="tight")
    plt.close(fig); print("wrote", case)

# %% FINAL figures of the LABEL-FREE method (2026-10-13): the pre-registered final tests, evaluated once.
#    Per case: four panels (discrete error, continuous error, |cost gap| box plots; per-instance solution-time CDF incl. the reference)
#    + one overview figure (mean cost gap per case; median solve time per case incl. the reference, log scale).
#    Arms: OUR METHOD (label-free: learned V [AC] + layer-trained integer cuts + general rows) vs SUPERVISED NN proxy; 3 seeds pooled.
#    Sources: acuc/runs/2026-10-13_*_FINAL_<case>_fresh4_lfx_seeds0-1-2/final.pkl,
#             acuc/runs/2026-10-13_*_vehicle_final_TEST_lf_pool_fresh2/test_results.pkl, references acuc/results/fresh4_refs_*.pkl,
#             dixit_vehicle/results/pool_fresh2/.   Colours = reference palette slots 1-3 (validated all-pairs, light).
#    Run from the release root:  python figures/make_figures_final_lf.py
import json, glob, os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS, AC, VP, OUT = f"{ROOT}/acuc/runs", f"{ROOT}/acuc/results", f"{ROOT}/dixit_vehicle/results/pool_fresh2", f"{ROOT}/figures/final_2026-10-13"
C_OURS, C_PX, C_REF = "#2a78d6", "#eb6834", "#1baf7a"          # palette slots 1 (blue), 2 (orange), 3 (aqua)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "figure.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


# %% load
def ac_case(case):
    d = pickle.load(open(sorted(glob.glob(f"{RUNS}/2026-10-13_*_FINAL_{case}_fresh4_lfx_seeds0-1-2/final.pkl"))[-1], "rb"))
    REF = pickle.load(open(f"{AC}/fresh4_refs_{case}.pkl", "rb"))
    ours = np.vstack([d["per_seed"][s]["OUR METHOD"] for s in sorted(d["per_seed"])])
    px = np.vstack([d["per_seed"][s]["supervised NN proxy"] for s in sorted(d["per_seed"])])
    ref_t = np.array([REF[i]["secs"] - REF[i].get("secs_polish", 0.0) for i in d["tid"]])
    # columns: 0 |gap| 1 disc 2 cont 3 time 4 signed gap 5 repaired
    return dict(ours=ours[:, :3], ours_t=ours[:, 3], px=px[:, :3], px_t=px[:, 3], ref_t=ref_t, n=len(d["tid"]),
                ref="reference: sequential QCAC", exact=False, unit="s")


RES = pickle.load(open(sorted(glob.glob(f"{RUNS}/2026-10-13_*_vehicle_final_TEST_lf_pool_fresh2/test_results.pkl"))[-1], "rb"))
def veh_case(names, kind):
    R = [r for r in RES if r[0] in names]
    ours = np.vstack([r[4] for r in R]); px = np.vstack([r[5] for r in R])
    if kind == "milp":
        ref_t = np.concatenate([[p["secs"] for p in json.load(open(f"{VP}/vehicle_{n[6:]}.json")) if p["mipgap"] <= 1e-4 + 1e-12] for n in names])
    else:
        ref_t = np.array([r["secs"] for r in json.load(open(f"{VP}/minlp.json"))["ref"].values() if r and r["mipgap"] <= 1e-4 + 1e-12])
    # columns: 0 |gap| 1 disc 2 cont 3 time 4 fallback 5 signed gap
    return dict(ours=ours[:, :3], ours_t=ours[:, 3], px=px[:, :3], px_t=px[:, 3], ref_t=ref_t, n=sum(len(r[2]) for r in R if r[1] == 0),
                ref="reference: exact solve (Gurobi)", exact=True, unit="s")


CASES = {"AC case118": ac_case("case118"), "AC case300": ac_case("case300"), "AC PGLib case118 (api)": ac_case("pglib_opf_case118_ieee__api"),
         "Vehicle MILP (8 vehicles)": veh_case([f"milp_v{v}" for v in (0, 1, 2, 4, 6, 7, 8, 9)], "milp"), "Vehicle MINLP": veh_case(["minlp"], "minlp")}


# %% per-case four-panel figure
def fname(case):
    return case.replace(" ", "_").replace("(", "").replace(")", "")


summary = {}
for case, cf in CASES.items():
    EX = cf["exact"]; ours, px = cf["ours"], cf["px"]
    ARMS = [("ours\n(label-free)", C_OURS, ours), ("supervised\nNN proxy", C_PX, px)]
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.1))
    rn = "exact optimum" if EX else "QCAC reference"
    panels = [(1, "Discrete decision error", f"% of integer decisions differing from {rn}"),
              (2, "Continuous decision error", f"relative L1 error vs {rn} (%)"),
              (0, "Cost gap", r"$|J-J^\star|/|J^\star|$ (%)" if EX else r"$|J-J^{\rm ref}|/|J^{\rm ref}|$ (%)")]
    for a_, (col, title, ylab) in zip(ax[:3], panels):
        data = [np.abs(A[:, col]) for _, _, A in ARMS]; data = [x[np.isfinite(x)] for x in data]
        rng = np.random.default_rng(0); top = max(float(x.max()) for x in data) or 1e-3
        for i, ((_, c, _), x) in enumerate(zip(ARMS, data), start=1):     # every instance-seed as a dot (zero-heavy data: boxes would collapse)
            a_.scatter(i + rng.uniform(-.2, .2, x.size), x, s=9, color=c, alpha=.45, linewidths=0, zorder=2)
            a_.plot([i - .3, i + .3], [np.median(x)]*2, color=INK, linewidth=1.6, zorder=3)
            a_.scatter([i + .36], [np.mean(x)], s=46, marker="D", color=c, edgecolor=INK, linewidth=1, zorder=4)
            a_.text(i, top*1.2, f"mean {np.mean(x):.3g}\nmedian {np.median(x):.3g}", ha="center", va="top", fontsize=7.8, color=INK2)
        a_.set_xticks([1, 2]); a_.set_xticklabels([n for n, _, _ in ARMS], fontsize=8.5, color=INK); a_.set_xlim(.4, 2.6)
        a_.set_title(title, fontsize=10.5, color=INK, loc="left"); a_.set_ylabel(ylab, fontsize=8.5)
        a_.set_ylim(-top*.03, top*1.32); a_.grid(axis="x", visible=False)
    a_ = ax[3]
    def cdf(v, c, lab):
        x = np.sort(np.asarray(v, float)); x = x[np.isfinite(x) & (x > 0)]; y = 100.0*np.arange(1, x.size + 1)/x.size
        a_.step(np.r_[x[0]*.7, x], np.r_[0., y], where="post", color=c, linewidth=2, label=f"{lab} (median {np.median(x):.3g} s)")
    cdf(cf["ours_t"], C_OURS, "ours (one short solve)"); cdf(cf["px_t"], C_PX, "NN proxy + repair"); cdf(cf["ref_t"], C_REF, cf["ref"])
    a_.set_xscale("log"); a_.set_xlabel("solution time per instance (s, log scale)"); a_.set_ylabel("% of instances solved")
    a_.set_title("Solution time (CDF)", fontsize=10.5, color=INK, loc="left"); a_.set_ylim(-2, 102)
    a_.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, labelcolor=INK)
    fig.text(0.01, -0.02, "dots = instances x seeds; black line = median; diamond = mean", fontsize=7.8, color=INK2)
    fig.suptitle(f"{case}: untouched test, {cf['n']} instances x 3 seeds", fontsize=11.5, color=INK, x=0.01, ha="left", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/QUAD_{fname(case)}.{ext}", dpi=185, bbox_inches="tight")
    plt.close(fig)
    summary[case] = {n: dict(gap_mean=float(np.nanmean(A[:, 0])), gap_median=float(np.nanmedian(A[:, 0])), disc=float(np.nanmean(A[:, 1])),
                             cont=float(np.nanmean(A[:, 2])), time_median_s=float(np.nanmedian(t)))
                     for n, A, t in (("ours", ours, cf["ours_t"]), ("proxy", px, cf["px_t"]))}
    summary[case]["reference_time_median_s"] = float(np.median(cf["ref_t"])); summary[case]["n_instances"] = cf["n"]

# %% overview: mean cost gap per case | median solve time per case (two panels -- two measures, no dual axis)
names = list(CASES); y = np.arange(len(names))[::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
for k, (arm, c, off) in enumerate((("ours", C_OURS, .18), ("proxy", C_PX, -.18))):
    v = [summary[n][arm]["gap_mean"] for n in names]
    a1.barh(y + off, v, height=.34, color=c, label="ours (label-free)" if arm == "ours" else "supervised NN proxy")
    for yy, vv in zip(y + off, v):
        a1.text(vv, yy, f" {vv:.3f}%", va="center", fontsize=8, color=INK2)
a1.set_yticks(y); a1.set_yticklabels(names, color=INK); a1.set_xlabel("mean cost gap vs reference (%)")
a1.set_title("Cost gap (lower is better)", loc="left", fontsize=10.5, color=INK); a1.grid(axis="y", visible=False)
a1.set_xlim(0, max(summary[n][a]["gap_mean"] for n in names for a in ("ours", "proxy"))*1.25)
a1.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=2, labelcolor=INK)
for arm, c, mk, lab in (("ours", C_OURS, "o", "ours (label-free)"), ("proxy", C_PX, "s", "supervised NN proxy"), ("ref", C_REF, "D", "reference solver")):
    v = [summary[n]["reference_time_median_s"] if arm == "ref" else summary[n][arm]["time_median_s"] for n in names]
    a2.scatter(v, y, s=60, marker=mk, color=c, edgecolor="white", linewidth=1.5, zorder=3, label=lab)
for n, yy in zip(names, y):
    a2.plot([summary[n]["ours"]["time_median_s"], summary[n]["reference_time_median_s"]], [yy, yy], color=GRID, linewidth=2, zorder=1)
    sp = summary[n]["reference_time_median_s"]/summary[n]["ours"]["time_median_s"]
    a2.text(summary[n]["reference_time_median_s"]*1.25, yy, f"{sp:.0f}x faster than reference", va="center", fontsize=8, color=INK2)
a2.set_xscale("log"); a2.set_yticks(y); a2.set_yticklabels([]); a2.set_xlabel("median solve time per instance (s, log scale)")
a2.set_xlim(min(summary[n]["proxy"]["time_median_s"] for n in names)*0.5, max(summary[n]["reference_time_median_s"] for n in names)*40)
a2.set_title("Solve time (median)", loc="left", fontsize=10.5, color=INK); a2.grid(axis="y", visible=False)
a2.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=3, labelcolor=INK)
fig.suptitle("Final untouched tests: label-free method vs supervised NN proxy (3 seeds)", fontsize=11.5, color=INK, x=0.01, ha="left", y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/OVERVIEW_gap_and_time.{ext}", dpi=185, bbox_inches="tight")
plt.close(fig)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
for n in names:
    s = summary[n]; print(f"{n:28s} ours {s['ours']['gap_mean']:.4f}% {s['ours']['time_median_s']:.3g}s | proxy {s['proxy']['gap_mean']:.4f}% "
                          f"{s['proxy']['time_median_s']:.3g}s | ref {s['reference_time_median_s']:.3g}s | n {s['n_instances']}")

# Evidence for AUDIT2.md

This folder holds the scripts, logs and per-instance outputs behind `../AUDIT2.md`, the second audit of `main` at `e64170f`. Each subfolder is one of the audit's six reviews, plus the re-run of the pre-registered final tests.

## Setup

```bash
pip install -r audit2/requirements-audit.txt      # needs a Gurobi licence; the Python version used was 3.12.12
```

**Run from a throwaway clone.** The scripts only read the committed artifacts in `acuc/results/` and `dixit_vehicle/results/`. But some write their outputs into their own folder or the current directory, and `55_final_eval.py` (used by `repro/run_repro.sh`) appends to `acuc/RESULTS.md`. A throwaway clone keeps all of that out of your working copy:

```bash
git clone <this repo> /tmp/audit2 && cd /tmp/audit2 && git checkout audit/second-audit
```

**How each script finds the repo root.**
- Most scripts find it from their own location (`audit2/<sub>/../..`), or from the `REPO` environment variable if set.
- The scripts in `f2_physics/` take the repo root as their first argument instead: `python audit2/f2_physics/t1_extgrid.py .`

The scripts were originally run from clones of `e64170f` in a scratch directory. Their hard-coded paths were rewritten to the scheme above, and scratch paths that appear inside logs are shown as `<scratch>`.

**Cached results.**
- `f3_ablations/abl.py` caches per (case, draw set) in `res_<case>_<set>.pkl`, and skips arms that are already present. Delete a pickle to recompute it.
- `f5_vehicle/veh_baselines.py` writes `bl_<case>.pkl`.

**Fast checks.** These re-derive the main tables from the committed per-instance outputs, without any solver:

```bash
python audit2/f3_ablations/verify_ablations.py   # AC ablation tables + paired tests; cross-checks A0/A8 against repro/runs (max diff 0.0)
python audit2/f5_vehicle/veh_analyse.py          # vehicle tables (pooled + vehicle-clustered), from bl_*.pkl
python audit2/f3_ablations/ana.py case118 fresh5+fresh6   # the full ablation table with paired tests vs A0
```

**Full re-run of the four AC final tests.** This takes about 30–60 minutes and runs in its own temporary clone:

```bash
PY=python3 bash audit2/repro/run_repro.sh 6      # 6 = worker pool size; results do not depend on it
```

## Layout and which AUDIT2.md item each part supports

| Folder | What it does | Supports |
|---|---|---|
| `repro/` | `run_repro.sh` re-runs the four pre-registered AC tests; `run_logs/` holds the re-run logs (minus repeated library warnings); `runs/*/final.pkl` holds the re-run outputs (`per_seed`, `rows`, `SR`, `tid`) | §2 Reproduction |
| `f1_protocol/` | protocol, leakage, data and selection checks | §7, C2, M3, Minor |
| `f2_physics/` | slack-bus fix, formulation, references, independent AC-OPF (PYPOWER) | M1, M4, M5, M6, M9, §7 |
| `f3_ablations/` | about 20 AC deployment ablations, run with `55_final_eval.py`'s own deployment code | §1, C1, C2, C3, M2, Appendix A |
| `f4_layer/` | the differentiable layer: gradients, FD checks, what training changes | C1, M8, Minor |
| `f5_vehicle/` | vehicle baselines on the fresh2 test pool | §1, C1, C2, C3, M2 |
| `f6_claims/` | timing of the label-free teacher (offline-compute accounting) | C3 |

### `f1_protocol/`
- `a7_valscores.py` (+ `.log`, `valscores.pkl`): re-computes the label-free validation scores that drove selection, at ρ = 1e4 and 3e5 (M3).
- `a10_seeds.py` (+ `.log`, `seeds.pkl`): the deployed configuration under Gurobi `Seed` 0/1/2. Results are bitwise invariant.
- `a11_perturb.py` (+ `perturb.pkl`): perturbs the V-net output by 1e-7 to measure the noise floor of the selection score (M3).
- `a5_proxy.py` (+ `.log`): replicates the 55 proxy on case118 validation and measures under- vs over-commitment and repair. Also runs a validation-tuned threshold and a label-free downward pass (C2).
- `a2_artifacts.py` (+ `.log`): the stored cut and row selection scores (`val` vs `v_cur`, `sel_hist`) and how far the selected state is from `last` (C1, M3).
- `a8_provenance.py` (+ `.log`): fixing agreement of the committed cut-nets with a BCE fit on own targets vs on reference labels, and units fixed per draw (§7, M3).
- `a9_teacher.py` (+ `.log`): reproduces committed own targets from label-free inputs (5/5 bit-exact).
- `a3_overlap.py`, `a4_blocks.py` (+ `loadscales.pkl`), `a6_refs.py`, `a1_pool2.py` (+ logs): data blocks, overlap, load scales, pool2 references (§7).

### `f2_physics/`
- `evalac.py`: the independent evaluator. It runs a PYPOWER AC-OPF (MATPOWER formulation) at a fixed commitment, and does a raw-voltage feasibility check with an independently built Ybus.
- `t3_case118_check.py <repo> <case> <final.pkl>`: strict price vs PYPOWER for every commitment in a re-run. It writes the `t3_<case>_*.pkl` caches that the PGLib scripts read.
- `t3b_pglib_stats.py` (+ `.log`): PGLib paired statistics under strict pricing vs PYPOWER, and the reference's QCAC cost vs the price of its own commitment (M5).
- `t3c_pglib_decomp.py` (+ `.log`): splits the PGLib saving into no-load and dispatch, and gives the condensers' share of discrete error (M5).
- `t3d_pglib_why.py` (+ `.log`): why the strict price stops above the optimum on PGLib (M5).
- `t4c_refmech.py` (+ `.log`): why the PGLib reference is weak; it stops at slack ≤ 1e-3 with no physics check (M5).
- `t5_excluded.py` (+ `.log`): are the excluded test instances infeasible, or reference-solver failures (M1)?
- `t2_case300_vband.py` (+ `.log`): case300 at its source voltage limits, 0.94/1.06 (M4).
- `t4_refs.py` (+ `.log`): reference quality, convergence and MIP gaps (M1, M6).
- `t4b_noise.py` (+ `t4b_case118.log`): tighter-gap re-solves of references (M6).
- `t3_exact_one.py`: Gurobi global AC-OPF for one commitment. No saved log.
- `t1_extgrid.py`, `t1b_units.py`, `t1c_pglib_fidelity.py` (+ logs): the slack-bus fix, unit limits, branch ratings, and PGLib fidelity to the `.m` file (§3, §7, M9).

### `f3_ablations/`
- `abl.py` (with worker helpers `f3work.py`): runs the arms `A0`–`A12` on case118 validation, the case118 fresh5/fresh6 draws (spent test draws, so these results are post hoc) and case300 fresh5. The arm key is in AUDIT2.md Appendix A.
- `run_f3*.sh`: the order in which the arms were run.
- `res_*.pkl`: the per-instance results.
- `ana_*.txt`: the tables. `ana_f56_full.txt` and `ana_300_full.txt` are the most complete.
- `log_*.txt`: the run logs.
- `verify_ablations.py`: my independent re-tabulation.
- `pairs.py`, `perinst.py`: helpers for paired comparisons.
- `probe768.py`, `probe_res.py`: the two blow-up mechanisms (M2).

### `f4_layer/`
- `common.py`: exact re-creation of the BCE warm start used by `46_unit_cuts_layer.py`.
- `a_warm_vs_deployed.py`, `a2_noise_floor.py` (+ logs): how many fixings layer training changes, against the retraining noise floor (C1).
- `d_cuts.py` (+ logs): the cut gradient equals the imitation gradient; rate of regularised KKT solves (C1, M8).
- `d_rows.py` (+ logs): row-gradient accuracy (M8).
- `b_rows_activity.py` with worker `brow.py` (+ logs and pkl): row activity at the anchor, at a deploy-like relaxation and at the deployed MIQCQP; also the distortion from the γ = 1000 smoothing (C1).
- `d_vgrad.py` (+ logs): FD check of the V-net gradient at the deployed `_env_nc` V-nets (M8).
- `i_tol.py <case> <instance ids>`: whether the small-h FD inconsistency comes from a kink or from solver tolerance (M8). No saved log.
- `c_vnet.py`, `c2_vnet_loss.py` (+ logs): what the V-net predicts, and its training objective (Minor, §7).
- `g_rank.py`, `g_run.py` (+ log and pkl): whether the layer does any work in label generation (Minor).
- `h_vehicle.py` (+ `.log`): the vehicle layer gradient vs the imitation gradient.
- `d_degen.sh` (+ log): re-runs `gradient_checks/03` and `degeneracy.py`.
- `chain*.sh`: the run order.

### `f5_vehicle/`
- `veh_baselines.py [case|all] [nproc]`: runs all vehicle arms on the fresh2 test pool and writes `bl_<case>.pkl` (Gurobi 13.0.3 set).
- `veh_analyse.py` → `analyse.txt`: pooled and vehicle-clustered tables, paired tests, and the excluded instances.
- `analyse_g12.txt`: the same analysis under Gurobi 12.0.3. Its pickles are not included; re-create them with `pip install gurobipy==12.0.3` and re-run.
- `test_rerun*.log`: re-runs of `dixit_vehicle/16_vehicle_final.py test` under Gurobi 13.0.3, under cvxpy 1.9.2 and under Gurobi 12.0.3.

### `f6_claims/`
- `time_teacher.py` (+ log): per-instance time of the label-free best-of teacher on case118 training draws (C3).

## Caveats

- **Timing.** Wall-clock times were measured while other audit jobs were running, so treat them as relative only. Gurobi runs use `Threads=1` with `WorkLimit` and are deterministic in a given environment, but outcomes on a few knife-edge instances change with the solver version (AUDIT2.md M2).
- **Post-hoc draws.** The fresh5/fresh6 ablations are diagnostics on test draws that were already spent. They are not new tests.
- **Seeds.** The vehicle analysis is seed-averaged over seeds 0/1/2. The AC ablations use proxy seed 0, and ours is a single deterministic pipeline.

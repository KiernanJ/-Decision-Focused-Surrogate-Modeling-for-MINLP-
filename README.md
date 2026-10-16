# Label-free learned cuts for MINLP through an exact differentiable optimization layer

A short Gurobi solve is made reliable by cuts that a neural network learns **without optimal solutions**. The network is trained through an
exact differentiable conic layer; its own best answers on training loads are the only targets. The baseline is a supervised NN proxy that
needs a converged reference solve for every training load.

Status (2026-10-16): the clean method is tested on **case118, case300, PGLib-118 (api) and the hybrid vehicle (MILP and MINLP)**, each on untouched test
draws scored once. Generalized integer cuts (a learned Hamming-ball row instead of per-unit fixings) are an exploratory, validation-only study (last section).
The full log (pre-registrations, test results, audits and disclosures) is `acuc/RESULTS.md`.

## The method in 6 steps
1. **Problem and data.** AC unit commitment (QCAC convexification of AC power flow; IEEE 118-bus, 54 units, load scale 0.55-1.30) and a
   30-step hybrid-vehicle problem (MILP of Dixit et al., TMLR 2025; and a convex MIQCQP). Input = the load / demand profile.
2. **V-net (AC only).** MLP load -> linearization point V of the relaxation; trained through the exact conic layer on the *uncut*
   relaxation, label-free (`acuc/40_vnet_layer.py ... nocuts`).
3. **Own targets.** For each training load, the best of {own short solve + exact-layer local search, two rounding restorations}, priced
   strictly; nothing longer than the deployment budget, no reference (`acuc/60_selfteacher.py`, `dixit_vehicle/18_lf_teacher.py`).
   Plus 300 extra unlabeled loads per case.
4. **Cut-net.** Per-unit integer cuts (units fixed on/off above a confidence c, chosen by a label-free validation score), fine-tuned
   through the layer (`acuc/46_unit_cuts_layer.py`).
5. **Row-net.** K=2 general rows `alpha'pg + a'u <= b`, anchored at the relaxation (`acuc/52_general_rows_on_top.py`).
6. **Deployment.** forward passes -> anchor solve -> ONE short MIQCQP (WorkLimit 10; 30 for case300; 0.01 for the vehicle) with the cuts
   and rows as soft constraints -> strict AC pricing. The deployment penalty rho is chosen by the label-free score of the full pipeline on
   the validation loads. Label-free retry/fallback only (`acuc/55_final_eval.py`, `dixit_vehicle/16_vehicle_final.py`).

## Results (untouched test draws, each scored once, 3 seeds; reference = converged sequential QCAC / exact Gurobi)
gap mean | gap median | discrete error | continuous error | median time

| problem | ours (no labels) | supervised NN proxy | paired test (pre-registered) |
|---|---|---|---|
| case118, set 1 (39 loads) | 1.025% / 0.056% / 6.03% / 7.77% / 10.9 s | 1.051% / 0.021% / 6.28% / 6.80% / 0.29 s | tie (Wilcoxon p=0.39) |
| case118, set 2 (40 loads) | 0.107% / 0.083% / 2.92% / 5.68% / 11.4 s | 0.807% / 0.021% / 5.45% / 6.48% / 0.36 s | lower mean, not significant (p=0.26) |
| case300 (40 loads) | 0.0825% / 0.0132% / 2.68% / 3.74% / 39.3 s | 0.0872% / 0.0103% / 2.74% / 3.07% / 1.17 s | tie (p=0.74) |
| PGLib-118 api (42 loads) | 0.391% / 0.296% / 7.10% / 6.58% / 10.1 s | 0.105% / 0.006% / 1.75% / 1.17% / 2.05 s | proxy closer to the reference (p=8.7e-6); signed cost: ours cheaper (-0.33 pp, p=8e-7) |
| vehicle MILP, 8 vehicles (734 profiles) | 0.084% / 0.001% / 9.25% / 9.21% / 10.9 ms | 0.876% / 0.007% / 9.91% / 9.51% / 4.0 ms | better on 8/8 vehicles (sign test p=0.0078) |
| vehicle MINLP (48 profiles) | 0.532% / 0.243% / 5.07% / 4.84% / 18.2 ms | 0.759% / 0.473% / 6.96% / 6.45% / 7.1 ms | better, p=0.018 |

On PGLib the two pre-registered metrics disagree (|gap| to the reference vs signed cost); the PGLib reference is a weak heuristic, so "cheaper than the reference" is not a proven advantage.

Training data: ours uses 444 / 422 / 355 loads on case118 / case300 / PGLib (own targets, incl. 300 extra unlabeled loads) and 400 / 444 profiles on the vehicle MILP / MINLP; the proxy uses the
labelled loads only (118 on case118, 114 on case300, 118 on PGLib, up to 100 / 144 on the vehicle). Disclosures, exclusions and audits: `acuc/RESULTS.md`.

## Layout
- `src/` library: conic layer (`conelayer.py`, `socp.py`), QCAC model and solvers (`acuc.py`, `combopt_ac.py`), cut networks
  (`cutnet_k.py`, `cuttrain.py`), label-free targets (`selfsup.py`), vehicle layer (`vehicle_layer.py`), data blocks (`data2.py`).
- `acuc/` AC pipeline scripts (numbered in run order) and `RESULTS.md`; `acuc/clean_redo_*.sh` run the full clean pipeline.
- `dixit_vehicle/` vehicle pipeline (`18_lf_teacher.py` -> `16_vehicle_final.py train | test`).
- `gradient_checks/` finite-difference checks of the layer and the V gradient.
- `figures/` vehicle figures (AC figures are regenerated after the case300 / PGLib tests).
- `acuc/results/` trained clean-pipeline artifacts (V-net `_env_nc`, cuts / rows `_lfc`, label-free targets `_nc`) and test references.

Superseded work (older method variants, previous paper, diagnostics, the long-solve teachers `teacher_*_w120/w180.json`, the old checkpoints and the old long-solve scripts) was removed from the repository; it remains in the git history.
Note: `40_vnet_layer.py` (without `nocuts`), `46_unit_cuts_layer.py` and `52_general_rows_on_top.py` still have a default teacher path that pointed to the removed long-solve files; always pass the label-free targets explicitly, exactly as `acuc/clean_redo_*.sh` does (the code is unchanged so the pre-registered hashes in `acuc/RESULTS.md` still match).

## Generalized integer cuts (exploratory, validation only, one model each)
The per-unit cuts `u_k >= l_k / u_k <= h_k` were replaced by general linear rows over the integer variables (no per-unit fixing anywhere; `acuc/67_generalized_integer_cuts.py`,
`dixit_vehicle/20_hamming_vehicle.py`). Learned integer rows (K = 2, 8, 32, with or without 2 mixed rows) did **not** work on case118 (label-free score 0.969-0.982 vs 0.962; validation gap 0.9-2.4%).
One learned **Hamming-ball row** (`sum_k p_k(1-u_k) + (1-p_k)u_k <= r`, centre from a network trained on our own targets, r chosen label-free) matched the current method on validation:
case118 0.119% vs 0.129% gap, case300 0.045% vs 0.052%; it was worse on the vehicle MINLP (1.40% vs 0.33%), roughly equal on the easy MILP vehicle 0, and failed the criterion at 118 training loads (3 seeds).
These are validation numbers selected on the same 24 draws (optimistic), with no test; the Hamming row is the local-branching / predict-and-search family, and r = 1-3 restricts about as strongly as fixing.

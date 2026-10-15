# Label-free learned cuts for MINLP through an exact differentiable optimization layer

A short Gurobi solve is made reliable by cuts that a neural network learns **without optimal solutions**. The network is trained through an
exact differentiable conic layer; its own best answers on training loads are the only targets. The baseline is a supervised NN proxy that
needs a converged reference solve for every training load.

Status (2026-10-14): **case118 and both hybrid-vehicle problems are final.** The clean case300 and PGLib-api pipelines are trained and
validated; their untouched-test results are pending and will be added here. The full log (pre-registrations, test results, audits and
disclosures) is `acuc/RESULTS.md`.

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
| case118, fresh5 (39 loads) | 1.025% / 0.056% / 6.03% / 7.77% / 10.9 s | 1.051% / 0.021% / 6.28% / 6.80% / 0.29 s | tie (Wilcoxon p=0.39) |
| case118, fresh6 (40 loads) | 0.107% / 0.083% / 2.92% / 5.68% / 11.4 s | 0.807% / 0.021% / 5.45% / 6.48% / 0.36 s | lower mean, not significant (p=0.26) |
| vehicle MILP, 8 vehicles (734 profiles) | 0.084% / 0.001% / 9.25% / 9.21% / 10.9 ms | 0.876% / 0.007% / 9.91% / 9.51% / 4.0 ms | better on 8/8 vehicles (sign test p=0.0078) |
| vehicle MINLP (48 profiles) | 0.532% / 0.243% / 5.07% / 4.84% / 18.2 ms | 0.759% / 0.473% / 6.96% / 6.45% / 7.1 ms | better, p=0.018 |

Training data: ours uses 444 loads on case118 (144 + 300 unlabeled) and 400 / 444 profiles on the vehicle MILP / MINLP; the proxy uses the
labelled loads only (118 on case118, up to 100 / 144 on the vehicle). Disclosures, exclusions and audits: `acuc/RESULTS.md`.

## Layout
- `src/` library: conic layer (`conelayer.py`, `socp.py`), QCAC model and solvers (`acuc.py`, `combopt_ac.py`), cut networks
  (`cutnet_k.py`, `cuttrain.py`), label-free targets (`selfsup.py`), vehicle layer (`vehicle_layer.py`), data blocks (`data2.py`).
- `acuc/` AC pipeline scripts (numbered in run order) and `RESULTS.md`; `acuc/clean_redo_*.sh` run the full clean pipeline.
- `dixit_vehicle/` vehicle pipeline (`18_lf_teacher.py` -> `16_vehicle_final.py train | test`).
- `gradient_checks/` finite-difference checks of the layer and the V gradient.
- `figures/` vehicle figures (AC figures are regenerated after the case300 / PGLib tests).
- `acuc/results/` trained clean-pipeline artifacts (V-net `_env_nc`, cuts / rows `_lfc`, label-free targets `_nc`) and test references.

Superseded work (older method variants, previous paper, diagnostics) was removed from the repository.

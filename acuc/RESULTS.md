# RESULTS — every run, one place (started 2026-10-10)
Metrics always: **gap mean | gap median | discrete error | continuous error** vs the converged reference
(sequential QCAC, physics-checked pricing). Ours = our method; proxy = supervised NN proxy (only comparison arm).

## Final tables so far (3 seeds; V-net trained with the OLD gradient — to be redone)
| case | method | gap mean | gap median | discrete err | continuous err |
|---|---|---|---|---|---|
| case118 (untouched test, 38) | ours: learned V + imitation fixing cuts | 0.064 ± 0.041% | 0.021% | 3.57% | 7.62% |
| | supervised NN proxy | 1.303 ± 0.228% | 0.054% | 8.07% | 8.23% |
| case300 (fresh test, 37) | ours | 0.063 ± 0.005% | 0.021% | 3.32% | 3.41% |
| | supervised NN proxy | 0.044 ± 0.006% | 0.008% | 2.55% | 3.26% |
| hard case118 PGLib api (split test, 38) | ours | 0.289 ± 0.004% | 0.117% | 4.37% | 1.64% |
| | supervised NN proxy | 0.118 ± 0.002% | 0.019% | 1.56% | 1.34% |
Caveats: case300/hard-case118 test sets were touched by design diagnostics; reference times include the polish step.

## 2026-10-10 method experiments (VALIDATION, case118, 21 instances with references unless noted)
| version | gap mean | gap median | discrete err | continuous err | note |
|---|---|---|---|---|---|
| current: learned V (corrected gradient) + imitation fixing cuts, MIQCQP W=10 | 0.018% | 0.0001% | 2.82% | 4.78% | best so far |
| no cuts, MIQCQP W=10 | 3.41% | 0.053% | 16.75% | 9.64% | reference point |
| per-unit integer cuts fine-tuned through exact layer | 0.021% | 0.0001% | 2.91% | 4.89% | = current (20/21 identical) |
| generalized integer cuts 2/8/16 rows through layer, MIQCQP | 3.41% (untrained) | | | | trained: +3.5–5% cost vs no cuts (weights not saved — mistake) |
| general+integer cuts through layer, MIQCQP | — | | | | +4–7% cost vs current (label-free score only) |
| hard case118: per-unit cuts, cost-aware loss (beta 0.01/0.1/1) | — | | | | tie with current (label-free score only) |
Lessons: V-net VJP gradient was wrong off flat V (fixed: envelope gradient); relaxation solution derivative needs smoothing
(gamma=1000); hard cut bounds made the training layer infeasible (fixed: soft rows); cuts inside a short MIQCQP need to
shrink the search — generalized cuts did not.

## Run log (one line per run from now on)
| run folder | summary |
|---|---|
| 2026-10-10_1634_surrogate_check_case118 | case118 VALIDATION surrogate WITH learned cuts (old checkpoint): gap mean | median | disc | cont | integral | no repair = 6.8830% | 7.7619% | 42.50% | 29.83% | 0/21 | 0/21 |
| 2026-10-10_1634_surrogate_check_case118 | case118 VALIDATION surrogate WITHOUT cuts (same V): gap mean | median | disc | cont | integral | no repair = 7.5595% | 7.7619% | 43.92% | 39.19% | 0/21 | 0/21 |
| 2026-10-10_1640_general_rows_on_top_case118 | case118 VALIDATION integer cuts only (layer-trained): gap mean | median | disc | cont = 10.3913% | 12.1870% | 44.36% | 19.40% |
| 2026-10-10_1640_general_rows_on_top_case118 | case118 VALIDATION + general rows, UNTRAINED: gap mean | median | disc | cont = 10.3913% | 12.1870% | 44.36% | 19.40% |
| 2026-10-10_1640_general_rows_on_top_case118 | case118 VALIDATION + general rows, TRAINED: gap mean | median | disc | cont = 10.5334% | 12.1870% | 45.59% | 21.51% |
| 2026-10-10_1652_general_rows_on_top_RESCORE_case118 | case118 VALIDATION integer cuts only (layer-trained): gap mean | median | disc | cont = 0.0205% | 0.0001% | 2.91% | 4.89% |
| 2026-10-10_1652_general_rows_on_top_RESCORE_case118 | case118 VALIDATION + general rows, UNTRAINED: gap mean | median | disc | cont = 0.6543% | 0.0244% | 5.56% | 7.03% |
| 2026-10-10_1652_general_rows_on_top_RESCORE_case118 | case118 VALIDATION + general rows, TRAINED: gap mean | median | disc | cont = 0.0205% | 0.0001% | 2.91% | 4.89% |

## !! BUG (found 2026-10-10 ~17:00): combopt_ac.miqcqp read 2G-column cut rows ([pg; u]) as u-only rows.
Every MIQCQP evaluation of the GENERAL/INTEGER cut categories before this fix is INVALID (training through the layer was
correct; the deployment/validation scoring was wrong): 44_cuts_layer_train, 45_cuts_tighten (both + int), 50_int_cuts_rows
(8/16 rows), 52 first report. Fixed + verified (u-only rows and 2G rows give the identical commitment).
| run (after fix) | version (case118 VALIDATION, 21 inst, W=10) | gap mean | gap median | discrete err | continuous err |
|---|---|---|---|---|---|
| 53_rescore (saved 52 weights) | integer cuts only (per-unit, layer-trained) | 0.0205% | 0.0001% | 2.91% | 4.89% |
| | + general rows UNTRAINED | 0.6543% | 0.0244% | 5.56% | 7.03% |
| | + general rows TRAINED through exact layer | 0.0205% | 0.0001% | 2.91% | 4.89% |
| 2026-10-10_1658_int_cuts_rows8_case118_postfix | case118 VALIDATION CURRENT METHOD (fixing cuts): gap mean | median | disc | cont = 0.0184% | 0.0001% | 2.82% | 4.78% |
| 2026-10-10_1658_int_cuts_rows8_case118_postfix | case118 VALIDATION no cuts: gap mean | median | disc | cont = 3.4135% | 0.0533% | 16.75% | 9.64% |
| 2026-10-10_1658_int_cuts_rows8_case118_postfix | case118 VALIDATION gen. integer cuts 8 rows, UNTRAINED: gap mean | median | disc | cont = 3.7294% | 0.1500% | 14.55% | 14.63% |
| 2026-10-10_1658_int_cuts_rows8_case118_postfix | case118 VALIDATION gen. integer cuts 8 rows, TRAINED (ep 14): gap mean | median | disc | cont = 4.0648% | 0.0533% | 19.84% | 11.24% |
| 2026-10-10_1658_int_cuts_rows16_case118_postfix | case118 VALIDATION CURRENT METHOD (fixing cuts): gap mean | median | disc | cont = 0.0184% | 0.0001% | 2.82% | 4.78% |
| 2026-10-10_1658_int_cuts_rows16_case118_postfix | case118 VALIDATION no cuts: gap mean | median | disc | cont = 3.4135% | 0.0533% | 16.75% | 9.64% |
| 2026-10-10_1658_int_cuts_rows16_case118_postfix | case118 VALIDATION gen. integer cuts 16 rows, UNTRAINED: gap mean | median | disc | cont = 6.0245% | 0.3973% | 23.99% | 14.29% |
| 2026-10-10_1658_int_cuts_rows16_case118_postfix | case118 VALIDATION gen. integer cuts 16 rows, TRAINED (ep 14): gap mean | median | disc | cont = 3.0415% | 0.0533% | 15.78% | 11.54% |
| 2026-10-10_1722_int_cuts_rows8_case118_postfix_selected_epoch | case118 VALIDATION CURRENT METHOD (fixing cuts): gap mean | median | disc | cont = 0.0184% | 0.0001% | 2.82% | 4.78% |
| 2026-10-10_1722_int_cuts_rows8_case118_postfix_selected_epoch | case118 VALIDATION no cuts: gap mean | median | disc | cont = 3.4135% | 0.0533% | 16.75% | 9.64% |
| 2026-10-10_1722_int_cuts_rows8_case118_postfix_selected_epoch | case118 VALIDATION gen. integer cuts 8 rows, SELECTED (ep 5): gap mean | median | disc | cont = 2.5607% | 0.0533% | 12.87% | 9.37% |
| 2026-10-10_1722_int_cuts_rows16_case118_postfix_selected_epoch | case118 VALIDATION CURRENT METHOD (fixing cuts): gap mean | median | disc | cont = 0.0184% | 0.0001% | 2.82% | 4.78% |
| 2026-10-10_1722_int_cuts_rows16_case118_postfix_selected_epoch | case118 VALIDATION no cuts: gap mean | median | disc | cont = 3.4135% | 0.0533% | 16.75% | 9.64% |
| 2026-10-10_1722_int_cuts_rows16_case118_postfix_selected_epoch | case118 VALIDATION gen. integer cuts 16 rows, SELECTED (ep 5): gap mean | median | disc | cont = 3.7956% | 0.0533% | 18.61% | 11.56% |
| 2026-10-10_1731_FINAL_case118_seeds0 | FINAL case118 untouched (38 inst, seeds [0]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.0411% | 0.0249% | 3.65% | 7.50% | 11.96s | 0 |
| 2026-10-10_1731_FINAL_case118_seeds0 | FINAL case118 untouched (38 inst, seeds [0]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 1.5507% | 0.0204% | 8.53% | 6.99% | 0.36s | 0 |

## Decisions + label-leakage check (2026-10-10 ~21:00)
- FINAL METHOD: learned V (V-net, exact layer, envelope gradient) + per-unit INTEGER cuts (imitation warm start of our own
  long solves, then fine-tuned through the exact layer; LABEL-FREE validation decides whether the fine-tune is kept
  = decision A) + GENERAL mixed rows (2, trained through the exact layer only) -> one short MIQCQP (W) at the learned V.
  case118: layer fine-tune selected; case300: warm start kept (layer version 0.25% worse on validation).
- Label check (code lines audited): no reference solution is used to train or select any part of our method; references
  appear only in [diag only] prints after selection, validation REPORTS, and final scoring. Supervised proxy uses them by design.
- Disclose: design-level choices (rho candidates, cut-constrained loss) were informed by reference-based diagnostics
  (oracle V, V accuracy); from now on method decisions use the label-free score only, four metrics are report-only.
| 2026-10-10_2056_general_rows_on_top_case300 | case300 VALIDATION integer cuts only (layer-trained): gap mean | median | disc | cont = 0.0435% | 0.0247% | 3.69% | 4.81% |
| 2026-10-10_2056_general_rows_on_top_case300 | case300 VALIDATION + general rows, UNTRAINED: gap mean | median | disc | cont = 0.1499% | 0.0440% | 5.07% | 7.00% |
| 2026-10-10_2056_general_rows_on_top_case300 | case300 VALIDATION + general rows, TRAINED: gap mean | median | disc | cont = 0.0557% | 0.0152% | 3.43% | 4.46% |

## HYBRID VEHICLE — headroom check (2026-10-10 late, VALIDATION only, dixit_vehicle/10_headroom.py), no cuts, short WorkLimit
| version | W | gap mean | gap median | discrete err | continuous err | time | exact time |
|---|---|---|---|---|---|---|---|
| MILP (Dixit) veh 1/2/4/9 | 0.01 | 0.15-0.23% | 0.07-0.17% | 15-23% | 15-23% | ~20 ms | 1.9-7.0 s |
| | 0.05 | 0.05-0.08% | 0-0.05% | 7.5-15% | 7.4-15% | ~85 ms | |
| | 0.20 | 0.009-0.020% | 0% | 1.9-5.7% | 1.9-5.7% | ~350 ms | |
| MINLP (convex MIQCQP) | 0.01 | 1.045% | 0.527% | 6.61% | 6.11% | ~20 ms | 0.9 s |
| | 0.05 | 0.049% | 0.033% | 2.64% | 2.68% | ~90 ms | |
=> real headroom at W=0.01 (~20 ms, 50-350x faster than exact) in BOTH versions -> same cut idea applies.

### PROPOSED vehicle plan (awaiting Parth's approval; nothing long started)
Same method as AC minus V: (1) teacher = OUR OWN longer solves on training profiles (label-free; W_long ~ 1.0);
(2) INTEGER cuts per time step on z_t in {0..3}: z_t >= l_t(D), z_t <= h_t(D) (cut-net imitation warm start, then
fine-tuned through the exact layer = smoothed LP/QP relaxation, soft rows; label-free validation keeps it or not);
(3) GENERAL mixed rows alpha'[E, P_eng] + a'z <= b (2 rows) trained through the exact layer;
(4) deploy = one short solve (W = 0.01) with the cuts; (5) only comparison = supervised NN proxy (round + fixed-z re-solve);
(6) four metrics + time; selection label-free on validation; untouched test once. Both MILP (8 non-trivial Dixit vehicles)
and MINLP. Gates (FD at trained points) before every training; unit test of the cut-row format before any experiment.
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v0 VALIDATION W=0.01 no cuts: gap mean | median | disc | cont | time = 0.026% | 0.000% | 3.83% | 3.74% | 23 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v0 VALIDATION W=0.01 imitation integer cuts (c=0.999, label-free): gap mean | median | disc | cont | time = 0.003% | 0.000% | 1.02% | 1.02% | 1 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v1 VALIDATION W=0.01 no cuts: gap mean | median | disc | cont | time = 0.154% | 0.073% | 16.69% | 16.90% | 20 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v1 VALIDATION W=0.01 imitation integer cuts (c=0.999, label-free): gap mean | median | disc | cont | time = 0.116% | 0.000% | 7.42% | 7.06% | 10 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v2 VALIDATION W=0.01 no cuts: gap mean | median | disc | cont | time = 0.193% | 0.085% | 14.92% | 14.94% | 18 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v2 VALIDATION W=0.01 imitation integer cuts (c=0.999, label-free): gap mean | median | disc | cont | time = 0.107% | 0.002% | 11.17% | 10.87% | 4 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v4 VALIDATION W=0.01 no cuts: gap mean | median | disc | cont | time = 0.165% | 0.166% | 22.30% | 22.14% | 17 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v4 VALIDATION W=0.01 imitation integer cuts (c=0.999, label-free): gap mean | median | disc | cont | time = 0.020% | 0.000% | 3.93% | 3.90% | 4 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v6 VALIDATION W=0.01 no cuts: gap mean | median | disc | cont | time = 0.188% | 0.082% | 17.51% | 17.09% | 23 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v6 VALIDATION W=0.01 imitation integer cuts (c=0.999, label-free): gap mean | median | disc | cont | time = 0.049% | 0.000% | 4.39% | 4.48% | 8 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v7 VALIDATION W=0.01 no cuts: gap mean | median | disc | cont | time = 0.069% | 0.021% | 12.19% | 12.17% | 26 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v7 VALIDATION W=0.01 imitation integer cuts (c=0.999, label-free): gap mean | median | disc | cont | time = 0.001% | 0.000% | 0.56% | 0.55% | 2 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v8 VALIDATION W=0.01 no cuts: gap mean | median | disc | cont | time = 0.257% | 0.259% | 25.43% | 25.28% | 23 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v8 VALIDATION W=0.01 imitation integer cuts (c=0.999, label-free): gap mean | median | disc | cont | time = 0.019% | 0.000% | 5.24% | 5.18% | 2 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v9 VALIDATION W=0.01 no cuts: gap mean | median | disc | cont | time = 0.225% | 0.155% | 23.10% | 22.98% | 24 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE milp_v9 VALIDATION W=0.01 imitation integer cuts (c=0.999, label-free): gap mean | median | disc | cont | time = 0.035% | 0.003% | 11.03% | 10.81% | 9 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE minlp VALIDATION W=0.01 no cuts: gap mean | median | disc | cont | time = 1.045% | 0.527% | 6.61% | 6.11% | 17 ms |
| 2026-10-10_2227_vehicle_stage1_fixcuts | VEHICLE minlp VALIDATION W=0.01 imitation integer cuts (c=0.9, label-free): gap mean | median | disc | cont | time = 0.313% | 0.173% | 4.59% | 4.29% | 19 ms |

## HYBRID VEHICLE stage 1 (2026-10-10 22:30, VALIDATION, W=0.01): imitation per-step integer cuts (label-free teacher)
| version | method | gap mean | gap median | discrete err | continuous err | time |
|---|---|---|---|---|---|---|
| MILP mean of 8 vehicles (0,1,2,4,6,7,8,9) | no cuts | 0.160% | 0.096% | 17.0% | 16.9% | ~22 ms |
| | imitation integer cuts (c=0.999) | 0.044% | 0.001% | 5.6% | 5.5% | ~5 ms |
| MINLP | no cuts | 1.045% | 0.527% | 6.61% | 6.11% | 17 ms |
| | imitation integer cuts (c=0.9) | 0.313% | 0.173% | 4.59% | 4.29% | 19 ms |
Per-vehicle lines in the run log below. Next (pending approval): layer fine-tune of the integer cuts, general rows,
supervised proxy, untouched test once.

**Vehicle exact layer, FD gate (2026-10-10 22:30, `dixit_vehicle/13_gate_layer.py`, `src/vehicle_layer.py`).** Per-step soft integer-cut rows
(-z_t <= -l_t, z_t <= h_t) in the LP (MILP) / QCQP (MINLP) relaxation, smoothing gamma=1, exact KKT VJP of the tracking loss w.r.t. (l, h),
evaluated at the stage-1 imitation cuts on 3 training profiles each. MILP vehicle 4: 3/3 PASS (corr 1.0000, signs 8/8). MINLP: 3/3 PASS
(corr 1.0000, signs 8/8). FD self-consistency >= 0.999 everywhere. The layer is ready for the fine-tune stage (still pending approval).
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v0 VALIDATION W=0.01 stage-1 imitation cuts: gap mean | median | disc | cont | time = 0.003% | 0.000% | 1.02% | 1.02% | 1 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v0 VALIDATION W=0.01 layer-trained cuts (ep -1): gap mean | median | disc | cont | time = 0.003% | 0.000% | 1.02% | 1.02% | 1 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v1 VALIDATION W=0.01 stage-1 imitation cuts: gap mean | median | disc | cont | time = 0.116% | 0.000% | 7.42% | 7.06% | 10 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v1 VALIDATION W=0.01 layer-trained cuts (ep 2): gap mean | median | disc | cont | time = 0.115% | 0.000% | 7.42% | 7.06% | 6 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v2 VALIDATION W=0.01 stage-1 imitation cuts: gap mean | median | disc | cont | time = 0.107% | 0.002% | 11.17% | 10.87% | 4 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v2 VALIDATION W=0.01 layer-trained cuts (ep -1): gap mean | median | disc | cont | time = 0.107% | 0.002% | 11.17% | 10.87% | 4 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v4 VALIDATION W=0.01 stage-1 imitation cuts: gap mean | median | disc | cont | time = 0.020% | 0.000% | 3.93% | 3.90% | 5 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v4 VALIDATION W=0.01 layer-trained cuts (ep -1): gap mean | median | disc | cont | time = 0.020% | 0.000% | 3.93% | 3.90% | 5 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v6 VALIDATION W=0.01 stage-1 imitation cuts: gap mean | median | disc | cont | time = 0.049% | 0.000% | 4.39% | 4.48% | 11 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v6 VALIDATION W=0.01 layer-trained cuts (ep -1): gap mean | median | disc | cont | time = 0.049% | 0.000% | 4.39% | 4.48% | 6 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v7 VALIDATION W=0.01 stage-1 imitation cuts: gap mean | median | disc | cont | time = 0.001% | 0.000% | 0.56% | 0.55% | 1 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v7 VALIDATION W=0.01 layer-trained cuts (ep -1): gap mean | median | disc | cont | time = 0.001% | 0.000% | 0.56% | 0.55% | 1 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v8 VALIDATION W=0.01 stage-1 imitation cuts: gap mean | median | disc | cont | time = 0.019% | 0.000% | 5.24% | 5.18% | 2 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v8 VALIDATION W=0.01 layer-trained cuts (ep -1): gap mean | median | disc | cont | time = 0.019% | 0.000% | 5.24% | 5.18% | 2 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v9 VALIDATION W=0.01 stage-1 imitation cuts: gap mean | median | disc | cont | time = 0.035% | 0.003% | 11.03% | 10.81% | 6 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE milp_v9 VALIDATION W=0.01 layer-trained cuts (ep 5): gap mean | median | disc | cont | time = 0.035% | 0.001% | 10.65% | 10.43% | 5 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE minlp VALIDATION W=0.01 stage-1 imitation cuts: gap mean | median | disc | cont | time = 0.313% | 0.173% | 4.59% | 4.29% | 11 ms |
| 2026-10-10_2233_vehicle_stage2_cuts_layer | VEHICLE minlp VALIDATION W=0.01 layer-trained cuts (ep -1): gap mean | median | disc | cont | time = 0.313% | 0.173% | 4.59% | 4.29% | 11 ms |

## HYBRID VEHICLE stage 2 (2026-10-10 23:30, VALIDATION, W=0.01): integer cuts fine-tuned through the EXACT LAYER (`dixit_vehicle/14_cuts_layer.py`)
Warm start = stage-1 cuts (epoch -1 reproduced stage 1 exactly on all 9 cases). Label-free validation picks the epoch (epoch -1 eligible).
Result: same as AC case300 / hard case118 — the layer fine-tune is nearly inert. Kept stage-1 cuts on 7/9; changed only MILP vehicles 1 and 9:
| case | cuts | gap mean | gap median | discrete err | continuous err | time |
|---|---|---|---|---|---|---|
| MILP vehicle 1 | stage-1 | 0.116% | 0.000% | 7.42% | 7.06% | 10 ms |
| | layer-trained (ep 2) | 0.115% | 0.000% | 7.42% | 7.06% | 6 ms |
| MILP vehicle 9 | stage-1 | 0.035% | 0.003% | 11.03% | 10.81% | 6 ms |
| | layer-trained (ep 5) | 0.035% | 0.001% | 10.65% | 10.43% | 5 ms |
| MILP mean of 8 | final | 0.043% | 0.001% | 5.5% | 5.4% | ~4 ms |
| MINLP | final (= stage-1) | 0.313% | 0.173% | 4.59% | 4.29% | 11 ms (1/24 no solution) |
Weights (warm/best/last) in `acuc/runs/2026-10-10_2233_vehicle_stage2_cuts_layer/`. Next (pending approval): general rows, supervised proxy, untouched test.
| 2026-10-11_0118_FINAL_case300_seeds0 | FINAL case300 untouched (37 inst, seeds [0]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.0469% | 0.0187% | 2.98% | 3.16% | 31.67s | 0 |
| 2026-10-11_0118_FINAL_case300_seeds0 | FINAL case300 untouched (37 inst, seeds [0]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.0770% | 0.0058% | 2.39% | 2.72% | 1.07s | 0 |
| 2026-10-11_0122_FINAL_pglib_opf_case118_ieee__api_seeds0 | FINAL pglib_opf_case118_ieee__api untouched (37 inst, seeds [0]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.3898% | 0.2108% | 5.96% | 3.31% | 4.74s | 0 |
| 2026-10-11_0122_FINAL_pglib_opf_case118_ieee__api_seeds0 | FINAL pglib_opf_case118_ieee__api untouched (37 inst, seeds [0]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.1020% | 0.0036% | 1.50% | 1.34% | 1.87s | 0 |

## AC FINAL, seed 0, untouched test (fresh2, seed 4242) — all three cases (2026-10-11 01:30). Reference = converged sequential QCAC (+polish); instances without a converged reference excluded (listed in each run log)
| case | method | gap mean | gap median | discrete err | continuous err | time median | paired ours−proxy (95% CI, Wilcoxon p) |
|---|---|---|---|---|---|---|---|
| case118 (38/48) | OUR METHOD | 0.041% | 0.025% | 3.65% | 7.50% | 12.0 s | −1.51 pp [−2.86, −0.44], p=0.030 → ours better |
| | supervised NN proxy | 1.551% | 0.020% | 8.53% | 6.99% | 0.36 s | |
| case300 (37/48) | OUR METHOD | 0.047% | 0.019% | 2.98% | 3.16% | 31.7 s | −0.03 pp [−0.10, +0.01], p=0.85 → tie (1 fallback) |
| | supervised NN proxy | 0.077% | 0.006% | 2.39% | 2.72% | 1.07 s | |
| hard case118 / PGLib api (37/48) | OUR METHOD | 0.390% | 0.211% | 5.96% | 3.31% | 4.7 s | +0.29 pp [+0.14, +0.41], p=1e-6 → proxy better (8 fallbacks) |
| | supervised NN proxy | 0.102% | 0.004% | 1.50% | 1.34% | 1.87 s | |
Phase B (V-net seeds 1,2 for all three cases, then 3-seed finals) started 01:30.
| 2026-10-11_0606_FINAL_case118_seeds0-1-2 | FINAL case118 untouched (38 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.0343+-0.0017% | 0.0193+-0.0006% | 3.53+-0.11% | 6.99+-0.23% | 9.34+-0.38s | 0 |
| 2026-10-11_0606_FINAL_case118_seeds0-1-2 | FINAL case118 untouched (38 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 1.3027+-0.2285% | 0.0537+-0.0310% | 8.07+-0.49% | 8.23+-1.41% | 0.29+-0.01s | 0 |
| 2026-10-11_0608_FINAL_pglib_opf_case118_ieee__api_seeds0-1-2 | FINAL pglib_opf_case118_ieee__api untouched (37 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.3919+-0.0019% | 0.2108+-0.0000% | 5.79+-0.18% | 3.17+-0.14% | 6.16+-2.25s | 0 |
| 2026-10-11_0608_FINAL_pglib_opf_case118_ieee__api_seeds0-1-2 | FINAL pglib_opf_case118_ieee__api untouched (37 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.1283+-0.0237% | 0.0031+-0.0007% | 1.47+-0.10% | 1.50+-0.26% | 1.81+-0.04s | 0 |
| 2026-10-11_0642_FINAL_case300_seeds0-1-2 | FINAL case300 untouched (37 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.0519+-0.0015% | 0.0186+-0.0014% | 3.02+-0.08% | 3.29+-0.14% | 32.23+-0.77s | 0 |
| 2026-10-11_0642_FINAL_case300_seeds0-1-2 | FINAL case300 untouched (37 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.0746+-0.0041% | 0.0075+-0.0030% | 2.39+-0.16% | 2.76+-0.16% | 1.10+-0.01s | 0 |

## AC FINAL, 3 SEEDS (0,1,2), untouched test — THE PAPER TABLE (2026-10-11 06:52). mean ± std over seeds
| case | method | gap mean | gap median | discrete err | continuous err | time median | paired ours−proxy per seed (Wilcoxon p) |
|---|---|---|---|---|---|---|---|
| case118 (38/48) | OUR METHOD | 0.034±0.002% | 0.019±0.001% | 3.53±0.11% | 6.99±0.23% | 9.3 s | −1.52 (p=0.013), −1.22 (p=2e-4), −1.07 (p=1e-3) → **ours better, all 3 seeds** |
| | supervised NN proxy | 1.303±0.229% | 0.054±0.031% | 8.07±0.49% | 8.23±1.41% | 0.29 s | |
| case300 (37/48) | OUR METHOD | 0.052±0.002% | 0.019±0.001% | 3.02±0.08% | 3.29±0.14% | 32.2 s | −0.02 (p=0.94), −0.03 (p=0.87), −0.02 (p=0.70) → **tie** |
| | supervised NN proxy | 0.075±0.004% | 0.008±0.003% | 2.39±0.16% | 2.76±0.16% | 1.10 s | |
| hard case118 / PGLib api (37/48) | OUR METHOD | 0.392±0.002% | 0.211±0.000% | 5.79±0.18% | 3.17±0.14% | 6.2 s | +0.29, +0.25, +0.26 (all p<1e-4) → **proxy better** (our fallbacks 8/5/5) |
| | supervised NN proxy | 0.128±0.024% | 0.003±0.001% | 1.47±0.10% | 1.50±0.26% | 1.81 s | |
Note: the seed-0 rows inside this 3-seed run differ slightly from the earlier seed-0-only run (case118 paired −1.516 vs −1.510; case300 −0.023 vs −0.030),
same weights, no decision taken between the two — run-to-run nondeterminism (not yet traced). Use THIS 3-seed table. Exclusions: instances without a converged
sequential-QCAC reference (listed with load scale in each run log). Logs: acuc/logs/fc_final{118,pg,300}_all.log.

## AUDIT #3 (2026-10-11 08:00, independent agent, Kiernan checklist) — findings on the AC finals above and what was done
The 3-seed AC table above (06:52) is SUPERSEDED: it is kept as a development record only. Findings:
1. BLOCKER — hard case118 (PGLib) deployed UNTRAINED random general rows (FD gate failed -> trained=None -> 55 fell back to GR["init"]). FIX: 55 deploys only
   label-free-selected rows; None = rows OFF.
2. General rows = LAST epoch, no selection (Kiernan #2 repeat); on case300 they lost on validation and were deployed anyway. FIX: 52 now scores rows OFF / init /
   every epoch with the LABEL-FREE validation score on ALL validation instances, saves every epoch, deploys the selected one (or none).
3. Nondeterminism (the seed-0 mismatch): cvxpy warm_start=True caches Clarabel per worker -> anchors depend on solve order (up to 4.2e-4; 0.06%/0.16% per-instance
   cost effect). FIX: warm_start=False in src/socp.py and src/vehicle_layer.py; verified bitwise-identical re-solves after other solves.
4. case118 fresh2 was scored (ours + proxy) on 10-09/10 (41/42) BEFORE the final method changes -> not untouched. case300/PG fresh2 results were seen before these
   fixes. FIX: new draw data2.fresh3_ids() (seed 9191, indices 336-383, disjoint) for ALL three cases; the fixed method is scored ONCE there.
5. Timing omitted the anchor solve and miqcqp's internal pricing (Kiernan repeat). FIX: ours = anchor + whole miqcqp call + strict pricing.
6. ref_polish dropped a converged reference when the polish broke convergence (case300 #293). FIX: keep the converged pre-polish solution.
7. V-net FD gate had no committed output and covered only case118 at an old V-net. FIX: gradient_checks/09_envelope_vgrad_fd_final.py at the final V-nets,
   all cases, seeds 0-2 -> gradient_checks/out/vgrad_fd_<case>.txt.
8. Statistics: add instance-level seed-averaged paired test, signed gaps, split by proxy-needed-repair, scored load range (all in 55 now).
MUST DISCLOSE (no code fix): exclusions are load-correlated (PG fresh2: all 11 at s>1.0); PG reference strict price is 1.1-7.2% below its own QCAC cost (untraced);
case300 references carry MIP gaps up to 54% on the polish step; the case118 "ours better" came entirely from instances where the proxy needed upward repair
(no-repair subset: tie); case300/PG integer cuts = the imitation warm start (layer fine-tune not selected); cuts/rows are trained once (seed 0) — seeds vary the
V-net and the proxy; V-nets were trained before the warm-start fix; W, W_long, K, gamma, LAM, epochs are fixed budgets (not selected).
Re-runs (deterministic): 46 x3, 52 x3, V-net FD gates (chain logs acuc/logs/a3_*.log, status logs/audit3_*.status); fresh3 references x3. Old weights backed up in
acuc/results/pre_audit3_2026-10-11/.
DECISION (Parth, 2026-10-11 08:50): the deployed cuts must be LEARNED THROUGH THE EXACT LAYER in both categories. Selection (label-free) is among layer-trained
epochs only; the imitation warm start, init general rows and "rows OFF" are scored for the report only, never deployed. A failed FD gate must be diagnosed and
fixed, not bypassed. Applied to acuc/46 + 52 and dixit_vehicle/16; chain A and the vehicle training restarted 08:54 (earlier partial outputs set aside).
| 2026-10-11_0854_vehicle_final_train | VEHICLE final TRAIN phase done (27 case-seed bundles in dixit_vehicle/results/final/; validation only) |
DECISION (Parth, 2026-10-11 10:45, option A, time): final AC test sets = case118 on the NEW draw fresh3 (its fresh2 was used in development);
case300 and hard case118 (PGLib) on fresh2 (never used for training/selection; their earlier results were SEEN before the audit-#3 fixes -> disclose).
fresh3 references for case300/PG not computed.
| 2026-10-11_1125_general_rows_on_top_case118 | case118 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.0205% | 0.0001% | 2.91% | 4.89% |
| 2026-10-11_1125_general_rows_on_top_case300 | case300 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.0805% | 0.0177% | 3.62% | 4.11% |
| 2026-10-11_1125_general_rows_on_top_case118 | case118 VALIDATION + general rows (selected: epoch 4): gap mean | median | disc | cont = 0.0165% | 0.0000% | 2.91% | 4.51% |
| 2026-10-11_1155_vehicle_final_TEST | FINAL VEHICLE MILP (8 vehicles pooled) FRESH untouched (701 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures | signed = 0.0845+-0.0184% | 0.0000+-0.0000% | 5.71+-0.11% | 5.65+-0.09% | 22.2+-0.7 ms | 0 | 0.0845+-0.0184% |
| 2026-10-11_1155_vehicle_final_TEST | FINAL VEHICLE MILP (8 vehicles pooled) FRESH untouched (701 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures | signed = 0.8116+-0.0522% | 0.0000+-0.0000% | 9.05+-0.07% | 8.70+-0.15% | 7.9+-0.7 ms | 0 | 0.8116+-0.0522% |
| 2026-10-11_1155_vehicle_final_TEST | FINAL VEHICLE MINLP FRESH untouched (48 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures | signed = 0.9319+-0.1869% | 0.5323+-0.0409% | 6.84+-0.14% | 6.40+-0.07% | 43.7+-5.1 ms | 0 | 0.9319+-0.1869% |
| 2026-10-11_1155_vehicle_final_TEST | FINAL VEHICLE MINLP FRESH untouched (48 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures | signed = 0.7428+-0.0453% | 0.4681+-0.0043% | 6.86+-0.34% | 6.44+-0.26% | 19.9+-2.6 ms | 0 | 0.7428+-0.0453% |

## VEHICLE FINAL — FRESH untouched test, 3 seeds (2026-10-11 11:56; dixit_vehicle/16_vehicle_final.py test, log dixit_vehicle/logs/16_test.log)
OUR METHOD = label-free per-step integer cuts + K=2 general rows, both trained through the exact layer (layer-trained epochs only), one short solve W=0.01.
Proxy = supervised MLP (wider grid + early stopping on validation), L1 projection + fixed-z re-solve. Fresh profiles (17_fresh_test_pool.py), refs TL 120 s.
| version | method | gap mean | gap median | discrete err | continuous err | time median | result (seed-averaged paired) |
|---|---|---|---|---|---|---|---|
| MILP, 8 vehicles pooled (701/800; 99 refs not at MIPGap 1e-4 excluded) | OUR METHOD | 0.085±0.018% | 0.000% | 5.71% | 5.65% | 22 ms | −0.73 pp [−0.87, −0.60], p=3e-43, better/tie/worse 301/337/63 → OURS BETTER (every seed, every vehicle) |
| | supervised NN proxy | 0.812±0.052% | 0.000% | 9.05% | 8.70% | 8 ms | |
| MINLP (48/48) | OUR METHOD | 0.932±0.187% | 0.532% | 6.84% | 6.40% | 44 ms | +0.19 pp [−0.07, +0.44], p=0.07, 17/0/31 → TIE (not significant; proxy lower mean) |
| | supervised NN proxy | 0.743±0.045% | 0.468% | 6.86% | 6.44% | 20 ms | |
Disclose: teacher (our W=1 solves) = reference z on 89-100% of training profiles; integer cuts deploy as fixings (layer fine-tune changed 1-5% of steps);
general rows forced on (label-free score preferred them OFF on several cases; on MINLP integer-cuts-only validation was ~0.49%); our MINLP fallbacks 5/3/5 per seed;
exclusions are the hard instances (v1/v2/v6/v9 ~25%); ours slower than the proxy (ms scale); gamma=1 smoothing distorts the layer's relaxation.
| 2026-10-11_1223_FINAL_case118_fresh3_seeds0-1-2 | FINAL case118 untouched (40 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.0382+-0.0038% | 0.0000+-0.0000% | 2.16+-0.12% | 5.19+-0.16% | 11.16+-0.18s | 0 |
| 2026-10-11_1223_FINAL_case118_fresh3_seeds0-1-2 | FINAL case118 untouched (40 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.4468+-0.3280% | 0.0173+-0.0142% | 3.53+-1.55% | 5.36+-0.66% | 0.35+-0.04s | 0 |

## AC FINAL case118 — NEW untouched draw fresh3, 3 seeds, all audit-#3 fixes (2026-10-11 12:26, logs/fc3_case118.log)
Our method = learned V (V-net, exact layer) + per-unit integer cuts (layer-trained, epoch 5) + K=2 general rows (layer-trained, label-free epoch 4); W=10.
| method | gap mean | gap median | discrete err | continuous err | time median | failures |
|---|---|---|---|---|---|---|
| OUR METHOD | 0.038±0.004% | 0.000% | 2.16% | 5.19% | 11.2 s | 0 |
| supervised NN proxy | 0.447±0.328% | 0.017% | 3.53% | 5.36% | 0.35 s | 0 |
Paired seed-averaged (n=40): ours − proxy −0.41 pp [−0.93, −0.05], Wilcoxon p=0.002, better/tie/worse 17/18/5 -> OURS BETTER. Per seed −0.73 / −0.41 / −0.08 pp.
Disclose: the win concentrates where the proxy's rounded commitment needs repair (9/40: −1.77 pp); where it does not (31/40) −0.014 pp (near tie).
40/48 with converged references (8 excluded, load 0.58-0.96, not load-extreme); scored load range = full drawn range [0.553, 1.288]. Our fallbacks 1/0/0.
Reference (sequential QCAC) median 321 s (measured under load). Our time includes anchor + whole MIQCQP call + strict pricing.
| 2026-10-11_1226_general_rows_on_top_case300 | case300 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.0805% | 0.0177% | 3.62% | 4.11% |
| 2026-10-11_1226_general_rows_on_top_case300 | case300 VALIDATION + general rows (selected: epoch 2): gap mean | median | disc | cont = 0.0586% | 0.0115% | 3.36% | 3.33% |
| 2026-10-11_1353_FINAL_case300_fresh2_seeds0-1-2 | FINAL case300 untouched (38 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.0369+-0.0048% | 0.0200+-0.0031% | 3.19+-0.22% | 3.47+-0.32% | 35.38+-3.27s | 0 |
| 2026-10-11_1353_FINAL_case300_fresh2_seeds0-1-2 | FINAL case300 untouched (38 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.0729+-0.0040% | 0.0072+-0.0031% | 2.33+-0.15% | 2.69+-0.16% | 1.28+-0.13s | 0 |
| 2026-10-11_1401_FINAL_pglib_opf_case118_ieee__api_fresh2_seeds0-1-2 | FINAL pglib_opf_case118_ieee__api untouched (37 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.5970+-0.1817% | 0.2168+-0.0014% | 5.62+-0.19% | 3.06+-0.23% | 14.67+-3.49s | 0 |
| 2026-10-11_1401_FINAL_pglib_opf_case118_ieee__api_fresh2_seeds0-1-2 | FINAL pglib_opf_case118_ieee__api untouched (37 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.1306+-0.0234% | 0.0000+-0.0000% | 1.47+-0.10% | 1.51+-0.26% | 2.31+-0.27s | 0 |

## AC FINALS case300 + hard case118 (2026-10-11 14:19; option A: fresh2 draws; logs/fc3_*.log)
| case | method | gap mean | gap median | discrete err | continuous err | time median | seed-averaged paired ours−proxy |
|---|---|---|---|---|---|---|---|
| case300 (38/48) | OUR METHOD | 0.037±0.005% | 0.020% | 3.19% | 3.47% | 35.4 s | −0.036 pp [−0.114, +0.019], p=0.30, 12/3/23 -> TIE |
| | supervised NN proxy | 0.073±0.004% | 0.007% | 2.33% | 2.69% | 1.28 s | |
| hard case118 / PGLib api (37/48) | OUR METHOD | 0.597±0.182% | 0.217% | 5.62% | 3.06% | 14.7 s | +0.47 pp [+0.13, +0.96], p=9e-6, 4/0/33 -> PROXY BETTER |
| | supervised NN proxy | 0.131±0.023% | 0.000% | 1.47% | 1.51% | 2.31 s | |
Notes: case300 rows epoch 2 (label-free 1.015368 < rows-off 1.015610), int cuts layer epoch 2; ref #293 re-solved with the polish fix; our fallbacks 0.
PG: rows epoch 3 (label-free 3.592741 > rows-off 3.591256: forced-on rows cost on validation), int cuts layer epoch 14; our fallbacks 5/4/5 per seed;
scored load range [0.852, 0.999] of drawn [0.852, 1.053] (all 11 exclusions at s>1.0, refs not converged).
General-row FD gate (h-retry): case300 2/3 instances self-consistent (corr 0.9999, 1.0000; third inst FD never self-consistent, corr 0.96-0.97); PG 3/3 (corr 1.0000).

## >>> PAPER TABLE (all final, 3 seeds, untouched test, ours = learned V + layer-trained integer cuts + layer-trained general rows; vehicle has no V) <<<
| case study | ours gap mean / median | proxy gap mean / median | ours disc / cont | proxy disc / cont | ours time | proxy time | verdict |
|---|---|---|---|---|---|---|---|
| AC case118 (fresh3, 40) | 0.038% / 0.000% | 0.447% / 0.017% | 2.16 / 5.19% | 3.53 / 5.36% | 11.2 s | 0.35 s | OURS BETTER (p=0.002) |
| AC case300 (fresh2, 38) | 0.037% / 0.020% | 0.073% / 0.007% | 3.19 / 3.47% | 2.33 / 2.69% | 35.4 s | 1.28 s | tie (p=0.30) |
| AC hard case118 (fresh2, 37) | 0.597% / 0.217% | 0.131% / 0.000% | 5.62 / 3.06% | 1.47 / 1.51% | 14.7 s | 2.31 s | PROXY BETTER (p=9e-6) |
| Vehicle MILP (fresh, 701) | 0.085% / 0.000% | 0.812% / 0.000% | 5.71 / 5.65% | 9.05 / 8.70% | 22 ms | 8 ms | OURS BETTER (p=3e-43) |
| Vehicle MINLP (fresh, 48) | 0.932% / 0.532% | 0.743% / 0.468% | 6.84 / 6.40% | 6.86 / 6.44% | 44 ms | 20 ms | tie (p=0.07) |
Reference (exact/sequential) solve time, median over the SCORED test instances (single-threaded Gurobi; measured while other jobs ran -> approximate):
AC = standard sequential QCAC (polish step excluded): case118 321 s, case300 566 s, hard case118 606 s. Vehicle = exact Gurobi (MIPGap 1e-4, Threads=1):
MILP 1.84 s (mean 10.4 s; instances that hit 120 s without MIPGap 1e-4 are excluded, so this understates), MINLP 1.78 s.
Speed-up of ours vs reference (median): case118 ~29x, case300 ~16x, hard case118 ~41x, vehicle MILP ~83x, vehicle MINLP ~40x.
FIGURES (final, same four-panel layout as the earlier release): figures/final_2026-10-11/QUAD_{AC_case118, AC_case300, AC_hard_case118_PGLib_api,
vehicle_MILP_Dixit, vehicle_MINLP}.{png,pdf} + summary.json, made by figures/make_figures_final.py from the final run pickles (3 seeds pooled).

V-NET GRADIENT CHECKS AT THE FINAL V-NETS (2026-10-11; gradient_checks/out/vgrad_fd_<case>.txt, 09_envelope_vgrad_fd_final.py): case118 PASS (all seeds,
corr >= 0.9999), case300 PASS (corr >= 0.9995). Hard case118 "FAILED" at h=1e-4 -> diagnostic 10_vgrad_fd_selfconsistency.py
(out/vgrad_fd_selfcons_pglib_opf_case118_ieee__api.txt): at the trained V the FINITE DIFFERENCE is not self-consistent at h=1e-4/1e-5
(forward vs backward corr down to -0.4, |fw-bw| up to 27x |fd|: the relaxed-QCAC value is KINKED in V on the stressed grid, binding limits switch).
Where FD IS self-consistent (mostly h=1e-3) the envelope gradient matches: 7 PASS, 1 borderline FAIL (corr 0.9896 vs bar 0.99). => the gradient code
is correct; the hard-case118 loss is non-smooth in V at the trained points (disclose; plausible reason the learned V helps less there).

CLEANUP (2026-10-11 15:30): 410 superseded outputs (old results, runs, logs, superseded vehicle bundles, old-study vehicle files) moved — not deleted — to
_archive_2026-10-11/ (same relative paths; list in _archive_2026-10-11/MANIFEST.json). Kept: everything the final pipeline loads + the final runs/logs.
All scripts kept in place (provenance). figures/QUAD_*.png (old release) kept because paper/main.tex still includes them — replace with figures/final_2026-10-11/.
Verified after the move: figures rebuild from kept files; all files 55_final_eval.py / 16_vehicle_final.py load are present.
CLEANUP 2 (2026-10-11 15:45, Parth: "clean up everything"): also archived the old paper/, old figures + scripts, AUDIT_RESPONSE.md, old README, the old pptx,
all non-final scripts (acuc 02-54 except the final pipeline + input provenance), old vehicle study, superseded gradient checks. acuc/checkpoints2 kept
(17/19 teacher provenance). New short README.md at the root. Verified: figures rebuild, all scripts parse, all final inputs present.
| 2026-10-11_1556_selfimprove_check_case118 | case118 TRAIN label-free teacher check: own deploy+improve vs teacher W=120: mean +1.2810% | median +0.0000% | p90 +1.0410% | max +18.4686% | within 0.01% 0.42 | cheaper 0.28 | unit disagreement 7.74% -> bar FAILED |
| 2026-10-11_1605_bestof_check_case118 | case118 TRAIN label-free best-of teacher [diag four metrics vs ref: 0.0276% | 0.0000% | 2.29% | 4.60%] vs teacher mean -0.0071% median +0.0000% -> bar PASSED |
| 2026-10-11_1628_general_rows_on_top_case118_lf | case118 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.6150% | 0.0000% | 5.47% | 5.48% |
| 2026-10-11_1628_general_rows_on_top_case118_lf | case118 VALIDATION + general rows (selected: epoch 7): gap mean | median | disc | cont = 1.1186% | 0.0000% | 7.58% | 5.95% |
| 2026-10-11_1710_cut_check_val_case118 | case118 VALIDATION integer cuts only, long-solve teacher, no check: gap mean | median | disc | cont | failures = 0.0205% | 0.0001% | 2.91% | 4.89% | 0 |
| 2026-10-11_1710_cut_check_val_case118 | case118 VALIDATION integer cuts only, long-solve teacher, with check: gap mean | median | disc | cont | failures = 0.0205% | 0.0001% | 2.91% | 4.89% | 0 |
| 2026-10-11_1710_cut_check_val_case118 | case118 VALIDATION integer cuts only, label-free teacher, no check: gap mean | median | disc | cont | failures = 0.6150% | 0.0000% | 5.47% | 5.48% | 0 |
| 2026-10-11_1710_cut_check_val_case118 | case118 VALIDATION integer cuts only, label-free teacher, with check: gap mean | median | disc | cont | failures = 0.5904% | 0.0000% | 5.47% | 5.20% | 0 |
| 2026-10-11_1715_selfteacher_case118_extra | case118 label-free best-of teacher for 300 'extra' draws -> selfteacher_case118_bestof_extra.json (444 targets) |
| 2026-10-11_1816_cut_check_val_case118 | case118 VALIDATION integer cuts only, long-solve teacher, no check: gap mean | median | disc | cont | failures = 0.0205% | 0.0001% | 2.91% | 4.89% | 0 |
| 2026-10-11_1816_cut_check_val_case118 | case118 VALIDATION integer cuts only, long-solve teacher, with check: gap mean | median | disc | cont | failures = 0.0205% | 0.0001% | 2.91% | 4.89% | 0 |
| 2026-10-11_1816_cut_check_val_case118 | case118 VALIDATION integer cuts only, label-free teacher, no check: gap mean | median | disc | cont | failures = 0.6150% | 0.0000% | 5.47% | 5.48% | 0 |
| 2026-10-11_1816_cut_check_val_case118 | case118 VALIDATION integer cuts only, label-free teacher, with check: gap mean | median | disc | cont | failures = 0.5904% | 0.0000% | 5.47% | 5.20% | 0 |
| 2026-10-11_1816_cut_check_val_case118 | case118 VALIDATION integer cuts only, label-free + 300 extra draws, no check: gap mean | median | disc | cont | failures = 0.0113% | 0.0000% | 2.91% | 4.13% | 0 |
| 2026-10-11_1816_cut_check_val_case118 | case118 VALIDATION integer cuts only, label-free + 300 extra draws, with check: gap mean | median | disc | cont | failures = 0.0113% | 0.0000% | 2.91% | 4.13% | 0 |
| 2026-10-11_2036_general_rows_on_top_case118_lfx | case118 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.0113% | 0.0000% | 2.91% | 4.13% |
| 2026-10-11_2036_general_rows_on_top_case118_lfx | case118 VALIDATION + general rows (selected: epoch 0): gap mean | median | disc | cont = 0.0090% | 0.0000% | 2.82% | 3.92% |
| 2026-10-11_2124_vehicle_lf_teacher | VEHICLE label-free teacher check (TRAIN profiles) [milp_v0] train n=100: new vs W=1 teacher mean +0.0252% median +0.0000% max +0.3579% -> PASSED | [diag vs ref, n=100] W=1 teacher 0.0000% | 0.0000% | 0.00% | 0.00% || label-free 0.0252% | 0.0000% | 8.77% | 8.72% | extra 300/300 | median 0.09s/inst  [37s] |
| 2026-10-11_2124_vehicle_lf_teacher | VEHICLE label-free teacher check (TRAIN profiles) [milp_v7] train n=100: new vs W=1 teacher mean +0.0729% median +0.0044% max +0.5188% -> FAILED | [diag vs ref, n=100] W=1 teacher 0.0000% | 0.0000% | 0.00% | 0.00% || label-free 0.0729% | 0.0044% | 12.14% | 12.05% | extra 300/300 | median 0.09s/inst  [39s] |
| 2026-10-11_2124_vehicle_lf_teacher | VEHICLE label-free teacher check (TRAIN profiles) [milp_v9] train n=100: new vs W=1 teacher mean +0.1787% median +0.1008% max +1.5018% -> FAILED | [diag vs ref, n=86] W=1 teacher 0.0007% | 0.0000% | 0.41% | 0.40% || label-free 0.1858% | 0.1002% | 17.52% | 17.50% | extra 300/300 | median 0.10s/inst  [39s] |
| 2026-10-11_2124_vehicle_lf_teacher | VEHICLE label-free teacher check (TRAIN profiles) [milp_v6] train n=100: new vs W=1 teacher mean +0.2081% median +0.0934% max +1.2955% -> FAILED | [diag vs ref, n=85] W=1 teacher 0.0027% | 0.0000% | 1.40% | 1.39% || label-free 0.2175% | 0.0927% | 16.86% | 16.69% | extra 300/300 | median 0.10s/inst  [40s] |
| 2026-10-11_2124_vehicle_lf_teacher | VEHICLE label-free teacher check (TRAIN profiles) [milp_v1] train n=100: new vs W=1 teacher mean +0.1988% median +0.0883% max +1.5500% -> FAILED | [diag vs ref, n=84] W=1 teacher 0.0000% | 0.0000% | 0.00% | 0.04% || label-free 0.2155% | 0.0922% | 17.11% | 17.18% | extra 300/300 | median 0.10s/inst  [40s] |
| 2026-10-11_2124_vehicle_lf_teacher | VEHICLE label-free teacher check (TRAIN profiles) [milp_v2] train n=100: new vs W=1 teacher mean +0.2426% median +0.1038% max +1.9598% -> FAILED | [diag vs ref, n=82] W=1 teacher 0.0012% | 0.0000% | 0.74% | 0.69% || label-free 0.2589% | 0.0969% | 17.27% | 17.09% | extra 300/300 | median 0.10s/inst  [40s] |
| 2026-10-11_2124_vehicle_lf_teacher | VEHICLE label-free teacher check (TRAIN profiles) [milp_v4] train n=100: new vs W=1 teacher mean +0.1811% median +0.1419% max +0.9528% -> FAILED | [diag vs ref, n=99] W=1 teacher 0.0000% | 0.0000% | 0.00% | 0.00% || label-free 0.1815% | 0.1410% | 16.96% | 16.81% | extra 300/300 | median 0.10s/inst  [40s] |
| 2026-10-11_2124_vehicle_lf_teacher | VEHICLE label-free teacher check (TRAIN profiles) [milp_v8] train n=100: new vs W=1 teacher mean +0.2250% median +0.2044% max +0.9672% -> FAILED | [diag vs ref, n=99] W=1 teacher 0.0041% | 0.0000% | 2.66% | 2.63% || label-free 0.2304% | 0.2070% | 19.96% | 19.81% | extra 300/300 | median 0.10s/inst  [41s] |
| 2026-10-11_2124_vehicle_lf_teacher | VEHICLE label-free teacher check (TRAIN profiles) [minlp] train n=144: new vs W=1 teacher mean +0.0912% median +0.0244% max +0.7640% -> FAILED | [diag vs ref, n=144] W=1 teacher 0.0007% | 0.0000% | 0.01% | 0.02% || label-free 0.0919% | 0.0244% | 2.77% | 2.55% | extra 300/300 | median 0.38s/inst  [142s] |
| 2026-10-11_2126_selfteacher_case300_train | case300 label-free best-of teacher for 136 'train' draws -> selfteacher_case300_bestof_train.json (136 targets) |
| 2026-10-11_2215_selfteacher_case300_extra | case300 label-free best-of teacher for 286 'extra' draws -> selfteacher_case300_bestof_extra.json (422 targets) |
| 2026-10-12_0142_general_rows_on_top_case300_lfx | case300 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.0369% | 0.0158% | 3.16% | 3.51% |
| 2026-10-12_0142_general_rows_on_top_case300_lfx | case300 VALIDATION + general rows (selected: epoch 8): gap mean | median | disc | cont = 0.0320% | 0.0103% | 2.44% | 2.63% |
| 2026-10-12_0341_selfteacher_pglib_opf_case118_ieee__api_train | pglib_opf_case118_ieee__api label-free best-of teacher for 118 'train' draws -> selfteacher_pglib_opf_case118_ieee__api_bestof_train.json (118 targets) |
| 2026-10-12_0514_selfteacher_pglib_opf_case118_ieee__api_extra | pglib_opf_case118_ieee__api label-free best-of teacher for 237 'extra' draws -> selfteacher_pglib_opf_case118_ieee__api_bestof_extra.json (355 targets) |
| 2026-10-12_1153_selfteacher_case118_train | case118 label-free best-of teacher for 144 'train' draws -> selfteacher_case118_bestof_train_nc.json (144 targets) |
| 2026-10-12_1205_selfteacher_case118_extra | case118 label-free best-of teacher for 300 'extra' draws -> selfteacher_case118_bestof_extra_nc.json (444 targets) |
| 2026-10-12_1318_general_rows_on_top_case118_lfc | case118 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.6727% | 0.0331% | 6.61% | 7.60% |
| 2026-10-12_1318_general_rows_on_top_case118_lfc | case118 VALIDATION + general rows (selected: epoch 6): gap mean | median | disc | cont = 0.6602% | 0.0337% | 6.88% | 7.71% |
| 2026-10-12_1359_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (clean no-cut V (check vs 52)): gap mean | median | disc | cont | time | fallbacks | failures = 0.6602% | 0.0337% | 6.88% | 7.71% | 9.9s (max 40.5s) | fallbacks 0 | failures 0 |
| 2026-10-12_1400_val_full_case118_env_lfx | case118 VALIDATION FULL METHOD (previous _lfx (check vs 52)): gap mean | median | disc | cont | time | fallbacks | failures = 0.0090% | 0.0000% | 2.82% | 3.92% | 11.6s (max 16.7s) | fallbacks 0 | failures 0 |
| 2026-10-12_1419_val_full_case118_env_r2_lfc | case118 VALIDATION FULL METHOD (ROUND 2: V-net trained with clean label-free cuts): gap mean | median | disc | cont | time | fallbacks | failures = 0.6069% | 0.0470% | 6.97% | 7.98% | 10.1s (max 16.7s) | fallbacks 0 | failures 0 |
| 2026-10-12_1421_val_full_case118_env_lfc | case118 VALIDATION FULL METHOD (A): gap mean | median | disc | cont | time | fallbacks | failures = 0.1009% | 0.0354% | 4.23% | 6.98% | 11.5s (max 32.0s) | fallbacks 1 | failures 0 |
| 2026-10-12_1422_val_full_case118_env_r2_lfx | case118 VALIDATION FULL METHOD (B): gap mean | median | disc | cont | time | fallbacks | failures = 1.1757% | 0.0000% | 8.11% | 6.22% | 10.5s (max 15.8s) | fallbacks 0 | failures 0 |
| 2026-10-12_1428_selfteacher_case118_train | case118 label-free best-of teacher for 144 'train' draws -> selfteacher_case118_bestof_train_r2.json (144 targets) |
| 2026-10-12_1438_selfteacher_case118_extra | case118 label-free best-of teacher for 300 'extra' draws -> selfteacher_case118_bestof_extra_r2.json (444 targets) |
| 2026-10-12_1859_general_rows_on_top_case118_lfv2 | case118 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.3201% | 0.2705% | 7.41% | 11.48% |
| 2026-10-12_1859_general_rows_on_top_case118_lfv2 | case118 VALIDATION + general rows (selected: epoch 4): gap mean | median | disc | cont = 0.3134% | 0.2705% | 7.32% | 11.44% |
| 2026-10-12_1916_val_full_case118_env_r2_lfv2 | case118 VALIDATION FULL METHOD (round2V_valuecuts): gap mean | median | disc | cont | time | fallbacks | failures = 0.3134% | 0.2705% | 7.32% | 11.44% | 11.9s (max 23.7s) | fallbacks 2 | failures 0 |
| 2026-10-12_1936_val_full_case118_env_r3v_lfv2 | case118 VALIDATION FULL METHOD (round3V_valuecuts): gap mean | median | disc | cont | time | fallbacks | failures = 3.8354% | 0.3656% | 19.31% | 11.94% | 11.7s (max 17.5s) | fallbacks 0 | failures 0 |

## PRE-REGISTRATION of the FINAL TEST (written 2026-10-12 20:35, BEFORE any fresh4 result exists)
Method frozen = `_lfx` (Parth, 2026-10-12): V-nets results/vnet_layer_<case>{,_s1,_s2}_env.pt; integer cuts unit_cuts_layer_<case>_soft_lfx.pt;
general rows general_rows_<case>_lfx.pt (PGLib rows: being trained now; sha1 recorded in final.pkl). Labels for cuts/rows = own
deployment-budget best-of solves (no reference anywhere). Command, run ONCE per case, alone on the machine:
`python 55_final_eval.py <case> 0,1,2 fresh4 _lfx`. Code snapshot: runs/2026-10-12_2032_CODE_SNAPSHOT_pre_final_lfx (code.tgz sha1 85279591065).
sha1 (first 12): case118 cuts 88c31bae034e rows 323e0921dbc1 V 976e9f68f4de/5cb16a1711e1/d543bdb96949; case300 cuts bc7d17c9b107
rows f7a242df12cc V a1df85ad0540/0572d0b9e586/f45afaccf4db; PGLib cuts 4faf8f3046df rows (pending) V f02a4cd2649c/e337bee7b38c/013f6c0bbcf3.
Report: the four metrics (gap mean | median | discrete | continuous) + time + failures for BOTH methods, as always.
Comparison (pre-registered, 55.paired): seed-averaged paired tests on |gap| AND on the signed gap (= paired cost difference; the
reference is a heuristic, so beating it must not count as error); tie = |difference| <= 1e-3 pp; Wilcoxon (zero_method wilcox) on the
thresholded differences with n_nonzero; sign test on better/worse; bootstrap 95% CI of the mean. Failures are NOT dropped: a method
failing on an instance (any seed) loses that instance. Repair split = descriptive/post hoc only. Exclusions = fresh4 instances whose
reference did not converge or needed repair (listed with load scale before scoring). PGLib rows are deployed as selected (layer-trained
epoch) even if worse than rows-off on the label-free validation score (Parth's rule) -> disclosed. If a row/cut FD gate fails, the
case is NOT tested until diagnosed. Timing = measured alone, machine load recorded; quoted as approximate (6P+6E cores).
MUST DISCLOSE (from the pre-final audit): V-net trained with fixings from a cut-net fitted on our own W=120/180 long solves; c from that
pipeline; version selection (_lfx vs clean variants) used reference-scored validation reports; 24 validation draws drove many choices;
integer-cut/row tracking gradient uses a regularised (inexact) KKT solve on most instances; layer training adds ~nothing on validation;
teacher compute per instance (case118 41 s, case300 189 s, PGLib 365 s worker-time) vs deployment and reference; seeds vary only V-net
and proxy; exclusions cluster at high load; reference certified gap = 1e-3 when the polish was not kept.

## PRE-REGISTRATION of the VEHICLE FINAL TEST (written 2026-10-12 ~21:55, BEFORE the label-free vehicle training or test pool exist)
Method = label-free vehicle method: `cd dixit_vehicle && VEH_LF=1 python 16_vehicle_final.py train` (27 bundles -> results/final_lf/;
teacher = 18_lf_teacher.py own deployment-budget solves, 100/144 original + 300 extra unlabeled profiles per case), then ONCE:
`VEH_LF=1 python 16_vehicle_final.py test` on the NEW untouched pool results/pool_fresh2/ (19_fresh2_test_pool.py, Pool(6) as 17).
Report: four metrics + time + failures for both methods (MILP 8 vehicles pooled; MINLP). Statistics = the AC rules (16.paired = 55.paired):
|gap| AND signed gap, tie <= 1e-3 pp, Wilcoxon on thresholded differences with n_nonzero, sign test, failures = losses, bootstrap CI;
MILP HEADLINE = VEHICLE-CLUSTERED sign test over the 8 vehicles (pooled instance-level p is NOT independent). Exclusion = reference
MIPGap > 1e-4 or no reference (same set for both). Partial results saved after every job; a crashed job is re-run alone (no re-look).
MUST DISCLOSE: the vehicle label-free teacher FAILED its own pre-registered bar on 8/9 cases (labels 0.07-0.26% / disc 9-20% vs the
reference, W=1 teacher ~0%); ours trains on 400/444 profiles (unlabeled extras) vs the proxy's 100/144 labels; teacher per-instance
compute (0.10 s MILP / 0.38 s MINLP) > deployment (22/44 ms) -- "nothing longer than deployment" holds per solve, not per instance;
the method was changed after the 2026-10-11 pool_fresh vehicle result had been seen (report both); exclusions = the hardest instances.
PRE-REGISTRATION addendum (2026-10-12 22:05, before any fresh4 result): PGLib _lfx general rows = epoch 0 (label-free 3.588486 vs rows-off 3.587907, i.e. +0.016% worse; deployed per the layer-trained-only rule, disclosed); sha1 general_rows_pglib_opf_case118_ieee__api_lfx.pt d5c9b315fee7.
| 2026-10-13_0155_FINAL_case118_fresh4_lfx_seeds0-1-2 | FINAL case118 untouched (40 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.0277+-0.0036% | 0.0000+-0.0000% | 1.88+-0.10% | 4.48+-0.29% | 9.20+-0.09s | 0 |
| 2026-10-13_0155_FINAL_case118_fresh4_lfx_seeds0-1-2 | FINAL case118 untouched (40 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.7072+-0.0064% | 0.0004+-0.0005% | 4.92+-0.19% | 5.17+-0.48% | 0.30+-0.01s | 0 |
| 2026-10-13_0157_FINAL_case300_fresh4_lfx_seeds0-1-2 | FINAL case300 untouched (37 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.1343+-0.0181% | 0.0237+-0.0023% | 3.28+-0.12% | 3.77+-0.05% | 33.94+-0.41s | 0 |
| 2026-10-13_0157_FINAL_case300_fresh4_lfx_seeds0-1-2 | FINAL case300 untouched (37 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.0963+-0.0065% | 0.0174+-0.0021% | 3.17+-0.04% | 3.54+-0.02% | 1.15+-0.00s | 0 |
| 2026-10-13_0207_FINAL_pglib_opf_case118_ieee__api_fresh4_lfx_seeds0-1-2 | FINAL pglib_opf_case118_ieee__api untouched (38 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.3262+-0.0091% | 0.2820+-0.0046% | 6.53+-0.24% | 5.51+-0.19% | 11.07+-0.18s | 0 |
| 2026-10-13_0207_FINAL_pglib_opf_case118_ieee__api_fresh4_lfx_seeds0-1-2 | FINAL pglib_opf_case118_ieee__api untouched (38 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.1739+-0.0830% | 0.0008+-0.0012% | 1.49+-0.18% | 1.45+-0.27% | 1.88+-0.03s | 0 |

## AC FINAL TEST on fresh4 (untouched, pre-registered, run ONCE 2026-10-13 01:55-02:17) + post-test AUDIT
| case | method | gap mean | gap median | discrete err | continuous err | time median | seed-avg paired ours-proxy (pre-registered) |
|---|---|---|---|---|---|---|---|
| case118 (40/48) | OURS (_lfx, label-free) | 0.028% | 0.000% | 1.88% | 4.48% | 9.2 s | |gap| -0.68 pp p=0.020, sign p=0.031 (14/22/4); signed -0.69 pp p=0.016 -> OURS BETTER |
| | supervised NN proxy | 0.707% | 0.000% | 4.92% | 5.17% | 0.30 s | (win concentrated on the 9 proxy-repair instances: -2.99 pp; other 31: -0.009 pp) |
| case300 (37/48) | OURS | 0.134% | 0.024% | 3.28% | 3.77% | 33.9 s | |gap| +0.04 pp p=0.31; signed +0.02 pp p=0.34 -> TIE |
| | supervised NN proxy | 0.096% | 0.017% | 3.17% | 3.54% | 1.15 s | |
| PGLib api (38/48) | OURS | 0.326% | 0.282% | 6.53% | 5.51% | 11.1 s | |gap| +0.15 pp p=0.003 (proxy closer to the reference, 8/0/30); SIGNED -0.40 pp p=6.5e-7 (ours CHEAPER, 33/0/5) |
| | supervised NN proxy | 0.174% | 0.001% | 1.49% | 1.45% | 1.88 s | the two pre-registered metrics DISAGREE -> report both |
Reference (sequential QCAC, converged only, polish excluded) median 104 / 336 / 595 s -> ~11x / 10x / 54x; understated (excluded refs ran longer).
POST-TEST AUDIT (independent agent): hashes = pre-registration; all numbers recompute from final.pkl; exclusions identical (8/11/10, PGLib all at load >= 0.996).
PGLib "ours cheaper than the reference" (signed mean -0.25%): our commitments are feasible (reserve, P/Q residual <= 9.7e-4 pu, lines <= 1.00002,
voltages, pg bounds) and commit fewer units (49.1 vs 51.4; no-load saving -0.64%, dispatch +0.39%; no-load cost is synthetic 15%). BUT the strict
pricing is path-dependent/not converged on PGLib (residual stalls at 1e-4..1e-3): extended re-pricing moves the gap 0.11-0.18 pp toward the
reference on differing commitments (n=5) -> true advantage likely ~ -0.1%, sign holds on the big wins. Needs an exact AC-OPF (u fixed) check
before it is a headline. The PGLib reference itself is a weak heuristic there (its QCAC cost 3.2% above the strict price of its own commitment).
case300 references certified only to ~1e-3 (pre-polish gap). Cheaper-than-reference is NEW behaviour of _lfx (old method on PGLib fresh2: +0.58%).
| 2026-10-13_0217_vehicle_final_train_lf | VEHICLE final TRAIN phase done (27 case-seed bundles in dixit_vehicle/results/final/; validation only) |
| 2026-10-13_0504_vehicle_final_TEST_lf_pool_fresh2 | FINAL VEHICLE MILP (8 vehicles pooled) [LF method, pool_fresh2] FRESH untouched (734 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures | signed = 0.0837+-0.0068% | 0.0013+-0.0021% | 9.25+-0.38% | 9.21+-0.38% | 10.9+-0.9 ms | 0 | 0.0837+-0.0068% |
| 2026-10-13_0504_vehicle_final_TEST_lf_pool_fresh2 | FINAL VEHICLE MILP (8 vehicles pooled) [LF method, pool_fresh2] FRESH untouched (734 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures | signed = 0.8763+-0.0803% | 0.0071+-0.0024% | 9.91+-0.09% | 9.51+-0.08% | 4.0+-0.1 ms | 0 | 0.8763+-0.0803% |
| 2026-10-13_0504_vehicle_final_TEST_lf_pool_fresh2 | FINAL VEHICLE MINLP [LF method, pool_fresh2] FRESH untouched (48 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures | signed = 0.5320+-0.1367% | 0.2430+-0.0398% | 5.07+-0.23% | 4.84+-0.31% | 18.2+-0.3 ms | 0 | 0.5320+-0.1367% |
| 2026-10-13_0504_vehicle_final_TEST_lf_pool_fresh2 | FINAL VEHICLE MINLP [LF method, pool_fresh2] FRESH untouched (48 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures | signed = 0.7590+-0.1700% | 0.4728+-0.0281% | 6.96+-0.77% | 6.45+-0.53% | 7.1+-0.1 ms | 0 | 0.7590+-0.1700% |

## VEHICLE FINAL TEST on pool_fresh2 (untouched, pre-registered, run ONCE 2026-10-13 05:04) + post-test AUDIT
| version | method | gap mean | gap median | discrete err | continuous err | time median | seed-avg paired ours-proxy (pre-registered) |
|---|---|---|---|---|---|---|---|
| MILP 8 vehicles (734/800) | OURS (label-free) | 0.084% | 0.001% | 9.25% | 9.21% | 10.9 ms | -0.79 pp, p=3e-28, 335/206/193; VEHICLE-CLUSTERED 8/0/0, sign p=0.0078 -> OURS BETTER |
| | supervised NN proxy | 0.876% | 0.007% | 9.91% | 9.51% | 4.0 ms | |
| MINLP (48/48) | OURS (label-free) | 0.532% | 0.243% | 5.07% | 4.84% | 18.2 ms | -0.23 pp, CI [-0.45,-0.02], p=0.018, sign p=0.002, 35/0/13 -> OURS BETTER (seed 1 alone: +0.07, n.s.) |
| | supervised NN proxy | 0.759% | 0.473% | 6.96% | 6.45% | 7.1 ms | |
Exact reference (Gurobi, MIPGap 1e-4) median 0.71 s MILP / 0.34 s MINLP -> ~65x / ~19x (understated; v0 only ~4x).
POST-TEST AUDIT: genuine run (16 s is plausible; 4 jobs re-run bit-identical), all numbers recompute, right bundles (lf, 400/444 targets) and pool
(provenance hashes), exclusions identical (66 MILP, 0 MINLP); excluded instances do NOT hide a loss (ours better 126/50 there).
DISCLOSE: (1) MILP: most of the win vs the proxy comes from running a 10 ms Gurobi solve at all -- plain no-cut W=0.01 solve already 0.163%
(audit diagnostic on the test, not an arm); learned cuts add ~0.08 pp of the 0.79 pp. MINLP: no-cut 0.891% (worse than the proxy) -> the
edge there comes from the cuts. (2) training data not like-for-like (ours 400/444 profiles incl. unlabeled extras; proxy 82-100/144 labels).
(3) v4, v8: ours has HIGHER discrete error and median gap than the proxy; the per-vehicle mean wins via the proxy's tail. (4) MINLP is seed-
dependent. (5) the vehicle label-free teacher failed its own bar on 8/9 cases. (6) 2/144 MINLP instance-seeds are no-cut fallbacks.
| 2026-10-13_1040_val_full_case118_env_lfx | case118 VALIDATION FULL METHOD (old_V+lfx): gap mean | median | disc | cont | time | fallbacks | failures = 0.0113% | 0.0000% | 2.91% | 4.13% | 9.1s (max 11.5s) | fallbacks 0 | failures 0 |
| 2026-10-13_1040_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (noCutV+lfc): gap mean | median | disc | cont | time | fallbacks | failures = 0.6727% | 0.0331% | 6.61% | 7.60% | 9.5s (max 39.0s) | fallbacks 1 | failures 0 |
| 2026-10-13_1041_val_full_case118_env_r3v_lfv2 | case118 VALIDATION FULL METHOD (round3_V + value cuts, NO rows): gap mean | median | disc | cont | time | fallbacks | failures = 0.2270% | 0.1799% | 6.26% | 8.39% | 11.2s (max 58.4s) | fallbacks 5 | failures 0 |
| 2026-10-13_1043_val_full_case118_env_r2_lfv2 | case118 VALIDATION FULL METHOD (round2_V + value cuts, NO rows): gap mean | median | disc | cont | time | fallbacks | failures = 0.3201% | 0.2705% | 7.41% | 11.48% | 10.2s (max 24.7s) | fallbacks 1 | failures 0 |
| 2026-10-13_1043_val_full_case118_env_r2_lfc | case118 VALIDATION FULL METHOD (round2_V + lfc cuts, NO rows): gap mean | median | disc | cont | time | fallbacks | failures = 0.1149% | 0.0470% | 4.06% | 7.04% | 10.4s (max 13.0s) | fallbacks 0 | failures 0 |

## DIAGNOSTIC 2026-10-13 (case118 validation, report-only) + PRE-REGISTRATION of a re-designed case118 method (written 11:2x, BEFORE any fresh5 result)
Diagnostic (61_val_full.py ... norows): the clean-V blow-ups (11-23%) come from the GENERAL ROWS trained at one V and deployed at another, not from the V-nets:
round3 V + value cuts 3.835% with rows -> 0.227% without; round2 V + lfc cuts 0.607% -> 0.115% (disc 4.06%); rows trained AT the deployed V are neutral
(round2 V + value cuts 0.313% vs 0.320%; no-cut V + lfc 0.660% vs 0.673%). fresh4 (case118) was scored with _lfx and its result seen -> a re-designed case118
method is scored ONCE on the NEW draw data2.fresh5_ids() (indices 732-779, seed 7373; refs results/fresh5_refs_case118.pkl).
Re-design (case118 only): V-net round 2 = vnet_layer_case118{,_s1,_s2}_env_r2.pt (trained through the relaxation with the LABEL-FREE round-1 cuts _lfc; no long-solve input);
label-free best-of teacher at V_r2 (selfteacher_case118_bestof_extra_r2.json, 444 targets); integer cuts + general rows trained AT V_r2 with the exact-envelope VALUE loss
hardened against over-tightening (tag _lfh; c re-selected label-free; V is NOT updated afterwards). Final test: `55_final_eval.py case118 0,1,2 fresh5 _lfh _r2`, same
pre-registered statistics/tie rule/failure handling as the earlier blocks. Case300 / PGLib remain _lfx (not re-done).
PRE-REGISTRATION ADDENDUM 2026-10-13 11:25 (audit of the redesign; BEFORE any validation table of the redesign and before fresh5 results):
(1) The case118 redesign is tested as `python 55_final_eval.py case118 0,1,2 fresh5 _lfh _r2 vseed0`: the V-net is ALWAYS seed 0 (V_r2) -- the cuts/rows were trained AT that V -- and the
3 seeds vary the supervised PROXY only (a mismatched-V deployment on seed-1/2 V-nets is the failure mode the diagnostic identified; V-nets s1/s2 are trained by the chain but NOT used).
(2) The method reported is the FULL method (V + integer cuts + general rows, both layer-trained epochs deployed, per the abstract). The `norows` validation report is diagnostic only and decides nothing.
(3) Not matched, to disclose: V_r2 was trained with the _lfc cuts (c=0.9) but is deployed with _lfh cuts/rows (c re-selected 0.95) -- V lags the cuts by one round; the value-loss
relaxations use rho=1e6 while the deployed V file uses rho=1e5; training relaxes fractional soft rows while deployment hard-rounds; hardened training barely moves the cuts (they are essentially the BCE
imitation of the label-free teacher) and rows are near-inert; the redesign was chosen after reference-scored validation reports and after seeing the fresh4 result of _lfx -> report BOTH tests (fresh4 _lfx, fresh5 _lfh).
| 2026-10-13_1118_general_rows_on_top_case118_lfh | case118 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.3099% | 0.2865% | 7.41% | 11.69% |
| 2026-10-13_1118_general_rows_on_top_case118_lfh | case118 VALIDATION + general rows (selected: epoch 3): gap mean | median | disc | cont = 0.3359% | 0.2705% | 7.58% | 11.29% |
| 2026-10-13_1133_val_full_case118_env_r2_lfh | case118 VALIDATION FULL METHOD (full_matchedV_hardened): gap mean | median | disc | cont | time | fallbacks | failures = 0.3359% | 0.2705% | 7.58% | 11.29% | 12.6s (max 24.3s) | fallbacks 2 | failures 0 |
| 2026-10-13_1134_val_full_case118_env_r2_lfh | case118 VALIDATION FULL METHOD (intcuts_only_matchedV_hardened): gap mean | median | disc | cont | time | fallbacks | failures = 0.3099% | 0.2865% | 7.41% | 11.69% | 11.8s (max 25.6s) | fallbacks 2 | failures 0 |
| 2026-10-13_1138_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (sanity): gap mean | median | disc | cont | time | fallbacks | failures = 0.6727% | 0.0331% | 6.61% | 7.60% | 10.4s (max 40.1s) | fallbacks 1 | failures 0 |
| 2026-10-13_1139_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (nc_intcuts_check): gap mean | median | disc | cont | time | fallbacks | failures = 0.6727% | 0.0331% | 6.61% | 7.60% | 11.4s (max 40.5s) | fallbacks 1 | failures 0 |
| 2026-10-13_1140_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (nc_full_check): gap mean | median | disc | cont | time | fallbacks | failures = 0.6602% | 0.0337% | 6.88% | 7.71% | 11.3s (max 42.2s) | fallbacks 0 | failures 0 |

## RESULT of the matched-V redesign attempt, case118 VALIDATION (2026-10-13 11:35-12:00; four metrics vs reference, report-only) -- NO fresh5 test was run
| V-net | cuts / rows | gap mean | gap median | discrete err | continuous err | note |
|---|---|---|---|---|---|---|
| old V (long-solve fixings in its training) = TESTED `_lfx` | _lfx | 0.0090% | 0.0000% | 2.82% | 3.92% | the tested version |
| clean round-1 V (no cuts, `_nc`) | _lfc (+rows), trained AT V_nc | 0.6602% | 0.0337% | 6.88% | 7.71% | one instance (#209, load 0.843) 12.8%: cut-net predicts ~42 units on, reference 13; cuts are valid (check releases 0), not a validity problem |
| round-2 V (`_r2`) | _lfh (hardened exact-gradient value loss) + rows, trained AT V_r2 | 0.3359% | 0.2705% | 7.58% | 11.29% | no blow-ups (worst 0.86%); labels at V_r2 are worse |
| round-2 V | _lfh integer cuts only | 0.3099% | 0.2865% | 7.41% | 11.69% | rows neutral when matched |
Findings: (1) the clean-V blow-ups (11-23%) came from rows trained at a different V than deployed; matching V removes them. (2) Label quality DEGRADES per clean round (own best-of labels vs reference,
118 train inst.): old V 0.028%|0.000%|2.29%|4.60%; clean round-1 V 0.093%|0.027%|2.87%|5.52%; round-2 V 0.264%|0.110%|5.38%|9.94% (V trained under imperfect fixings -> errors compound).
(3) The hardened exact-envelope value loss is verified (13/13 instances corr >= 0.9999 on free coordinates; saturated coordinates have a degenerate soft-row dual and are zeroed) and its gate passes (3/3, corr 1.0000), but it does not improve the cuts beyond the BCE imitation.
Conclusion: the fully label-free (no long-solve dependency) redesign does NOT beat the tested _lfx on case118 validation; the tested _lfx result stands with the disclosed V-net dependency. fresh5 stays untouched.
| 2026-10-13_1214_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_1e4): gap mean | median | disc | cont | time | fallbacks | failures = 0.6602% | 0.0337% | 6.88% | 7.71% | 10.1s (max 42.5s) | fallbacks 0 | failures 0 |
| 2026-10-13_1215_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_3e4): gap mean | median | disc | cont | time | fallbacks | failures = 1.5969% | 0.0447% | 10.23% | 8.31% | 9.2s (max 15.3s) | fallbacks 0 | failures 0 |
| 2026-10-13_1215_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_1e5): gap mean | median | disc | cont | time | fallbacks | failures = 0.0935% | 0.0523% | 4.59% | 8.49% | 9.6s (max 14.9s) | fallbacks 0 | failures 0 |
| 2026-10-13_1216_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_3e5): gap mean | median | disc | cont | time | fallbacks | failures = 0.1286% | 0.0523% | 4.76% | 8.71% | 10.0s (max 14.3s) | fallbacks 0 | failures 0 |
| 2026-10-13_1216_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_1e6): gap mean | median | disc | cont | time | fallbacks | failures = 0.1365% | 0.0523% | 4.85% | 8.90% | 10.6s (max 25.7s) | fallbacks 1 | failures 0 |

## PRE-REGISTRATION: the FULLY CLEAN case118 method (written 2026-10-13 ~13:15, after the rho diagnostic, BEFORE any fresh5 result)
Root cause of the last clean-pipeline blow-up (#209): the raw MIQCQP answer was not AC-feasible at the V-net's selected penalty rho=1e4 and the blind merit-order repair added 27 units; the V-net's rho
had been selected by a label-free score that deploys WITHOUT cuts. Rule adopted: the DEPLOYMENT rho is chosen from {1e4,3e4,1e5,3e5,1e6} by the label-free score of the FULL deployed pipeline
(strict price / flat-V relaxation cost on the validation draws, no reference). On the round-1 pipeline (_lfc) the rule picks 1e5 (0.9597; 1e4: 0.9652; 3e4: 0.9741; 3e5: 0.9601; 1e6: 0.9602) and
the four metrics at 1e5 are 0.0935% | 0.0523% | 4.59% | 8.49% (vs 0.6602% at 1e4) -- seen as a diagnostic.
Method (case118): V_nc = V-net trained with NO cuts (results/vnet_layer_case118_env_nc.pt, seed 0); labels = own best-of solves at V_nc (selfteacher_case118_bestof_extra_nc.json); integer cuts + general rows trained AT V_nc with the
hardened exact-envelope value loss (tag _lfn; layer-trained epochs deployed; c re-selected label-free at rho 1e5); deployment rho = argmin of the label-free sweep on the _lfn pipeline (grid above). No long solve and no reference
anywhere in the training of V, cuts or rows. Final test ONCE: `DEPLOY_RHO=<selected> python 55_final_eval.py case118 0,1,2 fresh5 _lfn _nc vseed0` (V-net seed 0 only; the 3 seeds vary the PROXY).
Not matched / disclose: V_nc trained at rho 1e4 vs deployed at the selected rho; value-loss relaxations use rho 1e6; training relaxes fractional soft rows while deployment hard-rounds; hardened training barely moves the cuts (~BCE imitation of the label-free labels);
design chosen after reference-scored validation reports and after seeing the fresh4 result of _lfx; the rho grid and rule were fixed after seeing the _lfc diagnostic. Case300 / PGLib NOT re-done (still _lfx).
| 2026-10-13_1226_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_1e4): gap mean | median | disc | cont | time | fallbacks | failures = 0.6602% | 0.0337% | 6.88% | 7.71% | 15.8s (max 66.2s) | fallbacks 1 | failures 0 |
| 2026-10-13_1228_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_3e4): gap mean | median | disc | cont | time | fallbacks | failures = 1.5969% | 0.0447% | 10.23% | 8.31% | 11.5s (max 19.2s) | fallbacks 0 | failures 0 |
| 2026-10-13_1229_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_1e5): gap mean | median | disc | cont | time | fallbacks | failures = 0.0935% | 0.0523% | 4.59% | 8.49% | 18.9s (max 29.1s) | fallbacks 0 | failures 0 |
| 2026-10-13_1230_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_3e5): gap mean | median | disc | cont | time | fallbacks | failures = 0.1286% | 0.0523% | 4.76% | 8.71% | 16.2s (max 23.7s) | fallbacks 1 | failures 0 |
| 2026-10-13_1231_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (rho_1e6): gap mean | median | disc | cont | time | fallbacks | failures = 0.1365% | 0.0523% | 4.85% | 8.90% | 17.9s (max 37.2s) | fallbacks 1 | failures 0 |
CORRECTION + TIE RULE (2026-10-13 ~12:50, still before any fresh5 result; supersedes the 'picks 1e5' statement above): that statement scored only the 21 validation draws with references. The label-free rule is scored on ALL 24
validation draws (the four-metric report stays on the 21). On the _lfc pipeline: score 1e4 0.9662 | 3e4 0.9738 | 1e5 0.9697 | 3e5 0.9618 | 1e6 0.9617; four metrics (21 draws) 1e4 0.6602%|0.0337%|6.88%|7.71% , 3e4 1.5969%|0.0447%|10.23%|8.31%,
1e5 0.0935%|0.0523%|4.59%|8.49%, 3e5 0.1286%|0.0523%|4.76%|8.71%, 1e6 0.1365%|0.0523%|4.85%|8.90%. PRE-REGISTERED TIE RULE: among grid values whose score is within 0.001 of the minimum, take the SMALLEST rho -> 3e5 on _lfc.
The _lfn chain (cuts c/epoch and rows selected at DEPLOY_RHO=3e5) was therefore restarted at 3e5; the final deployment rho = the rule applied to the _lfn sweep (disclose if it differs from 3e5).
| 2026-10-13_1248_general_rows_on_top_case118_lfn | case118 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.7191% | 0.0481% | 6.53% | 7.23% |
| 2026-10-13_1253_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (intcuts__lfc): gap mean | median | disc | cont | time | fallbacks | failures = 0.1167% | 0.0893% | 4.85% | 9.05% | 9.1s (max 11.7s) | fallbacks 1 | failures 0 |
| 2026-10-13_1254_val_full_case118_env_nc_lfn | case118 VALIDATION FULL METHOD (intcuts__lfn): gap mean | median | disc | cont | time | fallbacks | failures = 0.7191% | 0.0481% | 6.53% | 7.23% | 11.6s (max 41.5s) | fallbacks 1 | failures 0 |

## FINAL PRE-REGISTRATION of the FULLY CLEAN case118 method (2026-10-13 ~13:10; supersedes the _lfn plan; BEFORE any fresh5 result)
Candidates compared label-free at the V_nc pipeline, deployment rho 3e5 (24 validation draws): cuts _lfc (tracking loss, round-1) score 0.9617 [four metrics, integer cuts only: 0.1167%|0.0893%|4.85%|9.05%] vs cuts _lfn (hardened exact value loss, matched) score 0.9668
[0.7191%|0.0481%|6.53%|7.23%, #209 blow-up 12.76% re-introduced]; the _lfn rows gate also failed (no gradient signal on the 3 gate instances). The label-free score selects _lfc. FINAL CLEAN METHOD = V_nc (no-cut V-net, seed 0) + integer cuts _lfc + general rows _lfc
(both trained AT V_nc on the label-free best-of labels at V_nc, layer-trained epochs deployed, c re-selected label-free) + deployment rho = 3e5 (pre-registered rule: smallest rho within 0.001 of the minimum label-free score on all 24 validation draws; full-pipeline four metrics 0.1286%|0.0523%|4.76%|8.71%).
Command (ONCE): DEPLOY_RHO=3e5 python 55_final_eval.py case118 0,1,2 fresh5 _lfc _nc vseed0   (V-net seed 0 only; the 3 seeds vary the PROXY).  sha1 (first 12): unit_cuts_layer_case118_soft_lfc.pt d29d1c8d0190; general_rows_case118_lfc.pt da9810abe503; vnet_layer_case118_env_nc.pt e067d41927e0; fresh5_refs_case118.pkl 01635ee075bd
Gradient caveat (MUST DISCLOSE): _lfc cuts/rows were trained with the tracking loss whose KKT-based gradient is regularised/inexact on most AC solves (audit: 57-60% of solves, 40-85% off where it fires); the exact-gradient value-loss variants (verified 13/13 corr>=0.9999) did not beat the imitation start and, for cuts, re-introduced a blow-up.
Other disclosures as in the preceding blocks (V_nc trained at rho 1e4 vs deployed 3e5; value/tracking relaxations at rho 1e6; labels' per-instance compute > deployment although every MIQCQP <= W; design fixed after reference-scored validation reports and after the fresh4 result; 300/444 training draws unlabeled; case300/PGLib not redone).
| 2026-10-13_1255_FINAL_case118_fresh2_lfc_nc_vseed0_seeds0 | FINAL case118 untouched (38 inst, seeds [0]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.6007% | 0.0812% | 5.70% | 9.49% | 11.77s | 0 |
| 2026-10-13_1255_FINAL_case118_fresh2_lfc_nc_vseed0_seeds0 | FINAL case118 untouched (38 inst, seeds [0]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 1.5507% | 0.0204% | 8.53% | 6.99% | 0.36s | 0 |
| 2026-10-13_1305_FINAL_case118_fresh5_lfc_nc_vseed0_seeds0-1-2 | FINAL case118 untouched (39 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 1.0248+-0.0000% | 0.0558+-0.0000% | 6.03+-0.00% | 7.77+-0.00% | 10.92+-0.36s | 0 |
| 2026-10-13_1305_FINAL_case118_fresh5_lfc_nc_vseed0_seeds0-1-2 | FINAL case118 untouched (39 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 1.0505+-0.0283% | 0.0211+-0.0127% | 6.28+-0.26% | 6.80+-1.31% | 0.29+-0.02s | 0 |

## RESULT: FULLY CLEAN case118 method on the untouched draw fresh5 (run ONCE 2026-10-13 13:05-13:10; pre-registered above; machine load ~6.5 at start from macOS Spotlight/screensaver, timings approximate)
`DEPLOY_RHO=3e5 python 55_final_eval.py case118 0,1,2 fresh5 _lfc _nc vseed0`; hashes recorded in final.pkl = the pre-registered ones; 39/48 scored (9 refs not converged, load 0.563-1.122), same for both methods.
| method | gap mean | gap median | discrete err | continuous err | time median | seed-avg paired ours-proxy (pre-registered) |
|---|---|---|---|---|---|---|
| OURS, fully clean (no long solve, no reference in training V/cuts/rows) | 1.025% | 0.056% | 6.03% | 7.77% | 10.9 s | |gap| -0.026 pp p=0.39 (15/6/18); signed -0.023 pp p=0.33 (14/6/19) -> TIE |
| supervised NN proxy | 1.051% | 0.021% | 6.28% | 6.80% | 0.29 s | (ours vs proxy per seed: +0.002 / -0.055 / -0.024 pp) |
Driver of both means = the sparse low-load regime (load ~0.82-0.84, reference commits only 13 units): #733 ours 23.6% (all 54 units on, MIP gap 0.9999) vs proxy 12.9%; #768 ours 13.1% vs proxy 13.0%; #743 ours 0.18% vs proxy 12.7%.
Without the single worst instance: ours 0.432% | 0.050%, proxy 0.739% | 0.031% (n=38) -- descriptive/post hoc.
Comparison with the earlier tested version (_lfx on fresh4, V-net trained with long-solve fixings): ours 0.028% vs proxy 0.707% (ours better, p=0.02). Removing the long-solve dependency costs the case118 win on this draw: clean = TIE.
| 2026-10-13_1327_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (oracle_rho_1e4): gap mean | median | disc | cont | time | fallbacks | failures = 0.6602% | 0.0337% | 6.88% | 7.71% | 8.8s (max 39.2s) | fallbacks 1 | failures 0 |
| 2026-10-13_1329_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (oracle_rho_3e4): gap mean | median | disc | cont | time | fallbacks | failures = 1.5969% | 0.0447% | 10.23% | 8.31% | 10.3s (max 14.5s) | fallbacks 0 | failures 0 |
| 2026-10-13_1330_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (oracle_rho_1e5): gap mean | median | disc | cont | time | fallbacks | failures = 0.0935% | 0.0523% | 4.59% | 8.49% | 9.8s (max 14.7s) | fallbacks 0 | failures 0 |
| 2026-10-13_1332_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (oracle_rho_3e5): gap mean | median | disc | cont | time | fallbacks | failures = 0.1286% | 0.0523% | 4.76% | 8.71% | 10.6s (max 15.5s) | fallbacks 1 | failures 0 |
| 2026-10-13_1334_val_full_case118_env_nc_lfc | case118 VALIDATION FULL METHOD (oracle_rho_1e6): gap mean | median | disc | cont | time | fallbacks | failures = 0.1365% | 0.0523% | 4.85% | 8.90% | 11.9s (max 28.3s) | fallbacks 1 | failures 0 |

## DIAGNOSTIC + PRE-REGISTRATION: the RETRY safeguard (written 2026-10-13 ~14:35, BEFORE any fresh6 result; fresh6 references are running, results/fresh6_refs_case118.pkl)
Diagnostic (64_rho_band.py + stored validation sweeps; label-free: priced costs and MIP gaps only, no references read): 84 draws x 5 deployment rho = 420 deployments of the clean pipeline (V_nc + _lfc). 11 (2.6%) fail (cost >5% above the best rho of that instance,
median MIP gap 0.999, median 54 units on vs ~14); the failing rho differs from instance to instance (only 1 instance fails at >=2 rho) -> NOT predictable from the load (a rho classifier would learn noise) but DETECTABLE afterwards (MIP gap).
Offline policy simulation: single rho 3e5: mean cost vs per-instance-best 1.0038 (worst +23.3%); "deploy at 3e5; if the MIQCQP MIP gap > 0.05, deploy once more at rho 1e4 and keep the cheaper strictly-priced answer": 1.0009 (worst +0.6%), 1.07 deployments/instance. Designed and evaluated on the SAME draws (2 knobs).
Test (ONCE each, on the untouched draw fresh6 = data2.fresh6_ids(), indices 840-887, seed 9393; same instances, same proxy seeds, V-net seed 0, 3 proxy seeds):
 PRIMARY (clean method + retry):  DEPLOY_RHO=3e5 RETRY_RHO=1e4 RETRY_TAU=0.05 python 55_final_eval.py case118 0,1,2 fresh6 _lfc _nc vseed0
 ABLATION (clean method, single rho, = the earlier pre-registered method):  DEPLOY_RHO=3e5 python 55_final_eval.py case118 0,1,2 fresh6 _lfc _nc vseed0
Both are reported with the same four metrics + time + the pre-registered paired statistics against the same supervised proxy. Time of a retried instance = both solves + both pricings. Retry is label-free (solver MIP gap + own strict pricing; no reference, no heuristic; each solve <= W; up to 2 solves in ~7% of instances).
Files/hashes as in the 'FINAL PRE-REGISTRATION of the FULLY CLEAN case118 method' block (cuts _lfc d29d1c8d0190, rows _lfc da9810abe503, V_nc e067d41927e0). Design history to disclose: the rho problem was first seen as a reference-scored blow-up (#209) on validation; the retry rule's knobs were fixed on priced costs of the 84 draws above.
| 2026-10-13_1449_FINAL_case118_fresh2_lfc_nc_vseed0_retry_seeds0 | FINAL case118 untouched (38 inst, seeds [0]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.4952% | 0.0812% | 5.46% | 8.31% | 20.15s | 0 |
| 2026-10-13_1449_FINAL_case118_fresh2_lfc_nc_vseed0_retry_seeds0 | FINAL case118 untouched (38 inst, seeds [0]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 1.5507% | 0.0204% | 8.53% | 6.99% | 0.58s | 0 |
| 2026-10-13_1458_FINAL_case118_fresh2_lfc_nc_vseed0_retry_seeds0 | FINAL case118 untouched (38 inst, seeds [0]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.4952% | 0.0812% | 5.46% | 8.31% | 19.71s | 0 |
| 2026-10-13_1458_FINAL_case118_fresh2_lfc_nc_vseed0_retry_seeds0 | FINAL case118 untouched (38 inst, seeds [0]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 1.5507% | 0.0204% | 8.53% | 6.99% | 0.60s | 0 |
| 2026-10-13_1458_FINAL_case118_fresh2_lfc_nc_vseed0_retry_seeds0 | FINAL case118 untouched (38 inst, seeds [0]) OUR METHOD, single rho (ablation): gap mean | median | disc | cont | time | failures = 0.6007% | 0.0812% | 5.70% | 9.49% | 19.55s | 0 |
PRE-REGISTRATION AMENDMENT 2026-10-13 ~15:05 (after the independent audit of the retry; still BEFORE any fresh6 result; supersedes the 'two runs' plan above):
(1) ONE run only on fresh6: `DEPLOY_RHO=3e5 RETRY_RHO=1e4 RETRY_TAU=0.05 python 55_final_eval.py case118 0,1,2 fresh6 _lfc _nc vseed0`, on an IDLE machine after the fresh6 reference job has finished (load printed at start AND end). Gurobi is deterministic here (Threads=1, WorkLimit),
so the single-rho ablation is DERIVED IN THE SAME RUN from the saved first attempts (rows_out[s]['sol0'/'OP0']); no second run, no second look. The proxy is identical by construction (same seeds).
(2) PRIMARY ENDPOINT: clean method WITH retry vs the supervised proxy, the pre-registered seed-averaged paired tests on |gap| and on the signed gap (tie 1e-3 pp, Wilcoxon on thresholded differences, sign test, failures = losses). The single-rho ablation vs the proxy is reported with the same tests;
retry-vs-single-rho is DESCRIPTIVE on the instances where the retry fired. If the retry fires on 0 instances the two arms are identical and that is reported as 'retry not testable on this draw'. No change to the method, rho, tau or code after seeing the result.
(3) POWER: fresh6 has only ~5 draws in the sparse regime (load 0.78-0.88) -> expect 0-2 instances where the retry matters; a null difference between retry and single-rho is the likely outcome and is NOT evidence against the retry.
(4) DISCLOSE: (a) two mechanics smoke tests of the retry were run on the already-used draw fresh2 (seed 0; 14:49 and ~15:00): ours 0.6007% -> 0.4952% with retry (3/38 retried, 1 kept), the retry did NOT rescue a failure at load 1.146 (#319, all units on, +14.5%), i.e. failures are not confined to the sparse regime;
(b) the offline evidence (84 draws x 5 rho) rests on 2 instances (b832: 1.2326 -> 1.0005, b792: 1.0045 -> 1.0000); 4 of the 6 offline retries were fallback-triggered with no gain; retry rho 1e4 and tau 0.05 are not identified by the data (all alternatives equal);
(c) one of the 11 offline failures has a tiny MIP gap (0.0035) and is invisible to the rule; (d) retry fires on ~7-10% of instances and failed attempts are long, so mean time rises ~12% and max time up to ~95 s; (e) the design draws include the 24 validation draws already used in-sample, and the sparse regime was chosen after seeing the fresh5 failures.
| 2026-10-13_1542_FINAL_case118_fresh6_lfc_nc_vseed0_retry_seeds0-1-2 | FINAL case118 untouched (40 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.1074+-0.0000% | 0.0830+-0.0000% | 2.92+-0.00% | 5.68+-0.00% | 11.36+-0.09s | 0 |
| 2026-10-13_1542_FINAL_case118_fresh6_lfc_nc_vseed0_retry_seeds0-1-2 | FINAL case118 untouched (40 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.8074+-0.0274% | 0.0207+-0.0174% | 5.45+-0.12% | 6.48+-0.38% | 0.36+-0.01s | 0 |
| 2026-10-13_1542_FINAL_case118_fresh6_lfc_nc_vseed0_retry_seeds0-1-2 | FINAL case118 untouched (40 inst, seeds [0, 1, 2]) OUR METHOD, single rho (ablation): gap mean | median | disc | cont | time | failures = 0.1074+-0.0000% | 0.0830+-0.0000% | 2.92+-0.00% | 5.68+-0.00% | 11.36+-0.09s | 0 |

## RESULT: clean case118 method (V_nc + _lfc, rho 3e5) + RETRY safeguard on the untouched draw fresh6 (run ONCE 2026-10-13 15:42-15:45; idle machine, load 1.8 at start; pre-registered above)
`DEPLOY_RHO=3e5 RETRY_RHO=1e4 RETRY_TAU=0.05 python 55_final_eval.py case118 0,1,2 fresh6 _lfc _nc vseed0`; 40/48 scored (8 refs not converged), same instances for both methods.
| method | gap mean | gap median | discrete err | continuous err | time median |
|---|---|---|---|---|---|
| OURS, clean + retry (label-free end to end) | 0.107% | 0.083% | 2.92% | 5.68% | 11.4 s |
| supervised NN proxy | 0.807% | 0.021% | 5.45% | 6.48% | 0.36 s |
| ours, single rho (ablation, derived in the same run) | 0.107% | 0.083% | 2.92% | 5.68% | 11.4 s |
Pre-registered paired tests (seed-averaged, ours - proxy): |gap| -0.700 pp, 95% CI [-1.660, -0.029], Wilcoxon p=0.256, sign test p=0.136 (19/11/10); signed gap -0.700 pp, p=0.239 -> NOT SIGNIFICANT on the pre-registered rank tests (the bootstrap CI of the mean excludes 0; the median favours the proxy).
Where the proxy needed repair (10 instances): ours - proxy -2.70 pp; elsewhere (30): -0.035 pp (descriptive).
RETRY: fired on 1/40 instances (#857), kept the retried answer on 0 -> identical to the single-rho arm; the retry is NOT TESTABLE on this draw (as pre-registered: low power, ~5 sparse-regime draws). Time of the retried instance 41.4 s vs 33.4 s.
Two untouched tests of the same clean method (descriptive): fresh5 (39) tie, mean gap 1.025% vs 1.051%; fresh6 (40) 0.107% vs 0.807%, discrete 2.92% vs 5.45%, but not significant on the rank tests.

## CLEAN REDO case300 + PGLib (launched 2026-10-13, plan written BEFORE any result)
Recipe = the case118 `_lfc` pipeline: V-net with NO cuts -> label-free best-of teacher at that V (144 train + 300 extra draws) -> integer cuts (c label-free) + general rows (tracking loss) trained at that V ->
deployment rho chosen by the label-free score of the full pipeline on ALL 24 validation draws over {1e4,3e4,1e5,3e5,1e6} (smallest rho within 0.001 of the min). Nothing from a long solve or a reference.
Script acuc/clean_redo_300_pglib_2026-10-13.sh (status logs/clean300pg.status). Test draw = fresh5 (never scored for case300/PGLib; references made after training, never used in training/selection).
Seeds: V-net seed 0, proxy seeds 0,1,2. Pre-registration with hashes/statistics to be added BEFORE the single test per case. PGLib: no validation references -> label-free score only.
| 2026-10-13_1939_selfteacher_case300_train | case300 label-free best-of teacher for 136 'train' draws -> selfteacher_case300_bestof_train_nc.json (136 targets) |
| 2026-10-13_2022_selfteacher_case300_extra | case300 label-free best-of teacher for 286 'extra' draws -> selfteacher_case300_bestof_extra_nc.json (422 targets) |
| 2026-10-14_0017_general_rows_on_top_case300_lfc | case300 VALIDATION integer cuts only: gap mean | median | disc | cont = 0.0691% | 0.0103% | 3.82% | 4.77% |
| 2026-10-14_0017_general_rows_on_top_case300_lfc | case300 VALIDATION + general rows (selected: epoch 8): gap mean | median | disc | cont = 0.0521% | 0.0212% | 3.43% | 4.25% |
| 2026-10-14_0214_val_full_case300_env_nc_lfc | case300 VALIDATION FULL METHOD (rho_1e4): gap mean | median | disc | cont | time | fallbacks | failures = 0.0521% | 0.0212% | 3.43% | 4.25% | 41.3s (max 182.7s) | fallbacks 1 | failures 0 |
| 2026-10-14_0218_val_full_case300_env_nc_lfc | case300 VALIDATION FULL METHOD (rho_3e4): gap mean | median | disc | cont | time | fallbacks | failures = 0.0805% | 0.0289% | 3.89% | 5.00% | 40.6s (max 183.9s) | fallbacks 1 | failures 0 |
| 2026-10-14_0222_val_full_case300_env_nc_lfc | case300 VALIDATION FULL METHOD (rho_1e5): gap mean | median | disc | cont | time | fallbacks | failures = 0.1293% | 0.1062% | 4.41% | 6.07% | 42.0s (max 107.5s) | fallbacks 1 | failures 0 |
| 2026-10-14_0225_val_full_case300_env_nc_lfc | case300 VALIDATION FULL METHOD (rho_3e5): gap mean | median | disc | cont | time | fallbacks | failures = 0.1211% | 0.0938% | 4.41% | 5.35% | 40.4s (max 186.3s) | fallbacks 2 | failures 0 |
| 2026-10-14_0229_val_full_case300_env_nc_lfc | case300 VALIDATION FULL METHOD (rho_1e6): gap mean | median | disc | cont | time | fallbacks | failures = 0.1020% | 0.0368% | 3.95% | 4.80% | 41.0s (max 184.1s) | fallbacks 2 | failures 0 |
| 2026-10-14_0518_selfteacher_pglib_opf_case118_ieee__api_train | pglib_opf_case118_ieee__api label-free best-of teacher for 118 'train' draws -> selfteacher_pglib_opf_case118_ieee__api_bestof_train_nc.json (118 targets) |
| 2026-10-14_1013_selfteacher_pglib_opf_case118_ieee__api_extra | pglib_opf_case118_ieee__api label-free best-of teacher for 237 'extra' draws -> selfteacher_pglib_opf_case118_ieee__api_bestof_extra_nc.json (355 targets) |
| 2026-10-14_1505_vnet_vs_flat_rho_case118 | case118 V-NET vs FLAT V by deployment rho (validation, label-free score, no cuts): 1e+04 | 0.961692 | 0.977868 | -0.016176 (-1.65%) | 14 / 19 | +0.0043 || 3e+04 | 0.983818 | 0.965344 | +0.018474 (+1.91%) | 17 / 24 | +0.2092 || 1e+05 | 0.987592 | 0.983801 | +0.003791 (+0.39%) | 20 / 24 | +0.1544 || 3e+05 | 1.004816 | 1.011666 | -0.006850 (-0.68%) | 19 / 24 | +0.2092 || 1e+06 | 1.006203 | 1.055664 | -0.049461 (-4.69%) | 19 / 24 | +0.2635 |
| 2026-10-14_2014_val_full_pglib_opf_case118_ieee__api_env_nc_lfc | pglib_opf_case118_ieee__api VALIDATION FULL METHOD (rho_1e4): rho 1e+04, label-free score 3.587896 (no validation references -> no four-metric report), fallbacks 2 |
| 2026-10-14_2022_val_full_pglib_opf_case118_ieee__api_env_nc_lfc | pglib_opf_case118_ieee__api VALIDATION FULL METHOD (rho_3e4): rho 3e+04, label-free score 3.587884 (no validation references -> no four-metric report), fallbacks 2 |
| 2026-10-14_2030_val_full_pglib_opf_case118_ieee__api_env_nc_lfc | pglib_opf_case118_ieee__api VALIDATION FULL METHOD (rho_1e5): rho 1e+05, label-free score 3.587906 (no validation references -> no four-metric report), fallbacks 2 |
| 2026-10-14_2038_val_full_pglib_opf_case118_ieee__api_env_nc_lfc | pglib_opf_case118_ieee__api VALIDATION FULL METHOD (rho_3e5): rho 3e+05, label-free score 3.587794 (no validation references -> no four-metric report), fallbacks 1 |
| 2026-10-14_2046_val_full_pglib_opf_case118_ieee__api_env_nc_lfc | pglib_opf_case118_ieee__api VALIDATION FULL METHOD (rho_1e6): rho 1e+06, label-free score 3.587987 (no validation references -> no four-metric report), fallbacks 1 |

## PRE-REGISTRATION of the CLEAN case300 + PGLib FINAL TESTS (written 2026-10-15 07:15, BEFORE any fresh5 result exists for these cases)
Method = the clean label-free pipeline (acuc/clean_redo_300_pglib_2026-10-13.sh + resume2_pglib_2026-10-14.sh): V-net trained on the UNCUT relaxation (`_env_nc`, selected rho 1e4 both cases),
label-free best-of targets at that V (`_nc`; case300 422 targets, PGLib 355 = 118/144 train + 237/300 extra), integer cuts `_lfc` (c = 0.9 both; epoch 8 both) and general rows `_lfc` (case300 epoch 8, PGLib epoch 1), all trained at the deployed V.
Nothing from a long solve, a reference or a test/fresh draw reaches V-net, cuts, rows or the deployment rho (independent audit 2026-10-13: no leakage; chain scripts as run).
Deployment rho (label-free rule: smallest rho within 0.001 of the minimum of the full-pipeline score on ALL 24 validation draws): case300 1e4 (scores 1.015163 / 1.015479 / 1.015933 / 1.015876 / 1.015688 for 1e4..1e6);
PGLib 1e4 (3.587896 / 3.587884 / 3.587906 / 3.587794 / 3.587987; all within 2e-4, rule -> 1e4). RETRY: NOT used (a retry at rho 1e4 equals the deployment rho for both cases, so it is a no-op by construction).
Test draw = fresh5 (never scored for case300 / PGLib), scored ONCE per case, on an idle machine (load 2.1, 0 python processes at 07:10), order case300 then PGLib.
References (56_refs_fresh2.py, made after training, never used in training/selection): case300 40/48 converged (excluded 735 740 746 748 761 775 776 778), PGLib 42/48 converged (excluded 734 751 756 769 770 779). Exclusion = reference not converged, same set for both methods.
Commands (3 seeds; V-net seed 0 for all seeds, seeds vary the proxy):
  `DEPLOY_RHO=1e4 python acuc/55_final_eval.py case300 0,1,2 fresh5 _lfc _nc vseed0`
  `DEPLOY_RHO=1e4 python acuc/55_final_eval.py pglib_opf_case118_ieee__api 0,1,2 fresh5 _lfc _nc vseed0`
Statistics (same as every earlier AC test): paired per instance on the seed-averaged |gap| AND on the signed gap; tie <= 1e-3 pp; Wilcoxon on thresholded differences with n_nonzero; sign test; failures = losses; 95% CI.
Report: gap mean | median | discrete | continuous | time | failures for ours and the supervised NN proxy, per case, plus fallback / repair counts.
sha1 (first 12): case300 vnet 10a3c0d74f59, cuts a0f5eda7779a, rows ca1f05838846, fresh5 refs 363daf7009fe | PGLib vnet 9ba8ba90fc4e, cuts 5e9d7e5f2ed7, rows 61f219f9d2ea, fresh5 refs 36699d4a36c5.
Code: 55_final_eval.py 642c4edb19bf, combopt_ac.py 8d4cdb907757, data2.py 4355758a55bd, vnetio.py fbbb35ea8f9d, 60_selfteacher.py 263877b01887, 46_unit_cuts_layer.py 7459490bba94, 52_general_rows_on_top.py ae22c053dc70, 40_vnet_layer.py a2a75ec9dc54.
MUST DISCLOSE: n = 1 trained pipeline per case; the 24 validation draws were used for every design choice (c, epochs, rho); the proxy trains on 114 (case300) / 118 (PGLib) labelled loads vs our 422 / 355 own targets;
PGLib: the reference is a weak heuristic and strict pricing is path-dependent there (see the fresh4 audit), so |gap| and signed gap can disagree; case300 references are certified only to ~1e-3; some training loads had no own target (case300: 8/144 train + 14/300 extra; PGLib: 26/144 train + 63/300 extra).
| 2026-10-15_1104_FINAL_case300_fresh5_lfc_nc_vseed0_seeds0-1-2 | FINAL case300 untouched (40 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.0825+-0.0000% | 0.0132+-0.0000% | 2.68+-0.00% | 3.74+-0.00% | 39.28+-4.27s | 0 |
| 2026-10-15_1104_FINAL_case300_fresh5_lfc_nc_vseed0_seeds0-1-2 | FINAL case300 untouched (40 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.0872+-0.0124% | 0.0103+-0.0010% | 2.74+-0.15% | 3.07+-0.17% | 1.17+-0.04s | 0 |
| 2026-10-15_1114_FINAL_pglib_opf_case118_ieee__api_fresh5_lfc_nc_vseed0_seeds0-1-2 | FINAL pglib_opf_case118_ieee__api untouched (42 inst, seeds [0, 1, 2]) OUR METHOD: gap mean | median | disc | cont | time | failures = 0.3910+-0.0000% | 0.2956+-0.0000% | 7.10+-0.00% | 6.58+-0.00% | 10.10+-0.74s | 0 |
| 2026-10-15_1114_FINAL_pglib_opf_case118_ieee__api_fresh5_lfc_nc_vseed0_seeds0-1-2 | FINAL pglib_opf_case118_ieee__api untouched (42 inst, seeds [0, 1, 2]) supervised NN proxy: gap mean | median | disc | cont | time | failures = 0.1050+-0.0070% | 0.0057+-0.0013% | 1.75+-0.03% | 1.17+-0.10% | 2.05+-0.18s | 0 |

## CLEAN case300 + PGLib FINAL TESTS on fresh5 (untouched, pre-registered above, run ONCE 2026-10-15 11:04-11:17, DEPLOY_RHO=1e4, no retry, seeds 0,1,2; logs final_clean300_fresh5.log, final_cleanpglib_fresh5.log)
| case | method | gap mean | gap median | discrete err | continuous err | time median | seed-avg paired ours-proxy (pre-registered) |
|---|---|---|---|---|---|---|---|
| case300 (40/48) | OURS (clean, label-free) | 0.0825% | 0.0132% | 2.68% | 3.74% | 39.3 s | |gap| -0.005 pp, CI [-0.112,+0.102], Wilcoxon p=0.74, sign p=1.0 (17/7/16); signed +0.002 pp p=0.82 -> TIE |
| | supervised NN proxy | 0.0872% | 0.0103% | 2.74% | 3.07% | 1.17 s | fallbacks ours 1/seed; proxy repairs 2-3 |
| PGLib api (42/48) | OURS (clean, label-free) | 0.3910% | 0.2956% | 7.10% | 6.58% | 10.1 s | |gap| +0.286 pp, CI [+0.171,+0.394], p=8.7e-6, sign p=2.8e-6 (6/0/36) -> PROXY CLOSER TO THE REFERENCE |
| | supervised NN proxy | 0.1050% | 0.0057% | 1.75% | 1.17% | 2.05 s | SIGNED -0.326 pp, p=8.2e-7 (37/0/5) -> ours CHEAPER than the proxy; signed vs reference: ours -0.307% (min -1.57%), proxy +0.018% |
The two pre-registered metrics DISAGREE on PGLib (as on fresh4 with the earlier method): report both. Caveat from the fresh4 audit still applies: the PGLib reference is a weak heuristic and strict pricing is path-dependent there, so "cheaper than the reference" is not a proven advantage (needs an exact AC-OPF check with the commitment fixed).
Reference medians (sequential QCAC, converged only): case300 327 s, PGLib 663 s. Ours ran with a screensaver using ~1 core at the start (load 4.1 at 11:04), so times are slightly pessimistic.
Wall time of the two tests was 10 min + 3 min (3 seeds x 40/42 loads on 11 workers) -- far below my 1-1.5 h estimate.
| 2026-10-15_2212_geninteger_case118_ham_K0_M0_smoke | case118 VALIDATION gen-int-cuts ham K_I=0 M=0 r=3 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 0.4004% | 0.2883% | 5.56% | 18.35% | 8.8s | fallbacks 1; label-free score 0.968729 |

## GENERALIZED INTEGER CUTS, overnight experiment 1 (written 2026-10-15 22:30, BEFORE any full result; VALIDATION ONLY -- every case118 test draw and vehicle pool is already spent)
Question (PI): can the integer cuts be GENERAL linear rows over the integer variables (a'u <= b, a and b produced by the network) instead of the one-variable cuts u_k >= l_k / u_k <= h_k? The per-unit cut-net, its threshold c and all unit fixings are REMOVED from these runs (lo = 0, hi = 1 -> inert unit rows in the layer; the deployed MIQCQP receives only the learned rows).
Script acuc/67_generalized_integer_cuts.py (audited 2026-10-15 by an independent agent: no unit fixing survives, no reference/long solve/test/fresh draw reaches training or selection; fixes applied: DEPLOY_RHO assert, seeds, anchors only when used, validation-set-size assert, r grid up to 20). Chain acuc/night_geninteger_2026-10-15.sh, status logs/night_geninteger.status.
Variants (case118, V-net `_env_nc`, targets `_nc` (best-of, no cuts), DEPLOY_RHO=3e5, W=10, soft rows priced 1e6): none (control: V-net + one short solve) | rows K_I=2, 8, 32 (learned integer rows a2'u<=b2, CutNetK integer category, anchored at the uncut relaxation) | ham (ONE learned Hamming-ball row sum_k[p_k(1-u_k)+(1-p_k)u_k] <= r, centre p = MLP warm-started by BCE on our own targets, r from {1,2,3,5,8,12,20} by the label-free score) | ham + 2 mixed rows | rows K_I=8 + 2 mixed | rows K_I=32 + 2 mixed.
Training: tracking through the exact layer + validity (own target must satisfy every row), FD gate first (same pass rule as 52: a variant whose gate fails is reported as NOT TRAINED, never bypassed), epoch chosen by the LABEL-FREE score on ALL 24 validation draws among trained epochs.
Comparator (current method, per-unit fixings + 2 mixed rows, same V-net/rho/W/draws): label-free score 0.961770; validation four metrics 0.1286% | 0.0523% | 4.76% | 8.71% | 10.6 s (21 draws with references). No-cut V-net alone: label-free 1.0048 (diagnostic 65).
A variant COUNTS AS WORKING if label-free score <= 0.9638 (within 0.002 of the comparator) AND validation gap mean <= 0.26% (twice the comparator). A working variant would still need a fresh untouched draw + new references before any test claim.
MUST DISCLOSE: each variant's label-free score is a minimum over epochs (and over r for ham) on the same 24 validation draws, with 0 selections for none and 10-17 for the others -> mildly optimistic for the learned variants (the comparator also had epoch selection); one trained model per variant; the four metrics are report-only; the Hamming row is the local-branching / predict-and-search family (Fischetti-Lodi; Han et al. 2023), our difference is the label-free training through the exact layer.
| 2026-10-15_2225_geninteger_case118_none_K0_M0_n1 | case118 VALIDATION gen-int-cuts none K_I=0 M=0 r=1 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 4.2296% | 0.2403% | 19.40% | 13.12% | 10.0s | fallbacks 7; label-free score 1.004816 |
| 2026-10-15_2228_geninteger_case118_ham_K0_M0_n1 | case118 VALIDATION gen-int-cuts ham K_I=0 M=0 r=1 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 0.1189% | 0.0479% | 4.14% | 7.71% | 9.1s | fallbacks 3; label-free score 0.961620 |
| 2026-10-15_2308_geninteger_case118_rows_K8_M0_n1 | case118 VALIDATION gen-int-cuts rows K_I=8 M=0 r=1 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 1.0696% | 0.1918% | 7.85% | 10.03% | 25.0s | fallbacks 12; label-free score 0.969618 |
| 2026-10-15_2345_geninteger_case118_rows_K2_M0_n1 | case118 VALIDATION gen-int-cuts rows K_I=2 M=0 r=1 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 0.8671% | 0.1735% | 7.50% | 11.44% | 24.3s | fallbacks 12; label-free score 0.968767 |
| 2026-10-16_0021_geninteger_case118_rows_K32_M0_n1 | case118 VALIDATION gen-int-cuts rows K_I=32 M=0 r=1 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 2.4106% | 0.2755% | 13.14% | 12.25% | 9.5s | fallbacks 10; label-free score 0.981482 |
| 2026-10-16_0103_geninteger_case118_ham_K0_M2_n1 | case118 VALIDATION gen-int-cuts ham K_I=0 M=2 r=2 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 0.7351% | 0.1370% | 6.61% | 9.72% | 8.6s | fallbacks 2; label-free score 0.966948 |
| 2026-10-16_0142_geninteger_case118_rows_K8_M2_n1 | case118 VALIDATION gen-int-cuts rows K_I=8 M=2 r=1 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 1.4854% | 0.4563% | 10.58% | 17.67% | 9.1s | fallbacks 7; label-free score 0.974495 |
| 2026-10-16_0219_geninteger_case118_rows_K32_M2_n1 | case118 VALIDATION gen-int-cuts rows K_I=32 M=2 r=1 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 1.9543% | 0.2403% | 12.70% | 14.32% | 9.2s | fallbacks 9; label-free score 0.977823 |

## GENERALIZED INTEGER CUTS, overnight experiment 1 RESULTS (2026-10-15 22:25 -> 2026-10-16 03:00; validation only, case118, DEPLOY_RHO=3e5, no unit fixings in any variant; logs gi_n1_*.log)
| variant | label-free score | gap mean | gap median | discrete | continuous | time | fallbacks | verdict (score <= 0.9638 AND gap mean <= 0.26%) |
|---|---|---|---|---|---|---|---|---|
| current: per-unit fixings + 2 mixed rows (comparator) | 0.9618 | 0.1286% | 0.0523% | 4.76% | 8.71% | 10.6 s | 1 | - |
| none (V-net + short solve; control) | 1.0048 | 4.230% | 0.240% | 19.40% | 13.12% | 10.0 s | 7 | fails |
| **ham: 1 Hamming row, r=1** | **0.9616** | **0.1189%** | 0.0479% | 4.14% | 7.71% | 9.1 s | 3 | **WORKS** |
| ham r=2 + 2 mixed rows | 0.9669 | 0.735% | 0.137% | 6.61% | 9.72% | 8.6 s | 2 | fails |
| rows K_I=2 | 0.9688 | 0.867% | 0.174% | 7.50% | 11.44% | 24.3 s | 12 | fails |
| rows K_I=8 | 0.9696 | 1.070% | 0.192% | 7.85% | 10.03% | 25.0 s | 12 | fails |
| rows K_I=32 | 0.9815 | 2.411% | 0.276% | 13.14% | 12.25% | 9.5 s | 10 | fails |
| rows K_I=8 + 2 mixed | 0.9745 | 1.485% | 0.456% | 10.58% | 17.67% | 9.1 s | 7 | fails |
| rows K_I=32 + 2 mixed | 0.9778 | 1.954% | 0.240% | 12.70% | 14.32% | 9.2 s | 9 | fails |
Reading: learned integer rows from the CutNetK integer category do NOT replace the per-unit fixings (all fail; many fallbacks, up to 12/24). The single learned Hamming-ball row (r=1, centre = our own-target network) matches the comparator on validation (0.9616 vs 0.9618; gap 0.119% vs 0.129%).
CAVEATS: (a) the ham score is the minimum over 10 epochs that vary 0.9617-0.986 (and over r), so it is optimistic; the warm-start (initial) score was 0.9717; (b) r=1 means at most about one flip from the predicted commitment, i.e. nearly the same restriction strength as fixing 52/54 units -- general in FORM, similar in EFFECT; (c) one trained model per variant, no seeds; (d) validation only -- needs seeds, a fresh untouched draw with new references, and then a test.

## HAMMING ROW AT THE PROXY'S DATA SIZE (written 2026-10-16, before the runs; VALIDATION ONLY)
Request: train the learned Hamming-ball row (mode ham, 1 row, M=0, no unit fixings) on the SAME NUMBER of loads as the supervised NN proxy (118 labelled training loads on case118). Ours uses NTRAIN=118 loads drawn at random (seeded) from the 144 ORIGINAL training loads that have our own target; no extra unlabeled loads, no reference used to pick them (the proxy's 118 are the loads with a converged reference; ours are a different seeded 118, so the overlap is partial). V-net is still the clean `_env_nc` trained earlier on the 144 training loads without labels (not retrained).
Runs: SEED 0, 1, 2 (seed changes the subset, the centre-net init, the batch order); r from {1,2,3,5,8,12,20} and the epoch by the label-free score on ALL 24 validation draws; DEPLOY_RHO=3e5, W=10; same working criterion as before (label-free score <= 0.9638 and validation gap mean <= 0.26%); comparator 0.9618 / 0.1286%; the full-data (444) ham run scored 0.9616 / 0.1189%.
Script chain acuc/night_ham118_2026-10-16.sh (status logs/night_ham118.status). Same disclosures as above (selection optimism, validation reuse, no test).
| 2026-10-16_0839_geninteger_case118_ham_K0_M0_n118s0 | case118 VALIDATION gen-int-cuts ham K_I=0 M=0 r=1 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 0.7765% | 0.0533% | 6.88% | 7.30% | 11.4s | fallbacks 4; label-free score 0.967721 |
| 2026-10-16_0941_geninteger_case118_ham_K0_M0_n118s1 | case118 VALIDATION gen-int-cuts ham K_I=0 M=0 r=3 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 0.8816% | 0.2403% | 7.85% | 13.03% | 11.4s | fallbacks 7; label-free score 0.968949 |
| 2026-10-16_1006_geninteger_case118_ham_K0_M0_n118s2 | case118 VALIDATION gen-int-cuts ham K_I=0 M=0 r=2 rho=3e+05: gap mean | median | disc | cont | time | fallbacks = 0.3282% | 0.2187% | 4.06% | 11.33% | 11.1s | fallbacks 5; label-free score 0.963229 |

## HAMMING ROW AT THE PROXY'S DATA SIZE: RESULTS (2026-10-16 08:39 -> 10:31, validation only, case118, 118 training loads from the original training split, DEPLOY_RHO=3e5; logs gi_n118_ham_s{0,1,2}.log)
| run | r | epoch | label-free score | gap mean | gap median | discrete | continuous | time | fallbacks | criterion (<=0.9638 AND gap mean <=0.26%) |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 1 | 1 | 0.9677 | 0.7765% | 0.0533% | 6.88% | 7.30% | 11.4 s | 4 | fails |
| seed 1 | 3 | 1 | 0.9689 | 0.8816% | 0.2403% | 7.85% | 13.03% | 11.4 s | 7 | fails |
| seed 2 | 2 | 6 | 0.9632 | 0.3282% | 0.2187% | 4.06% | 11.33% | 11.1 s | 5 | fails (gap) |
| mean of 3 seeds | | | 0.9666 | 0.662% | 0.171% | 6.26% | 10.55% | 11.3 s | 5.3 | |
| 444 loads, seed 0 only (earlier) | 1 | 9 | 0.9616 | 0.1189% | 0.0479% | 4.14% | 7.71% | 9.1 s | 3 | works |
| current method (per-unit fixings + 2 mixed rows, 444 loads) | | | 0.9618 | 0.1286% | 0.0523% | 4.76% | 8.71% | 10.6 s | 1 | comparator |
Reading: at the proxy's data size (118 loads) the Hamming row does NOT meet the pre-registered criterion in any of 3 seeds, and the chosen r / epoch differ across seeds (r = 1, 3, 2). The 444-load result is one model, so data size and seed noise are not separated yet (444-load seeds 1, 2 not run). Validation only; no test number.

## HAMMING ROW ON OTHER PROBLEMS, a first look (written 2026-10-16 ~13:45, validation only, ONE seed, ONE model; exploratory "to get the idea", no test)
case300: `DEPLOY_RHO=1e4 python 67_generalized_integer_cuts.py case300 ham 0 0 _n1` (all 422 own targets, V-net `_env_nc`, W=30, r from {1,2,3,5,8,12,20} + epoch by the label-free score on ALL 24 validation draws). Comparator = current case300 full method at rho 1e4: label-free 1.015163; validation four metrics 0.0521% | 0.0212% | 3.43% | 4.25% | 41.3 s (22 draws with references).
Vehicle MILP (vehicle 0 only) and MINLP: new script dixit_vehicle/20_hamming_vehicle.py (integer Hamming row = L1 ball sum_t |z_t - zhat_t| <= r with aux variables, centre net trained through the exact vehicle layer; same label-free targets and validation as the final vehicle runs). Comparator = the current method's own bundle (per-step integer cuts + 2 general rows) deployed on the same validation profiles.
Same disclosures as the case118 Hamming experiments (selection optimism, validation reuse, one model).

## HAMMING (L1-ball) ROW ON THE VEHICLE, first look (2026-10-16 ~14:10, validation only, ONE seed, vehicle 0 for the MILP; script dixit_vehicle/20_hamming_vehicle.py; logs dixit_vehicle/logs/ham_veh_*.log)
Centre net = MLP warm-started by MSE on the label-free targets (zhat = 3*sigmoid), fine-tuned through the exact layer (gate passed: corr 1.000 / 0.999), r from {1,2,3,5,8,12} + epoch by the label-free score on all 24 validation profiles; one short solve W=0.01 with the soft L1 ball; no per-step bounds, no general rows. Four metrics on the 24 validation profiles with references (gap mean | median | disc | cont | time):
| problem | Hamming row (r) | label-free | current final method (same profiles) | label-free | supervised proxy (same profiles) |
|---|---|---|---|---|---|
| MILP vehicle 0 | 0.0118% / 0.0000% / 5.54% / 5.50% / 25.3 ms (r=3) | 1.003755 | 0.0000% / 0.0000% / 0.00% / 0.00% / 12.2 ms | 1.003637 | 0.0040% / 0.0000% / 2.03% / 2.02% / 5.6 ms |
| MINLP | 1.3959% / 1.3545% / 8.53% / 7.93% / 16.7 ms (r=1) | 1.034004 | 0.3319% / 0.1508% / 4.77% / 4.61% / 27.8 ms | 1.023163 | 0.6169% / 0.4990% / 6.58% / 6.21% / 15.0 ms |
Reading: MILP v0 is easy for everyone (all gaps ~0); the Hamming row is slightly worse than the current per-step cuts (0.012% vs 0.000%). On the MINLP the Hamming row is clearly WORSE than both the current method and the proxy (1.40% vs 0.33% vs 0.62%); the best epoch was epoch 0 (fine-tuning did not help) and r=1 was selected (larger r worse).
CAVEATS: untuned implementation (centre = MSE regression on ordinal levels; a classification centre may be better), one seed, validation only, vehicle 0 only for the MILP.
| 2026-10-16_1405_geninteger_case300_ham_K0_M0_n1 | case300 VALIDATION gen-int-cuts ham K_I=0 M=0 r=3 rho=1e+04: gap mean | median | disc | cont | time | fallbacks = 0.0449% | 0.0202% | 2.96% | 3.27% | 25.8s | fallbacks 4; label-free score 1.014999 |

## HAMMING ROW ON case300, first look (run 2026-10-16 13:50 -> 16:25, validation only, ONE model, 422 own targets, DEPLOY_RHO=1e4, W=30; log gi_case300_ham.log)
r = 3 (label-free), epoch 6; label-free 1.014999 (epochs all 1.0150-1.0159, flat); validation four metrics (22 draws with references): gap mean 0.0449% | median 0.0202% | discrete 2.96% | continuous 3.27% | 25.8 s | fallbacks 4.
Comparator (current case300 full method at rho 1e4, same V-net/draws): label-free 1.015163; 0.0521% | 0.0212% | 3.43% | 4.25% | 41.3 s | fallbacks 1. -> the Hamming row matches / is marginally better on validation. NOTE the time is not like-for-like: the Hamming run skips the anchor solve (the current method adds it), and its score is a min over epochs and r on the same 24 draws. No test number; all case300 draws other than fresh5 are validation/training and fresh5 is spent.

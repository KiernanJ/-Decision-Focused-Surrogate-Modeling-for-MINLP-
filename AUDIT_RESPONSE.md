# Response to the pre-publication audit (branch `audit-fixes`, 2026-10-06/07)

Status of every item in Kiernan's `AUDIT.md`, what was changed in code, and what the
re-run shows. Numbers marked PENDING were still running when this was written; the
files they will land in are given.

## What changed in the pipeline

| Audit item | Fix | Where |
|---|---|---|
| 6. slack-bus generator | limits read from `net.ext_grid` (case118 805.2 MW, Q ±300 MVAr; case300 2399 MW, Q 0–10 MVAr); treated as an ordinary committable unit; `always_on` no longer set | `src/acopf_data.py` |
| Major: forced-on units (qmin > 0) | the Q box always contains 0; `qmin*u <= qg <= qmax*u` alone controls it | `src/socp.py`, `src/acuc.py` |
| case300 band | with the true capacity (326.8 pu) the 10% reserve is infeasible above s≈1.25, so case300 draws s ~ U[0.55, 1.20] (case118 keeps U[0.55, 1.30]) | `src/data2.py` |
| 4. non-converged references | QCAC iteration warm-started from the complete previous iterate; Gurobi `WorkLimit` (deterministic) instead of a wall-clock limit; status/convergence/MIP gap stored per instance; only CONVERGED references are scored; `J_ref = price(u_ref)` (same evaluator as the methods) | `src/acuc.py`, `acuc/01_make_pool.py`, `acuc/07_select_eval.py` |
| split defined over drawn instances | the split no longer depends on which references succeeded | `src/data2.py` |
| 5. cut anchor | `b = <A, w_rlx(V_pred)> - depth`, the uncut relaxation solved at the network's OWN V (stop-gradient) | `acuc/02_train.py`, `acuc/07_select_eval.py` |
| 2. weights | trainer saves the validation-selected model as `ck["net"]` (last epoch kept as `ck["last"]`) | `acuc/02_train.py` |
| 3. settings chosen on test | order × threshold chosen per seed on the 24 VALIDATION instances, label-free (deployed cost / uncut relaxation cost); NDOWN is a fixed compute budget (25 / 40); training rounds at 0.5 for both cases (no test-derived 0.8) | `acuc/07_select_eval.py` |
| Protocol: trainer scores test | removed; the trainer never reads a reference | `acuc/02_train.py` |
| Protocol: failures dropped | failures counted in validation and test | `acuc/02_train.py`, `acuc/07_select_eval.py` |
| Proxy fairness | supervised proxy hyperparameters chosen on validation BCE; same label-free deployment selection as ours | `acuc/07_select_eval.py` |
| Baselines | AC-UC: learned-V-no-cut and untrained controls (same deployment setting as ours); vehicle: no-cut, ridge, constant schedule, validation-tuned MLP proxy | `acuc/07_select_eval.py`, `vehicle/05_baselines_timing.py` |
| Statistics | paired per-instance bootstrap CI + Wilcoxon, per-seed spread, failure counts | `acuc/08_aggregate.py` |
| Timing | anchor solve included; vehicle at Threads=1 both sides with trained weights; AC-UC re-timed on an idle machine (methods and reference) | `vehicle/05_baselines_timing.py`, `acuc/09_timing_ref.py`, `acuc/10_timing_methods.py` |
| Vehicle gradient | `eta*(E_max - E_T)` term added to layer 2 | `src/diffhv.py` |
| Gradient claims without a source | committed scripts + outputs: `gradient_checks/degeneracy.py`, `05_vs_cvxpylayers.py` (converged reference), `acuc_e2e_check.py`, `vehicle_grad_check.py` (w_int = 5) | `gradient_checks/out/` |

## Results of the re-run (corrected data, corrected protocol)

### case118 — complete (43 scored test instances: 48 minus 5 non-converged references; 4 seeds)

| method | mean \|gap\| | median \|gap\| | discrete | continuous | per-seed \|gap\| |
|---|---|---|---|---|---|
| supervised NN proxy | **0.046%** | 0.000% | **1.93%** | **4.02%** | 0.050 / 0.045 / 0.044 / 0.044 |
| learned V + cut (ours) | 1.43% | 0.56% | 8.29% | 25.2% | 1.74 / 1.11 / 2.35 / 0.52 |
| learned V, no cut | 1.27% | 0.60% | 8.71% | 24.9% | 2.11 / 0.71 / 1.28 / 0.96 |
| untrained network | 12.7% | 13.0% | 39.6% | 108.6% | 12.6 / 13.5 / 11.9 / 13.0 |

Paired per instance (seed-averaged |gap|, n = 43):
- ours − proxy = +1.38 pp, 95% CI [+1.04, +1.75], Wilcoxon p = 4e-12 (proxy better on 41/43)
- ours − no-cut = +0.16 pp, 95% CI [−0.05, +0.38], p = 0.16 (no detectable effect of the cut)
- ours − untrained = −11.3 pp, p = 2e-13 (training V is what helps)

Validation selected confidence order at every seed for every method; the threshold is inert under
confidence order (the reserve top-up re-commits exactly what the threshold removed).
The release's case118 number (0.176%) relied on the per-generator threshold fitted on training
instances; that fit is not part of this protocol.

### case300 — ours and controls complete; PROXY PENDING (training labels still solving)

37 scored test instances (48 minus 9 non-converged and 2 infeasible references).
Per-seed mean |gap|: ours 1.63 / 1.16 / 1.53 / 1.53; no-cut 1.77 / 1.10 / 1.55 / 1.49;
untrained (s1–s3) 2.90 / 1.32 / 1.32 (s0 pending). Selected epochs 22 / 1 / 2 / 2: the validation
loss rises after the first epochs, as in the release. Training does not help on case300.

### Vehicle — complete (48 test instances, 4 seeds, all solvers single-threaded)

| method | gap | discrete | continuous | median time |
|---|---|---|---|---|
| uncut relaxation + same projection | **0.088%** | 1.95% | 2.19% | 0.034 s |
| learned cut (ours) | 0.124% | 2.93% | 3.03% | 0.043 s |
| ridge regression on z\* | 0.461% | 5.67% | 5.43% | 0.034 s |
| MLP proxy (validation-tuned) | 1.06–1.64% | 8.2–10.0% | 7.0–8.1% | 0.03 s |
| constant training-mean schedule | 1.49% | 11.65% | 10.14% | 0.046 s |
| exact MINLP (Gurobi, Threads=1) | 0 | 0 | 0 | 1.03 s |

Retraining with the eta term reproduced the release models to 7 decimals: the term contributes no
gradient here (E_T does not respond to z in layer 2 unless the engine limit binds). The learned cut
is worse than no cut; speed-up vs Gurobi at equal threads is ~25x, not 92x.

### Gradient checks (committed outputs in `gradient_checks/out/`)

- Degeneracy (case118, flat V, rho = 1e6): 2185 active rows, rank 1805 (deficiency 380), KKT
  condition number 1.8e21 — confirms the qualitative claim.
- Layer vs cvxpylayers on the QCAC relaxation, against a CONVERGED finite-difference reference
  (self-consistency 8e-4): ours cosine 0.997, cvxpylayers 0.750 (its forward u is 0.059 off).
  "97% wrong" is withdrawn.
- END-TO-END through the deployment chain at the trained models (rho = 1e6), relative error per
  random direction where the FD reference converged: case118 cut median 33%, V median 53%;
  case300 cut 33%, V 113%. Many FD references do not converge (non-smooth loss). The paper's
  end-to-end cosines (0.962/0.940, V 0.864) have no source and are not reproduced.
- Vehicle (w_int = 5): dL/dA 10/10 within 10% (median 0.01%), dL/dthr 10/10, dL/db 9/10
  (instance 170: 106%).
- Exact vs smoothed estimator: RE-RUNNING (`out/estimator_*.json`).

## Bottom line

With the audit's corrections, the learned cut does not beat the no-cut control on any of the
three problems, and the supervised proxy is far better on case118. What survives: the cone layer
differentiates degenerate conic programs where cvxpylayers does not (0.997 vs 0.750), and training
the linearisation point helps on case118 (12.7% -> 1.4%). The paper's main claim does not hold in
its current form; the framing is a decision for Parth and Kiernan.

## Still running / not done
- case300 proxy (labels solving, ETA ~07:30, then ~1.5 h evaluation)
- idle-machine timing for AC-UC (reference and methods) — current AC-UC times were taken under load
- estimator comparison re-run
- paper not rewritten: its central claim no longer holds, so the framing needs your decision first

# Pre-publication audit (2026-10-05)

## Summary and what I'd like from you

This is a code and results audit of commit `85a9bba` ("Final release"), done ahead of AIChE and the journal submission. It aims to catch problems before a reviewer or the audience does.

The core of the method holds up well:
- The cone-layer implicit differentiation is correct.
- The Clarabel SOCP matches the Gurobi model.
- No reference labels leak into training.
- The reporting pipeline (results → `summary.json` → `numbers.tex` → PDF and figures) reproduces exactly.
- The case118 result is real and honestly reported.

However, several findings change what the case300 and vehicle results show. The main ones:
- the case300 numbers were scored with last-epoch rather than selected weights;
- the case300 deployment settings have no selection path other than the test split;
- some case300 references never converged;
- on the vehicle problem, solving with no cut does better than the learned cut.

I think these need resolving before we present or submit.

**Could you go through the six critical items** and either confirm them or tell me what I've missed? You know the history behind these choices and I don't. Some may have explanations that aren't visible in the code. After that, I'd like us to agree on the re-evaluation plan at the end.

**Evidence.** Every critical item has a script in `audit/` (see `audit/README.md`). Each script only reads the repo, and runs from the repo root. The slow re-evaluations also ship precomputed outputs, in `audit/out/precomputed/`. The audit used five parallel reviews: gradient math, AC-UC protocol, baselines and timing, formulations and references, and paper claims against results.

Items marked ✓ were re-run independently a second time. Items without a script reference were checked during the audit but aren't scripted here. Line numbers like "l.314" refer to `paper/main.tex`.

## Critical (these change the paper's conclusions)

1. ✓ **On the vehicle problem, the learned cut does worse than no cut.** *Script: `audit/c1_vehicle_baselines.py`.*
   - **No-cut baseline:** the uncut relaxation followed by the same `qz_restore` projection gives |gap| **0.088%** on all 48 test instances. The method gives **0.123%**.
   - The `qz_restore` docstring (`vehicle/02_train.py:26`) already quotes the +0.088% figure.
   - **Selected epochs:** the selected epochs are 1–3 of 25, so the deployed models are close to their initialisation.
   - **NN proxy:** the proxy (hidden width 16, lr 1e-2, 200 full-batch steps, no validation) does worse than predicting the training-mean schedule (1.51%). A ridge regression reaches 0.43%. A quickly tuned 256-unit MLP reached 0.77–0.82% (not scripted here).
   - **Consequence:** the "15× more accurate than the proxy" result (VRATIO; abstract, l.41/312/352) reflects the restoration step and a weak baseline, not the learned cut.

2. ✓ **The case300 numbers use last-epoch weights, not the epoch selected on validation.** *Scripts: `audit/c2_checkpoint_epochs.py`, `audit/c2_c3_case300_rescore.py`.*
   - **Which weights are loaded:** `04_eval_case300.py:64`, `05_timing.py:49` and `03_threshold_fit.py:73` load `ck["net"]`, which holds the final epoch. The selected epoch in `ck["best"]` is 0, 4, 0 or 1 depending on the seed.
   - **Cause:** `02_train.py` saves the checkpoint inside the epoch loop and doesn't re-save after loading `best`.
   - **Effect, re-scored with the selected weights:**
     - |gap| goes from 0.061 to 0.066.
     - The signed gap goes from −0.029% to **+0.008%**, so the "slightly cheaper than QCAC" statement at l.314 no longer holds.
     - The discrete error goes from 2.91% to 3.18%.
     - The script reproduces the published rows exactly when given `ck["net"]`.
   - **case118** is unaffected: its selected epoch is the last one, epoch 29.

3. ✓ **The case300 deployment settings were evaluated only on the test split.** *Script: `audit/c2_c3_case300_rescore.py --weights init`.*
   - **The sweep:** `04_eval_case300.py` sweeps repair order × threshold {0.5, 0.65, 0.80} × NDOWN {6, 25, 40} over `te = perm[:48]`, scored against the reference costs. The shipped setting (merit order, 0.80, NDOWN=40) is a corner of that grid.
   - **No other selection path:** I found no train or validation path in the repo that selects these values. Comments in `pipeline.py:80-84, 112-114` quote gap figures that look like test-split results.
   - **Leak into training:** if the 0.80 came from this sweep, it also reached training through `--dthr 0.8`.
   - **The label-free fit disagrees:** the committed label-free case300 threshold fits (`results/thr_lf/sh_case300_*`) chose different values.
   - **Untrained network:** with the same restoration, an untrained network scores 0.058% (seeds 0–1) against the published 0.059% on the same seeds, and it also beats the proxy. This suggests the case300 result comes from the restoration settings rather than from training.
   - **I may be missing context here.** If these values were chosen some other way, the paper should say how.

4. ✓ **Non-converged case300 references are used as ground truth.** *Script: `audit/c4_reference_convergence.py`.*
   - **Count:** 41 of 268 references have slack > 1e-3 (up to 2.35), and 10 of the 48 test instances are among them.
   - **Cause:** `acuc.solve_qcac_iterative` stops when a later iteration finds no incumbent within 120 s, and keeps the earlier high-slack iterate.
   - **Pricing:** our evaluator prices those reference commitments up to 0.94% above the stored cost. One of them (pool index 238, test position 39) can't be priced feasibly and fails for both arms, so case300 has n=188, not 192.
   - **Converged instances only:**
     - Ours: |gap| 0.025%, signed +0.002%.
     - Proxy: |gap| 0.032%, signed +0.006%.
     - The non-converged instances account for most of both arms' mean |gap|.
   - **Dropped instances:** 68 of 336 drawn instances returned no reference and were dropped (`poolload.py:51-53`). Failures concentrate at high load, so the evaluated band effectively tops out at s ≈ 1.20 rather than U[0.55, 1.30].
   - **Paper wording:** the paper reports "227/268 converged" but doesn't say the other 41 are still used as references.
   - **Training is affected too:** 19 training labels used by the proxy are non-converged, and the training pool only contains instances where the reference succeeded.

5. ✓ **On case118, the learned cuts are mostly slack at deployment.** *Script: `audit/c5_cut_activity.py`.*
   - **Cause:** the anchor `b = <A, w_rlx> − depth` uses `w_rlx` from the uncut relaxation at **flat V** (`02_train.py:88-92`), but layer 1 is solved at the **predicted V**.
   - **Measured on the trained checkpoints:**
     - 2–4 of the 4 rows are slack by 5–14 units, depending on the seed. Seed 1 has all 4 slack; seed 3 has 2.
     - At flat V the same rows are active.
     - Slack rows receive zero gradient.
   - **Consequence:**
     - The case118 gains likely come mainly from the learned V and the fitted threshold rather than from the cuts. A "learned V, no cuts" run would settle this.
     - It also conflicts with "anchoring keeps every cut active" (abstract (iii), l.37/155).
   - **case300:** the predicted |V| saturates at the tanh limits (0.50–1.58), and the relaxation slack is 12–25 pu, against 0.61 at flat V.

6. ✓ **The slack-bus (ext_grid) generator is effectively removed.** *Script: `audit/c6_slack_generator.py`.*
   - **Limits:** `acopf_data.py:124-147` reads pandapower's placeholder limit and caps pmax at twice the largest unit, 14.14 pu. The actual limit pandapower holds for case118 is 805.2 MW (8.05 pu); for case300 it is 2399 MW. It also sets qmin = qmax = 0, which gives a no-load cost of 10,050 against a median of 615.
   - **Never committed:** the unit isn't committed in any of the 240 case118 references (7 of 268 on case300).
   - **`always_on` isn't enforced:** `Grid.always_on` is set, but only `qcac.solve_exact` enforces it, and nothing calls that function.
   - **Which unit this is:** on case118 it is the MATPOWER bus-69 generator, which is the largest unit and one of the cheapest.
   - **Suggested fix:** take the limits from the `net.ext_grid` min/max columns, then either pin u=1 or treat it as a normal unit, and state the choice in the paper.

## Major

**Statistics** *(script: `audit/stats_paired.py`)*
- **No uncertainty is reported.** There are no confidence intervals or per-seed spreads. n=192 is 4 seeds × the same 48 test instances, so the effective n is 48: the split is fixed by `rng(0)`, and seeds change only the initialisation and batch order.
- **case300**, paired by instance:
  - ours − proxy = −0.014 pp, 95% CI [−0.054, +0.017], Wilcoxon p = 0.30.
  - Ours is better on 13 of 47 instances, worse on 19, tied on 15.
  - With the selected epoch, p ≈ 0.77 (not scripted here).
- **case118:** the proxy is better (Wilcoxon p = 0.003; the bootstrap CI on the mean difference is [−0.009, +0.143]).
- **Per-seed spread is large and unreported:** the per-seed |gap| for ours is 0.117 / 0.281 / 0.250 / 0.054.
- **The reference's own noise is about 0.1–0.2%.** That is as large as, or larger than, the differences being compared. Its sources:
  - MIP gap 1e-3.
  - A 120 s per-iteration time limit that binds on case300.
  - Slack ≤ 1e-3, while our evaluator accepts < 1e-4.
  - Reruns changed the commitment on 3 of 4 instances.
- Using J_ref = price(u_ref) would put the reference and our deployments through the same evaluator.

**Model description**
- **Formulation:** the code (`acuc.py:68-85`, `socp.py:82-91`) constrains |c − lin(v)| ≤ ξ, a full first-order expansion of both sides. That differs from the QCAC convex-concave inner approximation the paper describes. The unused `qcac.py:126-152` has the convex-concave form.
- **AC feasibility:** the pipeline never checks it (`verify` is never called). A post-hoc check found the priced points close to feasible: residuals ≤ 1.5e-5 pu on case118 and ≤ 5e-3 pu on case300.
- **Branch limits never bind:** all ratings are 99 pu (9900 MVA).
- **Not stated in the paper:** the UC is single-period, with no startup costs, min up/down times or ramping.
- **Forced-on units:** three case300 units (5, 44, 45) are forced on because qmin > 0 is used as a variable bound.

**Gradients.** The core implicit-differentiation math is correct and was verified against finite differences.
- **ε doesn't match the paper:**
  - `02_train.py:61` uses REG1 = REG2 = 1e-10; the paper (l.362) says 1e-8 for layer 1 and 1e-10 for layer 2.
  - The "broad plateau 1e-6 to 1e-9" holds at ρ = 1e4. At the training value ρ = 1e6, layer 2 only works at ε ≈ 1e-9 to 1e-10.
- **∂ℓ/∂V is less accurate than reported:** its cosine against finite differences is 0.3–0.6 at initialisation, against the paper's 0.864. The error comes through the dAᵀy term (V enters A), which the "dx is unique even though y is not" argument doesn't cover.
- **The "97% wrong" cvxpylayers figure** is measured against an unconverged finite-difference reference (h = 1e-2), using a max-abs metric:
  - Against h = 1e-3, its cosine is 0.80, against 0.9988 for ours.
  - Its forward solution also differs from Clarabel's.
  - The qualitative claim (cvxpylayers is clearly worse) survives.
- **Smaller layer issues:**
  - `ConeLayer.forward` accepts AlmostSolved (`conelayer.py:194`).
  - Failed instances are silently dropped from training and from the validation loss.
  - The reserve penalty is inactive by construction.
  - The vehicle gradient omits the η(E_max − E_T) term (negligible in practice).

**Claims without a committed source**
- **No script or output** exists in the release for these figures:
  - the degeneracy statistics (2154/1809/345, condition number ~1e21);
  - the linear-solver table (LSQR 0.18 / 8.4 s, SVD, LU, "8000×") and the ε plateau;
  - most of the gradient-validation table: 0.944 / 0.969, 0.990, 1.000000 vs 0.9955, 0.962 / 0.940, 0.864;
  - "0.9999 → 0.42" and "majority commitment within 0.35%".

  They appear in code comments or in scripts removed in `85a9bba`.
- The output of `07_estimator_compare.py` isn't committed, and the script fails when writing it because the `results/` folder doesn't exist.
- So the statement that the released code reproduces every number (l.363–364) doesn't currently hold.

**Timing**
- ✓ **The conclusion's speed-up range is wrong:** it says "24–92×" (`\SPDA--\SPDV`), but case300 is 11×, so the range should be 11–92×.
- **case300 reference:** its median time is set by the 120 s per-iteration limit. 87 instances hit the limit on every iteration.
- **The anchor solve is excluded:** both timing scripts leave out the uncut relaxation solve the cuts need. Including it, the vehicle speed-up goes from 92× to 79×.
- **Vehicle timing conditions:** it used untrained networks and multi-threaded Gurobi. With both sides at Threads=1 it is roughly 24–32×.
- **Vehicle restoration is an integer solve:** `qz_restore` is a MIQCP of the same size as the original problem, so "one conic solve instead of mixed-integer solves" doesn't describe the vehicle pipeline.
- **Other gaps:**
  - The vehicle proxy timer covers only the projection.
  - The case300 timing command and its threshold aren't recorded.
  - TRAINV = 1,420 s is hard-coded in `make_numbers.py`.

**Protocol**
- **Restoration differs between the arms:**
  - The proxy always rounds at 0.5.
  - Ours uses the fitted per-generator threshold on case118 and 0.80 on case300.
  - On case118 the fitted threshold accounts for much of the gain (shipped → fitted |gap| per seed: 0.51 → 0.12, 0.75 → 0.28, 1.15 → 0.25, 0.41 → 0.05; from the committed `thrlf_eval` files).
- **case118 selection:** the threshold was fitted at NDOWN=6 but deployed at 25. NDOWN=25 and confidence order have no documented label-free selection.
- **The trainer scores test** at the end of every run, including development runs (`ckptD/`).
- **The model-selection code contradicts its comments.** It selects on the surrogate validation loss, and `val_deploy` is computed on the last epoch and never used. The vehicle trainer selects on deployed cost instead.
- **Failures aren't counted:** failed instances are dropped with `continue`.

**Reproducibility**
- The case118 checkpoints predate `head_thr`: they have 271,412 parameters, while the committed `CutNetK` builds 285,290 and consumes the RNG differently.
- The vehicle result JSONs record `pool=results/poolorig`, `no_thr` and `tag _cutqz`, which come from an earlier script version.
- Output files were renamed by hand, and there's no requirements file.
- Pool generation (the `GATE_*` env vars) and the case300 timing and threshold commands are undocumented.
- Some code refers to external directories (`qcac_clean`, `difflayer_work`).

## Minor
- Pricing accepts slack < 1e-4; the paper says ≤ 1e-6.
- Tie-breaking differs between the four restoration copies (`argsort(uf)` vs `argsort(-uf)[::-1]`).
- Parameter counts exclude unused heads: the case300 checkpoint has 499,007 parameters, while the paper says 481k.
- Table 1 mixes mean and max training times. "About an hour per seed" for case300 is closer to 1.7 h.
- The vehicle gradient check reports a median that doesn't show one instance with 33% / 89% error, and it used w_int=0 while the paper configuration is w_int=5.
- `validate()` fails its own 1e-4 check, because it ignores BR_G.
- At l.314, "driven by a tail" is worth rewording: the proxy's median |gap| is about 18× lower than ours.
- Bibliography: `dixit2025dfsm` lacks a volume and URL; `besancon2023` should cite the journal version; the Gurobi manual year is out of date.

## What holds up
- The ConeLayer KKT and cone-projection derivatives, the soft-rounding derivatives, `v_chain` and `rows_torch` all match finite differences.
- The Clarabel SOCP equals the Gurobi build to about 1e-8.
- No label leakage into the loss, the cost scaling or the threshold fit.
- The splits are consistent across scripts with no duplicates, and normalisation uses the training set only.
- `summary.json`, `numbers.tex`, the PDF and the figures all match the raw results exactly.
- The vehicle references are all optimal, and the vehicle models are consistent across scripts.
- The soft-cut fallback never triggered.
- **case118 is a real result.** On seed 0, an untrained network with the same restoration gets 2.40% |gap|, against 0.12% for the trained one. The paper presents case118 honestly: the proxy wins there.

## Proposed fix plan (in priority order)
1. **Re-evaluate correctly.**
   - Load `ck["best"]`, or select on the deployed validation cost and save that model.
   - Use converged references only (or re-solve them), with J_ref = price(u_ref).
   - Choose order, threshold and NDOWN on the validation split without labels, then retrain case300 with that threshold.
   - Count failures, and report paired, instance-level confidence intervals.
2. **Add the baselines:**
   - no cut, with the same restoration;
   - untrained network, with the same restoration;
   - learned V with no cuts;
   - a tuned proxy given the same threshold fit;
   - ridge and constant-schedule baselines for the vehicle.
3. **Fix the cut anchor:** compute `w_rlx` at the predicted V (with a stop-gradient), so the cuts stay active.
4. **Fix the slack-generator data** and decide whether to regenerate the references. Store the MIP gap and status for each.
5. **Update the paper:**
   - the formulation wording, ε and the 11–92× range;
   - timing that includes the anchor solve;
   - branch limits and single-period UC;
   - the 47-instance case300 count;
   - sources for the gradient tables (release the scripts or drop the claims).

A framing that would hold up for AIChE in the meantime: exact gradients through degenerate conic layers are achievable and verified, demonstrated on case118, with case300 and the vehicle problem presented as open problems.

# Diagnostics: what drives the learned surrogate (2026-10-06/07)

This follows up `AUDIT.md`. That audit found that the case300 and vehicle results didn't hold under correct evaluation. These diagnostics ask why, and what the method does contribute. There are three studies:

1. **case118 component ablation** (§1): which parts produce the published case118 result.
2. **case300 cut diagnosis** (§2): why the learned cuts are weak and inactive, and why the training loss rises.
3. **Vehicle diagnosis** (§3): why the learned cut does worse than no cut.

All scripts only read the repo and write to their own `out/` folder. The outputs are included, so nothing needs re-running to follow this document. Section "Reproducing" below has the commands.

## Summary and what I'd like from you

**Main findings**

- **On case118, the learned convexification point V does most of the work.**
  - Learning V takes the deployed |gap| from 2.35% to 0.32%, better on 48 of 48 test instances.
  - The learned cuts add a further −0.15 pp (0.32% → 0.18%), but only on the 3 of 4 seeds where cut rows stayed active.
  - The supervised proxy is still more accurate: 0.10%, better on 39 of 48 instances.
- **On case300, the cuts are weak by construction, get almost no gradient, and go slack once V moves.**
  - The training loss rises because of a discrete reserve-margin escalation inside `DiffDeploy` that the gradient can't see. Moving V away from flat triggers it.
- **On the vehicle problem, the gradients are exact, but there is almost nothing left for a cut to fix.**
  - The uncut relaxation plus the L1 projection is already within 0.12%.
  - Every remaining error is a near-tie rounding.
  - The integrality penalty (`w_int=5`) dominates the gradient and makes deployment worse.
  - A no-learning local search over the near-ties reaches 0.012%.
- **Overall interpretation.** In single-period AC-UC, the error of a single convex surrogate solve comes mostly from the physics linearisation (about 2%), not the commitment decisions (about 0.1%).
  - That's why learning V matters far more than learning cuts here.
  - I propose framing the contribution as *decision-focused learning of the convexification point*, with learned constraints as a secondary component.
  - A literature search found no prior work on learning the convexification point.

**Questions for you**

1. Do you agree with the interpretation of the case118 ablation (§1), and with the framing in §4 for the AIChE talk?
2. Should we try the anchor fix (anchor the cuts at the predicted V instead of flat V) before the talk? Seed 1 on case118 suggests that keeping rows active is what makes the cuts useful (§1).
3. How do we handle the slack-bus generator bug from `AUDIT.md`? It affects every AC-UC number below.
4. For the vehicle problem: are you OK presenting it as a demonstration of the pipeline, shown next to the no-cut and tie-flip baselines, rather than as an accuracy result?

## Terms used below

- **V, flat V.** V = (Vre, Vim) is the bus-voltage point that QCAC linearises the AC power-flow products around. *Flat V* means every bus at 1.0 pu and angle 0 (`Vre = 1, Vim = 0`). It's the network's output at initialisation, the starting point of the QCAC iterative reference, and the point where the cut anchor is computed.
- **Anchor.** Cuts are built as `b = <A, w_rlx> − depth`, where `w_rlx` is the uncut relaxation solved at flat V (`acuc/02_train.py`).
- **Threshold.** The rounding threshold on the relaxed commitment. It's either 0.5, or a per-generator threshold fitted label-free on training-set deployed cost (`acuc/03_threshold_fit.py`).
- **|gap|.** The absolute cost deviation from the reference, in %. The AC-UC reference is the QCAC iterative heuristic; the vehicle reference is the Gurobi optimum.
- **pp.** Percentage points of |gap|.

## 1. case118 component ablation

**Setup.**
- Committed checkpoints (selected epoch = last epoch on case118), 48 test instances, 4 seeds.
- The paper's deployment: confidence order, reserve top-up, upward repair, NDOWN=25.
- Arms differ only in how the relaxed commitment is produced.
- Each arm gets its own label-free threshold fit with the paper's procedure: 32 training instances, NDOWN=6, 2 shards × 24 configurations, the same RNG seeds.
- Scripts: `case118_ablation/abl.py`, `case118_ablation/analyse.py`. Full output: `case118_ablation/out/analysis.txt`.

**Fidelity.** The paper arm reproduces the published results exactly. The fitted threshold coefficients match `acuc/results/thr_lf/` for all 4 seeds, and test |gap| is 0.117 / 0.281 / 0.250 / 0.054, identical to `thrlf_eval_case118_s*_d25.json`. The proxy (an exact replica of `acuc/06_nn_proxy.py`) matches on seeds 1–3. Seed 0 differs slightly (0.098 vs 0.102); I didn't investigate why.

### Results (mean |gap| over 48 instances × 4 seeds)

| Arm | Threshold 0.5 | Fitted threshold | Per-seed, fitted |
|---|---|---|---|
| Flat V, no cuts (no learning) | 11.59% | 2.35% | 2.26 / 2.24 / 2.54 / 2.35 |
| Flat V + learned cuts | 11.69% | 2.28% | 2.23 / 2.06 / 2.45 / 2.38 |
| Learned V, no cuts | 2.11% | 0.32% | 0.28 / 0.28 / 0.38 / 0.34 |
| **Learned V + cuts (paper)** | 0.71% | **0.18%** | 0.12 / 0.28 / 0.25 / 0.05 |
| Supervised proxy | **0.10%** | 0.33%* | 0.10 / 0.75 / 0.27 / 0.19 |

\*The proxy's threshold is fitted on the proxy's own training instances, so the fit is in-sample and unreliable. On seed 1 it picked a threshold that raised test |gap| from 0.05% to 0.75%. Compare against the proxy at 0.5, as published.

### Paired differences

Each arm is averaged over seeds per instance, then compared paired over the 48 instances. Negative means the first arm is better. The CI is a bootstrap 95% interval; p is from a Wilcoxon signed-rank test.

| Comparison | Δ\|gap\| (pp) | 95% CI | p | First arm better on |
|---|---|---|---|---|
| Learned V vs flat V (no cuts, fitted thr) | −2.02 | [−2.20, −1.85] | 7e-15 | 48/48 |
| Learned V vs flat V (no cuts, thr 0.5) | −9.48 | [−10.42, −8.56] | 7e-15 | 48/48 |
| Cuts on vs off (learned V, fitted thr) | −0.15 | [−0.20, −0.09] | 7e-6 | 32/48 |
| Cuts on vs off (learned V, thr 0.5) | −1.40 | [−1.63, −1.18] | 7e-15 | 48/48 |
| Cuts on vs off (flat V, fitted thr) | −0.07 | [−0.11, −0.03] | 0.003 | 26/48 |
| Fitted vs 0.5 threshold (paper arm) | −0.53 | [−0.77, −0.33] | 4e-6 | 28/48 |
| Paper method vs proxy (proxy at thr 0.5) | +0.07 | [−0.01, +0.14] | 0.003 | 9/48 |

### Takeaways

1. **The learned convexification point is the main effect.** It holds on every instance and every seed.
2. **The learned cuts help, but only while they're active.** On seed 1 every cut row is slack at the predicted V (see `AUDIT.md` item 5), and the cut and no-cut arms give identical results. On the other three seeds the cuts roughly halve the remaining gap (0.28 → 0.12, 0.38 → 0.25, 0.34 → 0.05). This is direct evidence that the stale flat-V anchor costs performance.
3. **The cuts do nothing on their own at flat V.** They were trained jointly with the learned V.
4. **The label-free threshold matters a lot:** −0.53 pp on the paper arm, and 11.6% → 2.35% with no learning at all.
5. **The proxy is more accurate, but needs labels.** It's better on 39/48 instances, at the cost of about 5 hours of reference solves. The mean difference (+0.07 pp) is driven by a few instances, so the bootstrap CI on the mean just touches zero, but the per-instance test is clear.

This partly revises `AUDIT.md` item 5, which concluded the case118 gains were "not from the cuts". The learned V and the threshold carry most of the gain, but the active cuts add a real, smaller improvement.

## 2. case300: why the cuts are weak and inactive

**Setup.** Committed checkpoints at three points: *init* (untrained, same seed), *best* (the epoch selected on validation) and *net* (the last epoch, which is what the paper evaluated). 24 validation instances, 4 seeds. The training objects mirror `acuc/02_train.py` exactly: REG 1e-10, ρ 1e6, τ 0.08, training threshold 0.8. Scripts are in `case300/`; the summary is in `case300/out/summary.txt`.

These diagnostics measure the training objective and the relaxation. I didn't re-run deployed cost here: the audit's re-scoring (`audit/out/precomputed/`) already shows the untrained network matches the trained one after restoration.

### 2.1 The cuts barely change anything, even at initialisation
- At init, the four rows change the relaxation cost by **0.03%**. They shift the relaxed commitment by Σ|Δu| ≈ **0.4**, out of about 45 committed units (`a_state.py`).
- **Only 57% of rows are active at init**, even at flat V where they were anchored.
  - Anchoring makes the *anchor point* violate each row; it doesn't make every row tight at the new optimum.
  - The rows are nearly parallel, so once one binds, the others become redundant.
  - Activity at init by row: general rows 81% / 61%, integer-only rows 43% / 44%.

### 2.2 V takes almost all of the gradient
- **Parameter gradients at init** (`a_state.py`): `head_v` 0.18; cut heads 2e-4 to 1.5e-3; trunk 3e-5.
- **Descent test** (`c_descent.py`, seed 0, 16 training instances): a step on the cut heads alone lowers the batch loss about 100× less than a V-head step of the same size.
- **Two paths for V's gradient** (`f_vsplit.py`): about 30–50% comes through layer 2, where V only changes the *price* of a fixed commitment. Deployment never uses that path, because it re-prices from flat V with the full re-linearisation loop (`pipeline.deploy` → `Socp.price`). Even counting only the decision path through layer 1, V's gradient is 3–55× the cut gradient.

### 2.3 Once V moves, the anchor goes stale and the cuts go slack

| | init | best | net |
|---|---|---|---|
| Rows active at the predicted V | 57% | 28% | 47% |
| Integer-only rows active | 43–44% | 6–10% | 24–55% |
| \|V\| range (seed average of extremes) | 1.00 | 0.84–1.15 | 0.50–1.58 (tanh limits) |
| Relaxation slack ξ at the predicted V (pu) | 0.52 | 1.34 | 25.4 |

Slack rows have exactly zero gradient. The finite-difference check agrees: about 1e-6 at h = 1e-3, against an analytic 0 (`b_fd.py`).

### 2.4 Gradient accuracy (`b_fd.py`, seed 0, 3 instances, init and best)
- **Cut gradients (`b`, `A`)** agree with finite differences wherever the rows are active, within the finite-difference noise.
- **The V gradient is inaccurate:** 3 of 12 random-direction derivatives have the wrong sign, and the rest are off by 30–300%. The loss is also noisy in V at small step sizes.
- **It's still a descent direction:** small steps lower the batch loss within about 10% of the first-order prediction (`c_descent.py`). The inaccuracy adds noise, but it doesn't explain the failure on its own.

### 2.5 Why the training loss rises: a reserve-margin cliff
When the layer-2 re-solve fails, `DiffDeploy` raises the reserve margin (+2% → +8% → … → +70%) and tries again. That is a discrete jump to more committed units, and it carries no gradient.

| | init | best | net |
|---|---|---|---|
| Instances needing escalation | 40% | 83% | 95% |
| Margins used | ≤ +8% | ≤ +28% | up to +70% |
| Spearman(loss, margin) | −0.11 | +0.38 | +0.54 |

`e_ablate.py` swaps V and the cuts independently:
- At flat V with the same cuts, escalation stays at **39–48%**.
- At the predicted V it's **70–96%**.

**So V is the trigger.** Each step lowers the loss locally, but moving V repeatedly tips instances over the escalation threshold, and the loss rises across epochs. That matches the committed training histories: on every seed the case300 training loss bottoms out within the first 5 epochs, then rises by 1.8–4.0% (seed 2 partly recovers by the end).

### 2.6 The learned V points the right way, at first
`d_price.py` prices the training loss's commitment three ways:

| | init | best | net |
|---|---|---|---|
| Loss (single linearisation at the predicted V) | 1.024 | 1.001–1.014 | 1.014–1.044 |
| Same commitment, single linearisation at flat V | 1.024 | 1.009–1.022 | 1.031–1.064 |
| Same commitment, converged `Socp.price` | 1.005 | 0.985–0.994 | 1.003–1.033 |

- A single linearisation at flat V overprices by about 2%.
- At the selected epoch, the learned V moves the loss toward the converged price, which is what predicting V is meant to do.
- Later, V saturates and the commitments grow (about 46 → up to 60 units), driven by the margin escalation.

### 2.7 What the cuts contribute
Measured as the training loss with cuts at the predicted V, minus the loss without them (`e_ablate.py`):
- **Selected epoch:** −0.5% to +0.05%.
- **Last epoch:** −0.7% to +2.4%, worse on 2 of 4 seeds.

For comparison, V is worth about 1–2%.

## 3. Vehicle: why the learned cut fails

**Setup.** Paper configuration (K=2, hidden 16, `w_int` 5, 25 epochs) next to `w_int` 0, seeds 0–1, with per-epoch diagnostics on the 24 validation instances. The test split isn't used. Scripts are in `vehicle/`; the summary is in `vehicle/out/summary.txt`.

These use the committed pool `vehicle/results/pool/`. Note that the committed vehicle training JSONs record `pool=results/poolorig`, which is not in the repo.

### 3.1 The gradient is exact
Finite-difference checks of dL/db and dL/dA (`diag_init.py`) give cosine 1.0000 and relative error ≤ 6e-3, with `w_int` 0 and 5.

### 3.2 There's almost nothing left to fix, and all of it is tie-breaking
- **Baseline:** the uncut relaxation plus `qz_restore` gives validation |gap| **0.119%**. It finds the exact optimum on 9/24 instances, with about 1.2 wrong steps out of 30. The integrality gap of the relaxation itself is 1.94%.
- **Where the errors are:** all 29 wrong steps (out of 720) have a relaxed z within 0.15 of a half-integer. On those 222 near-tie steps the projection is 86.9% correct; everywhere else it's 100% correct.
- **Why:** at the relaxation optimum `z = (S/P_max)·P_eng = 1.5·P_eng` exactly, so the fractional part says nothing about which integer is right.

### 3.3 The integrality penalty causes the degradation
- **Gradient size:** at init the `w_int` term's gradient is 12–14× the cost gradient, with cosine about −0.1 (4–19× over training).
- **Training with `w_int=5`:**

| | init | epoch 24 (s0 / s1) |
|---|---|---|
| Val \|gap\| | 0.147% / 0.176% | 0.43% / 0.51% |
| Wrong steps (Hamming) | 1.7 | 3.5 / 4.1 |
| Fractional steps (of 30) | 27.8 | 26.6 / 26.4 |

- **The objective is anti-aligned with deployed cost:** across epochs, the validation training objective and validation deployed cost have rank correlation −0.85 / −0.73.
- **Infeasibility:** the hard cuts make the relaxation infeasible on up to 3 validation and 9 training instances.
- **With `w_int=0`,** the objective tracks deployed cost (+0.75 / +0.88), but the run overfits after a few epochs and ends at 0.14% / 0.22%, no better than no cut.

### 3.4 Two more mismatches
- **The surrogate is optimistic.** The soft-rounded cost is about 1.011× the relaxation, but the deployed cost is 1.021×. That 1% offset is ten times the 0.12% headroom. The soft-rounded z differs from the deployed z by L1 ≈ 3.8 per instance, against a total error of about 1.2 steps.
- **Depth anchoring starts behind the baseline.** The untrained network scores 0.147% / 0.176%, against 0.119% for no cut.

### 3.5 A no-learning baseline beats every learned variant
`tie_flip.py` runs the projection, then flips each near-tie step to its other rounding and keeps the flip if the priced cost drops. It's label-free and costs about 16 extra convex solves per instance.

| | uncut + projection | + tie-flip |
|---|---|---|
| Validation | 0.119% (9/24 exact) | **0.012%** (15/24 exact) |
| Train (48) | 0.147% | **0.032%** |

## 4. Interpretation and proposed framing

**Where the error comes from.** In single-period AC-UC:
- A single linearisation at flat V misprices by about 2%.
- After restoration, the commitment decisions are off by about 0.05–0.1%.
- The commitment problem is shallow (no time coupling, only a reserve constraint), so relaxation plus rounding plus repair already handles most of it.

The learned convexification point therefore has a far larger lever than learned cuts. The gradient split in §2.2 shows the same thing.

**Proposed contribution.** *Decision-focused surrogates for MINLPs can be trained without labels or a bilevel program, by differentiating through the convex surrogate. For nonconvex problems the parameter worth learning is the convexification point; learned constraints help when they stay active, but they're secondary.*

This stays consistent with the submitted abstract: V enters the surrogate only through the linearised constraints, so learning V *is* learning problem-specific constraints. They're a physically structured family rather than generic cutting planes.

**Where learned constraints should matter.** In problems whose relaxation is weak because of integrality, for example multi-period UC with start-up costs and min up/down times. A convex multi-period UC (DC or SOC power flow, no V to learn) would be a clean test.

## 5. Suggested next steps

1. **Anchor the cuts at the predicted V** (stop-gradient), plus a hinge that keeps rows near-binding. Retrain case118 to check that the cut gain becomes consistent across seeds.
2. **Give the cut heads their own learning rate,** or train in stages (V first, then cuts with V frozen). Decorrelate the rows and start them deeper.
3. **Replace the discrete margin escalation** in `DiffDeploy` with a penalised, differentiable fallback.
4. **Limit V** (smaller `v_scale` or a pull toward flat). Consider removing V's layer-2 path, which only affects the price and not the deployed decision.
5. **Vehicle:** drop `w_int`, start from a cut that does nothing so initialisation equals the baseline, and report the no-cut and tie-flip baselines.
6. **From `AUDIT.md`:** fix the slack-bus generator, use converged references, select deployment settings on validation, and correct the speed-ups.

## Reproducing

**Environment:** same as the audit. Python 3.12, torch 2.14.1, cvxpy 1.9.3, clarabel 0.11.1, gurobipy 13.0.3 (the pip size-limited licence is enough), numpy 2.4.6, scipy 1.18.1, pandapower 3.5.5. Set `OMP_NUM_THREADS=1`. Scripts resolve the repo relative to their own location, so they run from any working directory.

**Vehicle (`diagnostics/vehicle/`)**
```bash
python diag_init.py                         # headroom, init state, FD check (~30 s)
python train_diag.py --w-int 5 --seed 0     # also --w-int 0, --seed 1 (~25 min each; also writes per-epoch ck_*.pt, not included here)
python summarise.py                         # tables in out/summary.txt (seconds)
python tie_flip.py                          # no-learning baseline (~5 s)
```

**case300 (`diagnostics/case300/`)**
```bash
python a_state.py SEED             # state, row activity, head gradients (seeds 0-3, ~1-2 min each)
python b_fd.py 0 init; python b_fd.py 0 best        # finite-difference check (a few minutes)
python c_descent.py 0 init; python c_descent.py 0 best   # descent test (a few minutes)
python d_price.py SEED             # loss vs converged pricing (a few minutes)
python e_ablate.py SEED            # V vs cuts, escalation (a few minutes)
python f_vsplit.py SEED            # dL/dV split into the decision and pricing paths
python summarise.py                # tables in out/summary.txt
```

**case118 ablation (`diagnostics/case118_ablation/`)** takes about 1 hour on 10 cores.
```bash
# 40 threshold fits (~17-21 min each), then 20 test evaluations (a few minutes each)
for a in pred_cut pred_nocut flat_cut proxy flat_nocut; do for s in 0 1 2 3; do for sh in 0 1; do
  python abl.py fit $a $s $sh; done; done; done
for a in pred_cut pred_nocut flat_cut proxy flat_nocut; do for s in 0 1 2 3; do python abl.py eval $a $s; done; done
python analyse.py                  # out/analysis.txt
```

## Files

```
diagnostics/
  README.md                this document
  case118_ablation/
    abl.py, analyse.py
    out/fit_*.json         threshold fits per arm / seed / shard
    out/eval_*.json        per-instance test rows per arm / seed / threshold
    out/analysis.txt       all tables in §1
  case300/
    c300common.py, a_state.py, b_fd.py, c_descent.py, d_price.py, e_ablate.py, f_vsplit.py, summarise.py
    out/*.json, out/*.txt  raw outputs and logs; out/summary.txt
  vehicle/
    common.py, diag_init.py, train_diag.py, summarise.py, tie_flip.py
    out/diag_init.json, out/train_*.json, out/log_*.txt, out/summary.txt, out/tie_flip.txt
```

## Caveats

- **Sample size.** case118 uses a single fixed split; the 4 seeds vary only initialisation and batch order, so the effective n is 48 instances. The case300 and vehicle diagnostics use 24 validation instances.
- **Limited checks.** The vehicle training comparison uses 2 seeds. The case300 finite-difference and descent checks are seed 0 only, on 3 and 16 instances.
- **The slack-bus generator bug** (`AUDIT.md` item 6) is present in every AC-UC arm. The comparisons hold, but the absolute numbers will shift once it's fixed.
- **The AC-UC reference is a heuristic,** with a noise floor of about 0.1–0.2%. That's comparable to the cut and proxy differences in §1.
- **Threshold-fit protocol.** The case118 threshold fit keeps the paper's protocol (fit at NDOWN=6, deploy at 25), applied identically to every arm.

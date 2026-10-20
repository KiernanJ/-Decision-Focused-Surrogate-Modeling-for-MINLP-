# Second audit (2026-10-19): the label-free pipeline at `e64170f`

## Summary and what I'd like from you

This audits `main` at `e64170f`, which covers the eight commits since the first audit (Oct 7–16):
- the label-free AC pipeline: V-net → per-unit fixing cuts and general rows learned from our own targets → one short MIQCQP → strict AC pricing;
- the vehicle study;
- the Hamming-row exploration.

It combines:
- six parallel reviews: protocol and leakage; formulation, references and pricing; deployment ablations; the differentiable layer; the vehicle; claims and statistics;
- a re-run of the four pre-registered AC final tests;
- about twenty ablation arms run with the pipeline's own deployment code.

A lot has improved since the first audit:
- Its critical bugs are fixed: the slack-bus unit, unconverged references, the wrong checkpoint, tuning on the test split, and timing.
- I found no label leakage.
- On case118, strict pricing matches an independent AC-OPF to within 3e-4%.
- Every committed model and script matches its pre-registered hash.
- `RESULTS.md` reports its nulls and negative results candidly, which made this audit much easier.

**The main finding is that the result rests on one ingredient.** That ingredient is a commitment classifier, trained on cheap self-generated targets, that fixes about 52 of 54 units (c = 0.9) before the short solve. The parts that make the method distinctive don't contribute:
- **The exact-layer parts don't help.** Fine-tuning the cuts through the exact layer, and the general rows, change nothing on AC. On the vehicle they make results worse: imitation-only cuts beat the deployed method on 8/8 vehicles.
- **The gain over the proxy comes from deployment.** When both arms get the same deployment, the AC advantage disappears and the vehicle advantage reverses. Plain Gurobi with about 5× the vehicle budget beats both.
- **Label-free matches supervised only with more data.** It needs about 4× more training instances, at an offline cost close to that of the reference labels.

On case118 and case300 the method ties the supervised proxy, at about 35× the proxy's runtime. On PGLib the proxy is closer to the reference (5× faster there).

**Could you go through the three critical items** (section 4) and either confirm them or tell me what I'm missing? After that, I'd like us to agree on the framing (section 8) and the fix plan (section 9).

**Evidence:**
- ✓ means I re-ran or re-tabulated the item myself. Everything else was verified by running, in an isolated clone, by one of the reviews, unless marked *(from reading)*.
- All re-runs used Gurobi 13.0.3, cvxpy 1.9.3, Clarabel 0.11.1, torch 2.14.1, pandapower 3.3.3 and Python 3.12. The original environment isn't recorded, and that matters (see M2).
- Ablation numbers on fresh5 and fresh6 are **post-hoc diagnostics on test draws that were already spent**, not new tests.
- The audit scripts and per-instance outputs aren't in this branch. I can add them.

## 1. What carries the result ✓

### AC

Gap is |cost − J_ref| / J_ref in %, where J_ref is the strict price of the reference commitment, exactly as in `55_final_eval.py`. A blow-up is a gap above 2%.
- case118: fresh5 + fresh6, 79 instances, post hoc.
- case300: fresh5, 40 instances.
- Deployment rho is 3e5 for case118 and 1e4 for case300; W = 10 / 30.

| Arm | What it isolates | case118: mean / median / blow-ups | case300: mean / median |
|---|---|---|---|
| **Ours as deployed** (V-net + fixings + rows) | — | **0.256 / 0.056 / 1** | **0.107 / 0.015** |
| Supervised proxy (round at 0.5 + upward repair) | the published comparator | 0.900 / 0.023 / 6 | 0.094 / 0.011 |
| Flat V, no cuts, same W | no learning | 7.24 / 5.62 / 75 | 0.327 / 0.076 |
| Flat V, no cuts, 4×W | more solver time | 5.82 / 5.57 / 79 | not run |
| V-net, no cuts | V-net alone | 1.87 / 0.23 / 7 | 0.038 / 0.014 (11 fallbacks) |
| Flat V + the same fixings | V-net, given fixings | 0.146 / 0.090 / 0 | 0.109 / 0.029 |
| V-net + fixings, rows off | the general rows | 0.557 / 0.044 / 2 (identical to ours on 73/79) | 0.083 / 0.018 |
| Imitation (warm-start) fixings, rows off | exact-layer fine-tune | 0.547 / 0.028 / 2 | not run |
| Our cut-net rounded at 0.5 + strict repair, **no MIQCQP** | the MIQCQP | 0.448 / 0.030 / 3 | **0.028** / 0.012 |
| Supervised fixings (reference labels), same deployment | label source | 0.834 / 0.024 / 3 (as hard bounds: 0.533 / 0.013 / 2) | **0.032** / 0.008 |
| Proxy's own net as hard fixings + same MIQCQP | deployment vs learning | **0.140** / 0.072 / 0 | 0.072 / 0.011 |

The supervised-fixings arm uses 118 labels; ours uses 444 own targets.

What this table shows:
- **Fixings are essential.** Without them, Gurobi fails even with 4× the budget.
- **The V-net helps typical instances a little, given fixings.** With the V-net, results are better on 18 instances and worse on 3 (sign p = 0.0015). It gives no protection against blow-ups.
- **The rows and the layer fine-tune do nothing measurable.**
- **Label source doesn't matter on case118 under the same deployment.** Supervised and imitation fixings: 23 better / 28 tie / 28 worse.
- **On case300, supervised labels are better** (p = 0.013).
- **The MIQCQP adds nothing over rounding the same network.** It ties on case118 and is worse on case300. Rounding is about 30× faster.

Full tables and paired tests are in Appendix A.

### Vehicle (fresh2 test pool)

| Arm | MILP mean (734) | Vehicles where the arm beats ours | MINLP mean (48) |
|---|---|---|---|
| **Ours as deployed** (layer cuts + rows, W = 0.01) | **0.084%** | — | **0.563%** |
| Supervised proxy (round + fixed-z re-solve) | 0.876% | 0/8 | 0.759% |
| Gurobi alone, W = 0.01 (same budget) | 0.163% | 0/8 | 0.891% |
| Gurobi alone, W = 0.05 (about 5×) | **0.035%** | **8/8** | **0.028%** |
| Imitation-only cuts (no layer, no rows) | **0.057%** | **8/8** ✓ | **0.307%** |
| Layer cuts, rows off | 0.065% | 7/8 | 0.312% |
| Supervised cut-net (reference labels), same solve | 0.088% | 5/8 | 0.495% |
| Proxy band [floor, ceil] + same solve | **0.042%** | 6/8 | 0.538% |
| Imitation trained on the 100 original profiles only | 0.137% | 1/8 | 0.527% |

- The exact-reference median time is 0.71 s (MILP) and 0.34 s (MINLP). Ours takes about 10–20 ms; Gurobi at W = 0.05 takes about 55 ms.
- The MINLP column uses Gurobi 13.0.3. Under 12.0.3: ours 0.342%, imitation-only 0.266%, proxy band 0.122%.

## 2. Reproduction ✓

All committed artifact and code hashes match the sha1s pre-registered in `RESULTS.md`. I re-ran the four AC tests with the pre-registered commands. The only change was pool size 6 instead of 11, which doesn't affect results.

| Test | Ours, reported → re-run | Proxy, reported → re-run | Verdict |
|---|---|---|---|
| case118, fresh6 | 0.107 → 0.107% (exact) | 0.807 → 0.639% | not significant in both |
| case118, fresh5 | 1.025 → **0.409%** | 1.051 → 1.022% | tie in both (p 0.39 → 0.15) |
| case300, fresh5 | 0.083 → 0.107% | 0.087 → 0.087% (exact) | tie in both |
| PGLib, fresh5 | 0.391 → 0.381% | 0.105 → 0.101% | same split verdict |
| Vehicle MILP | 0.0837 → 0.0835% | identical | 8/8 holds |
| Vehicle MINLP | 0.532 → 0.563% (Gurobi 13) / 0.342% (Gurobi 12) | — | p 0.018 → 0.048 / 9e-6 |

- **Ours on case118 fresh5 moves because of a single instance, #733.** In the published run, Gurobi's incumbent had all 54 units on (MIP gap ≈ 1, gap 23.6%). Under Gurobi 13.0.3 no incumbent is found within W, the label-free fallback runs, and it scores 0.18%.
- **case300 gains one extra fallback per seed.**
- **The proxy differences on case118 are untraced.** On case300 the proxy reproduces exactly.

## 3. Status of the first audit's items

| Item (2026-10-05) | Status | Evidence |
|---|---|---|
| Slack-bus unit removed | **Fixed** ✓ | `src/acopf_data.py:172-187`. The case118 unit (805.2 MW, ±300 Mvar) is now committed in 117/182 references (was 0/240). |
| qmin > 0 forces units on | **Fixed** | `src/acuc.py:53`, `src/socp.py:78` |
| Unconverged references; J_ref = price(u_ref) | **Fixed** | `55_final_eval.py:53-57`: converged references only, scored at the strict price of u_ref. Exclusions now raise a different issue (M1). |
| Wrong-epoch checkpoint | **Fixed / moot** | 46 saves the selected state separately from `last` (`46_unit_cuts_layer.py:123`). |
| Settings tuned on the test split | **Mostly fixed** | rho, c and epochs are now chosen label-free on validation. But test draws were reused across redesigns (M7), and `40_vnet_layer.py` still scores test unless `NOTEST=1` (l.110). |
| Cuts inactive / stale flat-V anchor | **Moot, new problem** | Cuts are now fixings. The rows are anchored at a relaxation the deployment never solves (C1). |
| Vehicle cut worse than no cut | **Changed** | New problem. Fixings beat no-cut at the same 10 ms, but lose to no-cut at about 55 ms (C2). |
| Weak proxies | **Partly fixed** | The vehicle proxy is tuned. The AC proxy still rounds and repairs upward only (C2). |
| Missing ablations | **Open** | The final tables compare ours vs the proxy only (section 1 fills the gap). |
| Statistics | **Improved** | Paired tests, failures counted as losses, bootstrap CIs. Ours is still one pipeline (M7). |
| Unsourced gradient claims | **Partly fixed** | The degeneracy and cvxpylayers outputs are committed. The V-net FD checks cover deleted models (M8). |
| Timing omissions | **Fixed** | `55_final_eval.py:132` includes the anchor, the whole MIQCQP call and strict pricing. |
| "QCAC" wording, branch limits, single period | **Open** | M9 |
| Reproducibility (versions, paths, external dirs) | **Open** | M2, Minor |

## 4. Critical (these change what the results show)

### C1. Training through the exact layer doesn't improve the deployed cuts or rows, which are deployed by rule anyway

**AC cuts:**
- **The layer gradient is the imitation gradient.** On the fixing rows, wherever the solve is exact (LU succeeds), the "layer" gradient equals the closed-form imitation gradient 2(u_k − u_own,k)/G (`cuttrain.track_soft`, `src/cuttrain.py:148`), with cosine 1.0000.
  - The gradient reaches only units that are already fixed, so fine-tuning mostly *removes* fixings.
  - It changes 1.2–3.1% of fixings, against a retraining noise floor of 0.3–0.5%.
- ✓ **The layer-trained epoch scores worse than its warm start in all three AC cases.** On the label-free validation score, the deployed epoch is worse than the BCE warm start it started from:

  | Case | Deployed epoch (`val`) | Warm start (`v_cur`) |
  |---|---|---|
  | case118 | 0.966475 | 0.966189 |
  | case300 | 1.015312 | 1.015094 |
  | PGLib | 3.588155 | 3.587858 |

  These are the `val` and `v_cur` fields of `unit_cuts_layer_*_soft_lfc.pt` (`46_unit_cuts_layer.py:96-123`). The pre-registered rule "layer-trained epochs only" (`46:97`; `dixit_vehicle/16_vehicle_final.py:16`) makes the warm start ineligible.
- **Deployed, the two are indistinguishable.** On case118 fresh5 + fresh6, imitation fixings vs layer fixings: 13 / 58 / 8, p = 0.23.

**AC general rows:**
- ✓ **The rows' gain is selection noise.** Their gain over rows-off is 2.9e-4 / 1.5e-4 / 2.6e-4 (the `sel_hist` field in `general_rows_*_lfc.pt`), against an epoch-to-epoch spread of up to 0.013. With and without rows, ours is identical on 73/79 case118 instances.
- **The rows are anchored at a point the deployment never solves.** `anchor6` (`55_final_eval.py:72,86`; `src/cuttrain.py:258`) differs from deployment in three ways:
  - it uses the γ = 1000 smoothed relaxation;
  - it uses fractional fixing rows instead of hard fixings;
  - it runs at ρ = 1e6.

  The `socp.py:127-130` comment says deployment never uses the smoothing; the anchor does. The anchor sits at L1 distance 1.16–2.95 from a deploy-like relaxation, against learned depths of 0.04–0.09.
- **Where the rows land:**

  | Case | Rows at the deployed solution | Effect |
  |---|---|---|
  | case118 | 98% slack | nil |
  | PGLib | bind on 45% | raise the MIQCQP's own cost by 1.07% on average |

**Vehicle:**
- ✓ **Imitation-only cuts beat the deployed method.** They score 0.057% (MILP) and 0.307% (MINLP), against 0.084% and 0.563% deployed. They are better on 8/8 vehicles (sign p = 0.0078), and on the MINLP p = 1e-4.
- **Validation already showed this:**
  - the layer epoch was worse than imitation in 19/24 MILP bundles;
  - the rows were worse than rows-off in 13/24 MILP bundles and 3/3 MINLP bundles.

**Consequence.** "Cuts learned through an exact differentiable layer" isn't supported for the cuts or the rows. The one layer-trained component that does work is the V-net: it helps typical instances a little given fixings, and helps a lot without them on case300.

### C2. The gain over the proxy comes from deployment, not from label-free learning

**AC case118:**
- **Where the published advantage sits.** It concentrates on the instances where the proxy needed repair (RESULTS: fresh6, 10 instances at −2.70 pp vs −0.035 pp elsewhere).
- **How the proxy fails.** It rounds at 0.5 and repairs upward only (`src/combopt_ac.py:571-593`). It under-commits by about one unit per instance. On the instances that then need repair (4–6 of 21 validation draws), the merit-order repair adds 8–14 units, which causes the blow-ups.
- **A cheap fix gives it parity.** With a label-free downward pass (at most 25 strict-priced switch-offs), the proxy reaches 0.12–0.18% on case118 validation, against ours at 0.130%. It also has a better median and discrete error, for about 15 extra pricings. Our own fallback already uses such a pass (`old_restore`); the proxy never gets one.
- ✓ **On the 79 spent case118 draws:**
  - ours − proxy is −0.64 pp [−1.31, −0.13] on the mean, but 24 / 25 / 30 instance by instance (Wilcoxon p = 0.62);
  - blow-ups are 1 vs 6;
  - the proxy's median is better (0.023% vs 0.056%).
- ✓ **The proxy's own network in our deployment beats ours.** As hard fixings it scores 0.140% with no blow-ups, against ours at 0.256%.
- ✓ **The MIQCQP adds nothing over rounding.** Our own cut-net, rounded at 0.5 with strict repair and no MIQCQP:
  - is indistinguishable from fixings + MIQCQP on case118 (13 / 49 / 17);
  - is better on case300 (0.028% vs 0.083%; −0.055 pp, CI [−0.135, −0.007]);
  - is about 30× faster.

**Vehicle:**
- ✓ **Supervised information in the same 10 ms solve does as well or better:**
  - supervised cut-net: 0.088% (ours better on 3/8 vehicles);
  - proxy band: 0.042% (beats ours on 6/8);
  - proxy as warm start: 0.062%.
- ✓ **Plain Gurobi at W = 0.05 (about 55 ms) beats ours on 8/8.** It scores 0.035% on the MILP and 0.028% on the MINLP.
- **The README omits a disclosed point.** `RESULTS.md` already says that a no-cut W = 0.01 solve reaches 0.163% against the proxy's 0.876%; the README doesn't.

### C3. Label-free reaches supervised only with about 4× more instances, and saves little offline

**Data size:**
- **At equal size, reference labels win.** Same deployment, same instance count (118): self-labels are better on 20 instances and worse on 33 (Wilcoxon p = 0.047; p = 0.014 with hard fixings).
- **The parity comes from more self-labelled instances.** 444 vs 118 instances: 25 / 48 / 6, p = 0.0009.
- **case300:** supervised fixings beat ours, 0.032% vs 0.083% (22 / 7 / 11, p = 0.013).
- **Vehicle:** imitation trained only on the original 100 profiles scores 0.137%. That is worse than the supervised cut-net (0.088%) on 6/8 vehicles.

**Offline compute.** Self-targets for 444 / 422 / 355 draws, compared with reference labels for the proxy's training split (from pool2 `secs`):

| Case | Self-targets vs reference labels |
|---|---|
| case118 | 2.6–3.4× cheaper |
| case300 | about 1.5× cheaper |
| PGLib | about 1.1× more expensive |

- These figures exclude our own training time, which isn't logged.
- Per instance, the teacher takes 3–36× the deployment time. So "nothing longer than the deployment budget" (README l.17) holds per solve, not per instance.
- On the vehicle MINLP, the teacher takes 0.30 s per instance against 0.34 s for the exact reference.

**Consequence.** The defensible claim is "no reference solver needed", not "cheaper than" or "as good as" supervised.

## 5. Major

### M1. The test sets exclude the hardest instances, and ours is never scored on them

- **What is excluded.** Exclusion means "no converged reference".
  - case118: all 17 excluded fresh5/fresh6 draws are AC-feasible. All-units-on prices strictly, and PYPOWER solves them. Their references failed at iteration 1, with no incumbent after 600 work units.
  - 16% of all case118 draws lack a converged reference, rising to 36% at load 0.84–1.01.
  - case300: 20% lack one, rising to 48% at load 1.10–1.20; 6 of the 8 fresh5 exclusions are feasible.
- **Why it matters.** These are the instances where Gurobi struggles on this MIQCQP family, which is also what the W = 10 deployment relies on.
- **What's needed.** Score both arms on them, against the best known cost or signed against each other, before claiming reliability.

### M2. Results depend on a few knife-edge instances, and the environment isn't recorded

- **Means mostly count blow-ups.** A mean over about 40 instances is mostly a count of 0–6 blow-ups, which come from two mechanisms:
  1. **The solver ignores the soft fixings.** It returns an incumbent with all 54 units on and MIP gap ≈ 1 (on #733, all 40 off-fixings were violated). Hard fixings return no incumbent within W on the same instances.
  2. **Blind merit-order repair after a pricing failure.** On #768, a 13-unit commitment was grown to 42 units (13.1%). All six proxy blow-ups are of this type.
- **Version sensitivity.** Which instances blow up depends on the solver version (#733 in section 2). The vehicle MINLP gives 0.532% as reported, 0.563% under Gurobi 13.0.3 (p = 0.048) and 0.342% under Gurobi 12.0.3.
- **Missing provenance:**
  - no requirements or lock file;
  - `acuc/runs/` (holding `final.pkl`) and the logs aren't committed, so README numbers can only be checked by re-running;
  - the vehicle test outputs and `pool_fresh/` aren't committed;
  - the chain scripts hard-code `~/.venvs/dfsm` (`acuc/clean_redo_2026-10-12.sh:8`).

### M3. Selection decisions sit below the noise floor, and case118's were made at a different ρ than deployment

- **Tiny margins.** c, the cut epoch and the rows epoch were selected on margins of 1.5–3e-4. The ρ tie tolerance is 1e-3.
- **The ρ = 1e4 validation scores don't reproduce in a fresh environment:**

  | Configuration | Logged | Fresh environment |
  |---|---|---|
  | Cuts only | 0.966475 | 0.972253 |
  | Cuts + rows | 0.966189 (val gap 0.66%) | 0.987744 (val gap 2.26%) |
  | No-cut V-net | 0.961692 | 0.977980 |

  - A 1e-7 perturbation of the V-net output moves the cuts-only score from 0.972 to 0.982.
  - The deployed ρ = 3e5 configuration does reproduce (0.961781 vs 0.961770).
- **The selection ρ wasn't the deployment ρ.** `clean_redo_2026-10-12.sh:20,22` runs 46 and 52 without `DEPLOY_RHO`. So case118's c, cut epoch and rows epoch were chosen at the V-file ρ = 1e4, while deployment uses 3e5. `RESULTS.md` discloses only the V-net ρ mismatch.
- **c = 0.9 is the edge of the grid** (`46_unit_cuts_layer.py:25`) in all three cases. At c = 0.9 the cut-net fixes 52.1/54 (case118), 64.9/69 (case300) and 51.0/54 (PGLib) units, so the MIQCQP decides only 2–4 commitments.
- **The ρ rule is degenerate for case300 and PGLib.** All five grid values fall within 1e-3, so the rule simply returns the smallest.

### M4. case300's voltage band was widened without disclosure ✓

- **What changed.** `src/pipeline.py:22-25` sets 0.90/1.10, with a comment saying that pandapower ships 0/2 placeholders and that 0.90/1.10 is case300's "own MATPOWER range". In fact, pandapower's case300 has 0.94/1.06 on every bus, as does MATPOWER's.
- **Why it matters.** Under 0.94/1.06, 9 of 10 sampled reference commitments become infeasible: repair adds 16–28 units, at +1.4 to +4.6% cost. All-units-on is still feasible on 9/10, so the case isn't infeasible at its real limits, contrary to the comment. The widening materially changes the case300 UC problem.

### M5. PGLib "cheaper than the reference" is real in direction but overstated, and the reference is weak

- **Independent re-pricing.** I re-priced with an independent AC-OPF: standalone PYPOWER, MATPOWER formulation, same data, with solutions feasible in our model.

  | Comparison | Independent AC-OPF | Strict pricing |
  |---|---|---|
  | Ours − proxy, signed | −0.224 pp (p = 5e-5, 35 / 7) | −0.312 pp |
  | Ours − reference | −0.21% (cheaper on 31/42) | −0.29% |

- **Why strict pricing overstates it.** `price_strict` (`src/combopt_ac.py:571-584`) accepts the first iterate that passes the 1e-3 pu physics check. On PGLib that iterate is above the converged cost for every commitment: reference +0.136%, ours +0.057%, proxy +0.145%. Iterating to residual < 1e-6 lands on PYPOWER's value (+0.0001%).
- **Where the saving comes from.** Ours commits 0.65 fewer real units (15.95 vs 16.60). That saves 0.70% of synthetic no-load cost and adds 0.58% of dispatch cost, so the result depends on the synthetic 15% no-load cost.
- **The reference is a weak heuristic.** Its QCAC cost is 3.62% (0.50–6.24%) above the converged price of its own commitment. It stops at slack ≤ 1e-3 with no physics check. Its last-iteration MIP gap has median 0.47% and maximum 19.5%. So |gap| to it isn't an accuracy measure.
- **The discrete error is mostly free decisions.** 35 of the 54 PGLib units are synchronous condensers (pmax = 0, no-load cost ≈ 1e-7), and 55–57% of discrete mismatches are condensers.
- **The band is narrower than stated.** The 0.85–1.05 band is effectively 0.85–0.99, because 93% of draws above 1.00 are infeasible.

### M6. case300 can't discriminate between methods

- **Reference noise is as big as the differences.**
  - Methods beat the case300 reference on 10–16 of 40 instances, by up to 0.31%.
  - Pre-polish MIP gaps reach 0.77%, and the polish improved none of the 40 commitments.
  - Reference noise of about 0.1–0.3% on at least a quarter of instances is comparable to the means being compared (0.08% vs 0.09%).
- **case118 references are good, by contrast.** Their QCAC cost equals the strict price to within 0.06%, and methods beat them on only 2–4 of about 40 instances, by at most 0.044%.

### M7. Statistics and claims

- **"3 seeds" (README l.26) is misleading.** On AC, ours is one deterministic pipeline (`vseed0`; cuts and rows trained with seed 0). The seeds retrain only the proxy, so ours shows ±0.000 and the paired tests compare one pipeline against a 3-seed proxy average. This needs at least 3 independently trained pipelines.
- **"Tie" means underpowered, not equivalent.** The minimum detectable difference (80% power) is:

  | Case | Minimum detectable difference | For scale |
  |---|---|---|
  | case300 | ≈ 0.15 pp | ours averages 0.08% |
  | PGLib | 0.16 pp | |
  | case118 fresh6 | 1.17 pp | paired SD 2.4–2.8 pp, driven by a few blow-ups |

  Use an equivalence test (TOST) with a pre-set margin before claiming parity.
- **Test draws were reused across method versions.** On case118:

  | Draw | Use | Result |
  |---|---|---|
  | te | — | — |
  | fresh | designing the V | — |
  | fresh2 | development | — |
  | fresh3 | long-solve teacher | ours better |
  | fresh4 | `_lfx` | ours better, p = 0.02 |
  | fresh5 | clean | tie |
  | fresh6 | clean + a retry designed from fresh5's failures | not significant |

  The redesigns removed label dependence and made results worse, which limits the p-hacking concern. But the retry and the ρ tie rule were fixed after looking at test results.
- **The pre-registrations can't be verified from git.**
  - `RESULTS.md` first appears in 0365beb (2026-10-14), and each pre-registration lands in the same commit as its results.
  - One header (~13:10, `RESULTS.md` l.503) postdates its own run (13:05, l.512).
  - The hashes match, so models weren't swapped. Committing or timestamping pre-registrations before running would settle this.
- **The README omits two things `RESULTS.md` discloses.** Most of the vehicle MILP win comes from running the 10 ms solve at all. And the pre-registration said to report both vehicle pools, because the method changed after the pool_fresh result had been seen. On pool_fresh, the earlier method version's MINLP was 0.932% vs the proxy's 0.743%, not significant.

### M8. The evidence for the layer is partly stale or scale-blind

- **The committed V-net FD checks cover deleted models.**
  - `gradient_checks/09_envelope_vgrad_fd_final.py:18,29` loads `vnet_layer_<case>_env.pt` and `teacher_<case>_w120.json`, both deleted in e64170f. The committed `vgrad_fd_*.txt` files therefore don't cover the deployed `_env_nc` V-nets, and the script can't run on main.
  - Re-run on `_env_nc` (3 training instances, h = 1e-3): cosine ≥ 0.993 on case118 and ≥ 0.9999 on case300 and PGLib, with relative error 1–12%. That passes, including PGLib, whose committed "FAILED" was for the old V-nets.
- **The "kinked value function" explanation isn't established.** At the trained V, FD becomes self-inconsistent at h ≤ 1e-4, and Clarabel returns AlmostSolved at tight tolerance. Solver precision explains this symptom as well as the kinked value function `RESULTS.md` proposes.
- **The gates ignore scale.** They use the Pearson correlation of 8 directional derivatives; one instance has correlation 0.9995 with 34% relative error. Use cosine plus relative error instead.
- **Many gradients come from a regularised fallback.**
  - Rows (`track_gen`) fall back to the regularised KKT solve on 75% / 100% / 38% of solves. Where FD is self-consistent, cosine is 0.98–1.00 but relative error is 18–36%.
  - Cuts fall back on 60% / 85% / 10% of solves, yet still reach cosine 0.92–0.9995 against FD at h = 1e-2.
  - `_solve` (`src/conelayer.py:335`) returns the regularised solution with no accuracy check.
- **Some claims have no committed output.** The cut/row FD gates, "13/13 corr ≥ 0.9999" and "57–60% inexact" have none.

### M9. Model and problem scope (mostly unchanged from the first audit)

- **It isn't QCAC.** `acuc.build` and `Socp` use a two-sided first-order linearisation |c − lin_V0(v)| ≤ ξ penalised by ρ, plus a rotated-SOC strengthening. The convex QCAC side, cii ≥ |v|², is absent, and `qcac.py`'s true constraints are unused. The README still says "QCAC convexification".
- **Line limits are vacuous on case118 and case300.** Ratings are 99 pu, and priced loadings stay at or below 5% and 15% of rating. Only PGLib has binding lines.
- **The UC is simplified.** It is single-period, the reserve is capacity-only, the no-load cost is synthetic (15% of the cost at pmax) and MATPOWER's c0 is dropped.
- **The instance families are effectively one-dimensional.**
  - Loads are one global scale times i.i.d. ±5% P / ±10% Q per-bus noise; total load varies with an SD of only 0.2–0.3%.
  - The vehicle family has one parameter u, and the nearest training u is a median 0.002 away.

  A 1-D interpolant is easy to learn, so showing generalization needs harder families.

## 6. Minor

- **Layer and V-net:**
  - `ConeLayer.forward` accepts AlmostSolved (`src/conelayer.py:194`).
  - The V-net output is unbounded (`src/vnetio.py:16-19`): on PGLib, |V| ranges over 0.65–1.43.
  - On case300 and PGLib, the V-net learns a low-slack linearisation point, not physical voltages. Its complex error is 0.14 against 0.16 for flat V, while a mean-reference predictor scores 0.04.
- **Label generation doesn't use the layer.** Replacing the layer-ranked flips in `combopt_ac.improve` with a random ranking gives the same targets on 15/15 case118 draws.
- **Proxy details:**
  - The PGLib proxy trains on 95 labels plus a 23-label hold-out (`55_final_eval.py:76`), not the README's 118.
  - The proxy trains on unpolished pool2 references (MIP gap 1e-3) but is scored against polished fresh references.
- **Thin validation:**
  - Validation has only 3/24 case118 draws in the sparse regime (load 0.78–0.88), where failures happen.
  - PGLib selection effectively uses about 17 draws, because 7 always fail and are scored 10.
- **Dead or stale code:**
  - `src/diffhv.py` (edited by the first audit's fix but never imported) and `src/cutnet_hv.py` are dead code.
  - `src/poolload.py:4-5` refers to external directories (`qcac_clean`, `difflayer_work`).
  - The README's figure note is stale; there are no AC figures.
- **Vehicle gates.** The vehicle FD gates (13, 15) cover vehicle 4 and the MINLP at the old W = 1 stage-1 cuts, not the final bundles. Vehicles 3 and 5 are excluded, which is documented only in git history.
- **Checks:**
  - `gradient_checks/05_vs_cvxpylayers.py:117-118` locates u in the canonical vector by matching values, which the layer's own docstring warns against; `var_slices` already exists.
  - `_phys` doesn't check raw voltage magnitudes. Measured violations are ≤ 1e-4 pu, so this is negligible.

## 7. What holds up

- **No leakage.**
  - The full clean chain was traced: 40 (nocuts) → 60 → 46 → 52 → 61/65 → 55.
  - References appear only in diagnostic prints and post-selection reports. The chain scripts pass the label-free targets explicitly, so the removed long-solve default paths are never used.
  - The committed cut-nets agree 98.6 / 97.2 / 98.3% with a BCE fit on the committed own targets, against 94.6 / 93.2 / 86.2% with a fit on reference labels.
- **The data is clean.**
  - The blocks are disjoint, with no exact or near duplicates.
  - The extra draws match the test distribution (KS p = 0.11 / 0.99 / 0.11).
  - Own targets cover only the train and extra draws.
  - The proxy labels were made after the slack-bus fix and use converged references only.
- **Scoring is sound.** J_ref is the strict price of u_ref, exclusions are identical across arms, failures count as losses, and `55_final_eval.py` has no silent drops.
- **The data fixes are correct.**
  - The slack-bus unit matches the source data, and the ppc generator-order assumption holds in pandapower 3.3.3.
  - The Q-box fix frees case300 units 5, 44 and 45.
  - The PGLib grid is identical to its `.m` file.
- **Pricing is physically sound where it can be checked.**
  - case118 strict prices match PYPOWER to within 3e-4%.
  - Feasibility was verified with an independently built Ybus.
  - case300 prices agree to within 0.07% wherever PYPOWER converges.
- **The layer math is correct.**
  - The KKT linearisation, cone-projection derivatives and reverse-mode chain pass toy checks.
  - The envelope V-gradient passes FD at the deployed V-nets, and the old VJP it replaced really was wrong.
  - The degeneracy diagnosis reproduces: rank 1804 of 2183 active rows, condition number 1.3e21.
- **The V-net learns something real on case118.**
  - Without labels, it learns load-dependent voltage structure: complex error 0.05 against 0.14 for flat V, angle correlation 0.77.
  - Given fixings, it gives a small typical gain.
- **Fixings make a tight-budget MIQCQP viable.** Gurobi alone fails even at 4×W.
- **The vehicle study is sound where checked.**
  - The MILP final test reproduces exactly, and the test pools are disjoint.
  - At 10 ms, fixings beat plain Gurobi on 8/8 vehicles.
  - The excluded instances don't hide a loss.
- **The deployed case118 configuration reproduces on validation** (0.961781 vs 0.961770) and is invariant to Gurobi's Seed parameter.

## 8. Selling points, and claims to avoid

**What I think we can claim:**
1. **Reference-free training.**
   - A pipeline that needs no converged reference solutions reaches parity with a supervised proxy on case118 and case300.
   - This is relevant because the reference solver fails on 16–20% of draws, and those are the hard ones.
2. **Learned fixings make a tight-budget MIQCQP work where Gurobi alone fails.**
   - The pattern is close to Xavier, Qiu & Ahmed (2021) and Neural Diving.
   - What's new is the label source.
3. **Method contributions:**
   - exact implicit differentiation through degenerate conic programs, verified and better than cvxpylayers;
   - label-free learning of the linearisation point through envelope gradients. I found no prior work on this, it is real on case118, and it gives a small gain on top of fixings.
4. **A rigorous evaluation protocol and useful negative results:**
   - training fixing cuts through the layer reduces to imitation;
   - the quality of self-training labels degrades from round to round;
   - 10 ms of Gurobi beats NN proxies on the Dixit MILP.

**Claims to avoid:**
- Beating the proxy on AC.
- Gains from the differentiable layer for the cuts or rows.
- "3 seeds" for our method.
- "Nothing longer than deployment".
- Cheaper than the reference on PGLib.
- The vehicle MILP win being due to learned cuts.
- Label-free being cheaper offline.
- Speed-ups over the reference without also saying that the proxy is 5–40× faster still.

## 9. Proposed fix plan (in priority order)

1. **Agree on the framing:** self-supervised commitment prediction plus a short solve. Report the layer-trained cuts and rows as a negative result, or drop them. The "differentiable layer" claim then rests on the V-net.
2. **Re-evaluate with matched deployments.**
   - Add these arms to the final tables:
     - our net rounded, with no MIQCQP;
     - the proxy with a downward pass;
     - supervised fixings in our deployment;
     - flat V + fixings;
     - Gurobi with 2–5× the budget;
     - imitation-only cuts.
   - Train at least 3 pipelines.
   - Score once on a fresh draw, with the environment pinned.
3. **Fix the evaluator and the test sets:**
   - price to convergence (residual ≤ 1e-6, or until the cost stops moving);
   - score the excluded hard instances;
   - report medians and blow-up counts beside means;
   - use TOST before claiming a tie;
   - restore case300's 0.94/1.06 limits, or disclose and justify the change;
   - report PGLib discrete error without the condensers.
4. **Fix the pipeline:**
   - select c and epochs at the deployment ρ;
   - extend the c grid below 0.9;
   - if the rows stay, anchor them at the deployed relaxation (unsmoothed, hard fixings, deployment ρ);
   - bound the V-net output;
   - make the soft-fixing and repair path robust against the two blow-up mechanisms in M2.
5. **Reproducibility:**
   - add a requirements or lock file;
   - commit `final.pkl` and the logs;
   - re-run the FD checks on the deployed models and commit them, using cosine plus relative error;
   - fix the paths in `gradient_checks/09`;
   - remove the dead code.
6. **Harder instances:** multi-period UC with min up/down and ramping, multi-dimensional load variation, and real line limits.
7. **Wording:** fix "3 seeds", "nothing longer than deployment" and "QCAC" in the README and `RESULTS.md`; add the vehicle short-solve disclosure, both vehicle pools, and the ρ mismatch in selection.

## Appendix A. Full AC ablation tables and paired tests ✓

**Arms:**
- A0: ours (V-net + fixings + rows).
- A1: flat V, no cuts. A9f: the same at 4×W.
- A2: V-net, no cuts. A9v: the same at 4×W.
- A3a: flat V + fixings. A3b: flat V + fixings + rows re-anchored.
- A4: V-net + fixings, rows off.
- A5: imitation (warm-start) fixings.
- A6: supervised fixings (reference labels, 118 instances). A6b: self-label fixings on the same 118 instances.
- A7: proxy probabilities as fixings.
- A8: proxy.
- A10: proxy + `improve`.
- A11: the label generator used as the method.
- A12: our cut-net rounded at 0.5 + strict repair, no MIQCQP. A12i: the same with the imitation net.
- Suffix h: fixings as hard variable bounds.

Each cell is mean / median / blow-ups (> 2%), in %.

| Arm | case118 val (21, in-sample) | case118 fresh5+6 (79, post hoc) | case300 fresh5 (40) |
|---|---|---|---|
| A0 | 0.130 / 0.052 / 0 | 0.256 / 0.056 / 1 | 0.107 / 0.015 / 0 |
| A4 | 0.110 / 0.052 / 0 | 0.557 / 0.044 / 2 | 0.083 / 0.018 / 0 |
| A3a | 0.149 / 0.152 / 0 | 0.146 / 0.090 / 0 | 0.109 / 0.029 / 0 |
| A3b | 0.154 / 0.152 / 0 | 0.144 / 0.089 / 0 | — |
| A5 | 0.775 / 0.063 / 1 | 0.547 / 0.028 / 2 | — |
| A6 | 2.346 / 0.140 / 3 | 0.834 / 0.024 / 3 | 0.032 / 0.008 / 0 |
| A6h | 0.779 / 0.047 / 1 | 0.533 / 0.013 / 2 | 0.030 / 0.007 / 0 |
| A6b | 0.713 / 0.061 / 1 | 0.601 / 0.056 / 3 | — |
| A7 | 2.182 / 0.192 / 2 | 0.605 / 0.081 / 2 | 0.098 / 0.006 / 0 |
| A7h | 1.219 / 0.176 / 1 | 0.140 / 0.072 / 0 | 0.072 / 0.011 / 0 |
| A8 | 1.330 / 0.026 / 2 | 0.900 / 0.023 / 6 | 0.094 / 0.011 / 0 |
| A10 | 1.324 / 0.026 / 2 | 0.884 / 0.017 / 6 | — |
| A11 | 0.100 / 0.039 / 0 | 0.195 / 0.081 / 1 | — |
| A12 | 0.187 / 0.136 / 0 | 0.448 / 0.030 / 3 | 0.028 / 0.012 / 0 |
| A12i | 0.347 / 0.180 / 1 | 0.380 / 0.025 / 2 | — |
| A2 | 6.299 / 0.619 / 8 | 1.866 / 0.225 / 7 | 0.038 / 0.014 / 0 |
| A9v | 0.956 / 0.824 / 1 | 1.171 / 0.856 / 8 | — |
| A1 | 4.762 / 4.673 / 19 | 7.235 / 5.624 / 75 | 0.327 / 0.076 / 0 |
| A9f | 5.487 / 5.104 / 21 | 5.820 / 5.569 / 79 | — |

**Paired tests.** Each line gives the difference in mean |gap| (first arm − second arm), its 95% bootstrap CI, the counts as first arm better / tie / worse (tie ≤ 1e-3 pp), and the Wilcoxon p-value.

case118, fresh5 + fresh6 (n = 79):
- A0 − A8: −0.644 pp [−1.312, −0.130], 24 / 25 / 30, p = 0.62
- A3a − A0: −0.110 pp [−0.460, +0.079], 3 / 58 / 18, p = 0.0057 (sign test p = 0.0015). The V-net is better on 18, worse on 3.
- A3a − A4: −0.411 pp [−1.167, +0.062], 4 / 59 / 16, p = 0.040
- A5 − A4: −0.010 pp [−0.034, +0.016], 13 / 58 / 8, p = 0.23
- A0 − A4: −0.301 pp [−0.899, +0.005], 3 / 73 / 3, p = 0.46
- A6 − A5: +0.286 pp [−0.609, +1.450], 23 / 28 / 28, p = 0.98
- A6b − A6: −0.232 pp [−1.261, +0.536], 20 / 26 / 33, p = 0.047
- A5 − A6b: −0.054 pp [−0.669, +0.695], 25 / 48 / 6, p = 0.0068 (sign test p = 0.0009)
- A12 − A4: −0.109 pp [−0.803, +0.380], 13 / 49 / 17, p = 0.28
- A7h − A8: −0.760 pp [−1.532, −0.129], 23 / 21 / 35, p = 0.21
- A11 − A0: −0.061 pp [−0.444, +0.166], 23 / 23 / 33, p = 0.090
- A2 − A0: +1.610 pp [+0.507, +2.939], 15 / 10 / 54, p = 4.7e-7

case300, fresh5 (n = 40):
- A0 − A8: +0.014 pp [−0.106, +0.136], 14 / 8 / 18, p = 0.82
- A6 − A4: −0.051 pp [−0.131, −0.003], 22 / 7 / 11, p = 0.013
- A12 − A4: −0.055 pp [−0.135, −0.007], 16 / 13 / 11, p = 0.068
- A12 − A8: −0.066 pp [−0.163, −0.000], 18 / 9 / 13, p = 0.24
- A2 − A1: −0.289 pp [−0.459, −0.137], 29 / 4 / 7, p = 1.0e-4
- A3a − A4: +0.026 pp [+0.005, +0.056], 5 / 20 / 15, p = 0.012

**Fixing diagnostics for A0:**
- case118 fixes 12–14 units on and 38–40 off, leaving about 2 of 54 free. 39–62% of instances have at least one fixing that disagrees with the reference.
- case300 leaves about 4 of 69 free.
- Median MIP gap at the end of W is about 6e-4 with fixings, against 3e-2 to 1.4e-1 without them (14–54% of instances above 0.05).

## Appendix B. How this was checked

- **Re-runs.** The pre-registered commands from `RESULTS.md` (e.g. `DEPLOY_RHO=3e5 RETRY_RHO=1e4 RETRY_TAU=0.05 python acuc/55_final_eval.py case118 0,1,2 fresh6 _lfc _nc vseed0`) were run in an isolated clone with pool size 6. The ablation harness reuses `55_final_eval.py`'s deployment code unchanged; its A0 and A8 match the re-runs instance for instance (maximum difference 0.0).
- **Vehicle.** The deployed bundles were re-run with `VEH_LF=1 python dixit_vehicle/16_vehicle_final.py test`. The imitation warm starts were rebuilt with each bundle's own c and seed, and match the stored validation scores on 22/24 MILP and 3/3 MINLP bundles. Baselines were run on the same fresh2 pool and compared both vehicle-clustered and instance-paired.
- **Physics.** I used standalone PYPOWER (MATPOWER formulation) at fixed commitment with the same data, costs and limits. I also checked power-flow residuals with an independently built Ybus.
- **Timing.** Wall-clock times were measured while other jobs ran, so they are relative only. Gurobi runs use Threads=1 with WorkLimit and are deterministic.
- **Repository.** It was read-only throughout; this file is the only addition.

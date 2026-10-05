# TODO

Updated after the final-release restructure (`85a9bba`). The earlier TODO
targeted `11_vcard.py`/`VCardNet`/`card_rows`, which no longer exist.

## Consistency issues found in the release (check before resubmitting)

- [ ] **Which epoch is evaluated on case300?** `02_train.py` saves the last
  epoch as `ck["net"]` and the selected epoch as `ck["best"]`, but
  `04_eval_case*.py`, `03_threshold_fit.py` and `05_timing.py` all load
  `ck["net"]`. Every committed case300 training JSON has `best_epoch` 0–4 of
  26, while `paper/main.tex` (l.251, l.347) says the selected epoch is used and
  that the selected case300 models are "close to their initialization". Either
  the eval is scoring last-epoch nets, or the checkpoints were rewritten. Load a
  checkpoint and compare `ck["ep"]` and `ck["best"][2]`, then fix the code or
  the paper. For case118 the best epoch is 29/30, so the difference is
  negligible there.
- [ ] **Selection criterion is inconsistent.** The AC-UC trainer picks `best` by
  *surrogate* validation loss, but its own comment says surrogate-based
  selection picked 4.6× worse models on case300. It also computes `val_deploy`
  (label-free deployed cost) *before* loading `best`, so that number describes
  the last epoch. The vehicle trainer selects on deployed validation cost.
  Use one rule for both problems.
- [ ] **Regularisation constant mismatch.** The paper (l.362) says ε=1e-8 for
  layer 1 and 1e-10 for layer 2, which are `DiffDeploy`'s defaults. But
  `02_train.py` passes `REG1 = REG2 = 1e-10`. The paper also reports that
  sharing one value cut the layer-2 cosine to 0.42.
- [ ] case118 training JSONs predate the `--dthr` flag (absent from their
  `args`), so they were made with an older trainer than the one committed.
  Confirm that the committed trainer reproduces them.
- [ ] `HVRelax` documents the cut as priced/soft, but the vehicle trainer and
  `DiffHV` call it with `cut_cap=0.0` (hard). `Socp` is hard by default too.
  Make the docstrings and the paper's description match.
- [ ] README's reproduce section omits pool generation (`01_make_pool.py` and
  its `GATE_*` env vars). Add it, or state that pools are shipped and not meant
  to be rebuilt.

## Results to address

- [ ] **case118: the supervised NN proxy beats the method** (|gap| 0.104 vs
  0.176, median 0.003 vs 0.058, discrete 2.4% vs 4.3%). case300 is close
  (0.061 vs 0.075). The clear win is the vehicle problem (0.12 vs 1.82). The
  paper should state this plainly, or case118 needs another look: training
  signal, threshold, restoration order.
- [ ] case300 training signal: the validation loss bottoms out within 5 epochs,
  so the soft-rounded surrogate is not a faithful proxy for deployed cost on
  case300. Candidates: anneal τ, train with the merit-order top-up inside
  `DiffDeploy`, or add a differentiable stand-in for downward repair.

## Generalising the cut (re-scoped from the old TODO)

Already done in the release:
- A second, non-UC problem class with **general integers** (hybrid vehicle,
  `z_t ∈ {0..3}`), sharing the same `ConeLayer` and the same depth anchoring.
- The cut is now a general learned linear form, not the all-ones cardinality
  band: two categories × K rows (`α'x_cont + a'x_int ≤ b`, `a2'x_int ≤ b2`).
  The `n_min` anchor was replaced by a generic one, `b = ⟨A, w_rlx⟩ − depth`.
- Integer rounding (`round_z`, a soft round that sums S sigmoids) and
  restoration generalised to {0..S}.

Still open:
- [ ] **Output size.** `CutNetK` emits `3·K·n_gen + 2K` cut outputs (≈650 on
  case118, more on case300), plus 2·n_bus for V. The old lesson was that large
  free-form cut outputs fail. Ablate against a small fixed family that learns
  only depths: all-ones, group cardinalities, or rows of the problem's own
  integer constraints.
- [ ] **Validity framing.** Anchoring the cut to be violated by `depth` at the
  relaxation point makes it a learned heuristic tightening, not a valid
  inequality. The paper calls the rows "parametric cutting planes" that keep
  every original constraint (l.33, l.55). Say explicitly that they can cut
  off the integer optimum, and report how often they do on test.
- [ ] **Restoration is still problem-specific.** AC-UC uses reserve top-up plus
  up/down repair. The vehicle uses a Gurobi L1 projection (`qz_restore`),
  which is itself a MIQCP solve. Either generalise one restoration, or report
  the vehicle timing with the projection's cost and dependence on Gurobi made
  explicit.
- [ ] Collapse the four restoration copies (`pipeline.deploy`, `restore.py`,
  `04_eval_case300.restore`, `src_hv.deploy`) into one parameterised function.
- [ ] A third problem class, ideally one with a nonconvex relaxation other
  than QCAC, before claiming the framework is fully generic.

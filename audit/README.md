# Audit scripts

Evidence for the critical and statistics findings in `../AUDIT.md`. Every script only
**reads** the repo; anything it writes goes to `audit/out/`. Run from the repo root:

```bash
export OMP_NUM_THREADS=1
python audit/c2_checkpoint_epochs.py
```

| Finding | Script | Runtime | What you should see |
|---|---|---|---|
| 1. Vehicle cut worse than no cut; weak proxy | `c1_vehicle_baselines.py` | 2–5 min, needs Gurobi | no-cut 0.0880%, constant 1.51%, ridge 0.43%, paper proxy 1.71% / 1.53%, method 0.1234% |
| 2. case300 scored with last-epoch weights | `c2_checkpoint_epochs.py` | seconds | case300 selected epoch 0 / 4 / 0 / 1 vs last 25; case118 identical |
| 2. Re-score with the selected weights | `c2_c3_case300_rescore.py --weights best` | ~30 s per instance per seed | mean over seeds: \|gap\| 0.066, signed +0.008 |
| 2. Reproduce the published rows | `c2_c3_case300_rescore.py --weights net --seeds 0 --positions 0,3` | ~1 min | +0.0770%, −0.4069% (as in `c300pipe_s0`) |
| 3. Untrained network, case300 | `c2_c3_case300_rescore.py --weights init --seeds 0,1` | ~30 s per instance per seed | \|gap\| 0.0515 (s0), 0.0653 (s1) |
| 3. Untrained network, case118 (contrast) | `c3_case118_untrained.py --weights init --seed 0` | ~10 min | \|gap\| 2.40% (fitted threshold), 11.5% with `--thr 0.5` |
| 4. Non-converged references | `c4_reference_convergence.py` | seconds | 41/268 case300; converged-only \|gap\| 0.0249 (ours) / 0.0318 (proxy) |
| 5. Cut rows slack at the predicted V | `c5_cut_activity.py case118 3` | 1–2 min | slack 5–14 at predicted V, about 0 at flat V |
| 6. Slack-bus generator | `c6_slack_generator.py` | seconds | pmax 14.14 pu vs 805.2 MW in pandapower; committed 0/240 |
| Statistics | `stats_paired.py` | seconds | case300 paired diff −0.014 pp, CI [−0.054, +0.017], p = 0.30 |

`_common.py` holds shared helpers: pool loading, the split, network loading and the
anchored-cut prediction. Each mirrors the corresponding code in `acuc/04_eval_case*.py`.

## Precomputed outputs (`out/precomputed/`)

The case300 re-scoring takes several hours for all seeds, so the audit's outputs are included:

- `case300_best_s{0..3}.json` — selected-epoch weights, published restoration (merit, 0.80, NDOWN 40)
- `case300_init_s{0,1}.json` — untrained network, same restoration
- `case118_init_{fitted,0.5}_s0.json` — untrained case118 network, with the fitted threshold or 0.5

Each file maps a **test position** (index into `te = perm[:48]`) to
`[gap %, discrete %, continuous %, n_on]` (case118: `[gap %, discrete %]`). The case300 files
omit position 39 (pool index 238): it fails restoration for every seed and both arms in the
published results, too. Spot-checked positions re-run with the scripts reproduce these rows
exactly.

## Environment used

Python 3.12, torch 2.14.1, cvxpy 1.9.3, clarabel 0.11.1, gurobipy 13.0.3, numpy 2.4.6,
scipy 1.18.1, pandapower 3.5.5. These are close to the versions in the main README. The pip
`gurobipy` size-limited licence is enough for the vehicle scripts. The AC-UC scripts here
don't call Gurobi.

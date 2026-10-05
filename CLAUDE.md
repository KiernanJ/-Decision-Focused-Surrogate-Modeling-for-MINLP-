# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code: **self-supervised learned cutting planes for MINLP, trained through exact differentiable conic layers.** A network maps instance parameters to parametric cut rows `A w ≤ b` that are added to a convex relaxation. On nonconvex AC unit commitment it also predicts the QCAC **linearisation point** `(Vre, Vim)`. Training backpropagates the deployed cost through two chained cone solves (relaxation with cuts → soft rounding → re-solve at the rounded integers). No optimal-solution labels are used.

There are three case studies: AC-UC on IEEE **case118** and **case300**, and a convex **hybrid-vehicle** MINLP with general integers (`T=30`, `z_t ∈ {0..3}`). `README.md` holds the layout, the reproduce commands and the caveats. `paper/main.tex` is the manuscript. Keep both in sync when results change.

History: commit `85a9bba` ("Final release") replaced the earlier tree, which had numbered top-level scripts `00_refs.py`…`31_*.py`, `11_vcard.py`/`VCardNet`, zeroth-order smoothing and `_dev/`. That commit says the old code is kept under tag `pre-final-release`. Fetch tags to see it. Code comments still cite old filenames (`21_tiered_train.py`, `42_proxy_sup.py`, `qcac_clean/`, `difflayer_work/`); those are not in this tree.

## Environment and commands

There is no package config, test suite, linter or committed environment. The versions used are listed at the end of `README.md` (torch, cvxpy 1.9.2, clarabel 0.11.1, Gurobi 13.0.3, numpy 2.4.6, scipy 1.18.1, pandapower). `gurobipy` is needed for the AC-UC references, the vehicle references and the vehicle restoration (`qz_restore` in `vehicle/02_train.py`). `cvxpylayers` is needed only by `gradient_checks/05_vs_cvxpylayers.py`.

**Run each script from its own folder.** Paths such as `results/wide`, `checkpoints/` and `results/pool/` are cwd-relative. Every script does `sys.path.insert(0, <script dir>/../src)` and imports modules by bare name. Scripts set `QCAC_CASE` from `--case` before importing `pipeline`, so the env var only matters when you import `src/` yourself. Jobs run single-threaded (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), and parallelism means separate OS processes, one per seed or shard.

```bash
cd acuc
python 02_train.py --case case118 --epochs 30 --lr 2e-3 --seed 0 --ckpt checkpoints/case118_s0.pt
python 02_train.py --case case300 --epochs 26 --lr 5e-4 --down 25 --dthr 0.8 --seed 0 --ckpt checkpoints/case300_s0.pt
python 03_threshold_fit.py --case case118 --seed 0 --shard 0     # -> results/thr_lf/sh_*.json
python 04_eval_case118.py --case case118 --seed 0 --down 25      # -> results/thrlf_eval_*.json
python 04_eval_case300.py --case case300 --seed 0 --order merit --thr 0.80 --downs 40
cd ../vehicle && python 02_train.py --seed 0
cd .. && python figures/make_figures.py && python paper/make_numbers.py   # summary.json -> numbers.tex
```

The full command list is in `README.md`. The authoritative record of the flags behind a result is the `args` block saved in each result JSON. For example, `acuc/results/training/train_case300_s*.json` record `ckpt: ckptD/c300_dt_s*.pt`; the files were renamed after the runs. Trainer output names (`results/{pool}_{case}_k{K}_s{seed}{tag}.json`) do not match the committed names (`results/training/train_*.json`).

Reference pools:
- AC-UC: `acuc/01_make_pool.py` is configured entirely by `GATE_*` env vars (`GATE_ARM/LO/HI/POUT/SEED/N/I0/I1/OUT/OFFSET`) and writes `results/wide/{case}_W_*.pkl`.
- Vehicle: `vehicle/01_make_pool.py --i0 --i1 --out results/pool/pool_K.json`.

Both are already committed. Rebuilding them is slow (QCAC iterative is roughly 80 s per instance on case118 and 350 s on case300).

## Architecture

**`src/` layers:**
- `acopf_data.py` — loads MATPOWER data from pandapower's `net._ppc` into a `Grid` (per-unit). `python src/acopf_data.py` validates it against an AC power flow.
- `qcac.py` — QCAC formulation helpers. `pipeline` imports `no_load_cost` from here.
- `acuc.py` — the single Gurobi model builder `build(exact=True|False)`, plus `solve_global`, `solve_qcac_iterative` (the AC-UC reference) and `verify`. Keep everything going through `build`; parallel model copies drifted in the past.
- `socp.py` — `Socp`: the continuous QCAC relaxation, compiled once as a DPP cvxpy problem and solved by Clarabel. `solve()` is the relaxation. `price()` fixes `u` and dispatches. Cuts are always `n_cuts` rows over `w = [pg; u]`. `A=0, b=1` disables them without changing the compiled structure. `cut_cap=0` (the default) makes cuts **hard**, with a soft fallback if the solve fails.
- `conelayer.py` (+ `conedesc.py`, `lu_server.py`) — `ConeLayer`: generic implicit differentiation of any DPP cvxpy problem. It runs a Clarabel forward pass and a reverse pass that solves the KKT system by regularised sparse LU (`K + εI`). The KKT systems are LICQ-degenerate, which is why there is a regulariser. The LU runs in a **subprocess** (`lu_server.py`) because SuperLU aborts the process after many factorisations. The helper gets restarted when it dies, and fork/spawn pools deadlocked or re-ran `__main__`.
- `framework.py` — `DiffDeploy`, the AC-UC differentiable deployment loss. Layer 1 is the relaxation with cuts at V. Then comes the soft round `z = zc + (1−2zc)·σ((uf−thr)/τ)`, with the reserve threshold found by bisection. Layer 2 prices at `z`. The loss is the cost divided by the uncut relaxation cost.
- `cutnet_k.py` — `CutNetK` and `rows_torch`: V head plus two cut categories with K rows each. The general category is `α'pg + a'u ≤ b` and the integer-only category is `a2'u ≤ b2`. The network emits a depth `d`, and `b = ⟨A, w_rlx⟩ − d` is built in torch so the anchor's gradient reaches `A`.
- `pipeline.py` — `grid()` with per-case `VBOUNDS`, the legacy `sample()`, and hard `deploy()`: threshold → reserve top-up in `order` → upward repair → `NDOWN` downward trials in the mirrored order. It also holds the pool workers `_init/_ref/_dep`. `NCUTS`/`NDOWN` are module globals set through `set_cuts`/`set_down`.
- `restore.py` — the same restoration as `deploy`, applied to an arbitrary `uf`. It is used by the NN proxy so that the arms differ only in how `uf` is produced.
- `poolload.py` — `load_pool(case, "wide"|"base", ...)` returns `[(pd, qd, u_ref, cost_ref)]`. Only `wide` (`s ~ U[0.55,1.30]`) works in this tree; `base` needs an external `qcac_clean/` directory. `hardgate.py` is the wide/outage sampler and reference worker used by `01_make_pool.py`.
- Vehicle: `src_hv.py` (pool loader, `splits`, `HVRelax` with cut rows over `[E; Peng; Pbatt; z]`, integer `round_z`, hard `deploy`), `cutnet_hv.py` (`CutNetHV`, cut only, no V), `diffhv.py` (`DiffHV`, with a soft round that sums S sigmoids). The final vehicle restoration is `qz_restore`, a Gurobi L1 projection in `vehicle/02_train.py`, not `src_hv.deploy`.

**Splits** are identical everywhere: `perm = default_rng(0).permutation(len(POOL))`, then 48 test, then 24 validation and 144 train taken from the remainder (validation first). 4 seeds (0–3) per method.

**AC-UC deployment settings differ by case.** case118 uses confidence order (`argsort(-uf)`), a per-generator threshold fitted label-free on training instances (`03_threshold_fit.py`), and NDOWN=25. case300 uses cheapest-first merit order, threshold 0.80 (also `--dthr 0.8` in training) and NDOWN=40.

## Invariants that are easy to break

- **Self-supervised claim.** References may be used only to score the test split and to train the supervised proxy baseline. The loss is normalised by the *uncut relaxation* cost. Threshold fitting and model selection use deployed or surrogate cost on train/validation only. Never reintroduce reference costs or commitments into the loss, the scaling or the selection, and never select on test.
- **Do not drop the SOC constraint** from the QCAC relaxation; solutions leave the rank-1 manifold and the cost drops below the proven optimum.
- **Problems fed to `ConeLayer` must stay DPP** (`HVRelax` asserts `is_dpp()`). Per-parameter directions are extracted once at build time. That is also why `socp.py` passes `V0sq`/`VV0` as their own parameters (a squared parameter is not DPP).
- **Cut initialisation is deliberate.**
  - `head_v` is zero-initialised, so training starts at flat V; `v_scale=0.5`.
  - The `a`/`α` heads are zero apart from a small random `spread` on the bias. Identical rows are redundant active constraints, which is an LICQ failure.
  - Depth anchoring makes every row start active. A slack row gets zero gradient and never trains.
  - `head_thr` is zero-initialised, so the threshold starts at 0.5.
- **Build `b` in torch** (`rows_torch`), not in numpy. Otherwise the dL/dA term that flows through the anchor is silently dropped.
- **Restoration exists in four copies:** `pipeline.deploy`, `restore.restore`, the local `restore()` in `acuc/04_eval_case300.py`, and `src_hv.deploy`. They must stay behaviourally identical, including the rule that the downward pass walks the top-up order reversed.
- **Instance sampling is one correlated system scale per instance**, with small per-load noise. i.i.d. per-load sampling collapses demand variation. Changing a sampler invalidates the cached pools: wide pool instances are identified by global index and regenerated from `(seed, N, lo, hi)`.
- pandapower's 0/2 voltage limits are placeholders. The real bounds come from `VBOUNDS`; case300 is infeasible at 0.94/1.06.
- Costs returned by `deploy`/`build`/`price` exclude penalty terms, so they compare directly with the reference.
- The AC-UC reference (QCAC iterative) is a heuristic upper bound, not a global optimum, so gaps can be negative. Vehicle references are Gurobi optima.
- Trained checkpoints, pools and result JSONs are committed and feed `figures/` and `paper/numbers.tex`. Do not overwrite them casually.

# Self-supervised learned cuts for MINLP via exact differentiable optimization layers

A neural network maps the instance parameters to **parametric cutting planes**
(and, for nonconvex AC unit commitment, to the **linearisation point V** of the
QCAC convexification). The cuts are added to a convex relaxation that is solved
as a **differentiable conic layer**; the network is trained end-to-end on the
cost of the deployed solution with **no optimal-solution labels**.

## Layout

```
src/              shared code
  conelayer.py    exact implicit differentiation of any DPP cvxpy problem (Clarabel forward,
                  KKT reverse mode, regularised direct LU for degenerate KKT systems)
  conedesc.py     cone projections and their derivatives
  lu_server.py    sparse-LU helper process (isolates SuperLU aborts)
  framework.py    AC-UC differentiable deployment: relaxation -> soft rounding -> priced re-solve
  cutnet_k.py     AC-UC network: V + 2 cut categories with depth anchoring
  socp.py         QCAC relaxation as an SOCP (Clarabel) + commitment pricing
  acuc.py         QCAC iterative reference (Constante-Flores & Li, Gurobi MIQCQP)
  pipeline.py     grid, deployment/restoration;  restore.py  restoration used by the proxy
  poolload.py, hardgate.py, acopf_data.py, qcac.py   data / instance generation
  src_hv.py, cutnet_hv.py, diffhv.py                 hybrid-vehicle relaxation, network, layer
acuc/             AC unit commitment, IEEE case118 and case300
  01_make_pool.py       reference pool (wide band s ~ U[0.55,1.30]) -> results/wide/*.pkl
  02_train.py           self-supervised training -> checkpoints/{case}_s{seed}.pt
  03_threshold_fit.py   label-free rounding-threshold fit on TRAINING instances (case118)
  04_eval_case118.py    final case118 evaluation (fitted threshold, confidence order, NDOWN=25)
  04_eval_case300.py    final case300 evaluation (threshold 0.80, merit order, NDOWN=40)
  05_timing.py          per-instance deployment time
  06_nn_proxy.py        supervised NN proxy baseline, same restoration
vehicle/          hybrid-vehicle convex MINLP (T=30, z_t in {0,..,3})
  01_make_pool.py  02_train.py  03_nn_proxy.py  04_timing.py
gradient_checks/  finite-difference / cvxpylayers / smoothed-estimator comparisons
figures/          make_figures.py -> QUAD_{case118,case300,hybrid_vehicle}.{png,pdf}, summary.json
paper/            manuscript (LaTeX)
```

## Reproduce (run each script from its own folder)

```bash
# AC-UC, per case and seed (cwd = acuc/)
python 02_train.py --case case118 --epochs 30 --lr 2e-3 --seed S --ckpt checkpoints/case118_sS.pt
python 02_train.py --case case300 --epochs 26 --lr 5e-4 --down 25 --dthr 0.8 --seed S --ckpt checkpoints/case300_sS.pt
python 03_threshold_fit.py --case case118 --seed S --shard K      # label-free, training split only
python 04_eval_case118.py --case case118 --seed S --down 25
python 04_eval_case300.py --case case300 --seed S --order merit --thr 0.80 --downs 40
python 06_nn_proxy.py --case case118 --seed S --down 25 --order conf
python 06_nn_proxy.py --case case300 --seed S --down 40 --order merit
python 05_timing.py --case case118 --down 25
# vehicle (cwd = vehicle/)
python 02_train.py --seed S          # defaults = paper config (K=2, hidden 16, w_int 5, 25 epochs)
python 03_nn_proxy.py --seed S
python 04_timing.py
# figures (cwd = repo root)
python figures/make_figures.py
```

Trained AC-UC checkpoints, reference pools and all result JSONs used in the
paper are included. Split: 48 test / 24 validation / 144 train, fixed by
`numpy.random.default_rng(0)`; 4 seeds per method.

## Notes

* AC-UC reference = QCAC iterative heuristic (Constante-Flores & Li). It is an
  inner convex approximation solved iteratively, **not a proven global optimum**,
  so AC-UC metrics are deviations from that reference.
* Training and model selection use no reference solutions; labels are used only
  to score the held-out test split (and to train the supervised proxy baseline).
* Environment: Python 3, torch 2.14, cvxpy 1.9.2, clarabel 0.11.1, Gurobi 13.0.3,
  numpy 2.4.6, scipy 1.18.1, pandapower. Measured on an Apple M3 Pro, one thread per job.

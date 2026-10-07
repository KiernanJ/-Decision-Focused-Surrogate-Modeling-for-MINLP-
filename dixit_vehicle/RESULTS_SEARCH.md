# Dixit et al. hybrid-vehicle MILP — learned cut + dual-guided search (2026-10-07, saved; work paused)

Method: LP relaxation -> L1 projection with learned z-only cut rows (CombOptNet-style training,
self-supervised on deployed cost, margin 0.5 = starts at no-cut) -> re-solve -> dual-guided local
search (LP duals rank +-1 mode moves, top-5 priced). 4 rows, 1 seed, 100 test profiles per vehicle.
Scripts: 01_make_pool.py, 02_combopt_train.py (--search), 03_nn_proxy.py, 04_metrics.py.
Results: results/combopt_v*_search.json / .pt, results/metrics_search.json, results/proxy_v*_s0.json.

| vehicle | gap | discrete | continuous | time ours | time exact | no-cut+search gap |
|---|---|---|---|---|---|---|
| 0 | 0.165% | 15.0% | 14.8% | 56 ms | 88 ms | 0.167% |
| 1 | 2.087% | 22.2% | 21.5% | 61 ms | 3.27 s | 4.478% |
| 2 | 1.897% | 25.0% | 24.0% | 71 ms | 3.49 s | 3.443% |
| 3 | 0 | 0 | 0 | 28 ms | 0 ms | 0 (trivial) |
| 4 | 0.543% | 16.5% | 16.4% | 48 ms | 4.66 s | 0.938% |
| 6 | 1.475% | 20.1% | 19.4% | 46 ms | 6.51 s | 2.867% |
| 7 | 0.344% | 19.0% | 18.8% | 45 ms | 0.24 s | 0.365% |
| 8 | 0.770% | 21.0% | 20.7% | 64 ms | 3.20 s | 1.523% |
| 9 | 1.866% | 22.9% | 21.9% | 63 ms | 4.87 s | 4.471% |
| mean (8 non-trivial) | 1.14% | 20.3% | 19.7% | ~57 ms | ~3.3 s | 2.28% |

Not done: vehicle 5 (reference pool unfinished), row-count test (16 / 60 rows) stopped, 1 seed only,
timings measured on a loaded machine, supervised proxy here had no search.

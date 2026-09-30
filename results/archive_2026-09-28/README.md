# Results of 2026-09-28 (archived before the consistent-config rerun)

These are the numbers behind the **+0.052% case118 headline**. They were
produced with settings that differ BETWEEN the two cases, which is why they are
being regenerated -- but each is internally valid and all controls passed.

## case118 -- 48 held out, 192 instances (144 train)
net_tier_k1best.pt | K=1 hard cardinality cut | down_train = down = 6 |
init_frac 0.25 (hand-set) | 60 epochs, full batch, lr 1e-3 | fixed 0.5 rounding

| arm | labels | discrete | continuous | gap |
|---|---|---|---|---|
| relaxation only (flat V) | - | 40.32% | 85.97% | +10.378% |
| NN proxy, SUPERVISED on u* | yes | 3.09% | 5.10% | +0.066% |
| NN proxy, self-supervised | no | 9.61% | 12.61% | +0.242% |
| ours: V only | no | 22.53% | 23.34% | +3.319% |
| ours: V + constant penalty | no | 4.17% | 5.29% | +0.103% |
| **ours: V + LEARNED cardinality cut** | no | **4.17%** | 6.26% | **+0.052%** |
| CONTROL dummy band | no | 41.05% | 41.98% | +9.217% |
| CONTROL random band | no | 41.05% | 45.54% | +9.446% |
| CONTROL mean band | no | 7.14% | 10.96% | +0.579% |

Learned cut beats every control (177x / 182x / 11x); per-instance spread sd 2.66.
Beats the SUPERVISED proxy using no labels.

## case300 -- 40 held out, 151 instances (111 train)
net_tier_c300big.pt | same cut | down_train = down = **2** (differs from case118)
| init_frac 0.70 (hand-set) | 60 epochs, full batch, lr 1e-3

| arm | labels | discrete | continuous | gap |
|---|---|---|---|---|
| relaxation only | - | 26.63% | 22.73% | +2.155% |
| NN proxy, SUPERVISED | yes | 1.78% | 1.36% | -0.004% |
| ours: V only | no | 25.11% | 18.83% | +1.878% |
| ours: V + constant penalty | no | 13.37% | 9.10% | +0.609% |
| ours: V + LEARNED cut | no | 14.57% | 7.79% | +0.638% |
| CONTROL mean band | no | 14.38% | 6.78% | +1.411% |

Passes its control, but the margin is thin (0.638 vs 0.737) and the learned cut
ties the constant penalty here. Data-limited: 111 instances for a 600-dim V.

## Why these are being regenerated
down_train/down was 6 on case118 and 2 on case300, and init_frac was hand-set
per case -- so no cross-case claim from this pair is safe. The rerun uses one
configuration for both plus a label-free band init and a learned rounding
threshold.

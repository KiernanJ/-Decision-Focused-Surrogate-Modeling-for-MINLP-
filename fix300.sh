#!/bin/zsh
# case300 retrain: TRAIN and EVAL restoration matched.
#
# The failed run trained with down_train=0 but was evaluated with 6 downward
# repair trials. Restoration that only switches generators ON makes
# over-commitment unrecoverable, so with down=0 the network learns to
# UNDER-commit to leave room for the upward climb -- the band drifted to ~24
# against a true n* of ~50.5, spread blew up to sd 18.3, and the learned cut
# lost to its own control.
#
# The fix is matching, not more repair: train AND evaluate at down=2. Cheap, and
# 24_final.py evaluates every arm at the same setting, so the comparison stays
# fair. init_frac 0.70 also starts the band near n* (48.3 vs 50.5) and still
# binding, instead of 41.4 with a 9-unit climb in the wrong direction.
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/fix300.log }

say "case300 K=1, down_train=2 == down_eval=2, init_frac 0.70"
QCAC_CASE=case300 $PY -u 21_tiered_train.py --tiers 1 --partition interleaved \
    --epochs 35 --procs 12 --batch 48 --down-train 2 --down 2 --log-every 5 \
    --lr 2e-3 --init-frac 0.70 --n-test 40 --tag c300fix > logs/f300.log 2>&1
say "$(grep -a 'V + LEARNED tier cuts' logs/f300.log)"
say "$(grep -a 'mean band' logs/f300.log)"
say "$(grep -a 'REAL\|DECORATIVE' logs/f300.log)"

say "case300 FINAL figure (all arms evaluated at down=2)"
QCAC_CASE=case300 NTEST=40 KTIERS=1 NDOWN=2 NET_TIER=net_tier_c300fix.pt \
    NET_VC=net_c300n.pt NET_PS=net_proxyself_case300.pt \
    $PY -u 24_final.py > logs/f300_final.log 2>&1
say "$(grep -a 'wrote' logs/f300_final.log)"
say "DONE"

#!/bin/zsh
# case300 on the enlarged pool: 345 instances, up from 151.
#
# Its V head predicts a 600-dimensional voltage correction and `V only` barely
# beats the baseline (+1.861% vs +2.154%), so the head is data-starved rather
# than mis-designed. Every architectural alternative tested came back negative
# (tiering, low-rank V basis, learned rounding threshold, variable fixing), so
# data is the only lever the measurements still support.
#
# Config matches case118 except batch, which is capped at 150 to keep the epoch
# affordable on a 265-instance training set. The threshold head is OFF: it tied
# fixed-0.5 rounding on both cases and is not part of the method any more.
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/data300.log }

say "case300 retrain, 345-instance pool, 80 held out"
QCAC_CASE=case300 $PY -u 21_tiered_train.py --tiers 1 --partition interleaved \
    --epochs 45 --procs 12 --batch 150 --down-train 6 --down 6 --log-every 5 \
    --lr 1e-3 --sigma 0.03 --sigma-tier 0.6 --learn-thr 0 --n-test 80 \
    --tag c300data > logs/d300.log 2>&1
say "$(grep -a 'V + LEARNED tier cuts' logs/d300.log)"
say "$(grep -a 'mean band' logs/d300.log)"
say "$(grep -a 'REAL\|DECORATIVE' logs/d300.log)"
say "figure"
QCAC_CASE=case300 NTEST=80 KTIERS=1 NDOWN=6 NET_TIER=net_tier_c300data.pt \
    NET_VC=net_c300n.pt NET_PS=net_proxyself_case300.pt \
    $PY -u 24_final.py > logs/d300_fig.log 2>&1
say "$(grep -a 'wrote\|timestamped' logs/d300_fig.log | tr '\n' ' ')"
say "DONE"

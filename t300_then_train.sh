#!/bin/zsh
# Timing FIRST on an idle machine -- a contended timing measurement is a wrong
# one, and the stored reference times were already inflated 2x by 12-way
# contention (132s vs 68s re-timed single-process on case118). Only after that
# does the retrain start and take the whole machine.
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/t300.log }

say "case300 timing (machine idle)"
OMP_NUM_THREADS=12 QCAC_CASE=case300 NTEST=40 NDOWN=6 NRETIME=4 \
    NET_TIER=net_tier_cons300.pt $PY -u 29_timing.py > logs/t300_timing.log 2>&1
say "$(grep -a 'ours: V + learned cut' logs/t300_timing.log)"
say "$(grep -a 'single process' logs/t300_timing.log)"
say "$(grep -a 'speed-up' logs/t300_timing.log)"
say "$(grep -a 'wrote' logs/t300_timing.log)"

say "case300 retrain on the 345-instance pool (down=6, matches case118)"
OMP_NUM_THREADS=1 QCAC_CASE=case300 $PY -u 21_tiered_train.py --tiers 1 \
    --partition interleaved --epochs 45 --procs 12 --batch 150 --down-train 6 \
    --down 6 --log-every 5 --lr 1e-3 --sigma 0.03 --sigma-tier 0.6 \
    --learn-thr 0 --n-test 80 --tag c300data > logs/d300.log 2>&1
say "$(grep -a 'V + LEARNED tier cuts' logs/d300.log)"
say "$(grep -a 'mean band' logs/d300.log)"
say "$(grep -a 'REAL\|DECORATIVE' logs/d300.log)"
OMP_NUM_THREADS=1 QCAC_CASE=case300 NTEST=80 KTIERS=1 NDOWN=6 \
    NET_TIER=net_tier_c300data.pt NET_VC=net_c300n.pt \
    NET_PS=net_proxyself_case300.pt $PY -u 24_final.py > logs/d300_fig.log 2>&1
say "$(grep -a 'wrote\|timestamped' logs/d300_fig.log | tr '\n' ' ')"
say "ALL DONE"

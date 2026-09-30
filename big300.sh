#!/bin/zsh
# case300 on case118's winning recipe. The previous run saw only 48 of 111
# instances per epoch for 35 epochs; case118's winner used the FULL batch for
# 60. That is ~3x more gradient information, and case300 needs more, not less:
# it predicts a 600-dimensional voltage correction from 111 instances, against
# case118's 236 from 144.
#
# Not attempted: a low-rank V basis. A label-free PCA of the relaxation's own V
# captures 99.76% of ITS variance in 24 components, but leaves residual
# ||dV||=1.17 against the true V* (do-nothing is 3.67) -- it would cap accuracy
# below what is needed rather than make it easier to reach.
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/big300.log }

say "case300 K=1, FULL batch 111, 60 epochs, down 2 == eval 2"
QCAC_CASE=case300 $PY -u 21_tiered_train.py --tiers 1 --partition interleaved \
    --epochs 60 --procs 12 --batch 111 --down-train 2 --down 2 --log-every 5 \
    --lr 1e-3 --init-frac 0.70 --n-test 40 --tag c300big > logs/b300.log 2>&1
say "$(grep -a 'V + LEARNED tier cuts' logs/b300.log)"
say "$(grep -a 'mean band' logs/b300.log)"
say "$(grep -a 'REAL\|DECORATIVE' logs/b300.log)"

say "case300 FINAL figure"
QCAC_CASE=case300 NTEST=40 KTIERS=1 NDOWN=2 NET_TIER=net_tier_c300big.pt \
    NET_VC=net_c300n.pt NET_PS=net_proxyself_case300.pt \
    $PY -u 24_final.py > logs/b300_final.log 2>&1
say "$(grep -a 'wrote' logs/b300_final.log)"
say "DONE"

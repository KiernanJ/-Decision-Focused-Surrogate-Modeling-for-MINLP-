#!/bin/zsh
# IDENTICAL configuration for case118 and case300. Every previous comparison
# differed somewhere -- down-train/down were 6 vs 2, and init-frac was hand-set
# per case (0.25 vs 0.70) -- which makes any cross-case claim a tuning artifact.
#
# Now shared by both: tiers 1, 45 epochs, FULL batch, down-train 6 == down 6,
# lr 1e-3, sigma 0.03 / 0.6 / 0.08, learned per-generator threshold ON.
# The band init is derived from n_min (n*/n_min measured 1.97 and 2.24), so it
# adapts per case WITHOUT a per-case constant and uses no labels.
#
# n-test stays 48 / 40, matching the existing splits so the new numbers remain
# comparable with everything already reported.
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/consistent.log }

run(){   # $1=case  $2=n_test  $3=tag
  say "$1  K=1 + learned threshold, identical config"
  QCAC_CASE=$1 $PY -u 21_tiered_train.py --tiers 1 --partition interleaved \
      --epochs 45 --procs 12 --batch 0 --down-train 6 --down 6 --log-every 5 \
      --lr 1e-3 --sigma 0.03 --sigma-tier 0.6 --sigma-thr 0.08 --learn-thr 1 \
      --n-test $2 --tag $3 > logs/c_$3.log 2>&1
  say "$1  $(grep -a 'V + LEARNED tier cuts' logs/c_$3.log)"
  say "$1  $(grep -a 'threshold 0.5' logs/c_$3.log)"
  say "$1  $(grep -a 'MEAN threshold' logs/c_$3.log)"
  say "$1  $(grep -a 'mean band' logs/c_$3.log)"
  say "$1  $(grep -a 'learned thresholds' logs/c_$3.log)"
  say "$1  $(grep -a 'REAL\|DECORATIVE' logs/c_$3.log)"
  say "$1  FINAL figure"
  QCAC_CASE=$1 NTEST=$2 KTIERS=1 NDOWN=6 NET_TIER=net_tier_$3.pt \
      NET_VC=$4 NET_PS=net_proxyself_$1.pt \
      $PY -u 24_final.py > logs/c_final_$1.log 2>&1
  say "$1  $(grep -a 'wrote' logs/c_final_$1.log)"
}
run case118 48 cons118 net_c118_big_vcard.pt
run case300 40 cons300 net_c300n.pt
say "ALL DONE"

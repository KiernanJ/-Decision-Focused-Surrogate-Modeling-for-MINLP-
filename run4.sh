#!/bin/zsh
# Does the PARTITION carry meaning, or is the gain just K constraints instead of 1?
#   block   contiguous merit order -- what "tier" should mean
#   random  THE CONTROL. If it matches, the partition is arbitrary and the
#           story is "K constraints", not "merit tiers".
#   K=1     does tiering earn its complexity, trained identically
# Explicit invocations: zsh does not word-split unquoted vars, so `set -- $spec`
# passed --tiers "3 block" and all three runs died in 2 seconds.
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/run4.log }
run(){   # $1=K  $2=partition
  say "K=$1 $2  starting"
  $PY -u 21_tiered_train.py --tiers $1 --partition $2 --epochs 60 --procs 12 \
      --batch 96 --down-train 2 --down 6 --log-every 10 --lr 1e-3 \
      --tag k$1_$2 > logs/p_k$1_$2.log 2>&1
  say "K=$1 $2  $(grep -a 'LEARNED CUT IS REAL\|DECORATIVE' logs/p_k$1_$2.log)"
  say "K=$1 $2  learned: $(grep -a 'V + LEARNED tier cuts' logs/p_k$1_$2.log)"
  say "K=$1 $2  mean:    $(grep -a 'mean band' logs/p_k$1_$2.log)"
}
run 3 block
run 3 random
run 1 interleaved
say "ALL DONE"

#!/bin/zsh
# Wait for case300's timing, then regenerate BOTH combined figures so their
# time panels use the converged reference medians (n=15, drift 0.0%) rather
# than the 4-sample draws that gave 50x and 25x.
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/finish.log }
while pgrep -f 29_timing.py >/dev/null; do sleep 30; done
say "case300 timing: $(grep -a 'speed-up' logs/tfin300.log)"

say "regenerating combined figures with converged times"
QCAC_CASE=case118 POOL_SEEDS=0,1,2,3,4 NTEST=48 KTIERS=1 NDOWN=6 \
    NET_TIER=net_tier_cons118.pt NET_VC=net_c118_big_vcard.pt \
    NET_PS=net_proxyself_case118.pt $PY -u 24_final.py > logs/ff118.log 2>&1
say "case118: $(grep -a 'wrote' logs/ff118.log)"
QCAC_CASE=case300 POOL_SEEDS=0,1,2,3 NTEST=40 KTIERS=1 NDOWN=6 \
    NET_TIER=net_tier_cons300.pt NET_VC=net_c300n.pt \
    NET_PS=net_proxyself_case300.pt $PY -u 24_final.py > logs/ff300.log 2>&1
say "case300: $(grep -a 'wrote' logs/ff300.log)"
say "FIGURES DONE"

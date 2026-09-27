#!/bin/zsh
# Overnight plan. Sequential so the 12 cores are never oversubscribed; every
# phase logs to logs/ and a failure in one phase does not stop the rest.
#
# Goal: reduce DISCRETE and CONTINUOUS decision error on case118 AND case300.
#   - downward repair (measured 0.103% -> 0.029% on case118) is on everywhere
#   - both label leaks are closed, so the self-supervised claim is real
#   - case118 gets FRESH instances, so seeds are independent rather than
#     re-splits of one pool (the caveat on the earlier 3-seed result)
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/overnight.log }

say "P0  waiting for the case118 downward-repair runs already in flight"
while pgrep -f "11_vcard.py --arm" >/dev/null; do sleep 60; done
say "P0  done"

# ---- P1  case300 references -------------------------------------------------
say "P1  case300 references (2 x 40 instances)"
for s in 0 1; do
  QCAC_CASE=case300 SEED=$s NINST=40 PROCS=6 $PY -u 00_refs.py \
      > logs/refs300_s$s.log 2>&1
  say "P1  case300 seed $s: $(grep -a 'solved' logs/refs300_s$s.log | tail -1)"
done

# ---- P2  case300 training ---------------------------------------------------
say "P2  case300 training, v vs vcard, downward repair on"
for arm in vcard v; do
  # case300 starts ~24 units below its n*, so it needs more epochs and a
  # bigger cardinality step than case118 to climb there.
  QCAC_CASE=case300 $PY -u 11_vcard.py --arm $arm --epochs 45 --procs 12 \
      --down 6 --n-test 20 --sigma-card 2.0 --tag c300_$arm > logs/c300_$arm.log 2>&1
  say "P2  case300 $arm: $(sed -n '/^=\{20,\}/,$p' logs/c300_$arm.log | grep -a learned)"
done

# ---- P3  case118 FRESH instances (independent seeds, not re-splits) ---------
say "P3  case118 extra references (seeds 3,4 -> +80 instances)"
for s in 3 4; do
  QCAC_CASE=case118 SEED=$s NINST=40 PROCS=12 $PY -u 00_refs.py \
      > logs/refs118_s$s.log 2>&1
  say "P3  case118 seed $s: $(grep -a 'solved' logs/refs118_s$s.log | tail -1)"
done

# ---- P4  case118 final on the expanded pool --------------------------------
say "P4  case118 final, v vs vcard on ~192 instances"
for arm in vcard v; do
  QCAC_CASE=case118 $PY -u 11_vcard.py --arm $arm --epochs 40 --procs 12 \
      --down 6 --n-test 48 --tag c118_big_$arm > logs/c118_big_$arm.log 2>&1
  say "P4  case118 $arm: $(sed -n '/^=\{20,\}/,$p' logs/c118_big_$arm.log | grep -a learned)"
done

# ---- P5  error decomposition + figures for BOTH cases ----------------------
say "P5  error figures"
QCAC_CASE=case118 NET_V=net_c118_big_v.pt NET_VC=net_c118_big_vcard.pt \
    NTEST=48 $PY -u 14_errors.py > logs/fig118.log 2>&1
say "P5  case118: $(grep -a 'V + card' logs/fig118.log)"
QCAC_CASE=case300 NET_V=net_c300_v.pt NET_VC=net_c300_vcard.pt \
    NTEST=20 $PY -u 14_errors.py > logs/fig300.log 2>&1
say "P5  case300: $(grep -a 'V + card' logs/fig300.log)"
say "ALL DONE"

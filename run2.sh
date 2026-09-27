#!/bin/zsh
# Queue 2. Priority order = what most constrains the claim.
#   Q1 the SELF-SUPERVISED proxy. Decides whether the relaxation in the loop is
#      load-bearing. The supervised proxy already beats us (0.063 vs 0.103 on
#      case118), but it uses labels we never touch; this one uses our signal.
#   Q2 more case300 references. 58 training instances for a 600-dim voltage
#      target is the likeliest reason our V head trails the proxy there.
#   Q3 retrain case300 on the larger pool, then re-run the proxy comparison.
#   Q4 cut-family ablation: is 2 rows right, and is the advantage a tuning
#      artifact of the band init?
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
P=12
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/run2.log }

say "Q1  self-supervised NN proxy, case118 (the decisive baseline)"
QCAC_CASE=case118 $PY -u 16_proxy_self.py --epochs 40 --procs $P --n-test 48 \
    > logs/q1_proxyself118.log 2>&1
say "Q1  $(grep -a 'SELF-SUPERVISED' logs/q1_proxyself118.log)"

say "Q2  case300 references, seeds 2,3 (+80 -> ~158 instances)"
for s in 2 3; do
  QCAC_CASE=case300 SEED=$s NINST=40 PROCS=$P $PY -u 00_refs.py \
      > logs/q2_refs300_s$s.log 2>&1
  say "Q2  seed $s: $(grep -a 'solved' logs/q2_refs300_s$s.log | tail -1)"
done

say "Q3  case300 retrain on the larger pool"
for arm in vcard v; do
  QCAC_CASE=case300 $PY -u 11_vcard.py --arm $arm --epochs 45 --procs $P \
      --down 6 --n-test 40 --sigma-card 2.0 --tag c300b_$arm \
      > logs/q3_c300b_$arm.log 2>&1
  say "Q3  $arm: $(sed -n '/^=\{20,\}/,$p' logs/q3_c300b_$arm.log | grep -a learned)"
done
say "Q3  proxy comparison on the larger case300 pool"
QCAC_CASE=case300 NTEST=40 NET_V=net_c300b_v.pt NET_VC=net_c300b_vcard.pt \
    $PY -u 15_proxy.py > logs/q3_proxy300.log 2>&1
say "Q3  $(grep -a 'NN proxy' logs/q3_proxy300.log)"
say "Q3  $(grep -a 'cardinality cut' logs/q3_proxy300.log)"

say "Q4  cut-family ablation on case118"
QCAC_CASE=case118 $PY -u 17_cut_ablation.py > logs/q4_ablation.log 2>&1
say "Q4  done -- see logs/q4_ablation.log"
say "ALL DONE"

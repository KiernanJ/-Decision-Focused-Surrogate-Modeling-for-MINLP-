#!/bin/zsh
# Queue 3 (Q1 done: self-supervised proxy = 0.242% vs ours 0.103%).
#   Q2 more case300 references -- 58 train instances for a 600-dim voltage
#      target is the likeliest reason our V head trails the proxy there.
#   Q3 retrain case300 on the larger pool, then re-run the proxy comparison
#      INCLUDING the self-supervised proxy, so case300 gets the same
#      label-free head-to-head that decided case118.
#   Q4 cut ablation: which rows matter, and does the gain survive mistuning
#      the band init (the one hyperparameter set by hand, not by search).
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/run3.log }

say "Q2  case300 references, seeds 2,3"
for s in 2 3; do
  QCAC_CASE=case300 SEED=$s NINST=40 PROCS=12 $PY -u 00_refs.py \
      > logs/q2_refs300_s$s.log 2>&1
  say "Q2  seed $s: $(grep -a 'solved' logs/q2_refs300_s$s.log | tail -1)"
done

say "Q3  case300 retrain on the larger pool"
for arm in vcard v; do
  QCAC_CASE=case300 $PY -u 11_vcard.py --arm $arm --epochs 45 --procs 12 \
      --down 6 --n-test 40 --sigma-card 2.0 --tag c300b_$arm \
      > logs/q3_c300b_$arm.log 2>&1
  say "Q3  $arm: $(sed -n '/^=\{20,\}/,$p' logs/q3_c300b_$arm.log | grep -a learned)"
done
say "Q3  supervised proxy + ours, case300"
QCAC_CASE=case300 NTEST=40 NET_V=net_c300b_v.pt NET_VC=net_c300b_vcard.pt \
    $PY -u 15_proxy.py > logs/q3_proxy300.log 2>&1
say "Q3  $(grep -a 'NN proxy' logs/q3_proxy300.log)"
say "Q3  $(grep -a 'cardinality cut' logs/q3_proxy300.log)"
say "Q3  SELF-supervised proxy, case300 (the label-free head-to-head)"
QCAC_CASE=case300 $PY -u 16_proxy_self.py --epochs 45 --procs 12 --n-test 40 \
    --sigma 0.15 > logs/q3_proxyself300.log 2>&1
say "Q3  $(grep -a 'SELF-SUPERVISED' logs/q3_proxyself300.log)"

say "Q4  cut-family ablation, case118"
QCAC_CASE=case118 NTEST=48 NET_VC=net_c118_big_vcard.pt $PY -u 17_cut_ablation.py \
    > logs/q4_ablation.log 2>&1
say "Q4  done -- logs/q4_ablation.log"
say "ALL DONE"

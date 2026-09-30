#!/bin/zsh
# Resume. case118 is finished (+0.052%); its figure goes first so there is a
# result in hand immediately. case300 then runs at a much cheaper budget:
# P2a took FOUR hours because 11_vcard trains with the full 6 downward repair
# trials in the loop. Training-time restoration does not affect the final
# comparison -- 24_final.py evaluates EVERY arm with the same 6 trials -- so
# down_train=0 costs nothing in fairness and roughly 10x in time.
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/night2.log }

say "A  case118 FINAL figure (result already in hand)"
QCAC_CASE=case118 NTEST=48 KTIERS=1 NET_TIER=net_tier_k1best.pt \
    NET_VC=net_c118_big_vcard.pt NET_PS=net_proxyself_case118.pt \
    $PY -u 24_final.py > logs/n2_final118.log 2>&1
say "A  $(grep -a 'wrote' logs/n2_final118.log)"

say "B  case300 K=1 learned cardinality (down_train 0 for speed)"
QCAC_CASE=case300 $PY -u 21_tiered_train.py --tiers 1 --partition interleaved \
    --epochs 40 --procs 12 --batch 64 --down-train 0 --down 6 --log-every 10 \
    --lr 1e-3 --init-frac 0.60 --n-test 40 --tag c300k1 > logs/n2_c300_k1.log 2>&1
say "B  $(grep -a 'V + LEARNED tier cuts' logs/n2_c300_k1.log)"
say "B  $(grep -a 'REAL\|DECORATIVE' logs/n2_c300_k1.log)"

say "C  case300 self-supervised NN proxy"
QCAC_CASE=case300 $PY -u 16_proxy_self.py --epochs 40 --procs 12 --n-test 40 \
    --sigma 0.15 --down 0 > logs/n2_c300_ps.log 2>&1
say "C  $(grep -a 'SELF-SUPERVISED' logs/n2_c300_ps.log)"

say "D  case300 FINAL figure"
QCAC_CASE=case300 NTEST=40 KTIERS=1 NET_TIER=net_tier_c300k1.pt \
    NET_VC=net_c300n.pt NET_PS=net_proxyself_case300.pt \
    $PY -u 24_final.py > logs/n2_final300.log 2>&1
say "D  $(grep -a 'wrote' logs/n2_final300.log)"

say "E  preflight, case118"
QCAC_CASE=case118 NTEST=48 NET_VC=net_c118_big_vcard.pt $PY -u 18_preflight.py \
    > logs/n2_pre118.log 2>&1
say "E  $(grep -a 'passed,' logs/n2_pre118.log)"
say "COMPLETE"

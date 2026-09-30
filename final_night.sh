#!/bin/zsh
# Ordered so the DELIVERABLES complete first and the improvement run uses
# whatever time is left. If the night is cut short, the tables are still whole.
#
#  A  case300 self-supervised proxy on the CURRENT 151-instance pool -- the one
#     missing row in the case300 table -> regenerate its figure.      ~1.5 h
#  B  +200 case300 references. Its V head predicts a 600-dim target from 111
#     instances and `V only` barely beats the baseline (+1.861% vs +2.154%),
#     so data is the only lever the measurements still support.        ~1 h
#  C  retrain case300 on the enlarged pool, then its figure.           ~6 h
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/final_night.log }

say "A  case300 self-supervised proxy (completes the table)"
QCAC_CASE=case300 $PY -u 16_proxy_self.py --epochs 40 --procs 12 --n-test 40 \
    --sigma 0.15 --down 6 > logs/fn_ps300.log 2>&1
say "A  $(grep -a 'SELF-SUPERVISED' logs/fn_ps300.log)"
say "A  case300 figure WITH the proxy row"
QCAC_CASE=case300 NTEST=40 KTIERS=1 NDOWN=6 NET_TIER=net_tier_cons300.pt \
    NET_VC=net_c300n.pt NET_PS=net_proxyself_case300.pt \
    $PY -u 24_final.py > logs/fn_fig300.log 2>&1
say "A  $(grep -a 'wrote\|timestamped' logs/fn_fig300.log | tr '\n' ' ')"

say "B  +200 case300 references (seeds 4-8)"
for s in 4 5 6 7 8; do
  QCAC_CASE=case300 SEED=$s NINST=40 PROCS=12 $PY -u 00_refs.py \
      > logs/fn_ref300_s$s.log 2>&1
  say "B  seed $s: $(grep -a 'solved' logs/fn_ref300_s$s.log | tail -1)"
done

say "C  case300 retrain on the enlarged pool"
QCAC_CASE=case300 $PY -u 21_tiered_train.py --tiers 1 --partition interleaved \
    --epochs 45 --procs 12 --batch 150 --down-train 6 --down 6 --log-every 5 \
    --lr 1e-3 --sigma 0.03 --sigma-tier 0.6 --learn-thr 0 --n-test 80 \
    --tag c300data > logs/fn_c300data.log 2>&1
say "C  $(grep -a 'V + LEARNED tier cuts' logs/fn_c300data.log)"
say "C  $(grep -a 'mean band' logs/fn_c300data.log)"
QCAC_CASE=case300 NTEST=80 KTIERS=1 NDOWN=6 NET_TIER=net_tier_c300data.pt \
    NET_VC=net_c300n.pt NET_PS=net_proxyself_case300.pt \
    $PY -u 24_final.py > logs/fn_fig300b.log 2>&1
say "C  $(grep -a 'wrote' logs/fn_fig300b.log)"
say "ALL DONE"

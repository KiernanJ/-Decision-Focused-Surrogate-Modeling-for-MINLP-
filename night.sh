#!/bin/zsh
# Overnight v2. The partition comparison settled the cut family:
#   K=3 block  +1.757%   K=3 random +0.807%   K=3 interleaved +0.351%
#   K=1 GLOBAL +0.074%  <- best, and 6.6x better than its own mean-band control
# Tiering hurts; every added tier is worse. So the method is a LEARNED GLOBAL
# CARDINALITY cut, hard-enforced -- the original idea, which only ever failed
# because soft cuts made the band inert (a dummy [0,0] band reproduced it
# bit for bit).
#
# K=1 goes through 21_tiered_train.py, which carries the refutation controls
# (dummy / random / mean band) and the per-instance spread check in-process.
cd "$(dirname "$0")"
PY=~/.venvs/dfsm/bin/python
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
say(){ print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a logs/night.log }

say "P1  case118 K=1 learned cardinality, FULL budget"
$PY -u 21_tiered_train.py --tiers 1 --partition interleaved --epochs 60 --procs 12 \
    --batch 144 --down-train 6 --down 6 --log-every 10 --lr 1e-3 --init-frac 0.25 \
    --n-test 48 --tag k1best > logs/n_c118_k1.log 2>&1
say "P1  $(grep -a 'V + LEARNED tier cuts' logs/n_c118_k1.log)"
say "P1  $(grep -a 'mean band' logs/n_c118_k1.log)"
say "P1  $(grep -a 'REAL\|DECORATIVE' logs/n_c118_k1.log)"

say "P2a case300 V + cardinality (supplies V-only and constant-penalty arms)"
QCAC_CASE=case300 $PY -u 11_vcard.py --arm vcard --epochs 40 --procs 12 \
    --down 6 --n-test 40 --sigma-card 2.0 --tag c300n > logs/n_c300_vcard.log 2>&1
say "P2a $(sed -n '/^=\{20,\}/,$p' logs/n_c300_vcard.log | grep -a learned)"

say "P2b case300 K=1 learned cardinality (init-frac 0.60: it commits ~75%)"
QCAC_CASE=case300 $PY -u 21_tiered_train.py --tiers 1 --partition interleaved \
    --epochs 45 --procs 12 --batch 80 --down-train 2 --down 6 --log-every 10 \
    --lr 1e-3 --init-frac 0.60 --n-test 40 --tag c300k1 > logs/n_c300_k1.log 2>&1
say "P2b $(grep -a 'V + LEARNED tier cuts' logs/n_c300_k1.log)"
say "P2b $(grep -a 'REAL\|DECORATIVE' logs/n_c300_k1.log)"

say "P2c case300 self-supervised NN proxy (the baseline that matters)"
QCAC_CASE=case300 $PY -u 16_proxy_self.py --epochs 40 --procs 12 --n-test 40 \
    --sigma 0.15 > logs/n_c300_proxyself.log 2>&1
say "P2c $(grep -a 'SELF-SUPERVISED' logs/n_c300_proxyself.log)"

say "P3  preflight, case118"
QCAC_CASE=case118 NTEST=48 NET_VC=net_c118_big_vcard.pt $PY -u 18_preflight.py \
    > logs/n_pre118.log 2>&1
say "P3  $(grep -a 'passed,' logs/n_pre118.log)"

say "P4  FINAL figure, case118"
QCAC_CASE=case118 NTEST=48 KTIERS=1 NET_TIER=net_tier_k1best.pt \
    NET_VC=net_c118_big_vcard.pt NET_PS=net_proxyself_case118.pt \
    $PY -u 24_final.py > logs/n_final118.log 2>&1
say "P4  $(grep -a 'wrote' logs/n_final118.log)"

say "P4  FINAL figure, case300"
QCAC_CASE=case300 NTEST=40 KTIERS=1 NET_TIER=net_tier_c300k1.pt \
    NET_VC=net_c300n.pt NET_PS=net_proxyself_case300.pt \
    $PY -u 24_final.py > logs/n_final300.log 2>&1
say "P4  $(grep -a 'wrote' logs/n_final300.log)"

say "NIGHT COMPLETE"
grep -aE "^(relaxation|NN proxy|ours|CONTROL)" logs/n_final118.log >> logs/night.log
grep -aE "^(relaxation|NN proxy|ours|CONTROL)" logs/n_final300.log >> logs/night.log

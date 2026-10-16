#!/bin/bash
# Hamming row trained on 118 loads (= the proxy's data size), seeds 0,1,2. Validation only. Status logs/night_ham118.status
cd "$(dirname "$0")"; P=~/.venvs/dfsm/bin/python3; S=logs/night_ham118.status; export OMP_NUM_THREADS=1 DEPLOY_RHO=3e5 NTRAIN=118
st() { echo "$1 $(date '+%m-%d %H:%M:%S')" >> "$S"; }
: > "$S"; st "CHAIN START"
for sd in 0 1 2; do st "START seed $sd"; SEED=$sd $P 67_generalized_integer_cuts.py case118 ham 0 0 _n118s$sd > logs/gi_n118_ham_s$sd.log 2>&1; st "END seed $sd exit $?"; done
st "CHAIN DONE"

#!/bin/bash
# RESUME of night_ham118_2026-10-16.sh after the user stopped it: seed 0 is DONE (logs/gi_n118_ham_s0.log); run seeds 1 and 2 with the same settings. Validation only.
cd "$(dirname "$0")"; P=~/.venvs/dfsm/bin/python3; S=logs/resume_ham118.status; export OMP_NUM_THREADS=1 DEPLOY_RHO=3e5 NTRAIN=118
st() { echo "$1 $(date '+%m-%d %H:%M:%S')" >> "$S"; }
: > "$S"; st "CHAIN START"
for sd in 1 2; do st "START seed $sd"; SEED=$sd $P 67_generalized_integer_cuts.py case118 ham 0 0 _n118s$sd > logs/gi_n118_ham_s$sd.log 2>&1; st "END seed $sd exit $?"; done
st "CHAIN DONE"

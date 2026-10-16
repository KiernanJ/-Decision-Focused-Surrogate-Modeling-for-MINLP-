#!/bin/bash
# FINAL TESTS (2026-10-15), exactly the pre-registered commands (acuc/RESULTS.md "PRE-REGISTRATION of the CLEAN case300 + PGLib FINAL TESTS"): each case ONCE on fresh5, seeds 0,1,2, DEPLOY_RHO=1e4, no retry.
cd "$(dirname "$0")"; P=~/.venvs/dfsm/bin/python3; S=logs/final300pg.status; export OMP_NUM_THREADS=1 DEPLOY_RHO=1e4
st() { echo "$1 $(date '+%m-%d %H:%M:%S')" >> "$S"; }
: > "$S"; st "START"
st "START case300"; $P 55_final_eval.py case300 0,1,2 fresh5 _lfc _nc vseed0 > logs/final_clean300_fresh5.log 2>&1; st "END case300 exit $?"
st "START pglib"; $P 55_final_eval.py pglib_opf_case118_ieee__api 0,1,2 fresh5 _lfc _nc vseed0 > logs/final_cleanpglib_fresh5.log 2>&1; st "END pglib exit $?"
st "DONE"

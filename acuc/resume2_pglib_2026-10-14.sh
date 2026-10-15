#!/bin/bash
# RESUME 2 (2026-10-14): PGLib rows are DONE (gate passed, selected epoch 1; the earlier chain aborted by mistake: its check matched the normal "[val] ... rows OFF <score>" line).
# Remaining: PGLib rho sweep (label-free score on all 24 validation draws), then fresh5 references for case300 + PGLib, then the reference check. No tests.
cd "$(dirname "$0")"; P=~/.venvs/dfsm/bin/python3; S=logs/resume2_pglib.status; export OMP_NUM_THREADS=1
st() { echo "$1 $(date '+%m-%d %H:%M:%S')" >> "$S"; }
run() { local name=$1; shift; st "START $name"; "$@"; st "END $name exit $?"; }
: > "$S"; st "CHAIN START"; C=pglib_opf_case118_ieee__api
grep -q "gate\] PASSED" logs/c3p_rows_$C.log && [ -f results/general_rows_${C}_lfc.pt ] || { st "ABORT rows not trained"; exit 1; }
for r in 1e4 3e4 1e5 3e5 1e6; do run "${C}_val_rho_$r" bash -c "$P 61_val_full.py $C _env_nc _lfc rho_$r full none $r > logs/c3p_val_${C}_rho_$r.log 2>&1"; done
st "CASE DONE $C"
for X in case300 $C; do run "${X}_fresh5_refs" bash -c "$P 56_refs_fresh2.py $X fresh5 > logs/c3p_refs_fresh5_$X.log 2>&1"; done
run refs_check bash -c "$P 62_refs_check.py fresh5 > logs/c3p_refs_check.log 2>&1"
st "CHAIN DONE (stop: pre-registration, then 55 once per case on fresh5)"

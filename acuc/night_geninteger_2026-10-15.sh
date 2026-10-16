#!/bin/bash
# OVERNIGHT 1 (2026-10-15): generalized integer cuts, validation only (see RESULTS.md "GENERALIZED INTEGER CUTS"). One job at a time. Status logs/night_geninteger.status
cd "$(dirname "$0")"; P=~/.venvs/dfsm/bin/python3; S=logs/night_geninteger.status; export OMP_NUM_THREADS=1 DEPLOY_RHO=3e5
st() { echo "$1 $(date '+%m-%d %H:%M:%S')" >> "$S"; }
: > "$S"; st "CHAIN START"
for cfg in "none 0 0" "ham 0 0" "rows 8 0" "rows 2 0" "rows 32 0" "ham 0 2" "rows 8 2" "rows 32 2"; do
  set -- $cfg; name="$1_K$2_M$3"; st "START $name"
  $P 67_generalized_integer_cuts.py case118 $1 $2 $3 _n1 > logs/gi_n1_$name.log 2>&1; st "END $name exit $?"
done
st "CHAIN DONE"

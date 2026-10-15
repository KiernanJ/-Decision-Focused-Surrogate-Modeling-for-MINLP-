#!/bin/bash
# FULLY CLEAN case118 (2026-10-13): V-net = round-1 V trained with NO cuts (vnet_layer_case118_env_nc.pt); labels = the label-free best-of teacher AT that V
# (selfteacher_case118_bestof_extra_nc.json, 444 targets); cuts and rows trained AT that V with the hardened EXACT-envelope value loss (no inexact KKT gradient),
# c re-selected label-free; deployment rho = DEPLOY_RHO=3e5 (chosen by the label-free score of the full deployed method on the _lfc pipeline; re-checked below).
#   1. cuts (valueh, selc, rho 1e5) -> unit_cuts_layer_case118_soft_lfn.pt   2. rows (valueh) -> general_rows_case118_lfn.pt
#   3. rho sweep on the new pipeline: label-free score + four metrics (report-only)         STOP -> audit -> 55 case118 0,1,2 fresh5 _lfn _nc vseed0 (DEPLOY_RHO = argmin of the sweep)
cd "$(dirname "$0")"; P=~/.venvs/dfsm/bin/python3; S=logs/clean118.status; export OMP_NUM_THREADS=1
T=results/selfteacher_case118_bestof_extra_nc.json
st() { echo "$1 $(date '+%m-%d %H:%M:%S')" >> "$S"; }
run() { local name=$1; shift; st "START $name"; "$@"; st "END $name exit $?"; }
: > "$S"; st "CHAIN START"
run cuts bash -c "DEPLOY_RHO=3e5 $P 46_unit_cuts_layer.py soft case118 $T _lfn extra _nc selc valueh > logs/cl_cuts_118.log 2>&1"
[ -f results/unit_cuts_layer_case118_soft_lfn.pt ] || { st "ABORT (no cuts: gate failed or crash)"; exit 1; }
run rows bash -c "DEPLOY_RHO=3e5 $P 52_general_rows_on_top.py case118 $T _lfn extra _nc valueh > logs/cl_rows_118.log 2>&1"
[ -f results/general_rows_case118_lfn.pt ] || { st "ABORT (no rows)"; exit 1; }
for r in 1e4 3e4 1e5 3e5 1e6; do run "val_rho_$r" bash -c "$P 61_val_full.py case118 _env_nc _lfn rho_$r full none $r > logs/cl_val_rho_$r.log 2>&1"; done
st "CHAIN DONE (stop: audit, then the final test on fresh5)"

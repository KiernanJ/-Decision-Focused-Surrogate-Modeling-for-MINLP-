#!/bin/bash
# CLEAN LABEL-FREE REDO (2026-10-12). AC: nothing may come from a long solve, so per case:
#   V-net seed 0 on the UNCUT relaxation (40 ... nocuts) -> label-free best-of teacher at that V (60, train + 300 extra)
#   -> integer cuts with c re-selected label-free (46 ... _nc selc) -> general rows (52) -> V-nets seeds 1, 2 (uncut).
#   Outputs: vnet_layer_<case>[_sS]_env_nc.pt, selfteacher_<case>_bestof_{train,extra}_nc.json, unit_cuts_layer_<case>_soft_lfc.pt,
#   general_rows_<case>_lfc.pt.  Then vehicle (label-free mode, unaffected by the V-net issue) and the untouched test sets.
#   One job at a time. Status: acuc/logs/clean_redo.status. No final test here (audit first).
cd "$(dirname "$0")"; A=$(pwd); V="$A/../dixit_vehicle"; P=~/.venvs/dfsm/bin/python3; S="$A/logs/clean_redo.status"
export OMP_NUM_THREADS=1
st() { echo "$1 $(date '+%m-%d %H:%M:%S')" >> "$S"; }
run() { local name=$1; shift; st "START $name"; "$@"; st "END $name exit $?"; }
: > "$S"; st "CHAIN START"
for C in case118 case300 pglib_opf_case118_ieee__api; do
  W=10; [ "$C" = "case300" ] && W=30
  run "${C}_vnet_s0" bash -c "NOTEST=1 $P 40_vnet_layer.py $C $W 0 env nocuts > logs/cr_vnet_${C}_s0.log 2>&1"
  [ -f "results/vnet_layer_${C}_env_nc.pt" ] || { st "ABORT $C (no clean V-net)"; continue; }
  run "${C}_teacher_train" bash -c "$P 60_selfteacher.py $C train - _nc > logs/cr_teacher_train_$C.log 2>&1"
  run "${C}_teacher_extra" bash -c "$P 60_selfteacher.py $C extra results/selfteacher_${C}_bestof_train_nc.json _nc > logs/cr_teacher_extra_$C.log 2>&1"
  [ -f "results/selfteacher_${C}_bestof_extra_nc.json" ] || { st "ABORT $C (no teacher)"; continue; }
  run "${C}_cuts" bash -c "$P 46_unit_cuts_layer.py soft $C results/selfteacher_${C}_bestof_extra_nc.json _lfc extra _nc selc > logs/cr_cuts_$C.log 2>&1"
  [ -f "results/unit_cuts_layer_${C}_soft_lfc.pt" ] || { st "ABORT $C (no cuts: gate failed or crash)"; continue; }
  run "${C}_rows" bash -c "$P 52_general_rows_on_top.py $C results/selfteacher_${C}_bestof_extra_nc.json _lfc extra _nc > logs/cr_rows_$C.log 2>&1"
  for SD in 1 2; do run "${C}_vnet_s$SD" bash -c "NOTEST=1 $P 40_vnet_layer.py $C $W $SD env nocuts > logs/cr_vnet_${C}_s$SD.log 2>&1"; done
  st "CASE DONE $C"
done
run veh_train bash -c "cd '$V' && VEH_LF=1 $P 16_vehicle_final.py train > logs/cr_lf_train.log 2>&1"
for C in case118 case300 pglib_opf_case118_ieee__api; do
  run "${C}_fresh4_refs" bash -c "$P 56_refs_fresh2.py $C fresh4 > logs/cr_refs_fresh4_$C.log 2>&1"
done
run veh_fresh2_pool bash -c "cd '$V' && $P 19_fresh2_test_pool.py > logs/cr_fresh2_pool.log 2>&1"
st "ALL DONE"

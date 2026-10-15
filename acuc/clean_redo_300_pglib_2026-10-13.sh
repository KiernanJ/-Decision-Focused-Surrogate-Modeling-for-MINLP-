#!/bin/bash
# CLEAN LABEL-FREE REDO of case300 and PGLib-case118-api (2026-10-13), the recipe that worked on case118 (_lfc):
#   V-net seed 0 on the UNCUT relaxation (40 ... nocuts) -> label-free best-of teacher AT that V (60 train, 60 extra: 144 + 300 draws)
#   -> integer cuts, c re-selected label-free (46 ... _lfc extra _nc selc, tracking loss) -> general rows (52 ... _lfc extra _nc)
#   -> deployment-rho sweep over {1e4,3e4,1e5,3e5,1e6} with the label-free score of the FULL pipeline on ALL 24 validation draws (61)
#   -> references for the untouched test draw fresh5 (56) + 62_refs_check.   STOPS here: audit, pre-registration, then ONE test per case (55).
# V-net seed 0 only (vseed0): seeds 1,2 vary the supervised proxy only, as in the case118 finals. One job at a time. Status: logs/clean300pg.status
cd "$(dirname "$0")"; P=~/.venvs/dfsm/bin/python3; S=logs/clean300pg.status; export OMP_NUM_THREADS=1
st() { echo "$1 $(date '+%m-%d %H:%M:%S')" >> "$S"; }
run() { local name=$1; shift; st "START $name"; "$@"; st "END $name exit $?"; }
: > "$S"; st "CHAIN START"
for C in case300 pglib_opf_case118_ieee__api; do
  W=10; [ "$C" = "case300" ] && W=30
  T=results/selfteacher_${C}_bestof_extra_nc.json
  run "${C}_vnet_s0" bash -c "NOTEST=1 $P 40_vnet_layer.py $C $W 0 env nocuts > logs/c3p_vnet_${C}.log 2>&1"
  [ -f "results/vnet_layer_${C}_env_nc.pt" ] || { st "ABORT $C (no clean V-net)"; continue; }
  run "${C}_teacher_train" bash -c "$P 60_selfteacher.py $C train - _nc > logs/c3p_teacher_train_$C.log 2>&1"
  run "${C}_teacher_extra" bash -c "$P 60_selfteacher.py $C extra results/selfteacher_${C}_bestof_train_nc.json _nc > logs/c3p_teacher_extra_$C.log 2>&1"
  [ -f "$T" ] || { st "ABORT $C (no teacher)"; continue; }
  run "${C}_cuts" bash -c "$P 46_unit_cuts_layer.py soft $C $T _lfc extra _nc selc > logs/c3p_cuts_$C.log 2>&1"
  [ -f "results/unit_cuts_layer_${C}_soft_lfc.pt" ] || { st "ABORT $C (no cuts: gate failed or crash)"; continue; }
  run "${C}_rows" bash -c "$P 52_general_rows_on_top.py $C $T _lfc extra _nc > logs/c3p_rows_$C.log 2>&1"
  [ -f "results/general_rows_${C}_lfc.pt" ] || { st "ABORT $C (no rows)"; continue; }
  for r in 1e4 3e4 1e5 3e5 1e6; do run "${C}_val_rho_$r" bash -c "$P 61_val_full.py $C _env_nc _lfc rho_$r full none $r > logs/c3p_val_${C}_rho_$r.log 2>&1"; done
  st "CASE DONE $C"
done
for C in case300 pglib_opf_case118_ieee__api; do
  run "${C}_fresh5_refs" bash -c "$P 56_refs_fresh2.py $C fresh5 > logs/c3p_refs_fresh5_$C.log 2>&1"
done
run refs_check bash -c "$P 62_refs_check.py fresh5 > logs/c3p_refs_check.log 2>&1"
st "CHAIN DONE (stop: audit, pre-registration, then 55 once per case on fresh5)"

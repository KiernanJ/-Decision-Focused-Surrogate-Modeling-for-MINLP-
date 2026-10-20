#!/bin/bash
cd "$(dirname "$0")"
PY=${PY:-python3}
export OMP_NUM_THREADS=1
until grep -q "ALL DONE" log_300.txt; do sleep 15; done
echo STAGE3DONE > stage3.flag
for SET in val fresh5 fresh6; do $PY abl.py case118 $SET A12,A12i > log_${SET}_d.txt 2>&1; done
$PY abl.py case300 fresh5 A12 > log_300_d.txt 2>&1
echo STAGE5DONE > stage5.flag
for SET in val fresh5 fresh6; do $PY abl.py case118 $SET A9v > log_${SET}_c.txt 2>&1; done
echo STAGE4DONE > stage4.flag

#!/bin/bash
cd "$(dirname "$0")"
PY=${PY:-python3}
export OMP_NUM_THREADS=1
while [ ! -f stage1.flag ]; do sleep 20; done
for SET in val fresh5 fresh6; do $PY abl.py case118 $SET A4h,A5h,A6h,A7h,A0h,A6bh,A3ah > log_${SET}_h.txt 2>&1; done
echo STAGEHDONE > stageH.flag
for SET in val fresh5 fresh6; do $PY abl.py case118 $SET A9f,A11,A10 > log_${SET}_b.txt 2>&1; done
echo STAGE2DONE > stage2.flag
$PY abl.py case300 fresh5 A0,A8,A1,A2,A4,A4h,A3a,A6,A6h,A7,A7h > log_300.txt 2>&1
echo STAGE3DONE > stage3.flag
for SET in val fresh5 fresh6; do $PY abl.py case118 $SET A9v > log_${SET}_c.txt 2>&1; done
echo STAGE4DONE > stage4.flag

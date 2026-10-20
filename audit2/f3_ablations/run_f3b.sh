#!/bin/bash
cd "$(dirname "$0")"
PY=${PY:-python3}
export OMP_NUM_THREADS=1
while [ ! -f stage1.flag ]; do sleep 20; done
$PY abl.py case118 val A9f,A11,A10 > log_val_b.txt 2>&1
$PY abl.py case118 fresh5 A9f,A11,A10 > log_f5_b.txt 2>&1
$PY abl.py case118 fresh6 A9f,A11,A10 > log_f6_b.txt 2>&1
echo STAGE2DONE > stage2.flag
$PY abl.py case300 fresh5 A0,A8,A1,A2,A4,A3a,A6,A7 > log_300.txt 2>&1
echo STAGE3DONE > stage3.flag
$PY abl.py case118 val A9v > log_val_c.txt 2>&1
$PY abl.py case118 fresh5 A9v > log_f5_c.txt 2>&1
$PY abl.py case118 fresh6 A9v > log_f6_c.txt 2>&1
echo STAGE4DONE > stage4.flag

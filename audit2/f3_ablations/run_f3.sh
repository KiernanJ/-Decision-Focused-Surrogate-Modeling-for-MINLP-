#!/bin/bash
cd "$(dirname "$0")"
PY=${PY:-python3}
export OMP_NUM_THREADS=1
$PY abl.py case118 val A0,A8,A1,A2,A4,A3a,A3b,A5,A6,A6b,A7 > log_val.txt 2>&1
$PY abl.py case118 fresh5 A0,A8,A1,A2,A4,A3a,A3b,A5,A6,A6b,A7 > log_f5.txt 2>&1
$PY abl.py case118 fresh6 A0,A8,A1,A2,A4,A3a,A3b,A5,A6,A6b,A7 > log_f6.txt 2>&1
echo STAGE1DONE > stage1.flag

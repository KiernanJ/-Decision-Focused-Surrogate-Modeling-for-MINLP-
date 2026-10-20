#!/bin/bash
# Re-run the four pre-registered AC final tests of acuc/RESULTS.md (AUDIT2.md section 2).
# 55_final_eval.py appends to acuc/RESULTS.md and writes acuc/runs/, so this runs in a throwaway clone of the current commit.
# Usage: PY=<python with audit2/requirements-audit.txt> bash audit2/repro/run_repro.sh [pool size, default 6]
# Output: audit2/repro/logs_rerun/ (logs + runs/*/final.pkl); compare with audit2/repro/run_logs/ and audit2/repro/runs/.
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; PY="${PY:-python3}"; NP="${1:-6}"
OUT="$REPO/audit2/repro/logs_rerun"; mkdir -p "$OUT"
W="$(mktemp -d)/repo"; git clone -q "$REPO" "$W"; cd "$W/acuc"
# Only the worker-pool size changes (11 -> NP). Results do not depend on it: Gurobi runs Threads=1 with WorkLimit and Clarabel
# runs without warm start; wall-clock times do.
sed -i.bak "s/Pool(11,/Pool($NP,/g" 55_final_eval.py
export OMP_NUM_THREADS=1
DEPLOY_RHO=3e5 RETRY_RHO=1e4 RETRY_TAU=0.05 "$PY" 55_final_eval.py case118 0,1,2 fresh6 _lfc _nc vseed0 > "$OUT/repro_case118_fresh6.log" 2>&1
DEPLOY_RHO=3e5 "$PY" 55_final_eval.py case118 0,1,2 fresh5 _lfc _nc vseed0 > "$OUT/repro_case118_fresh5.log" 2>&1
DEPLOY_RHO=1e4 "$PY" 55_final_eval.py pglib_opf_case118_ieee__api 0,1,2 fresh5 _lfc _nc vseed0 > "$OUT/repro_pglib_fresh5.log" 2>&1
DEPLOY_RHO=1e4 "$PY" 55_final_eval.py case300 0,1,2 fresh5 _lfc _nc vseed0 > "$OUT/repro_case300_fresh5.log" 2>&1
cp -r "$W/acuc/runs" "$OUT/runs"
echo "done: logs and runs/*/final.pkl in $OUT (throwaway clone left at $W)"

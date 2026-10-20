cd "$(dirname "$0")"; P=${PY:-python3}; export OMP_NUM_THREADS=1
$P b_rows_activity.py case300 12 1e4 30 > b_rows_case300.log 2>&1
$P b_rows_activity.py pglib_opf_case118_ieee__api 24 1e4 10 > b_rows_pglib.log 2>&1
echo done > chain1.done

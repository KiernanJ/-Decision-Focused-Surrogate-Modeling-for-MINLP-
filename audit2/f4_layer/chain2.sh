cd "$(dirname "$0")"; P=${PY:-python3}; export OMP_NUM_THREADS=1
($P d_rows.py case118 20 3 > d_rows_case118.log 2>&1; $P d_rows.py case300 12 2 > d_rows_case300.log 2>&1; $P d_rows.py pglib_opf_case118_ieee__api 16 3 > d_rows_pglib.log 2>&1; echo done > chain2a.done) &
($P d_cuts.py case300 20 4 > d_cuts_case300.log 2>&1; $P d_cuts.py pglib_opf_case118_ieee__api 20 4 > d_cuts_pglib.log 2>&1; $P c2_vnet_loss.py case118 3e5 > c2_case118.log 2>&1; $P c2_vnet_loss.py case300 1e4 > c2_case300.log 2>&1; $P c2_vnet_loss.py pglib_opf_case118_ieee__api 1e4 > c2_pglib.log 2>&1; echo done > chain2b.done) &
wait

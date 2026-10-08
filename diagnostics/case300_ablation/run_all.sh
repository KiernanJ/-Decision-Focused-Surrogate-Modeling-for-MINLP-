#!/bin/sh
# Run the full case300 ablation (69 jobs, ~25 CPU-hours; ~2.5-3 h on 10 cores),
# then write out/analysis.txt. Resumable: jobs whose output file already exists
# are skipped, so it is safe to stop (pkill -f abl300.py) and re-run.
#
#   sh run_all.sh [N_PARALLEL]        # default 10;  PY=/path/to/python sh run_all.sh
cd "$(dirname "$0")" || exit 1
N=${1:-10}
PY=${PY:-python}
export OMP_NUM_THREADS=1 PY
mkdir -p out

{
  echo "refs"
  echo "val flat_nocut 0"
  for a in pred_cut pred_nocut flat_cut proxy; do for s in 0 1 2 3; do echo "val $a $s"; done; done
  for t in 0.8 0.65 0.5; do
    echo "test flat_nocut 0 $t"
    for a in pred_cut pred_nocut flat_cut proxy; do for s in 0 1 2 3; do echo "test $a $s $t"; done; done
  done
} | while read -r mode arm seed thr; do
  case $mode in
    refs) f=out/refs.json ;;
    val)  f=out/val_${arm}_s${seed}.json ;;
    test) f=out/test_${arm}_s${seed}_t${thr}.json ;;
  esac
  # no trailing blank: xargs -L treats a line ending in a blank as continued
  [ -f "$f" ] || echo "$mode${arm:+ $arm}${seed:+ $seed}${thr:+ $thr}"
done > out/jobs_todo.txt

echo "$(wc -l < out/jobs_todo.txt | tr -d ' ') jobs to run, $N in parallel"
xargs -P "$N" -L 1 sh -c '$PY abl300.py "$@" > "out/log_$(echo "$@" | tr " " "_").txt" 2>&1' _ < out/jobs_todo.txt
grep -l Traceback out/log_*.txt && echo "some jobs failed (see logs above)"
$PY analyse300.py > out/analysis.txt 2>&1 && echo "wrote out/analysis.txt"

cd "$(dirname "$0")/../.." && OMP_NUM_THREADS=1 ${PY:-python3} gradient_checks/03_conelayer_check.py && OMP_NUM_THREADS=1 ${PY:-python3} gradient_checks/degeneracy.py

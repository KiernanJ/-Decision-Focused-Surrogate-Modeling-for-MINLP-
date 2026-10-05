"""Sparse LU in a PERSISTENT helper process, spoken to over pipes.

SuperLU on this machine aborts the process after enough factorisations
(cblas_dtrsv / cblas_dgemv / SIGSEGV). It is an abort, not an exception, so the
caller cannot catch it -- it killed training runs at a fixed epoch, and a
checkpoint-resume died at the same epoch every time.

Both multiprocessing routes failed here: fork deadlocked at 0% CPU (the parent
holds clarabel/torch state) and spawn re-imports __main__ and re-runs the
training script. This helper is launched with subprocess, imports only
numpy/scipy, and is simply RESTARTED when it dies -- the abort then costs one
factorisation instead of the whole run.

Protocol, little-endian, on stdin/stdout:
    request :  n_rows n_cols nnz len_r   then data, indices, indptr, r
    reply   :  b"OK" + x  |  b"ER" + message
"""
import struct
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _read_exact(f, n):
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        hdr = _read_exact(fin, 32)
        if hdr is None:
            return
        nr, nc, nnz, lr = struct.unpack("<4q", hdr)
        data = np.frombuffer(_read_exact(fin, 8*nnz), dtype=np.float64)
        ind = np.frombuffer(_read_exact(fin, 4*nnz), dtype=np.int32)
        indptr = np.frombuffer(_read_exact(fin, 4*(nc+1)), dtype=np.int32)
        r = np.frombuffer(_read_exact(fin, 8*lr), dtype=np.float64)
        try:
            M = sp.csc_matrix((data, ind, indptr), shape=(nr, nc))
            x = spla.splu(M).solve(r)
            fout.write(b"OK"); fout.write(np.ascontiguousarray(x, np.float64).tobytes())
        except Exception as e:
            msg = str(e).encode()[:200]
            fout.write(b"ER"); fout.write(struct.pack("<q", len(msg))); fout.write(msg)
        fout.flush()


if __name__ == "__main__":
    main()

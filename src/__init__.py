import os

# Cap BLAS threads before numpy loads: avoids OpenBLAS buffer-allocation failures when several
# processes start at once (DataLoader / ProcessPool workers) and stops CPU oversubscription.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

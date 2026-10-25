"""Windows file-locking test for the results CSV (CPU, ~1 min).

  python tests/test_csv_concurrency.py

Parallel queue windows append to one classifier_runs.csv while other processes read it, and Windows refuses
to replace a file that any process has open (pipeline status, another window, Excel). Every append must land
exactly once, and a long hold must make the append wait and say why instead of failing the run.
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WRITER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.classify.train import ROW_COLS, append_rows
w, n, csv = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
if len(sys.argv) > 5:  # handshake: imports done, append on the go line
    print("ready", flush=True)
    sys.stdin.readline()
for i in range(n):
    append_rows(csv, [{**dict.fromkeys(ROW_COLS), "run_id": f"w{w}_{i}", "dataset": ds} for ds in ("aptos_test", "idrid")])
"""
READER = r"""
import sys, time
csv, hold, end = sys.argv[1], float(sys.argv[2]), time.time() + float(sys.argv[3])
while time.time() < end:
    try:
        with open(csv) as f:
            f.read()
            time.sleep(hold)
    except OSError:
        pass
    time.sleep(0.03)
"""
HOLDER = "import sys, time; f = open(sys.argv[1]); print('held', flush=True); time.sleep(float(sys.argv[2]))"
LOCKED_READER = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.classify.train import run_ids
csv, end = Path(sys.argv[2]), time.time() + float(sys.argv[3])
while time.time() < end:
    run_ids(csv)
"""


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="csv_concurrency_"))
    csv, py, n, writers = tmp / "classifier_runs.csv", sys.executable, 8, 4
    readers = [subprocess.Popen([py, "-c", READER, str(csv), "0.02", "300"]),  # unlocked reads: pipeline status, Excel refresh
               subprocess.Popen([py, "-c", LOCKED_READER, str(ROOT), str(csv), "300"])]  # another window's queue/train
    try:
        procs = [subprocess.Popen([py, "-c", WRITER, str(ROOT), str(w), str(n), str(csv)], stderr=subprocess.PIPE, text=True)
                 for w in range(writers)]
        errs = [p.communicate(timeout=600)[1] for p in procs]
    finally:
        for r in readers:
            r.kill()
    assert all(p.returncode == 0 for p in procs), errs
    df = pd.read_csv(csv)
    assert len(df) == writers * n * 2 and not df.duplicated(["run_id", "dataset"]).any(), df

    # the CSV open in another program for 6 s (Excel keeps it open): the append waits, says why, then lands
    p = subprocess.Popen([py, "-c", WRITER, str(ROOT), "x", "1", str(csv), "handshake"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    holder = subprocess.Popen([py, "-c", HOLDER, str(csv), "6"], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    t0 = time.time()
    _, err = p.communicate("go\n", timeout=120)
    holder.wait()
    assert p.returncode == 0 and "open in another program" in err and time.time() - t0 > 4, err
    assert len(pd.read_csv(csv)) == writers * n * 2 + 2
    shutil.rmtree(tmp, ignore_errors=True)
    print("CSV_CONCURRENCY_OK")


if __name__ == "__main__":
    main()

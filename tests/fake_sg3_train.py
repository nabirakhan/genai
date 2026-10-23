"""Stand-in for stylegan3 train.py used by tests/test_kaggle_train.py.

Accepts the same flags, creates <outdir>/<id>-stylegan2-fake-<desc>/, prints tick lines and writes
network-snapshot-<kimg>.pkl every tick*snap kimg (counted from 0, like stylegan3 on resume) plus a
final one at --kimg. Speed comes from FAKE_SEC_PER_KIMG (default 0.01 s).
"""
import os
import sys
import time
from pathlib import Path

args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
out = Path(args["outdir"])
out.mkdir(parents=True, exist_ok=True)
run = out / f"{len([d for d in out.iterdir() if d.is_dir()]):05d}-stylegan2-fake-{args['desc']}"
run.mkdir()
total, tick, snap = int(args["kimg"]), int(args["tick"]), int(args["snap"])
sec = float(os.environ.get("FAKE_SEC_PER_KIMG", "0.01"))
print(f"resume {args['resume']} data {args['data']} aug {args['aug']}", flush=True)
kimg = 0
while True:
    done = kimg >= total
    if kimg % (tick * snap) == 0 or done:
        p = run / f"network-snapshot-{kimg:06d}.pkl"
        p.write_bytes(b"x" * 1000 + b".")
        (run / f"fakes{kimg:06d}.png").write_bytes(b"png")
    if done:
        break
    time.sleep(sec * tick)
    kimg += tick
    print(f"tick {kimg // tick} kimg {kimg:.1f} sec/kimg {sec}", flush=True)
print("Exiting...", flush=True)

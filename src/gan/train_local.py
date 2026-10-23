"""Train one fraction's conditional StyleGAN2-ADA on the lab PC, mirroring the Kaggle notebook.

  python -m src.gan.train_local --fraction f100 --aug ada --kimg 1000 --hours 3
  python -m src.gan.train_local --fraction f010 --aug noaug --kimg 1000 --hours 2 --resume auto

Needs stylegan3's CUDA kernels to compile (nvcc + a C++ compiler; see LAB_RUNBOOK.md). Runs are
written to <GENAI_DATA>/gan_runs/<id>-...-<fraction>-<aug>-from<kimg>/ and resume from the latest
snapshot of the same fraction/aug (--resume auto). Stops after --hours at the next snapshot and
keeps only the last two snapshots, so a session can end any time. --kimg is the TOTAL target.
"""
import glob
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import click

from src.genai.paths import DATA_ROOT, SG3, STYLEGAN3

FFHQ = ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/"
        "ffhq-res256-mirror-paper256-noaug.pkl")
RUNS = DATA_ROOT / "gan_runs"


def total_kimg(path: str) -> int:
    m = re.search(r"-from(\d+)[/\\].*snapshot-(\d+)", path)
    return int(m.group(1)) + int(m.group(2)) if m else 0


def latest_snapshot(fraction: str, aug: str) -> str | None:
    snaps = [s for s in glob.glob(str(RUNS / "**" / "network-snapshot-*.pkl"), recursive=True) if f"-{fraction}-{aug}-from" in s]
    return max(snaps, key=total_kimg) if snaps else None


@click.command()
@click.option("--fraction", type=click.Choice(["f010", "f025", "f050", "f100"]), required=True)
@click.option("--aug", type=click.Choice(["ada", "noaug"]), default="ada")
@click.option("--kimg", type=int, default=1000, help="total kimg target across sessions")
@click.option("--hours", type=float, default=3.0, help="wall-clock budget for this session")
@click.option("--resume", default="auto", help="auto | ffhq | path to a snapshot")
@click.option("--batch", type=int, default=32)
@click.option("--batch-gpu", type=int, default=16)
@click.option("--snap", type=int, default=10, help="snapshot every N ticks (tick = 4 kimg)")
@click.option("--gamma", type=float, default=1.0)
def main(fraction, aug, kimg, hours, resume, batch, batch_gpu, snap, gamma):
    data = SG3 / f"aptos256_{fraction}.zip"
    if not data.exists():
        raise SystemExit(f"{data} missing: run stage S6 first")
    RUNS.mkdir(parents=True, exist_ok=True)
    if resume == "ffhq":
        resume_pkl = FFHQ
    elif resume == "auto":
        resume_pkl = latest_snapshot(fraction, aug) or FFHQ
    else:
        resume_pkl = resume
    start = 0 if resume_pkl == FFHQ else total_kimg(resume_pkl)
    remain = max(0, kimg - start)
    click.echo(f"resume from {resume_pkl}\nalready trained {start} kimg; this session trains up to {remain} kimg, {hours} h budget")
    if remain == 0:
        click.echo("target reached")
        return
    if resume_pkl == FFHQ:
        subprocess.run([sys.executable, "scripts/check_resume_compat.py", "--pkl", FFHQ], check=True)

    desc = f"{fraction}-{aug}-from{start}"
    cmd = [sys.executable, str(STYLEGAN3 / "train.py"), f"--outdir={RUNS}", "--cfg=stylegan2", f"--data={data}", "--cond=1",
           "--gpus=1", f"--batch={batch}", f"--batch-gpu={batch_gpu}", f"--gamma={gamma}", "--cbase=16384", "--cmax=512",
           "--map-depth=8", "--mbstd-group=8", "--mirror=1", f"--aug={aug}", "--target=0.6", f"--kimg={remain}", "--tick=4",
           f"--snap={snap}", "--metrics=none", "--workers=2", f"--resume={resume_pkl}", f"--desc={desc}"]
    click.echo(" ".join(cmd))

    def run_dir():
        ds = sorted(glob.glob(str(RUNS / f"*-{desc}")))
        return ds[-1] if ds else None

    def snapshots():
        d = run_dir()
        return sorted(glob.glob(os.path.join(d, "network-snapshot-*.pkl"))) if d else []

    def prune(keep=2):
        for s in snapshots()[:-keep]:
            os.remove(s)

    log = open(RUNS / f"train_{desc}.log", "a")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **kw)
    t0 = time.time()
    deadline = t0 + hours * 3600
    stop_after = None
    last_print = 0.0
    try:
        while p.poll() is None:
            time.sleep(30)
            prune()
            if time.time() - last_print > 600:
                last_print = time.time()
                d = run_dir()
                tail = ""
                if d and os.path.exists(os.path.join(d, "log.txt")):
                    with open(os.path.join(d, "log.txt"), errors="replace") as f:
                        tail = f.read()[-300:].strip().splitlines()[-1:] or [""]
                        tail = tail[0]
                click.echo(f"[{(time.time() - t0) / 3600:.2f} h] {tail}")
            if stop_after is None and time.time() > deadline:
                stop_after = snapshots()[-1] if snapshots() else ""
                click.echo("time budget reached; stopping after the next snapshot")
            if stop_after is not None and snapshots() and snapshots()[-1] != stop_after:
                time.sleep(45)
                break
            if stop_after is not None and time.time() > deadline + 50 * 60:
                click.echo("no new snapshot within 50 min; stopping")
                break
    except KeyboardInterrupt:
        click.echo("interrupted; the last snapshot is kept")
    finally:
        if p.poll() is None:
            if os.name == "nt":
                p.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(5)
            p.terminate()
        log.close()
        prune()
    snaps = snapshots()
    if snaps:
        click.echo(f"latest snapshot: {snaps[-1]}  (total {total_kimg(snaps[-1])} kimg)")
    else:
        click.echo("no snapshot written; check the log for errors (CUDA kernels compiled?)")


if __name__ == "__main__":
    main()

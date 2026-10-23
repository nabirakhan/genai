"""One Kaggle GAN-training session over the run queue (driven by notebooks/kaggle_gan_train.ipynb).

  python -m src.gan.kaggle_train                    # settings from notebooks/kaggle_session.json
  python -m src.gan.kaggle_train --hours 3 --dry-run

Queue: f100-ada -> f010-ada -> f010-noaug -> f025-ada -> f050-ada, each trained to --kimg in total.
A session attaches the genai-data-prep output and the previous version's output of the train
notebook. The runner continues the first unfinished entry from its newest complete snapshot (the
FFHQ-256 pickle for an entry that never started) and moves on to the next entry while at least
--min-start minutes of the --hours budget remain. After the deadline it stops at the next snapshot.

Safety rules:
- No silent restarts: without an attached previous output (no ledger.json, no snapshots) the session
  refuses to start unless --first-session; an entry that ledger.json records as started is never
  restarted from FFHQ because its snapshot is missing.
- Data provenance: f100 trains only on a zip holding exactly the committed split's train ids;
  f010-f050 only on a zip from a data-prep output whose splits/aptos_splits.csv equals the committed one.
- Finished GANs are carried forward as final/<tag>_kimg<NNNN>.pkl and ledger.json records every
  entry's kimg, so the newest version's output is all the next session and the eval notebook need.
"""
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import click
import pandas as pd

from src.data.split import digest
from src.genai.paths import SPLITS

REPO = Path(__file__).resolve().parents[2]
SESSION_JSON = REPO / "notebooks" / "kaggle_session.json"
QUEUE = ("f100-ada", "f010-ada", "f010-noaug", "f025-ada", "f050-ada")
FFHQ = ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/"
        "ffhq-res256-mirror-paper256-noaug.pkl")
RUN_RE = re.compile(r"-(f\d{3}-(?:ada|noaug))-from(\d+)$")
SNAP_RE = re.compile(r"^network-snapshot-(\d+)\.pkl$")
FINAL_RE = re.compile(r"^(f\d{3}-(?:ada|noaug))_kimg(\d+)\.pkl$")
DEFAULTS = dict(kimg=1000, hours=10.0, run="auto", min_start=35.0, grace=50.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pickle_complete(p: Path) -> bool:
    """A finished pickle ends with the STOP opcode; a snapshot cut off mid-write almost never does."""
    try:
        with open(p, "rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) == b"."
    except OSError:
        return False


def pickle_tag(p: Path) -> tuple[str, int] | None:
    m = FINAL_RE.match(p.name)
    if m:
        return m.group(1), int(m.group(2))
    s, r = SNAP_RE.match(p.name), RUN_RE.search(p.parent.name)
    return (r.group(1), int(r.group(2)) + int(s.group(1))) if s and r else None


def scan(roots: list[Path]) -> tuple[dict[str, tuple[int, Path]], dict[str, int]]:
    """Newest complete pickle per tag ({tag: (total_kimg, path)}) and the merged ledger.json."""
    best, ledger = {}, {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pkl"):
            hit = pickle_tag(p)
            if hit is None:
                continue
            tag, total = hit
            if (tag not in best or total > best[tag][0]) and pickle_complete(p):
                best[tag] = (total, p)
        for p in root.rglob("ledger.json"):
            for tag, total in json.loads(p.read_text()).items():
                ledger[tag] = max(ledger.get(tag, 0), int(total))
    return best, ledger


def committed_split() -> pd.DataFrame | None:
    p = SPLITS / "aptos_splits.csv"
    return pd.read_csv(p) if p.exists() else None


def find_zip(name: str, roots: list[Path], committed: pd.DataFrame | None) -> tuple[Path | None, str]:
    """aptos256_<name>.zip whose provenance matches the committed split, else (None, reason)."""
    reasons = []
    for z in sorted(p for r in roots if r.exists() for p in r.rglob(f"aptos256_{name}.zip")):
        csv = z.parent.parent / "splits" / "aptos_splits.csv"
        if name == "f100" and committed is not None:
            names = {Path(n).stem for n, _ in json.loads(zipfile.ZipFile(z).read("dataset.json"))["labels"]}
            train = set(committed.loc[committed.split == "train", "image_id"])
            if names == train:
                return z, "f100 zip == committed train split"
            reasons.append(f"{z}: {len(names)} images vs {len(train)} committed train ids ({len(names ^ train)} differ)")
        elif committed is None:
            if name == "f100":
                return z, "WARNING: no committed split in the repo, f100 zip unchecked"
            reasons.append("no committed splits/aptos_splits.csv in the repo")
        elif not csv.exists():
            reasons.append(f"{z}: no splits/aptos_splits.csv beside it")
        elif (d := digest(pd.read_csv(csv))) != digest(committed):
            reasons.append(f"{z}: built from split {d}, committed split is {digest(committed)}")
        else:
            return z, f"split digest {d} == committed"
    return None, "; ".join(reasons) or f"aptos256_{name}.zip not found under {', '.join(map(str, roots))}"


def carry_forward(best: dict, kimg: int, work: Path) -> None:
    """Copy each finished GAN to work/final and drop its now-redundant run-dir snapshots in work."""
    for tag, (total, src) in best.items():
        if total < kimg:
            continue
        dst = work / "final" / f"{tag}_kimg{total:04d}.pkl"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            log(f"final: {dst.name} <- {src}")
        for d in (work / "runs").glob(f"*-{tag}-from*"):
            for s in d.glob("network-snapshot-*.pkl"):
                s.unlink()


def prune(run_dir: Path, keep_snaps: int = 2, keep_fakes: int = 3) -> None:
    for pattern, keep in (("network-snapshot-*.pkl", keep_snaps), ("fakes[0-9]*.png", keep_fakes)):
        for p in sorted(run_dir.glob(pattern))[:-keep]:
            p.unlink(missing_ok=True)


def last_line(path: Path) -> str:
    try:
        lines = [l for l in path.read_text(errors="replace").replace("\r", "\n").splitlines() if l.strip()]
        return lines[-1][-240:] if lines else ""
    except OSError:
        return ""


def stop(p: subprocess.Popen, why: str, pause: float) -> None:
    log(why)
    try:
        if os.name == "nt":
            p.terminate()
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        p.wait(timeout=90)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        p.kill()
    except ProcessLookupError:
        pass
    time.sleep(pause)  # let the GPUs release memory before the next entry starts


def train(tag: str, start: int, resume: str, data: Path, kimg: int, deadline: float, o: dict) -> tuple[int, bool]:
    """Run train.py for one entry under the session deadline. Returns (exit code, stopped by the deadline)."""
    aug = tag.split("-")[1]
    desc = f"{tag}-from{start}"
    work = o["work"]
    base = shlex.split(o["train_cmd"]) if o["train_cmd"] else [sys.executable, str(REPO / "third_party/stylegan3/train.py")]
    cmd = base + [f"--outdir={work / 'runs'}", "--cfg=stylegan2", f"--data={data}", "--cond=1", f"--gpus={o['gpus']}",
                  f"--batch={o['batch']}", f"--batch-gpu={o['batch_gpu']}", f"--gamma={o['gamma']}", "--cbase=16384",
                  "--cmax=512", "--map-depth=8", "--mbstd-group=8", "--mirror=1", f"--aug={aug}", "--target=0.6",
                  f"--kimg={kimg - start}", "--tick=4", f"--snap={o['snap']}", "--metrics=none", "--workers=2",
                  f"--resume={resume}", f"--desc={desc}"]
    log("$ " + " ".join(cmd))
    log_path = work / f"train_{desc}.log"

    def run_dir() -> Path | None:
        ds = sorted((work / "runs").glob(f"*-{desc}"))
        return ds[-1] if ds else None

    def snaps() -> list[Path]:
        d = run_dir()
        return sorted(d.glob("network-snapshot-*.pkl")) if d else []

    kw = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else dict(start_new_session=True)
    stopped, stop_after, t_print = False, None, 0.0
    with open(log_path, "a") as lf:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, **kw)
        while p.poll() is None:
            time.sleep(o["poll"])
            if (d := run_dir()) is not None:
                prune(d)
            now = time.time()
            if now - t_print >= o["print_every"]:
                t_print = now
                log(f"{tag}: {last_line(log_path)}")
            s = snaps()
            if stop_after is None and now > deadline:
                stop_after = s[-1].name if s else ""
                log(f"{tag}: time budget reached; stopping after the next snapshot")
            if stop_after is not None and s and s[-1].name != stop_after:
                size = s[-1].stat().st_size
                time.sleep(min(10.0, o["poll"]))
                if s[-1].stat().st_size == size and pickle_complete(s[-1]):
                    stop(p, f"{tag}: stopped after {s[-1].name}", min(5.0, o["poll"]))
                    stopped = True
                    break
            if stop_after is not None and now > deadline + o["grace"] * 60:
                stop(p, f"{tag}: no new snapshot within {o['grace']:.0f} min of the deadline; stopped", min(5.0, o["poll"]))
                stopped = True
                break
        rc = p.wait()
    if (d := run_dir()) is not None:
        prune(d)
    log(f"{tag}: exit code {rc}; last log line: {last_line(log_path)}")
    return rc, stopped


def status(best: dict, kimg: int) -> str:
    rows = []
    for t in QUEUE:
        total = best.get(t, (0, None))[0]
        rows.append(f"  {t:<11} {total:>5} / {kimg} kimg  {'done' if total >= kimg else 'running' if total else 'not started'}")
    return "\n".join(rows)


@click.command()
@click.option("--kimg", type=int, default=None, help="total kimg per GAN (default: kaggle_session.json)")
@click.option("--hours", type=float, default=None, help="training budget of this session")
@click.option("--run", "only", default=None, help="'auto' = queue order, or one tag such as f010-ada")
@click.option("--min-start", type=float, default=None, help="minutes left needed to start an entry")
@click.option("--grace", type=float, default=None, help="minutes after the deadline to wait for a snapshot")
@click.option("--first-session", is_flag=True, help="allow starting without any previous output attached")
@click.option("--gpus", type=int, default=2)
@click.option("--batch", type=int, default=32)
@click.option("--batch-gpu", type=int, default=16)
@click.option("--gamma", type=float, default=1.0)
@click.option("--snap", type=int, default=10)
@click.option("--input-root", type=click.Path(path_type=Path), default=Path("/kaggle/input"))
@click.option("--work", type=click.Path(path_type=Path), default=Path("/kaggle/working"))
@click.option("--poll", type=float, default=30.0)
@click.option("--print-every", type=float, default=600.0)
@click.option("--train-cmd", default=None, hidden=True, help="test hook: replaces 'python train.py'")
@click.option("--skip-compat", is_flag=True, hidden=True)
@click.option("--dry-run", is_flag=True)
def main(kimg, hours, only, min_start, grace, first_session, gpus, batch, batch_gpu, gamma, snap, input_root, work,
         poll, print_every, train_cmd, skip_compat, dry_run):
    cfg = {**DEFAULTS, **(json.loads(SESSION_JSON.read_text()) if SESSION_JSON.exists() else {})}
    kimg = kimg if kimg is not None else int(cfg["kimg"])
    hours = hours if hours is not None else float(cfg["hours"])
    only = only or cfg["run"]
    min_start = min_start if min_start is not None else float(cfg["min_start"])
    grace = grace if grace is not None else float(cfg["grace"])
    first_session = first_session or bool(cfg.get("first_session", False))
    deadline = time.time() + hours * 3600
    o = dict(work=work, gpus=gpus, batch=batch, batch_gpu=batch_gpu, gamma=gamma, snap=snap, poll=poll,
             print_every=print_every, grace=grace, train_cmd=train_cmd)
    (work / "runs").mkdir(parents=True, exist_ok=True)
    roots = [input_root, work]
    best, ledger = scan(roots)
    log(f"session: kimg={kimg} hours={hours} run={only} min_start={min_start} first_session={first_session}")
    log("attached state:\n" + status(best, kimg))
    if not best and not ledger and not first_session:
        raise SystemExit("REFUSED: no previous output of this notebook is attached (no ledger.json, no snapshots). "
                         "Attach the newest version's output, or set first_session for the very first session.")
    for tag, k in sorted(ledger.items()):
        if k > best.get(tag, (0, None))[0]:
            raise SystemExit(f"REFUSED: ledger.json records {tag} at {k} kimg but no complete pickle that far is "
                             "attached; attach the newest version's output instead of restarting it")
    queue = QUEUE if only == "auto" else (only,)
    assert all(t in QUEUE for t in queue), f"unknown run {only}; choose from {QUEUE}"
    carry_forward(best, kimg, work)
    (work / "ledger.json").write_text(json.dumps({t: v[0] for t, v in sorted(best.items())}, indent=1))
    committed = committed_split()
    compat_done = skip_compat
    progressed, exit_code = False, 0
    while True:
        todo = [t for t in queue if best.get(t, (0, None))[0] < kimg]
        if not todo:
            log("QUEUE_DONE")
            break
        tag = todo[0]
        left = (deadline - time.time()) / 60
        if left < min_start:
            log(f"NEXT_SESSION: {left:.0f} min left < {min_start:.0f}; {tag} continues next session")
            break
        data, why = find_zip(tag.split("-")[0], [input_root], committed)
        if data is None:
            log(f"REFUSED {tag}: {why}")
            break
        start, src = best.get(tag, (0, None))
        resume = str(src) if start > 0 else FFHQ
        log(f"TRAIN {tag}: {start} -> {kimg} kimg, resume {resume}\n           data {data} ({why})")
        if dry_run:
            break
        if resume == FFHQ and not compat_done:
            subprocess.run([sys.executable, str(REPO / "scripts/check_resume_compat.py"), "--pkl", FFHQ], cwd=REPO, check=True)
            compat_done = True
        rc, stopped = train(tag, start, resume, data, kimg, deadline, o)
        best, _ = scan(roots)
        carry_forward(best, kimg, work)
        (work / "ledger.json").write_text(json.dumps({t: v[0] for t, v in sorted(best.items())}, indent=1))
        reached = best.get(tag, (0, None))[0]
        progressed = progressed or reached > start
        if reached >= kimg:
            log(f"DONE {tag} at {reached} kimg")
            continue
        if stopped:
            log(f"NEXT_SESSION: {tag} at {reached} kimg")
            break
        log(f"TRAIN_FAILED {tag}: exit code {rc} at {reached} kimg; see {work}/train_{tag}-from{start}.log")
        exit_code = 3
        break
    log("final state:\n" + status(best, kimg))
    if not progressed and exit_code == 0 and not dry_run and any(best.get(t, (0, None))[0] < kimg for t in queue):
        exit_code = 4  # refused before training anything
    log(f"RUNNER_EXIT {exit_code} (0 ok, 3 train failed, 4 nothing trained)")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

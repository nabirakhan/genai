"""Run the 84-run grid, skipping runs already in results/classifier_runs.csv.

  python -m src.classify.queue --dry-run          # list what would run
  python -m src.classify.queue                    # run everything that is pending
  python -m src.classify.queue --condition 5 --fraction 1.0

Order: conditions 1-5 at 100% first (week-2 baselines), then 1-5 at 50/25/10%, then condition 6
at ratio 1 for every fraction, then the ratio sweep (0.5x, 2x at 10% and 100%).
Condition-6 runs are skipped with a message until the matching synth/f{ppp}/ folder exists.
Each run is a subprocess so a crash or OOM does not take the queue down. Ctrl+C stops cleanly
after the current run (its last.pt checkpoint lets it resume next session).
"""
import subprocess
import sys
import time

import click
import pandas as pd

from src.classify.data import synth_files
from src.classify.train import RESULTS_CSV, run_id_of
from src.data.split import fraction_mask, load_splits
from src.genai.paths import FRACTIONS, RARE_GRADES, frac_tag

SEEDS = (0, 1, 2)
RATIO_SWEEP = (0.5, 2.0)
RATIO_SWEEP_FRACTIONS = (0.1, 1.0)
KEEP_WEIGHTS = {run_id_of(5, 1.0, 0.0, 0), run_id_of(6, 1.0, 1.0, 0)}


def grid() -> list[dict]:
    runs = []
    for frac in (1.0, 0.5, 0.25, 0.1):
        for cond in range(1, 6):
            for seed in SEEDS:
                runs.append(dict(condition=cond, fraction=frac, ratio=0.0, seed=seed))
    for frac in (1.0, 0.5, 0.25, 0.1):
        for seed in SEEDS:
            runs.append(dict(condition=6, fraction=frac, ratio=1.0, seed=seed))
    for frac in RATIO_SWEEP_FRACTIONS:
        for ratio in RATIO_SWEEP:
            for seed in SEEDS:
                runs.append(dict(condition=6, fraction=frac, ratio=ratio, seed=seed))
    for r in runs:
        r["run_id"] = run_id_of(r["condition"], r["fraction"], r["ratio"], r["seed"])
    assert len(runs) == 84, len(runs)
    return runs


def done_ids(results_csv) -> set[str]:
    if not results_csv.exists():
        return set()
    df = pd.read_csv(results_csv, usecols=["run_id", "dataset"])
    return set(df[df.dataset == "aptos_test"].run_id)


def synth_ready(fraction: float, ratio: float) -> str | None:
    try:
        df = load_splits()
        tr = df[fraction_mask(df, fraction)]
        synth_files(fraction, ratio, {g: int((tr.grade == g).sum()) for g in RARE_GRADES})
        return None
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:  # splits missing etc.
        return f"{type(e).__name__}: {e}"


@click.command()
@click.option("--dry-run", is_flag=True)
@click.option("--condition", type=int, default=None)
@click.option("--fraction", type=float, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--epochs", type=int, default=20)
@click.option("--workers", type=int, default=6)
@click.option("--batch-size", type=int, default=64)
@click.option("--results-csv", default=str(RESULTS_CSV))
@click.option("--extra", default="", help="extra args passed to train.py, e.g. '--no-pretrained'")
def main(dry_run, condition, fraction, seed, epochs, workers, batch_size, results_csv, extra):
    from pathlib import Path
    results_csv = Path(results_csv)
    runs = grid()
    if condition is not None:
        runs = [r for r in runs if r["condition"] == condition]
    if fraction is not None:
        runs = [r for r in runs if abs(r["fraction"] - fraction) < 1e-9]
    if seed is not None:
        runs = [r for r in runs if r["seed"] == seed]
    done = done_ids(results_csv)
    pending = [r for r in runs if r["run_id"] not in done]
    click.echo(f"{len(runs)} selected, {len(runs) - len(pending)} done, {len(pending)} pending")
    skipped_synth = {}
    todo = []
    for r in pending:
        if r["condition"] == 6:
            msg = synth_ready(r["fraction"], r["ratio"])
            if msg:
                skipped_synth[(frac_tag(r["fraction"]), r["ratio"])] = msg
                continue
        todo.append(r)
    for key, msg in skipped_synth.items():
        click.echo(f"  waiting for synthetic images {key}: {msg}")
    if dry_run:
        for r in todo:
            click.echo(f"  {r['run_id']}")
        return
    durations = []
    for i, r in enumerate(todo):
        cmd = [sys.executable, "-m", "src.classify.train", "--condition", str(r["condition"]), "--fraction", str(r["fraction"]),
               "--ratio", str(r["ratio"]), "--seed", str(r["seed"]), "--epochs", str(epochs), "--workers", str(workers),
               "--batch-size", str(batch_size), "--results-csv", str(results_csv)]
        if r["run_id"] in KEEP_WEIGHTS:
            cmd.append("--keep-weights")
        cmd += extra.split()
        eta = (sum(durations) / len(durations)) * (len(todo) - i) / 60 if durations else float("nan")
        click.echo(f"\n=== [{i + 1}/{len(todo)}] {r['run_id']}  (ETA {eta:.0f} min) ===")
        t0 = time.time()
        try:
            rc = subprocess.run(cmd).returncode
        except KeyboardInterrupt:
            click.echo("\ninterrupted; the current run can resume from its last.pt next time")
            return
        durations.append(time.time() - t0)
        if rc != 0:
            click.echo(f"!! {r['run_id']} exited with {rc}; continuing")
    click.echo("queue finished")


if __name__ == "__main__":
    main()

"""Run the 84-run grid, skipping runs already in results/classifier_runs.csv.

  python -m src.classify.queue --dry-run          # list what would run
  python -m src.classify.queue                    # run everything that is pending
  python -m src.classify.queue --condition 5 --fraction 1.0
  python -m src.classify.queue --checks magnitude # pre-registered check runs (LAB_RUNBOOK S8a / S10a)

Order: conditions 1-5 at 100% first (week-2 baselines), then 1-5 at 50/25/10%, then condition 6
at ratio 1 for every fraction, then the ratio sweep (0.5x, 2x at 10% and 100%).
Condition-6 runs are skipped with a message until the matching synth/f{ppp}/ folder exists.
--checks NAME: the 6 baseline runs (main-grid runs, main CSV) + the 6 variant runs (val only, written to
results/checks/classifier_runs.csv) of a check in CHECKS; no APTOS test / IDRiD metrics are printed.
Each run is a subprocess so a crash or OOM does not take the queue down. Ctrl+C stops cleanly
after the current run (its last.pt checkpoint lets it resume next session).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import click
import pandas as pd

from src.classify.data import config_path, load_config, synth_files
from src.classify.train import KEEP_WEIGHTS, RESULTS_CSV, run_id_of
from src.data.split import fraction_mask, load_splits
from src.genai.paths import RARE_GRADES

SEEDS = (0, 1, 2)
RATIO_SWEEP = (0.5, 2.0)
RATIO_SWEEP_FRACTIONS = (0.1, 1.0)
CHECK_FRACTIONS = (0.1, 1.0)
CHECKS = {  # pre-registered in report/prereg_checks.md; affects = conditions whose runs the setting changes
    "magnitude": dict(condition=5, ratio=0.0, key="ra_magnitude", baseline=9, variant=15, flag="--ra-magnitude", affects=(5, 6)),
    "psi": dict(condition=6, ratio=1.0, key="synth_variant", baseline="", variant="psi1", flag="--synth-variant", affects=(6,)),
}
assert KEEP_WEIGHTS == {run_id_of(5, 1.0, 0.0, 0), run_id_of(6, 1.0, 1.0, 0)}


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


def check_runs(name: str) -> list[dict]:
    """6 baseline runs (plain main-grid ids) + 6 variant runs scored on val only."""
    c = CHECKS[name]
    runs = []
    for variant in ({}, {c["key"]: c["variant"], "eval_split": "val"}):
        for frac in CHECK_FRACTIONS:
            for seed in SEEDS:
                r = dict(condition=c["condition"], fraction=frac, ratio=c["ratio"], seed=seed, **variant)
                runs.append({**r, "run_id": run_id_of(**r)})
    return runs


def decision(name: str, checks_dir: Path) -> dict | None:
    """The applied decision of a pre-registered check, else None."""
    p = checks_dir / f"decision_{name}.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return d if d.get("applied") else None


def check_config(checks_dir: Path) -> None:
    """config.json is what plain-id runs use: it must match every applied decision."""
    conf = load_config()
    for name, c in CHECKS.items():
        d = decision(name, checks_dir)
        if d and conf[c["key"]] != (want := c["variant"] if d["variant_adopted"] else c["baseline"]):
            raise click.ClickException(f"{config_path()} has {c['key']}={conf[c['key']]!r} but decision_{name}.json applied "
                                       f"{want!r}: restore and commit src/classify/config.json")


def done_ids(results_csv) -> set[str]:
    return set(pd.read_csv(results_csv, usecols=["run_id"]).run_id) if Path(results_csv).exists() else set()


def synth_ready(fraction: float, ratio: float, variant: str | None = None) -> str | None:
    try:
        df = load_splits()
        tr = df[fraction_mask(df, fraction)]
        synth_files(fraction, ratio, {g: int((tr.grade == g).sum()) for g in RARE_GRADES}, variant)
        return None
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:  # splits missing etc.
        return f"{type(e).__name__}: {e}"


@click.command()
@click.option("--dry-run", is_flag=True)
@click.option("--checks", type=click.Choice(sorted(CHECKS)), default=None, help="run a pre-registered check instead of the grid")
@click.option("--condition", type=int, default=None)
@click.option("--fraction", type=float, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--epochs", type=int, default=20)
@click.option("--workers", type=int, default=6)
@click.option("--batch-size", type=int, default=64)
@click.option("--results-csv", default=str(RESULTS_CSV), help="main CSV; check variants go to checks/ next to it")
@click.option("--extra", default="", help="extra args passed to train.py, e.g. '--no-pretrained --allow-cpu'")
def main(dry_run, checks, condition, fraction, seed, epochs, workers, batch_size, results_csv, extra):
    results_csv = Path(results_csv)
    checks_csv = results_csv.parent / "checks" / results_csv.name
    check_config(checks_csv.parent)
    if checks:
        c = CHECKS[checks]
        now = load_config()[c["key"]]
        if now != c["baseline"]:
            raise click.ClickException(f"config.json has {c['key']}={now!r}, not the baseline {c['baseline']!r}: "
                                       f"the {checks} check is decided (results/checks/decision_{checks}.json)")
        if checks == "psi" and not decision("magnitude", checks_csv.parent):
            raise click.ClickException("run S8a first (queue --checks magnitude, then checks magnitude --apply): "
                                       "the psi check runs at the magnitude adopted there")
    runs = check_runs(checks) if checks else grid()
    if condition is not None:
        runs = [r for r in runs if r["condition"] == condition]
    if fraction is not None:
        runs = [r for r in runs if abs(r["fraction"] - fraction) < 1e-9]
    if seed is not None:
        runs = [r for r in runs if r["seed"] == seed]
    done = {results_csv: done_ids(results_csv), checks_csv: done_ids(checks_csv)}
    for r in runs:
        r["csv"] = checks_csv if r.get("eval_split") == "val" else results_csv
    pending = [r for r in runs if r["run_id"] not in done[r["csv"]]]
    click.echo(f"{len(runs)} selected, {len(runs) - len(pending)} done, {len(pending)} pending")
    if not checks:  # pre-registered order: no grid run before the magnitude decision, no c6 before psi
        mag, psi = decision("magnitude", checks_csv.parent), decision("psi", checks_csv.parent)
        held = [r for r in pending if not mag or (r["condition"] == 6 and not psi)]
        if held:
            todo_checks = [s for s, ok in (("magnitude (S8a)", mag), ("psi (S10a)", psi)) if not ok]
            click.echo(f"  {len(held)} runs wait for the pre-registered check(s): {', '.join(todo_checks)}")
            pending = [r for r in pending if r not in held]
    waiting = {}
    todo = []
    for r in pending:
        if r["condition"] == 6:
            msg = synth_ready(r["fraction"], r["ratio"], r.get("synth_variant"))
            if msg:
                waiting.setdefault(msg, []).append(r["run_id"])
                continue
        todo.append(r)
    for msg, ids in waiting.items():
        click.echo(f"  waiting for synthetic images ({len(ids)} runs): {msg}")
    if dry_run:
        for r in todo:
            click.echo(f"  {r['run_id']}")
        return
    durations = []
    for i, r in enumerate(todo):
        cmd = [sys.executable, "-m", "src.classify.train", "--condition", str(r["condition"]), "--fraction", str(r["fraction"]),
               "--ratio", str(r["ratio"]), "--seed", str(r["seed"]), "--epochs", str(epochs), "--workers", str(workers),
               "--batch-size", str(batch_size), "--results-csv", str(r["csv"])]
        if r.get("eval_split") == "val":
            cmd += [CHECKS[checks]["flag"], str(r[CHECKS[checks]["key"]]), "--eval-split", "val"]
        if checks:
            cmd.append("--hide-test")
        if r["run_id"] in KEEP_WEIGHTS:
            cmd.append("--keep-weights")
        cmd += extra.split()
        eta = (sum(durations) / len(durations)) * (len(todo) - i) / 60 if durations else float("nan")
        click.echo(f"\n=== [{i + 1}/{len(todo)}] {r['run_id']}  (ETA {eta:.0f} min) ===")
        t0 = time.time()
        # own process group/session: the console's Ctrl+C reaches only the queue, which lets the run finish;
        # poll with sleep because Popen.wait() is not interruptible on Windows
        grp = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
        p = subprocess.Popen(cmd, **grp)
        try:
            while p.poll() is None:
                time.sleep(0.5)
            rc = p.returncode
        except KeyboardInterrupt:
            click.echo(f"\nCtrl+C: stopping after {r['run_id']} finishes (Ctrl+C again aborts it; it then resumes from last.pt)")
            try:
                while p.poll() is None:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                p.kill()
            return
        durations.append(time.time() - t0)
        if rc != 0:
            click.echo(f"!! {r['run_id']} exited with {rc}; continuing")
    click.echo("queue finished" + (f"; decide with: python -m src.analysis.checks {checks} --apply" if checks else ""))


if __name__ == "__main__":
    main()

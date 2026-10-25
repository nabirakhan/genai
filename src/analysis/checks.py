"""Pre-registered validation checks (report/prereg_checks.md, 2026-10-24).

  python -m src.analysis.checks magnitude [--apply]   # condition 5: RandAugment magnitude 9 (baseline) vs 15
  python -m src.analysis.checks psi [--apply]         # condition 6 at 1x: synth/f<ppp> (psi 0.8) vs synth/f<ppp>_psi1

Reads validation metrics only. Baseline arm: the val_* columns of the 6 baseline runs (main-grid runs at 10% and
100%, seeds 0-2) in results/classifier_runs.csv (results/checks/superseded_runs.csv once they are superseded).
Variant arm: the dataset == "val" rows of results/checks/classifier_runs.csv (python -m src.classify.queue --checks NAME).

PRE-REGISTERED RULE: the variant is adopted only if, at BOTH fractions (10% and 100%), its mean val macro-F1 over the
3 seeds exceeds the baseline mean by more than the baseline's seed standard deviation (ddof=1), AND its mean val
rare-grade recall ((recall_3 + recall_4)/2) is not lower than the baseline's. Otherwise keep the baseline.

Prints the table and the decision and writes results/checks/decision_<name>.json. --apply with a winning variant sets
it in src/classify/config.json and supersedes every main-CSV run of an affected condition trained with the old setting
(by the runbook order exactly the 6 baseline runs): rows of all datasets -> results/checks/superseded_runs.csv, preds ->
results/checks/preds_superseded/, so the main queue re-runs them with the adopted setting. --apply with the baseline
winning only records the decision.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd
from filelock import FileLock

from src.classify.data import config_path, load_config
from src.classify.queue import CHECK_FRACTIONS, CHECKS, SEEDS, decision
from src.classify.train import RESULTS_CSV, ROW_COLS, replace_retry, run_id_of

RULE = ("the variant is adopted only if, at BOTH fractions (10% and 100%), its mean val macro-F1 over the 3 seeds "
        "exceeds the baseline mean by more than the baseline's seed standard deviation (ddof=1), AND its mean val "
        "rare-grade recall ((recall_3 + recall_4)/2) is not lower than the baseline's. Otherwise keep the baseline.")
EPS = 1e-12  # float noise only: a tie is "not lower" for recall and "not more than" for macro-F1
COLS = [*ROW_COLS, "superseded_by"]


def read_runs(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"git_sha": str}) if path.exists() else pd.DataFrame(columns=COLS)


def setting(df: pd.DataFrame, key: str) -> pd.Series:
    return df.config.map(lambda s: json.loads(s).get(key))


def arm_rows(df: pd.DataFrame, c: dict, value, dataset: str) -> pd.DataFrame:
    m = ((df.dataset == dataset) & (df.condition == c["condition"]) & ((df.ratio - c["ratio"]).abs() < 1e-9)
         & df.fraction.round(4).isin(CHECK_FRACTIONS) & df.seed.isin(SEEDS) & (setting(df, c["key"]) == value))
    return df[m]


def concat(frames) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


def atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    replace_retry(tmp, path)


def decide(name: str, results_csv: Path) -> tuple[list[dict], list[dict], bool]:
    c = CHECKS[name]
    checks_dir = results_csv.parent / "checks"
    sup = read_runs(checks_dir / "superseded_runs.csv")
    pool = concat([sup[sup.superseded_by == name], read_runs(results_csv)])  # main CSV last: it wins duplicates
    base = arm_rows(pool, c, c["baseline"], "aptos_test").drop_duplicates("run_id", keep="last")
    var = arm_rows(read_runs(checks_dir / "classifier_runs.csv"), c, c["variant"], "val")
    bad = []
    for arm, df, kw in (("baseline", base, {}), ("variant", var, {c["key"]: c["variant"], "eval_split": "val"})):
        for f in CHECK_FRACTIONS:
            for s in SEEDS:
                n = int(((df.fraction.round(4) == f) & (df.seed == s)).sum())
                if n != 1:
                    bad.append(f"{arm} {run_id_of(c['condition'], f, c['ratio'], s, **kw)} ({n} rows)")
    if bad:
        raise click.ClickException(f"{name}: need exactly one row for each of the 12 runs; missing or ambiguous: " + ", ".join(bad))
    both = pd.concat([base, var], ignore_index=True)
    if both[["val_macro_f1", "val_recall_3", "val_recall_4"]].isna().any().any():
        raise click.ClickException(f"{name}: val_* metrics missing in {both[both.val_macro_f1.isna()].run_id.tolist()}")
    rest = {json.dumps({k: v for k, v in json.loads(s).items() if k not in (c["key"], "eval_split")}, sort_keys=True)
            for s in both.config}
    if len(rest) != 1:
        raise click.ClickException(f"{name}: the arms differ in more than {c['key']}: {sorted(rest)}")
    if name == "psi":  # pre-registered: both arms at the magnitude adopted by the first check
        if not decision("magnitude", checks_dir):
            raise click.ClickException("psi: decide the magnitude check first (checks magnitude --apply)")
        mags = set(setting(both, "ra_magnitude"))
        if mags != {load_config()["ra_magnitude"]}:
            raise click.ClickException(f"psi: runs used ra_magnitude {sorted(mags)}, config has {load_config()['ra_magnitude']}")
    both = both.assign(arm=["baseline"] * len(base) + ["variant"] * len(var),
                       val_rare_recall=(both.val_recall_3 + both.val_recall_4) / 2)
    table = []
    for f in CHECK_FRACTIONS:
        b, v = (both[(both.arm == a) & (both.fraction.round(4) == f)] for a in ("baseline", "variant"))
        r = dict(fraction=f, baseline_f1=float(b.val_macro_f1.mean()), baseline_f1_sd=float(b.val_macro_f1.std(ddof=1)),
                 variant_f1=float(v.val_macro_f1.mean()))
        r["delta_f1"] = r["variant_f1"] - r["baseline_f1"]
        r["f1_ok"] = r["delta_f1"] > r["baseline_f1_sd"] + EPS
        r.update(baseline_rare_recall=float(b.val_rare_recall.mean()), variant_rare_recall=float(v.val_rare_recall.mean()))
        r["recall_ok"] = r["variant_rare_recall"] >= r["baseline_rare_recall"] - EPS
        r["passed"] = r["f1_ok"] and r["recall_ok"]
        table.append(r)
    runs = [dict(arm=r.arm, run_id=r.run_id, fraction=float(r.fraction), seed=int(r.seed), val_macro_f1=float(r.val_macro_f1),
                 val_rare_recall=float(r.val_rare_recall)) for r in both.itertuples()]
    return table, runs, all(r["passed"] for r in table)


def supersede(name: str, results_csv: Path) -> None:
    c = CHECKS[name]
    checks_dir = results_csv.parent / "checks"
    if not results_csv.exists():
        return
    with FileLock(str(results_csv) + ".lock"):  # same lock as train.append_rows
        df = pd.read_csv(results_csv, dtype={"git_sha": str})
        stale = df.condition.isin(c["affects"]) & (setting(df, c["key"]) != c["variant"])
        if not stale.any():
            return
        moved = concat([read_runs(checks_dir / "superseded_runs.csv"), df[stale].assign(superseded_by=name)])
        moved = moved.drop_duplicates(["run_id", "dataset", "superseded_by"], keep="last")
        atomic_write(checks_dir / "superseded_runs.csv", lambda t: moved.to_csv(t, index=False))
        (checks_dir / "preds_superseded").mkdir(parents=True, exist_ok=True)
        for rid, ds in zip(df[stale].run_id, df[stale].dataset):
            p = results_csv.parent / "preds" / f"{rid}_{ds}.csv"
            if p.exists():
                replace_retry(p, checks_dir / "preds_superseded" / p.name)
        atomic_write(results_csv, lambda t: df[~stale].to_csv(t, index=False))
    click.echo(f"superseded {int(stale.sum())} rows of {sorted(set(df[stale].run_id))}: the main queue re-runs them")


@click.command()
@click.argument("name", type=click.Choice(sorted(CHECKS)))
@click.option("--apply", "apply_", is_flag=True, help="act on the decision (config.json + superseded runs)")
@click.option("--results-csv", type=click.Path(path_type=Path), default=RESULTS_CSV)
def main(name, apply_, results_csv):
    c = CHECKS[name]
    table, runs, adopted = decide(name, results_csv)
    click.echo(f"{name}: {c['key']} baseline {c['baseline']!r} vs variant {c['variant']!r}, val metrics, mean over seeds {SEEDS}")
    click.echo(pd.DataFrame(table).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    click.echo(f"decision: {'ADOPT variant' if adopted else 'KEEP baseline'} {c['key']}="
               f"{c['variant'] if adopted else c['baseline']!r}")
    out = results_csv.parent / "checks" / f"decision_{name}.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if prev.get("applied"):  # the pre-registered decision stands; never rewrite it
        if prev["variant_adopted"] != adopted:
            raise click.ClickException(f"{out} was applied as {'variant' if prev['variant_adopted'] else 'baseline'}, but the "
                                       "rows now give the opposite verdict (runs redone after the decision?): restore the rows")
        click.echo(f"{out} already applied ({prev['decided_at']}); verdict unchanged, record kept")
        return
    if apply_ and adopted:
        supersede(name, results_csv)
        conf = {**load_config(), c["key"]: c["variant"]}
        atomic_write(config_path(), lambda t: t.write_text(json.dumps(conf) + "\n", encoding="utf-8"))
        click.echo(f"{config_path()}: {conf}  (commit it with results/)")
    elif not apply_:
        click.echo("not applied (rerun with --apply)")
    sup = read_runs(results_csv.parent / "checks" / "superseded_runs.csv")
    decision = dict(check=name, rule=RULE, decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    setting=c["key"], baseline=c["baseline"], variant=c["variant"], variant_adopted=adopted,
                    applied=bool(apply_ or prev.get("applied")), per_fraction=table, runs=runs, config=load_config(),
                    superseded=sorted(set(sup[sup.superseded_by == name].run_id)))
    atomic_write(out, lambda t: t.write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8"))
    click.echo(f"wrote {out}")


if __name__ == "__main__":
    main()

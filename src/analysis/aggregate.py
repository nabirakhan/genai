"""S11a: aggregate results/classifier_runs.csv into report tables + web/public/results/tables.json.

results/tables/summary_long.csv        mean/std/n per (condition, fraction, ratio, dataset, metric)
results/tables/main_{dataset}_{metric}.csv|.md   condition x fraction, "mean +- std" (ratio 1x for cond 6)
results/tables/delta_6_vs_5_{dataset}.csv        paired-by-seed differences cond6(1x) - cond5
results/tables/ratio_{dataset}.csv               cond 6 at 0.5x/1x/2x (fractions 10%, 100%) with cond 5 as 0x
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd

from src.genai.paths import RESULTS, WEB_RESULTS

METRICS = ["macro_f1", "qwk", "recall_3", "recall_4", "acc"]
COND_LABELS = {1: "none", 2: "class weights", 3: "oversampling", 4: "classical aug",
               5: "RandAugment + mixup", 6: "RandAugment + mixup + GAN"}


def load_runs(path: Path = RESULTS / "classifier_runs.csv") -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates(["run_id", "dataset"], keep="last")  # --force re-runs append; preds/ holds the latest
    df["fraction"] = df.fraction.round(4)
    df["ratio"] = df.ratio.round(4)
    return df


def to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines) + "\n"


def summary_long(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["condition", "condition_name", "fraction", "ratio", "dataset"])[METRICS]
    out = g.agg(["mean", "std", "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def fmt_ms(mean, std) -> str:
    if pd.isna(mean):
        return ""
    return f"{mean:.3f} ± {std:.3f}" if not pd.isna(std) else f"{mean:.3f}"


def main_table(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    sub = df[(df.dataset == dataset) & (df.ratio.isin([0.0, 1.0]))]
    g = sub.groupby(["condition", "fraction"])[metric].agg(["mean", "std"]).reset_index()
    g["cell"] = [fmt_ms(m, s) for m, s in zip(g["mean"], g["std"])]
    piv = g.pivot(index="condition", columns="fraction", values="cell").sort_index()
    piv.index = [f"({c}) {COND_LABELS[c]}" for c in piv.index]
    piv.columns = [f"{int(round(f * 100))}%" for f in piv.columns]
    return piv


def paired_delta(df: pd.DataFrame, dataset: str, ratio: float = 1.0) -> pd.DataFrame:
    c6 = df[(df.dataset == dataset) & (df.condition == 6) & (df.ratio == ratio)]
    c5 = df[(df.dataset == dataset) & (df.condition == 5)]
    m = c6.merge(c5, on=["fraction", "seed"], suffixes=("_6", "_5"))
    rows = []
    for frac, grp in m.groupby("fraction"):
        for metric in METRICS:
            d = grp[f"{metric}_6"] - grp[f"{metric}_5"]
            rows.append(dict(fraction=frac, ratio=ratio, metric=metric, n=len(d), delta_mean=d.mean(), delta_std=d.std(),
                             wins=int((d > 0).sum()), losses=int((d < 0).sum())))
    return pd.DataFrame(rows)


def ratio_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    sub = df[(df.dataset == dataset) & (((df.condition == 6)) | (df.condition == 5))].copy()
    sub["ratio_eff"] = sub.ratio.where(sub.condition == 6, 0.0)
    sub = sub[sub.fraction.isin([0.1, 1.0])]
    rows = []
    for (frac, ratio), grp in sub.groupby(["fraction", "ratio_eff"]):
        row = dict(fraction=frac, ratio=ratio, n=len(grp))
        for metric in METRICS:
            row[metric] = fmt_ms(grp[metric].mean(), grp[metric].std())
        rows.append(row)
    out = pd.DataFrame(rows, columns=["fraction", "ratio", "n", *METRICS])
    return out.sort_values(["fraction", "ratio"]).reset_index(drop=True)


@click.command()
@click.option("--runs-csv", type=click.Path(path_type=Path), default=RESULTS / "classifier_runs.csv")
@click.option("--out-dir", type=click.Path(path_type=Path), default=RESULTS / "tables")
@click.option("--web-dir", type=click.Path(path_type=Path), default=WEB_RESULTS)
def main(runs_csv, out_dir, web_dir):
    df = load_runs(runs_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    web_dir.mkdir(parents=True, exist_ok=True)
    long = summary_long(df)
    long.to_csv(out_dir / "summary_long.csv", index=False)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_rows": int(len(df)),
               "n_runs": int(df.run_id.nunique()), "metrics": METRICS, "condition_labels": COND_LABELS,
               "summary_long": json.loads(long.to_json(orient="records")), "main": {}, "delta_6_vs_5": {}, "ratio": {}}
    for dataset in sorted(df.dataset.unique()):
        payload["main"][dataset] = {}
        for metric in METRICS:
            t = main_table(df, dataset, metric)
            t.to_csv(out_dir / f"main_{dataset}_{metric}.csv")
            (out_dir / f"main_{dataset}_{metric}.md").write_text(f"### {dataset} — {metric}\n\n" + to_markdown(t), encoding="utf-8")
            payload["main"][dataset][metric] = json.loads(t.to_json(orient="index"))
        d = paired_delta(df, dataset)
        d.to_csv(out_dir / f"delta_6_vs_5_{dataset}.csv", index=False)
        payload["delta_6_vs_5"][dataset] = json.loads(d.to_json(orient="records"))
        r = ratio_table(df, dataset)
        r.to_csv(out_dir / f"ratio_{dataset}.csv", index=False)
        payload["ratio"][dataset] = json.loads(r.to_json(orient="records"))
    gan_csv = RESULTS / "gan_metrics.csv"
    if gan_csv.exists():
        payload["gan_metrics"] = json.loads(pd.read_csv(gan_csv).to_json(orient="records"))
    (web_dir / "tables.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    click.echo(f"{len(df)} rows / {df.run_id.nunique()} runs -> {out_dir} and {web_dir / 'tables.json'}")
    for dataset in sorted(df.dataset.unique()):
        click.echo(f"\n{dataset} macro_f1\n" + to_markdown(main_table(df, dataset, "macro_f1")))


if __name__ == "__main__":
    main()

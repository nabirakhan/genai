"""S11c: figures for the report and the web Results page (copies go to web/public/results/).

learning_curve_{dataset}_{metric}.png   metric vs real rare-grade fraction, one line per condition
ratio_map_{dataset}.png                 heatmap of cond6 - cond5 by fraction x synthetic ratio
recall_bars_{dataset}_{frac}.png        per-grade recall, all conditions, at 10% and 100%
gan_kid.png                             per-class KID per GAN (if results/gan_metrics.csv exists)
"""
import shutil
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.aggregate import COND_LABELS, load_runs  # noqa: E402
from src.genai.paths import RESULTS, WEB_RESULTS  # noqa: E402

COLORS = {1: "#9e9e9e", 2: "#4e79a7", 3: "#59a14f", 4: "#f28e2b", 5: "#b07aa1", 6: "#e15759"}
METRIC_LABELS = {"macro_f1": "macro F1", "qwk": "quadratic weighted kappa", "recall_3": "recall grade 3",
                 "recall_4": "recall grade 4", "acc": "accuracy"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def learning_curve(df: pd.DataFrame, dataset: str, metric: str, out: Path) -> None:
    sub = df[(df.dataset == dataset) & (df.ratio.isin([0.0, 1.0]))]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for cond, grp in sub.groupby("condition"):
        g = grp.groupby("fraction")[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(g.fraction * 100, g["mean"], yerr=g["std"].fillna(0), marker="o", ms=4, capsize=2,
                    color=COLORS[cond], label=f"({cond}) {COND_LABELS[cond]}", lw=1.5 if cond in (5, 6) else 1)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("real grade-3/4 training images kept (%)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{dataset}: {METRIC_LABELS[metric]} (mean ± std over 3 seeds)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def ratio_map(df: pd.DataFrame, dataset: str, out: Path, metrics=("macro_f1", "recall_3", "recall_4")) -> None:
    c5 = df[(df.dataset == dataset) & (df.condition == 5)].groupby("fraction")[list(metrics)].mean()
    c6 = df[(df.dataset == dataset) & (df.condition == 6)].groupby(["fraction", "ratio"])[list(metrics)].mean()
    fracs = sorted(df.fraction.unique())
    ratios = sorted(df[df.condition == 6].ratio.unique())
    if not ratios:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.2))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        grid = np.full((len(ratios), len(fracs)), np.nan)
        for i, r in enumerate(ratios):
            for j, f in enumerate(fracs):
                if (f, r) in c6.index and f in c5.index:
                    grid[i, j] = c6.loc[(f, r), metric] - c5.loc[f, metric]
        vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 0.1
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(ratios)):
            for j in range(len(fracs)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(fracs)), [f"{int(round(f * 100))}%" for f in fracs])
        ax.set_yticks(range(len(ratios)), [f"{r:g}x" for r in ratios])
        ax.set_xlabel("real grade-3/4 fraction")
        ax.set_ylabel("synthetic ratio")
        ax.set_title(f"Δ {METRIC_LABELS[metric]} (GAN − strong aug)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{dataset}: when does the GAN help?", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def recall_bars(df: pd.DataFrame, dataset: str, frac: float, out: Path) -> None:
    sub = df[(df.dataset == dataset) & (df.fraction == frac) & (df.ratio.isin([0.0, 1.0]))]
    if sub.empty:
        return
    conds = sorted(sub.condition.unique())
    width = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, cond in enumerate(conds):
        g = sub[sub.condition == cond][[f"recall_{k}" for k in range(5)]].agg(["mean", "std"])
        ax.bar(np.arange(5) + i * width, g.loc["mean"], width, yerr=g.loc["std"].fillna(0), capsize=1.5,
               color=COLORS[cond], label=f"({cond}) {COND_LABELS[cond]}")
    ax.set_xticks(np.arange(5) + 0.4 - width / 2, [f"grade {k}" for k in range(5)])
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1)
    ax.set_title(f"{dataset}: per-grade recall at {int(round(frac * 100))}% rare-grade data")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def gan_kid(gan_csv: Path, out: Path) -> None:
    g = pd.read_csv(gan_csv)
    fig, ax = plt.subplots(figsize=(5, 3))
    for grade, grp in g.groupby("grade"):
        ax.bar(np.arange(len(grp)) + (0.4 if grade == 4 else 0), grp.kid, 0.4, yerr=grp.kid_std, capsize=2,
               label=f"grade {grade}")
        ax.set_xticks(np.arange(len(grp)) + 0.2, grp.gan, rotation=20)
    ax.set_ylabel("KID (lower is better)")
    ax.set_title("GAN quality per rare grade vs held-out val images")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


@click.command()
@click.option("--runs-csv", type=click.Path(path_type=Path), default=RESULTS / "classifier_runs.csv")
@click.option("--out-dir", type=click.Path(path_type=Path), default=RESULTS / "figures")
@click.option("--web-dir", type=click.Path(path_type=Path), default=WEB_RESULTS)
def main(runs_csv, out_dir, web_dir):
    df = load_runs(runs_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    web_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for dataset in sorted(df.dataset.unique()):
        for metric in ("macro_f1", "qwk", "recall_3", "recall_4"):
            p = out_dir / f"learning_curve_{dataset}_{metric}.png"
            learning_curve(df, dataset, metric, p)
            made.append(p)
        if (df.condition == 6).any():
            p = out_dir / f"ratio_map_{dataset}.png"
            ratio_map(df, dataset, p)
            made.append(p)
        for frac in (0.1, 1.0):
            p = out_dir / f"recall_bars_{dataset}_f{int(round(frac * 100)):03d}.png"
            recall_bars(df, dataset, frac, p)
            made.append(p)
    gan_csv = RESULTS / "gan_metrics.csv"
    if gan_csv.exists():
        p = out_dir / "gan_kid.png"
        gan_kid(gan_csv, p)
        made.append(p)
    for p in made:
        if p.exists():
            shutil.copy2(p, web_dir / p.name)
    click.echo(f"{sum(p.exists() for p in made)} figures -> {out_dir} (+ copies in {web_dir})")


if __name__ == "__main__":
    main()

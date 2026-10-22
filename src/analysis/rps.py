"""S13: Retinal Pigmentation Score (Rajesh et al. 2023) on real vs generated images.

The reference implementation (uw-biomedical-ml/retinal-pigmentation-score) pins torch 1.7 /
Python 3.8, so it runs in its Docker image on CPU. Three sub-commands:

  python -m src.analysis.rps stage   [--per-grade 300]   # copy PNGs + manifest to DATA_ROOT/rps/
  python -m src.analysis.rps run     [--dry-run]          # docker run arajesh17/rps ... (prints the command)
  python -m src.analysis.rps merge                        # join RPS csv with manifest -> tables + figure

Outputs: results/tables/rps_scores.csv (image, source, grade, a, b, rps), rps_summary.csv (per
source/grade: n, n_unscored, median, IQR; KS test real vs synthetic), figures/rps_hist.png.
The exact in-container command follows the repo README; verify once with `docker run -it ... /bin/bash`
and adjust RPS_CMD below if the image layout differs.
"""
import shutil
import subprocess
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from src.data.split import load_splits  # noqa: E402
from src.genai.paths import APTOS256, DATA_ROOT, RESULTS, SYNTH  # noqa: E402

RPS_ROOT = DATA_ROOT / "rps"
IMAGES, OUT = RPS_ROOT / "images", RPS_ROOT / "results"
DOCKER_IMAGE = "arajesh17/rps"
RPS_CMD = "source activate automorph && python /home/retinal-pigementation-score/src/main.py"


@click.group()
def cli():
    pass


@cli.command()
@click.option("--per-grade", type=int, default=300, help="real train images sampled per grade (0 = all)")
@click.option("--seed", type=int, default=0)
def stage(per_grade, seed):
    IMAGES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_splits()
    tr = df[df.split == "train"]
    rng = np.random.default_rng(seed)
    rows = []
    for g, grp in tr.groupby("grade"):
        ids = grp.image_id.to_numpy()
        if per_grade and len(ids) > per_grade:
            ids = rng.choice(ids, per_grade, replace=False)
        for i in ids:
            name = f"real_g{g}_{i}.png"
            if not (IMAGES / name).exists():
                shutil.copy2(APTOS256 / f"{i}.png", IMAGES / name)
            rows.append(dict(image=name, source="real", grade=int(g), origin=i))
    for tag_dir in sorted(SYNTH.glob("f*")):
        for gdir in sorted(tag_dir.glob("g*")):
            g = int(gdir.name[1:])
            for f in sorted(gdir.glob("*.png")):
                name = f"synth_{tag_dir.name}_g{g}_{f.stem}.png"
                if not (IMAGES / name).exists():
                    shutil.copy2(f, IMAGES / name)
                rows.append(dict(image=name, source=f"synth_{tag_dir.name}", grade=g, origin=f.name))
    pd.DataFrame(rows).to_csv(RPS_ROOT / "manifest.csv", index=False)
    click.echo(f"{len(rows)} images staged in {IMAGES}")


@cli.command()
@click.option("--dry-run", is_flag=True)
def run(dry_run):
    cmd = ["docker", "run", "--rm", "-v", f"{IMAGES}:/home/images/", "-v", f"{OUT}:/home/results/",
           DOCKER_IMAGE, "/bin/bash", "-lc", RPS_CMD]
    click.echo(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


@cli.command()
def merge():
    man = pd.read_csv(RPS_ROOT / "manifest.csv")
    cands = list(OUT.rglob("retinal_pigmentation_score.csv"))
    if not cands:
        raise SystemExit(f"no retinal_pigmentation_score.csv under {OUT}; run the docker step first")
    rps = pd.read_csv(cands[0])
    name_col = next(c for c in rps.columns if "name" in c.lower() or "image" in c.lower() or "path" in c.lower())
    rps["image"] = rps[name_col].astype(str).map(lambda s: Path(s).name)
    score_col = next(c for c in rps.columns if "rps" in c.lower() or "pigment" in c.lower())
    merged = man.merge(rps[["image", score_col]].rename(columns={score_col: "rps"}), on="image", how="left")
    tables = RESULTS / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    merged.to_csv(tables / "rps_scores.csv", index=False)
    summ = merged.groupby(["source", "grade"]).rps.agg(n="size", n_unscored=lambda s: int(s.isna().sum()), median="median",
                                                    q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75)).reset_index()
    real = merged[(merged.source == "real") & merged.rps.notna()].rps
    ks_rows = []
    for src, grp in merged[merged.source != "real"].groupby("source"):
        s = grp.rps.dropna()
        if len(s) and len(real):
            ks = ks_2samp(real, s)
            ks_rows.append(dict(source=src, ks_stat=ks.statistic, ks_p=ks.pvalue, n=len(s)))
    summ.to_csv(tables / "rps_summary.csv", index=False)
    pd.DataFrame(ks_rows).to_csv(tables / "rps_ks.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    bins = np.linspace(merged.rps.min(), merged.rps.max(), 30) if merged.rps.notna().any() else 30
    ax.hist(real, bins=bins, density=True, alpha=0.5, label="real (train)")
    for src, grp in merged[merged.source != "real"].groupby("source"):
        ax.hist(grp.rps.dropna(), bins=bins, density=True, alpha=0.4, label=src)
    ax.set_xlabel("Retinal Pigmentation Score")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "rps_hist.png", dpi=150)
    click.echo(summ.to_string(index=False))
    click.echo(pd.DataFrame(ks_rows).to_string(index=False))


if __name__ == "__main__":
    cli()

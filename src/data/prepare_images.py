"""S3: crop + resize APTOS train images and IDRiD grading images to 256px PNGs.

APTOS  -> APTOS256/{image_id}.png
IDRiD  -> IDRID256/{split}_{IDRiD_xxx}.png  (train and test file names collide in IDRiD, so
          the original split is part of the id) and splits/idrid_test.csv (image_id, grade, orig_split).
IDRiD must be placed manually under RAW_IDRID (zip or extracted folder from IEEE DataPort).
"""
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from src.data.fundus import crop_fundus, load_rgb, save_png
from src.genai.paths import APTOS256, IDRID256, RAW_APTOS, RAW_IDRID, SPLITS, ensure_dirs

IDRID_URL = "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid"


def _process(src: str, dst: str) -> str:
    if not Path(dst).exists():
        save_png(crop_fundus(load_rgb(src)), dst)
    return dst


def _run(jobs: list[tuple[str, str]], workers: int, desc: str) -> None:
    jobs = [(s, d) for s, d in jobs if not Path(d).exists()]
    if not jobs:
        click.echo(f"{desc}: nothing to do")
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process, s, d) for s, d in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            f.result()  # re-raise worker errors instead of silently dropping the image


def prepare_aptos(workers: int) -> None:
    df = pd.read_csv(RAW_APTOS / "train.csv")
    jobs = [(str(RAW_APTOS / "train_images" / f"{i}.png"), str(APTOS256 / f"{i}.png")) for i in df["id_code"]]
    _run(jobs, workers, "aptos->256")
    n = len(list(APTOS256.glob("*.png")))
    assert n == len(df), f"{n} outputs for {len(df)} inputs"


def _find_idrid_files() -> tuple[list[Path], list[Path]]:
    for z in RAW_IDRID.glob("*.zip"):
        marker = RAW_IDRID / f".extracted_{z.stem}"
        if not marker.exists():
            click.echo(f"extracting {z.name}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(RAW_IDRID)
            marker.touch()
    jpgs = [p for p in RAW_IDRID.rglob("*.jpg")
            if re.fullmatch(r"IDRiD_\d+\.jpg", p.name) and "Disease Grading" in str(p) and "Original Images" in str(p)]
    csvs = [p for p in RAW_IDRID.rglob("*.csv") if "Disease Grading" in str(p) and "Labels" in p.name]
    return jpgs, csvs


def prepare_idrid(workers: int) -> None:
    jpgs, csvs = _find_idrid_files()
    if not jpgs or not csvs:
        raise SystemExit(f"IDRiD not found under {RAW_IDRID}. Download part B (Disease Grading) from {IDRID_URL} "
                         f"and place the zip there.")
    rows = []
    for c in csvs:
        split = "train" if "Training" in c.name else "test"
        df = pd.read_csv(c)
        df.columns = [col.strip() for col in df.columns]
        name_col = next(col for col in df.columns if "Image" in col)
        grade_col = next(col for col in df.columns if "Retinopathy" in col)
        df = df.dropna(subset=[name_col, grade_col])
        for name, grade in zip(df[name_col], df[grade_col]):
            name = str(name).strip()
            if name:
                rows.append({"image_id": f"{split}_{name}", "grade": int(grade), "orig_split": split, "name": name})
    labels = pd.DataFrame(rows).drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
    by_key = {("train" if "Training" in str(p) else "test", p.stem): p for p in jpgs}
    jobs, missing = [], []
    for r in labels.itertuples():
        src = by_key.get((r.orig_split, r.name))
        if src is None:
            missing.append(r.image_id)
        else:
            jobs.append((str(src), str(IDRID256 / f"{r.image_id}.png")))
    if missing:
        click.echo(f"WARNING: {len(missing)} labelled IDRiD images not found, e.g. {missing[:5]}")
        labels = labels[~labels.image_id.isin(missing)]
    _run(jobs, workers, "idrid->256")
    labels[["image_id", "grade", "orig_split"]].to_csv(SPLITS / "idrid_test.csv", index=False)
    click.echo(f"IDRiD: {len(labels)} images, grade counts {labels.grade.value_counts().sort_index().to_dict()}")
    if len(labels) != 516:
        click.echo(f"WARNING: expected 516 IDRiD grading images, got {len(labels)}")


@click.command()
@click.option("--source", type=click.Choice(["aptos", "idrid", "both"]), default="both")
@click.option("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
def main(source: str, workers: int) -> None:
    ensure_dirs()
    if source in ("aptos", "both"):
        prepare_aptos(workers)
    if source in ("idrid", "both"):
        prepare_idrid(workers)


if __name__ == "__main__":
    main()

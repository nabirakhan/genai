"""S1: download APTOS 2019 train set from Kaggle and extract train.csv + train_images/.

Requires kaggle.json in ~/.kaggle (or KAGGLE_USERNAME/KAGGLE_KEY) and the competition
rules accepted on the website. Idempotent: skips download/extraction already done.
"""
import subprocess
import sys
import zipfile

import click
import pandas as pd
from tqdm import tqdm

from src.genai.paths import RAW_APTOS, ensure_dirs

COMP = "aptos2019-blindness-detection"
EXPECTED_TOTAL = 3662
EXPECTED_COUNTS = {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}


def verify() -> pd.DataFrame:
    df = pd.read_csv(RAW_APTOS / "train.csv")
    counts = df["diagnosis"].value_counts().to_dict()
    n_png = len(list((RAW_APTOS / "train_images").glob("*.png")))
    assert len(df) == EXPECTED_TOTAL, f"train.csv has {len(df)} rows, expected {EXPECTED_TOTAL}"
    assert counts == EXPECTED_COUNTS, f"grade counts {counts} != {EXPECTED_COUNTS}"
    assert n_png == EXPECTED_TOTAL, f"{n_png} PNGs found, expected {EXPECTED_TOTAL}"
    return df


@click.command()
@click.option("--skip-download", is_flag=True, help="Only extract/verify an existing zip.")
def main(skip_download: bool) -> None:
    ensure_dirs()
    zpath = RAW_APTOS / f"{COMP}.zip"
    have_pngs = len(list((RAW_APTOS / "train_images").glob("*.png")))
    if (RAW_APTOS / "train.csv").exists() and have_pngs == EXPECTED_TOTAL:
        verify()
        click.echo("APTOS already extracted and verified.")
        return
    if not zpath.exists():
        if skip_download:
            raise SystemExit(f"{zpath} missing and --skip-download given")
        click.echo("Downloading APTOS 2019 (~9.5 GB)...")
        subprocess.run([sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMP, "-p", str(RAW_APTOS)],
                       check=True)  # the venv's kaggle, even when .venv\Scripts is not on PATH
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n == "train.csv" or n.startswith("train_images/")]
        for n in tqdm(names, desc="extract"):
            if not (RAW_APTOS / n).exists():
                z.extract(n, RAW_APTOS)
    verify()
    click.echo("APTOS extracted and verified. Keep the zip as a backup (copy to F:).")


if __name__ == "__main__":
    main()

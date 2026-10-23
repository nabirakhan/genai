"""S7: publish the processed data as a private Kaggle dataset for the GAN notebooks.

Stages sg3/*.zip + splits/*.csv into DATA_ROOT/kaggle_dataset/ with dataset-metadata.json,
then `kaggle datasets create` (first time) or `kaggle datasets version`.
Kaggle auto-extracts uploaded zips into folders named after the zip; stylegan3 reads either
a zip or a folder containing dataset.json, and the notebooks handle both.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import click

from src.genai.paths import DATA_ROOT, SG3, SPLITS

SLUG = "aptos-dr-256-genai"


def kaggle_username() -> str:
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    raise SystemExit("kaggle.json not found; put it in ~/.kaggle/ (remove it at session end on shared PCs)")


@click.command()
@click.option("--user", default=None, help="Kaggle username (default: from kaggle.json)")
@click.option("--message", "-m", default="update", help="version message")
def main(user: str | None, message: str) -> None:
    user = user or kaggle_username()
    stage = DATA_ROOT / "kaggle_dataset"
    stage.mkdir(parents=True, exist_ok=True)
    files = sorted(SG3.glob("aptos256_*.zip")) + sorted(SPLITS.glob("*.csv"))
    assert files, "nothing to upload; run make_sg3_zips first"
    for f in files:
        dst = stage / f.name
        if not dst.exists() or dst.stat().st_size != f.stat().st_size:
            shutil.copy2(f, dst)
    meta = {"title": "APTOS 2019 DR 256px splits (GenAI project)", "id": f"{user}/{SLUG}",
            "licenses": [{"name": "other"}]}
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    status = subprocess.run(["kaggle", "datasets", "status", f"{user}/{SLUG}"], capture_output=True, text=True)
    exists = status.returncode == 0 and "not found" not in (status.stdout + status.stderr).lower()
    if exists:
        cmd = ["kaggle", "datasets", "version", "-p", str(stage), "-m", message, "--dir-mode", "skip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(stage), "--dir-mode", "skip"]
    click.echo(" ".join(cmd))
    subprocess.run(cmd, check=True)
    (stage / ".uploaded").write_text(f"{user}/{SLUG}\n" + "\n".join(f.name for f in files))
    click.echo(f"dataset: https://www.kaggle.com/datasets/{user}/{SLUG}  (private)")


if __name__ == "__main__":
    main()

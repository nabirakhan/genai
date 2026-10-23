"""S6: build stylegan3 training zips, one per rare-grade fraction, plus a labelled val zip
for FID/KID. Images are already 256px PNGs, so files are stored as-is (no re-encode).

sg3/aptos256_f010.zip ... aptos256_f100.zip : train grades 0-2 (all) + fraction of grades 3/4
sg3/aptos256_val.zip                         : the full val split (held-out reals for metrics)
Each zip contains dataset.json = {"labels": [["<image_id>.png", grade], ...]}.
Test images never enter any zip (asserted).
"""
import json
import zipfile

import click
from tqdm import tqdm

from src.data.split import fraction_mask, load_splits
from src.genai.paths import APTOS256, FRACTIONS, SG3, frac_tag


def write_zip(df_sub, out_path) -> None:
    labels = [[f"{r.image_id}.png", int(r.grade)] for r in df_sub.itertuples()]
    tmp = out_path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        for r in tqdm(df_sub.itertuples(), total=len(df_sub), desc=out_path.name):
            z.write(APTOS256 / f"{r.image_id}.png", arcname=f"{r.image_id}.png")
        z.writestr("dataset.json", json.dumps({"labels": labels}))
    tmp.replace(out_path)


@click.command()
@click.option("--force", is_flag=True)
def main(force: bool) -> None:
    SG3.mkdir(parents=True, exist_ok=True)
    df = load_splits()
    test_ids = set(df[df.split == "test"].image_id)
    for p in FRACTIONS:
        sub = df[fraction_mask(df, p)]
        assert not (set(sub.image_id) & test_ids)
        assert (sub.split == "train").all()
        assert set(sub.grade) == {0, 1, 2, 3, 4}, "every grade must be present for --cond=1"
        out = SG3 / f"aptos256_{frac_tag(p)}.zip"
        if force or not out.exists():
            write_zip(sub, out)
        click.echo(f"{out.name}: {len(sub)} images, grades {sub.grade.value_counts().sort_index().to_dict()}")
    val = df[df.split == "val"]
    assert not (set(val.image_id) & test_ids)
    out = SG3 / "aptos256_val.zip"
    if force or not out.exists():
        write_zip(val, out)
    click.echo(f"{out.name}: {len(val)} images")


if __name__ == "__main__":
    main()

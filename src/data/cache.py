"""S5: pack the 256px PNGs into uint8 arrays for fast classifier training.

cache/aptos256.npy  (N,256,256,3) in the row order of splits/aptos_splits.csv
cache/idrid256.npy  (M,256,256,3) in the row order of splits/idrid_test.csv
The classifier opens them with mmap_mode='r', so DataLoader workers share pages.
"""
import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.split import load_splits
from src.genai.paths import APTOS256, CACHE, IDRID256, IMG_SIZE, SPLITS


def build(ids, src_dir, out_path) -> None:
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(len(ids), IMG_SIZE, IMG_SIZE, 3))
    for i, img_id in enumerate(tqdm(ids, desc=out_path.name)):
        im = np.asarray(Image.open(src_dir / f"{img_id}.png").convert("RGB"))
        assert im.shape == (IMG_SIZE, IMG_SIZE, 3), (img_id, im.shape)
        arr[i] = im
    arr.flush()
    del arr


@click.command()
@click.option("--force", is_flag=True)
def main(force: bool) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / "aptos256.npy"
    if force or not out.exists():
        build(load_splits().image_id.tolist(), APTOS256, out)
    idrid_csv = SPLITS / "idrid_test.csv"
    out2 = CACHE / "idrid256.npy"
    if idrid_csv.exists() and (force or not out2.exists()):
        build(pd.read_csv(idrid_csv).image_id.tolist(), IDRID256, out2)
    for p in (out, out2):
        if p.exists():
            a = np.load(p, mmap_mode="r")
            click.echo(f"{p.name}: shape {a.shape}, {p.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

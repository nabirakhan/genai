"""Per-class FID / KID between generated images of one grade and held-out real images of the
same grade, using the same Inception features as stylegan3's fid50k_full (so the overall and
per-class numbers are comparable). KID is the honest metric here: the real sets are small
(about 29 grade-3 and 44 grade-4 val images), which biases FID upward.

  python -m src.gan.metrics_per_class --gan f100-ada --synth-dir synth_f100 \
      --real /kaggle/input/aptos-dr-256-genai/aptos256_val --classes 3,4

--real is the val zip or the folder Kaggle extracted it to (both contain dataset.json).
Appends rows to results/gan_metrics.csv: gan, grade, n_real, n_gen, fid, kid, kid_std.
"""
import io
import json
import zipfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.linalg
import torch
from PIL import Image
from tqdm import tqdm

import src.gan  # noqa: F401
from metrics import metric_utils  # noqa: E402  (stylegan3)
from src.genai.paths import RESULTS  # noqa: E402

DETECTOR_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"


def _read_real(real: Path, grade: int) -> list[np.ndarray]:
    if real.is_dir():
        labels = json.loads((real / "dataset.json").read_text())["labels"]
        return [np.asarray(Image.open(real / f).convert("RGB")) for f, g in labels if int(g) == grade]
    with zipfile.ZipFile(real) as z:
        labels = json.loads(z.read("dataset.json"))["labels"]
        return [np.asarray(Image.open(io.BytesIO(z.read(f))).convert("RGB")) for f, g in labels if int(g) == grade]


def _read_dir(d: Path) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert("RGB")) for p in sorted(d.glob("*.png"))]


@torch.no_grad()
def features(imgs: list[np.ndarray], detector, device, batch: int = 32) -> np.ndarray:
    out = []
    for i in tqdm(range(0, len(imgs), batch), desc="inception", leave=False):
        x = torch.from_numpy(np.stack(imgs[i:i + batch])).permute(0, 3, 1, 2).to(device)
        out.append(detector(x, return_features=True).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def fid(fr: np.ndarray, fg: np.ndarray) -> float:
    mu_r, mu_g = fr.mean(0), fg.mean(0)
    s_r, s_g = np.cov(fr, rowvar=False), np.cov(fg, rowvar=False)
    m = np.square(mu_g - mu_r).sum()
    s = scipy.linalg.sqrtm(np.dot(s_g, s_r))  # SciPy >= 1.18 removed disp; without it every version returns the matrix only
    return float(np.real(m + np.trace(s_g + s_r - s * 2)))


def kid(fr: np.ndarray, fg: np.ndarray, num_subsets: int = 100, max_subset_size: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Polynomial-kernel MMD^2, unbiased, averaged over random subsets (stylegan3 kid50k_full)."""
    rng = np.random.default_rng(seed)
    n = fr.shape[1]
    m = min(min(len(fr), len(fg)), max_subset_size)
    vals = []
    for _ in range(num_subsets):
        x = fg[rng.choice(len(fg), m, replace=False)]
        y = fr[rng.choice(len(fr), m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        vals.append((a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m)
    vals = np.array(vals) / m
    return float(vals.mean()), float(vals.std())


@click.command()
@click.option("--gan", required=True, help="tag, e.g. f100-ada, f010-noaug")
@click.option("--synth-dir", type=click.Path(exists=True, path_type=Path), required=True, help="folder with g3/, g4/")
@click.option("--real", type=click.Path(exists=True, path_type=Path), required=True, help="aptos256_val zip or folder")
@click.option("--classes", default="3,4")
@click.option("--out", type=click.Path(path_type=Path), default=RESULTS / "gan_metrics.csv")
def main(gan, synth_dir, real, classes, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = metric_utils.get_feature_detector(DETECTOR_URL, device=device, num_gpus=1, rank=0, verbose=True)
    rows = []
    for grade in [int(c) for c in classes.split(",")]:
        reals = _read_real(real, grade)
        gens = _read_dir(synth_dir / f"g{grade}")
        fr, fg = features(reals, detector, device), features(gens, detector, device)
        k, ks = kid(fr, fg)
        rows.append(dict(gan=gan, grade=grade, n_real=len(reals), n_gen=len(gens), fid=round(fid(fr, fg), 2),
                         kid=round(k, 5), kid_std=round(ks, 5)))
        click.echo(rows[-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        old = old[~((old.gan.isin(df.gan)) & (old.grade.isin(df.grade)))]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    click.echo(f"-> {out}")


if __name__ == "__main__":
    main()

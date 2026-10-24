"""Per-class KID between generated images of one grade and the held-out real val images of that
grade, on the Inception features of stylegan3's fid50k_full / kid50k_full (one feature space for the
per-class and the overall numbers).

  python -m src.gan.metrics_per_class --gan f100-ada --synth-dir synth_f100-ada --classes 3,4 \
      --real <prep>/sg3/aptos256_val.zip --real-train <prep>/sg3/aptos256_f100.zip

--real / --real-train: a stylegan3 zip or the folder Kaggle extracted it to (both hold dataset.json).

KID: unbiased MMD^2 with the polynomial kernel k(x, y) = (x.y / d + 1)^3 (Binkowski et al., as in
stylegan3), one point estimate on ALL real vs ALL generated images of the grade. After dedup the val
split holds only 25 grade-3 / 39 grade-4 images, so the real side dominates the noise. Uncertainty,
reported as kid_std = sqrt(kid_se_real^2 + kid_se_gen^2):
- kid_se_real: delete-one jackknife over the real images (closed form on the kernel matrices).
- kid_se_gen: half-samples of the generated images drawn WITHOUT replacement (delete-d jackknife with
  d = n/2, so the SD over half-samples is the SE of the full estimate). Never resample with
  replacement: a duplicated image pairs with itself in the within-set term and inflates KID.
Both are conservative when the two sets share one distribution (the estimator is then degenerate).
stylegan3's kid_std (SD over subsets of min(n_real, n_gen) images) is not used: it describes subsets
of 25 images, not this estimate.

Real-vs-real floor (--real-train): the same estimate between the val images and the f100 TRAIN images
of the grade (different eyes, same distribution) as gan "real-f100-train", at the same real sample
size: what "as good as real" looks like. Its expected value is 0; its spread is the noise level.

Per-class FID is not computed (fid stays NaN in these rows): a covariance from 25-39 images has rank
<= 24-38 in 2048 dimensions, so the value would measure the estimator, not the GAN. The overall FID/KID
rows (grade "all") come from calc_metrics.py in the eval notebook.

Rows replace earlier rows of the same gan/grade in results/gan_metrics.csv:
gan, grade, n_real, n_gen, fid, kid, kid_std, kid_se_real, kid_se_gen.
"""
import io
import json
import re
import zipfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import src.gan  # noqa: F401
from metrics import metric_utils  # noqa: E402  (stylegan3)
from src.genai.paths import RESULTS  # noqa: E402

DETECTOR_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
COLS = ["gan", "grade", "n_real", "n_gen", "fid", "kid", "kid_std", "kid_se_real", "kid_se_gen"]


def _read_real(real: Path, grade: int) -> list[np.ndarray]:
    if real.is_dir():
        labels = json.loads((real / "dataset.json").read_text())["labels"]
        return [np.asarray(Image.open(real / f).convert("RGB")) for f, g in labels if int(g) == grade]
    with zipfile.ZipFile(real) as z:
        labels = json.loads(z.read("dataset.json"))["labels"]
        return [np.asarray(Image.open(io.BytesIO(z.read(f))).convert("RGB")) for f, g in labels if int(g) == grade]


def _read_dir(d: Path) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert("RGB")) for p in sorted(d.glob("*.png"))]


class StubDetector(torch.nn.Module):
    """Tests only (--detector stub): a fixed random projection of 16x16 thumbnails to 2048 features."""

    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.randn(768, 2048, generator=torch.Generator().manual_seed(0)) / 768 ** 0.5)

    def forward(self, x, return_features=True):
        return (F.adaptive_avg_pool2d(x.float() / 255, 16).flatten(1) @ self.w).relu()


def load_detector(device, kind: str = "inception"):
    if kind == "stub":
        click.echo("WARNING: stub feature detector (tests only), not Inception")
        return StubDetector().to(device)
    return metric_utils.get_feature_detector(DETECTOR_URL, device=device, num_gpus=1, rank=0, verbose=True)


@torch.no_grad()
def features(imgs, detector, device, batch: int = 32) -> np.ndarray:
    """imgs: a list of HxWx3 uint8 arrays or one (N,H,W,3) uint8 array."""
    out = []
    for i in tqdm(range(0, len(imgs), batch), desc="inception", leave=False):
        x = torch.from_numpy(np.stack(imgs[i:i + batch])).permute(0, 3, 1, 2).to(device)
        out.append(detector(x, return_features=True).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def poly_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T / a.shape[1] + 1) ** 3


def kid(fr: np.ndarray, fg: np.ndarray, n_half: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """(KID on all images, SE from the real side, SE from the generated side); see the module docstring."""
    m, n = len(fr), len(fg)
    assert m >= 2 and n >= 2, (m, n)
    kxx, kyy, kxy = poly_kernel(fr, fr), poly_kernel(fg, fg), poly_kernel(fr, fg)
    xx_off = kxx.sum() - np.trace(kxx)
    xx, yy = xx_off / (m * (m - 1)), (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    est = xx + yy - 2 * kxy.mean()
    se_real = se_gen = float("nan")
    if m >= 3:
        loo = ((xx_off - 2 * (kxx.sum(1) - np.diag(kxx))) / ((m - 1) * (m - 2)) + yy
               - 2 * (kxy.sum() - kxy.sum(1)) / ((m - 1) * n))
        se_real = np.sqrt((m - 1) / m * np.square(loo - loo.mean()).sum())
    r = n // 2
    if r >= 2:
        rng = np.random.default_rng(seed)
        half = []
        for _ in range(n_half):
            s = rng.choice(n, r, replace=False)
            k = kyy[np.ix_(s, s)]
            half.append(xx + (k.sum() - np.trace(k)) / (r * (r - 1)) - 2 * kxy[:, s].mean())
        se_gen = np.sqrt(r / (n - r) * np.var(half, ddof=1))
    return float(est), float(se_real), float(se_gen)


def kid_row(gan: str, grade: int, fr: np.ndarray, fg: np.ndarray) -> dict:
    k, se_r, se_g = kid(fr, fg)
    return dict(gan=gan, grade=str(grade), n_real=len(fr), n_gen=len(fg), fid=float("nan"), kid=round(k, 6),
                kid_std=round(float(np.hypot(se_r, se_g)), 6), kid_se_real=round(se_r, 6), kid_se_gen=round(se_g, 6))


def floor_tag(real_train: Path) -> str:
    m = re.search(r"f\d{3}", real_train.name)
    return f"real-{m.group(0)}-train" if m else "real-train"


@click.command()
@click.option("--gan", required=True, help="tag, e.g. f100-ada, f010-noaug, f010-ada_psi1")
@click.option("--synth-dir", type=click.Path(exists=True, path_type=Path), required=True, help="folder with g3/, g4/")
@click.option("--real", type=click.Path(exists=True, path_type=Path), required=True, help="aptos256_val zip or folder")
@click.option("--real-train", type=click.Path(exists=True, path_type=Path), default=None,
              help="aptos256_f100 zip or folder: adds the real-vs-real floor rows (optional)")
@click.option("--classes", default="3,4")
@click.option("--out", type=click.Path(path_type=Path), default=RESULTS / "gan_metrics.csv")
@click.option("--detector", "detector_kind", type=click.Choice(["inception", "stub"]), default="inception", hidden=True)
def main(gan, synth_dir, real, real_train, classes, out, detector_kind):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(device, detector_kind)
    rows = []
    for grade in [int(c) for c in classes.split(",")]:
        fr = features(_read_real(real, grade), detector, device)
        rows.append(kid_row(gan, grade, fr, features(_read_dir(synth_dir / f"g{grade}"), detector, device)))
        if real_train is not None:
            rows.append(kid_row(floor_tag(real_train), grade, fr, features(_read_real(real_train, grade), detector, device)))
    for r in rows:
        click.echo(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLS)
    if out.exists():
        old = pd.read_csv(out, dtype={"grade": str})
        key = lambda d: d.gan.astype(str) + "/" + d.grade.astype(str)  # noqa: E731
        df = pd.concat([old[~key(old).isin(key(df))], df], ignore_index=True)
    df.to_csv(out, index=False)
    click.echo(f"-> {out}")


if __name__ == "__main__":
    main()

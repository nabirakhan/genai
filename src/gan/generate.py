"""Generate class-conditional images from a stylegan3 pickle (runs on Kaggle GPU or CPU).

  python -m src.gan.generate --pkl runs/.../network-snapshot-000800.pkl --outdir synth_f100 \
      --classes 3,4 --n 500 --psi 0.8

Files: {outdir}/g{grade}/g{grade}_s{seed:05d}.png with Synthetic=true metadata (no visible
label: these are the training copies). Seeds 0..n-1 per class, so 0.5x / 1x / 2x ratios are
nested prefixes of the same set.
"""
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import src.gan  # noqa: F401  (adds stylegan3 to sys.path)
import dnnlib  # noqa: E402
import legacy  # noqa: E402
from src.gan.watermark import save_synthetic  # noqa: E402


def load_G(pkl: str, device: torch.device):
    with dnnlib.util.open_url(pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"]
    return G.eval().requires_grad_(False).to(device)


@torch.no_grad()
def class_w_avg(G, grade: int, device: torch.device, n: int = 10000) -> torch.Tensor:
    """Mean w of one grade (fixed z seed), the centre for class-conditional truncation.

    G.mapping.w_avg is ONE running mean over the training label mix (about half grade 0), so the
    default truncation pulls every grade-3/4 sample towards an average, mostly healthy, fundus.
    """
    c = torch.zeros([n, G.c_dim], device=device)
    c[:, grade] = 1
    z = torch.from_numpy(np.random.RandomState(2**31 - 1).randn(n, G.z_dim)).to(device)
    return G.mapping(z, c, truncation_psi=1)[:, 0].mean(0)


@torch.no_grad()
def generate_one(G, seed: int, grade: int, psi: float, device: torch.device,
                 center: torch.Tensor | None = None) -> Image.Image:
    """center=None: stylegan3's global truncation (w_avg); else truncate towards this grade's mean w."""
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    c = torch.zeros([1, G.c_dim], device=device)
    c[0, grade] = 1
    if center is None:
        img = G(z, c, truncation_psi=psi, noise_mode="const")
    else:
        ws = G.mapping(z, c, truncation_psi=1)
        img = G.synthesis(center + psi * (ws - center), noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    return Image.fromarray(img, "RGB")


@click.command()
@click.option("--pkl", required=True)
@click.option("--outdir", type=click.Path(path_type=Path), required=True)
@click.option("--classes", default="3,4")
@click.option("--n", type=int, default=500)
@click.option("--psi", type=float, default=0.8)
@click.option("--tag", default="", help="generator tag written into metadata, e.g. f100-ada")
@click.option("--trunc-center", type=click.Choice(["global", "class"]), default="global", show_default=True,
              help="truncate towards G's single w_avg (stylegan3 default) or towards each grade's mean w")
def main(pkl, outdir, classes, n, psi, tag, trunc_center):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = load_G(pkl, device)
    assert G.c_dim > 0, "unconditional pickle"
    for grade in [int(c) for c in classes.split(",")]:
        d = outdir / f"g{grade}"
        d.mkdir(parents=True, exist_ok=True)
        center = class_w_avg(G, grade, device) if trunc_center == "class" else None
        for seed in tqdm(range(n), desc=f"grade {grade}"):
            p = d / f"g{grade}_s{seed:05d}.png"
            if p.exists():
                continue
            save_synthetic(generate_one(G, seed, grade, psi, device, center), p, visible=False,
                           generator=tag or Path(pkl).name, grade=grade, seed=seed, psi=psi, trunc_center=trunc_center)
    click.echo(f"done -> {outdir}")


if __name__ == "__main__":
    main()

"""Verify that a class-conditional StyleGAN2 (--cond=1, c_dim=5) can be initialised from the
unconditional FFHQ-256 pickle with the shape-tolerant copy. CPU is fine (slow reference ops).

  python scripts/check_resume_compat.py            # uses GAN_PKLS/ffhq-res256-mirror-paper256-noaug.pkl or downloads it
  python scripts/check_resume_compat.py --pkl path/or/url

Prints the copied / partially-copied / skipped / new tensors and the train.py flags implied by
the source network's architecture, then runs one G and D forward pass.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import src.gan  # noqa: F401,E402
import dnnlib  # noqa: E402
import legacy  # noqa: E402
from torch_utils import misc  # noqa: E402
from training.networks_stylegan2 import Discriminator, Generator  # noqa: E402
from src.genai.paths import GAN_PKLS  # noqa: E402

FFHQ_URL = ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/"
            "ffhq-res256-mirror-paper256-noaug.pkl")
C_DIM = 5


def flatten_G_kwargs(kw: dict) -> dict:
    kw = dict(kw)
    kw.update(kw.pop("synthesis_kwargs", {}))
    return kw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    pkl = args.pkl or (str(GAN_PKLS / "ffhq-res256-mirror-paper256-noaug.pkl")
                       if (GAN_PKLS / "ffhq-res256-mirror-paper256-noaug.pkl").exists() else FFHQ_URL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; source {pkl}")
    with dnnlib.util.open_url(pkl) as f:
        data = legacy.load_network_pkl(f)
    src_G, src_D = data["G_ema"], data["D"]
    g_kw = flatten_G_kwargs(src_G.init_kwargs)
    d_kw = dict(src_D.init_kwargs)
    print("source G kwargs:", {k: v for k, v in g_kw.items() if k != "mapping_kwargs"}, g_kw.get("mapping_kwargs"))
    print("source D kwargs:", {k: v for k, v in d_kw.items() if not k.endswith("_kwargs")}, d_kw.get("epilogue_kwargs"))
    cbase, cmax = g_kw.get("channel_base", 32768), g_kw.get("channel_max", 512)
    map_depth = (g_kw.get("mapping_kwargs") or {}).get("num_layers", 8)
    mbstd = (d_kw.get("epilogue_kwargs") or {}).get("mbstd_group_size", 4)
    print(f"\n=> train.py flags: --cfg=stylegan2 --cond=1 --cbase={cbase} --cmax={cmax} --map-depth={map_depth} "
          f"--mbstd-group={min(mbstd, 8)}  (batch-gpu must be a multiple of mbstd-group)")

    g_kw["c_dim"] = C_DIM
    d_kw["c_dim"] = C_DIM
    G = Generator(**g_kw).to(device)
    D = Discriminator(**d_kw).to(device)
    print("\nG <- source G_ema")
    rg = misc.copy_params_and_buffers_tolerant(src_G, G, verbose=True)
    print("D <- source D")
    rd = misc.copy_params_and_buffers_tolerant(src_D, D, verbose=True)
    assert "mapping.fc0.weight [512, 512]->[512, 1024]" in rg["partial"], rg["partial"]
    assert any(s.startswith("b4.out.weight") for s in rd["skipped"]), rd["skipped"]
    assert not rg["skipped"], f"unexpected G skips {rg['skipped']}"
    assert rg["copied"] and rd["copied"]

    G.eval()
    D.eval()
    z = torch.randn(args.batch, G.z_dim, device=device)
    c = torch.nn.functional.one_hot(torch.arange(args.batch, device=device) % C_DIM, C_DIM).float()
    t0 = time.time()
    with torch.no_grad():
        img = G(z, c, truncation_psi=0.8, noise_mode="const", force_fp32=True)
        logits = D(img, c, force_fp32=True)
    print(f"\nforward OK: img {tuple(img.shape)} range [{img.min():.2f},{img.max():.2f}], D {tuple(logits.shape)}, "
          f"{time.time() - t0:.1f}s on {device} (custom-op fallback warnings above are expected without nvcc)")
    print("RESUME_COMPAT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

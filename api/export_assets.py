"""S15: build the demo assets folder (DATA_ROOT/demo_assets) and optionally upload it to HF Hub.

  python -m api.export_assets --pkl <GAN_PKLS>/f100-ada.pkl [--upload nabirakhan/dr-gan-demo]

Needs checkpoints/c5_f100_r0_s0/best.pt and checkpoints/c6_f100_r1_s0/best.pt (the queue keeps
them) and the cached arrays + splits. Produces G_ema_f100.pkl (D stripped), effnet_c5/c6.pt,
nn_index.npz + nn_thumbs.zip (f100 real train set), bank/ (20 watermarked samples per grade).
"""
import io
import os
import pickle
import shutil
import zipfile
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import src.gan  # noqa: F401
import legacy  # noqa: E402
from api.core import Backend  # noqa: E402
from src.data.split import fraction_mask, load_splits  # noqa: E402
from src.gan.watermark import save_synthetic  # noqa: E402
from src.genai.paths import CACHE, CHECKPOINTS, DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "demo_assets"


@click.command()
@click.option("--pkl", required=True, help="final f100 (ADA) network pickle")
@click.option("--upload", default=None, help="HF model repo id to upload to (needs HF_TOKEN)")
@click.option("--bank-per-grade", type=int, default=20)
def main(pkl, upload, bank_per_grade):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(pkl, "rb") as f:
        data = legacy.load_network_pkl(f)
    G = data["G_ema"].eval().requires_grad_(False).cpu()
    with open(OUT / "G_ema_f100.pkl", "wb") as f:
        pickle.dump({"G_ema": G}, f)
    click.echo(f"G_ema_f100.pkl {os.path.getsize(OUT / 'G_ema_f100.pkl') / 1e6:.0f} MB")
    for rid, name in (("c5_f100_r0_s0", "effnet_c5_f100.pt"), ("c6_f100_r1_s0", "effnet_c6_f100.pt")):
        src = CHECKPOINTS / rid / "best.pt"
        if not src.exists():
            raise SystemExit(f"{src} missing (queue keeps weights for these two runs)")
        shutil.copy2(src, OUT / name)

    be = Backend(OUT, device="cuda" if torch.cuda.is_available() else "cpu")
    df = load_splits()
    arr = np.load(CACHE / "aptos256.npy", mmap_mode="r")
    idx = np.where(fraction_mask(df, 1.0))[0]
    feats = np.stack([be.embed(np.ascontiguousarray(arr[i])) for i in tqdm(idx, desc="nn index")]).astype(np.float16)
    ids = df.image_id.to_numpy()[idx].astype(str)
    np.savez(OUT / "nn_index.npz", feats=feats, ids=ids)
    with zipfile.ZipFile(OUT / "nn_thumbs.zip", "w", zipfile.ZIP_STORED) as z:
        for i, img_id in zip(idx, ids):
            im = Image.fromarray(np.ascontiguousarray(arr[i])).resize((96, 96), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=80)
            z.writestr(f"{img_id}.jpg", buf.getvalue())
    bank = OUT / "bank"
    bank.mkdir(exist_ok=True)
    for g in range(5):
        for s in range(bank_per_grade):
            seed = 100000 + s
            save_synthetic(be.generate(g, seed, 0.8), bank / f"g{g}_s{seed}.png", visible=True, grade=g, seed=seed, psi=0.8,
                           generator=Path(pkl).name)
    click.echo(f"assets in {OUT}: " + ", ".join(p.name for p in sorted(OUT.iterdir())))
    if upload:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(upload, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(repo_id=upload, folder_path=str(OUT), repo_type="model")
        click.echo(f"uploaded to https://huggingface.co/{upload}")


if __name__ == "__main__":
    main()

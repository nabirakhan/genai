"""S14: Grad-CAM for the strong-aug (cond 5) vs GAN-augmented (cond 6) f100 models.

Reusable: cam_overlay(model, rgb_uint8, target=None) -> (overlay_rgb_uint8, probs) used by the demo API.
CLI writes results/figures/gradcam_examples.png with grade-3/4 test images: original | cond 5 | cond 6.
"""
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

from src.classify.data import MEAN, STD  # noqa: E402
from src.classify.model import create_model, gradcam_target_layer, load_weights  # noqa: E402
from src.data.split import load_splits  # noqa: E402
from src.genai.paths import CACHE, CHECKPOINTS, RESULTS  # noqa: E402

RUN_IDS = {"strong aug (5)": "c5_f100_r0_s0", "strong aug + GAN (6)": "c6_f100_r1_s0"}


def to_tensor(rgb: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
    x = (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
    return x[None].to(device)


def cam_overlay(model: torch.nn.Module, rgb: np.ndarray, target: int | None = None):
    device = next(model.parameters()).device
    x = to_tensor(rgb, device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0].cpu().numpy()
    target = int(probs.argmax()) if target is None else target
    cam = GradCAM(model=model, target_layers=[gradcam_target_layer(model)])
    gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
    overlay = show_cam_on_image(rgb.astype(np.float32) / 255, gray, use_rgb=True)
    return overlay, probs


def load_run_model(run_id: str, device):
    path = CHECKPOINTS / run_id / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; the queue keeps weights only for {list(RUN_IDS.values())}")
    return load_weights(create_model(pretrained=False), path, device)


@click.command()
@click.option("--n-per-grade", type=int, default=3)
@click.option("--out", type=click.Path(path_type=Path), default=RESULTS / "figures" / "gradcam_examples.png")
def main(n_per_grade, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {name: load_run_model(rid, device) for name, rid in RUN_IDS.items()}
    df = load_splits()
    arr = np.load(CACHE / "aptos256.npy", mmap_mode="r")
    picks = []
    for g in (3, 4):
        idx = np.where((df.split == "test") & (df.grade == g))[0][:n_per_grade]
        picks += [(int(i), g) for i in idx]
    fig, axes = plt.subplots(len(picks), 1 + len(models), figsize=(2.2 * (1 + len(models)), 2.2 * len(picks)))
    for r, (i, g) in enumerate(picks):
        rgb = np.ascontiguousarray(arr[i])
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{df.image_id[i]} (true grade {g})", fontsize=7)
        for c, (name, m) in enumerate(models.items(), start=1):
            overlay, probs = cam_overlay(m, rgb)
            axes[r, c].imshow(overlay)
            axes[r, c].set_title(f"{name}\npred {probs.argmax()} (p={probs.max():.2f})", fontsize=7)
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    click.echo(f"-> {out}")


if __name__ == "__main__":
    main()

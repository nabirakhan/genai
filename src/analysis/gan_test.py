"""GAN-test (lab PC, after S9): does a classifier trained on real images only see the conditioned grade
in the generated images?

  python -m src.analysis.gan_test

Judge: c5_f100_r0_s0 (RandAugment + mixup on every real training image, no synthetic data; its best.pt is
kept by src/classify/train.py KEEP_WEIGHTS). It scores synth/f<ppp>/g{3,4} and the synth/f<ppp>_psi1
variants with the eval transform, and the real val images of grades 3/4 as reference (val, not test; val
also picked the checkpoint's epoch, so its recall is slightly optimistic).

Per set and conditioned grade g, compare share_as_grade (share predicted as g) with real_val_recall
(the same classifier's recall on real grade-g val images):
- far above real recall: the images are more typical than real grade-g eyes, which hints at memorised or
  over-typical (truncated, mode-collapsed) images; cross-check with the memorisation check (S12).
- well below, with pred_0..pred_2 high: drift towards grades 0-2, i.e. class leakage: the GAN ignores the
  rare label and falls back on the grades it saw far more often (results/tables/fraction_label_shares.md).
- pred_3 high for grade-4 images or the reverse: the GAN mixes the two rare grades.

results/tables/gan_test.csv   set, fraction, psi, grade, n, share_as_grade, real_val_recall, mean_p_grade,
                              pred_0..pred_4 (aggregates only, no per-image rows)
results/figures/gan_test.png  predicted-grade distribution per set, one panel per conditioned grade
"""
import re
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from PIL import Image  # noqa: E402

from src.classify.data import NUM_CLASSES, build_transform  # noqa: E402
from src.classify.model import DEFAULT_MODEL, create_model  # noqa: E402
from src.data.cache import check_rows  # noqa: E402
from src.data.split import load_splits  # noqa: E402
from src.genai.paths import CACHE, CHECKPOINTS, RARE_GRADES, RESULTS, SYNTH  # noqa: E402

JUDGE = "c5_f100_r0_s0"
SET_RE = re.compile(r"f(\d{3})(_psi1)?")
REAL = "real val"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # predicted grade 0..4: one hue, light to dark


@torch.no_grad()
def predict(model, imgs, device, batch: int = 64) -> np.ndarray:
    """Softmax probabilities for HxWx3 uint8 images: eval transform, fp16 autocast on CUDA as in training."""
    tf = build_transform(1, False)
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.stack([tf(torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)) for a in imgs[i:i + batch]])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device).contiguous(memory_format=torch.channels_last))
        out.append(logits.float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def summarise(name: str, fraction: float, psi: str, g: int, probs: np.ndarray, recall: float) -> dict:
    share = np.bincount(probs.argmax(1), minlength=NUM_CLASSES) / len(probs)
    return dict(set=name, fraction=fraction, psi=psi, grade=g, n=len(probs), share_as_grade=round(float(share[g]), 4),
                real_val_recall=recall, mean_p_grade=round(float(probs[:, g].mean()), 4),
                **{f"pred_{k}": round(float(share[k]), 4) for k in range(NUM_CLASSES)})


def figure(res: pd.DataFrame, run_id: str, out: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    order = [REAL] + sorted(s for s in res["set"].unique() if s != REAL)
    psi = res.drop_duplicates("set").set_index("set").psi
    labels = [s if s == REAL else f"{s.split('_')[0]}  psi {psi[s] or '?'}" for s in order]
    grades = sorted(res.grade.unique())
    fig, axes = plt.subplots(1, len(grades), figsize=(3.6 * len(grades), 0.34 * len(order) + 1.4), sharey=True, squeeze=False)
    for ax, g in zip(axes[0], grades):
        sub = res[res.grade == g].set_index("set").reindex(order)
        left = np.zeros(len(order))
        for k in range(NUM_CLASSES):
            w = sub[f"pred_{k}"].fillna(0).to_numpy()
            ax.barh(np.arange(len(order)), w, left=left, height=0.6, color=RAMP[k], edgecolor="white", linewidth=1,
                    label=f"grade {k}")
            if k == g:
                for y, (x0, wk) in enumerate(zip(left, w)):
                    if wk >= 0.12:
                        ax.text(x0 + wk / 2, y, f"{wk:.0%}", ha="center", va="center", color="white", fontsize=7)
            left += w
        ax.set_title(f"conditioned / true grade {g}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0, 0].set_yticks(np.arange(len(order)), labels)
    axes[0, 0].invert_yaxis()
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, title="predicted", ncol=NUM_CLASSES, loc="lower center", frameon=False, fontsize=7, title_fontsize=7)
    fig.suptitle(f"GAN-test: {run_id} (trained on real images only) classifies generated images\n"
                 "labels: share predicted as the conditioned grade (real val: recall)", fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


@click.command()
@click.option("--run-id", default=JUDGE, show_default=True, help="classifier under checkpoints/<run-id>/best.pt")
@click.option("--out-csv", type=click.Path(path_type=Path), default=RESULTS / "tables" / "gan_test.csv")
@click.option("--out-fig", type=click.Path(path_type=Path), default=RESULTS / "figures" / "gan_test.png")
def main(run_id, out_csv, out_fig):
    sets = sorted(d.name for d in SYNTH.glob("f*") if d.is_dir() and SET_RE.fullmatch(d.name)) if SYNTH.exists() else []
    if not sets:
        raise SystemExit(f"no synth/f<ppp> sets under {SYNTH}: import them first (S9)")
    path = CHECKPOINTS / run_id / "best.pt"
    if not path.exists():
        raise SystemExit(f"{path} missing: the classifier queue keeps it for {run_id} (S8)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    runs_csv = RESULTS / "classifier_runs.csv"
    t = pd.to_datetime(pd.read_csv(runs_csv).query("run_id == @run_id").timestamp, utc=True) if runs_csv.exists() else pd.Series()
    if ck.get("run_id") != run_id or t.empty or path.stat().st_mtime < t.max().timestamp() - 5:
        raise SystemExit(f"{path} is not the finished {run_id} recorded in {runs_csv} (superseded or unfinished run?)")
    model = create_model(ck.get("model_name", DEFAULT_MODEL), pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    df = load_splits()
    check_rows(CACHE / "aptos256.npy", df.image_id.tolist())
    arr = np.load(CACHE / "aptos256.npy", mmap_mode="r")
    rows, recall = [], {}
    for g in RARE_GRADES:
        probs = predict(model, arr[np.where((df.split == "val") & (df.grade == g))[0]], device)
        recall[g] = round(float((probs.argmax(1) == g).mean()), 4)
        rows.append(summarise(REAL, float("nan"), "", g, probs, recall[g]))
    for s in sets:
        for g in RARE_GRADES:
            files = sorted((SYNTH / s / f"g{g}").glob("*.png"))
            if not files:
                continue
            with Image.open(files[0]) as im:
                psi = im.text.get("psi", "")
            probs = predict(model, [np.asarray(Image.open(f).convert("RGB")) for f in files], device)
            rows.append(summarise(s, int(SET_RE.fullmatch(s).group(1)) / 100, psi, g, probs, recall[g]))
    res = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    figure(res, run_id, out_fig)
    click.echo(res.to_string(index=False))
    click.echo(f"-> {out_csv}, {out_fig}")


if __name__ == "__main__":
    main()

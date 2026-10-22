"""S12: nearest-neighbour memorisation check for every imported synthetic set.

For synth/f{ppp}/g{3,4}: Inception features (same detector as FID) of each synthetic image vs
the real grade-3/4 TRAIN images of that fraction (what the GAN saw). Reference distribution:
real-vs-real leave-one-out nearest-neighbour distances within the same set. A synthetic image
closer than the 1st percentile of real-real distances is flagged as a possible copy.

results/tables/memorization_{tag}.csv       per synthetic image: file, grade, nn_image_id, nn_dist
results/tables/memorization_summary.csv     tag, grade, n_synth, median_dist, p1_real_real, n_flagged
results/figures/memorization_{tag}.png      10 closest synthetic/real pairs
"""
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import src.gan  # noqa: F401,E402
from metrics import metric_utils  # noqa: E402
from src.data.split import fraction_mask, load_splits  # noqa: E402
from src.gan.metrics_per_class import DETECTOR_URL, features  # noqa: E402
from src.gan.watermark import add_visible_label  # noqa: E402
from src.genai.paths import CACHE, FRACTIONS, RARE_GRADES, RESULTS, SYNTH, frac_tag  # noqa: E402


def cosine_nn(a: np.ndarray, b: np.ndarray, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    d = 1 - a @ b.T
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1), d.argmin(axis=1)


def run_tag(p: float, detector, device, out_tables: Path, out_figs: Path) -> list[dict]:
    tag = frac_tag(p)
    synth_root = SYNTH / tag
    if not synth_root.exists():
        return []
    df = load_splits()
    arr = np.load(CACHE / "aptos256.npy", mmap_mode="r")
    rows_summary, rows = [], []
    pairs = []
    for g in RARE_GRADES:
        real_idx = np.where(fraction_mask(df, p) & (df.grade == g))[0]
        real_imgs = [np.ascontiguousarray(arr[i]) for i in real_idx]
        real_ids = df.image_id.to_numpy()[real_idx]
        cache = CACHE / f"inception_{tag}_g{g}.npy"
        if cache.exists():
            fr = np.load(cache)
        else:
            fr = features(real_imgs, detector, device)
            np.save(cache, fr)
        files = sorted((synth_root / f"g{g}").glob("*.png"))
        synth_imgs = [np.asarray(Image.open(f).convert("RGB")) for f in files]
        fs = features(synth_imgs, detector, device)
        rr, _ = cosine_nn(fr, fr, exclude_self=True)
        p1 = float(np.percentile(rr, 1)) if len(rr) > 1 else 0.0
        ds, nn = cosine_nn(fs, fr)
        for f, d, j in zip(files, ds, nn):
            rows.append(dict(file=f.name, grade=g, nn_image_id=real_ids[j], nn_dist=float(d), flagged=bool(d < p1)))
        rows_summary.append(dict(tag=tag, grade=g, n_synth=len(files), n_real=len(real_idx), median_dist=float(np.median(ds)),
                                 median_real_real=float(np.median(rr)), p1_real_real=p1, n_flagged=int((ds < p1).sum())))
        order = np.argsort(ds)[:5]
        pairs += [(synth_imgs[k], real_imgs[nn[k]], ds[k], g) for k in order]
    pd.DataFrame(rows).to_csv(out_tables / f"memorization_{tag}.csv", index=False)
    fig, axes = plt.subplots(2, len(pairs), figsize=(1.3 * len(pairs), 2.9))
    for k, (s, r, d, g) in enumerate(pairs):
        axes[0, k].imshow(np.asarray(add_visible_label(Image.fromarray(s))))
        axes[0, k].set_title(f"g{g} d={d:.3f}", fontsize=6)
        axes[1, k].imshow(r)
        for a in (axes[0, k], axes[1, k]):
            a.axis("off")
    axes[0, 0].set_ylabel("synthetic")
    axes[1, 0].set_ylabel("nearest real")
    fig.suptitle(f"{tag}: closest synthetic / real-train pairs (Inception cosine distance)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / f"memorization_{tag}.png", dpi=150)
    plt.close(fig)
    return rows_summary


@click.command()
@click.option("--fractions", default="all", help="comma list like 0.1,1.0 or 'all'")
def main(fractions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = metric_utils.get_feature_detector(DETECTOR_URL, device=device, num_gpus=1, rank=0, verbose=True)
    out_tables, out_figs = RESULTS / "tables", RESULTS / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    fr = FRACTIONS if fractions == "all" else [float(x) for x in fractions.split(",")]
    summary = []
    for p in fr:
        summary += run_tag(p, detector, device, out_tables, out_figs)
    if summary:
        pd.DataFrame(summary).to_csv(out_tables / "memorization_summary.csv", index=False)
        click.echo(pd.DataFrame(summary).to_string(index=False))
    else:
        click.echo("no synthetic sets found under " + str(SYNTH))


if __name__ == "__main__":
    main()

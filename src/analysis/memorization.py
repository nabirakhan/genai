"""S12: memorisation check for every imported synthetic set (synth/f<ppp>/ and synth/f<ppp>_psi1/).

Each synthetic image is compared with ALL real training images of its fraction, every grade (what
that GAN saw), in both orientations: the GANs train with x-flips, so a memorising GAN can emit a
mirrored copy. The held-out reference is scored the same way.
1. Feature space: cosine distance to the nearest training image in Inception features (the FID
   detector). Reference: the same distance for the held-out real val images (all grades), i.e. how
   close a new real eye gets to the training set. n_below_p1 / n_below_p5 count synthetic images
   closer than the reference's 1st / 5th percentile. An excess means "more typical than new real
   eyes", which truncation (psi 0.8) and grade-3/4-only sets produce without copying: compare with
   the psi-1 set of the same GAN, and use the vessel check as the copy test.
2. Pixel space (the copy test): the dedup vessel-pattern correlation (src/data/dedup.py vessel_field
   + aligned_corr, shift-tolerant; >= 0.90 means the same photograph) against the K nearest training
   images in features plus the K best by plain vessel correlation. n_vessel_copies: >= 0.90.
   n_vessel_above_ref: closer to a training eye than any held-out real eye gets (ref_max_vessel).

  python -m src.analysis.memorization                  # every set; --sets f010,f100_psi1 for some

results/tables/memorization_summary.csv   per set and grade, aggregate counts only (committed)
results/tables/memorization_<set>.csv     per synthetic image with nearest-neighbour ids (gitignored)
results/figures/memorization_<set>.png    closest pairs + every vessel match (gitignored: real APTOS eyes)
results/tables/memorization_<set>_pairs.csv  one row per figure panel: the drawn pair and its own scores (gitignored)
Needs cache/aptos256.npy (S5) and the aptos256/ PNGs (S3); Inception features of the train+val
images are cached in cache/inception_trainval.npy.
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
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.cache import check_rows  # noqa: E402
from src.data.dedup import MASK, aligned_corr, unit, vessel_field  # noqa: E402
from src.data.split import fraction_mask, load_splits  # noqa: E402
from src.gan.metrics_per_class import features, load_detector  # noqa: E402
from src.gan.watermark import add_visible_label  # noqa: E402
from src.genai.paths import APTOS256, CACHE, RARE_GRADES, RESULTS, SYNTH  # noqa: E402

SET_RE = re.compile(r"f(\d{3})(_psi1)?")
VESSEL_COPY = 0.90  # src/data/dedup.py duplicate threshold
K = 5


def nearest(a: np.ndarray, b: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distances and indices of the k nearest rows of b for every row of a, nearest first."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    d = 1 - a @ b.T
    idx = np.argsort(d, axis=1)[:, :k]
    return np.take_along_axis(d, idx, 1), idx


def nearest_mirrored(fa: np.ndarray, fa_flip: np.ndarray, fb: np.ndarray):
    """nearest() over both orientations of every query: distance/indices of the closer one, whether that
    was the flip, and the candidates of both orientations (for the vessel check)."""
    d1, i1 = nearest(fa, fb)
    d2, i2 = nearest(fa_flip, fb)
    flip = d2[:, 0] < d1[:, 0]
    return np.where(flip[:, None], d2, d1), np.where(flip[:, None], i2, i1), flip, np.concatenate([i1, i2], axis=1)


def vessel_best(fq: np.ndarray, ft: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best aligned vessel correlation of each query field, either orientation, over its candidates (given rows
    of ft plus the K best rows by plain correlation): (score, row of ft, matched via the flip)."""
    out = []
    for q in (fq, np.ascontiguousarray(fq[:, :, ::-1])):  # fields commute with x-flips (symmetric blur/CLAHE/mask)
        plain = unit(q[:, MASK]) @ unit(ft[:, MASK]).T
        c_all = np.concatenate([cand, np.argsort(-plain, axis=1)[:, :K]], axis=1)
        best, arg = np.zeros(len(q)), np.zeros(len(q), int)
        for i in tqdm(range(len(q)), desc="vessel corr", leave=False):
            c = np.unique(c_all[i])
            s = [aligned_corr(q[i], ft[j]) for j in c]
            best[i], arg[i] = max(s), c[int(np.argmax(s))]
        out.append((best, arg))
    (s1, a1), (s2, a2) = out
    flip = s2 > s1
    return np.where(flip, s2, s1), np.where(flip, a2, a1), flip


def pair_vessel(q: np.ndarray, r: np.ndarray) -> float:
    """Vessel correlation of one drawn pair, either orientation (as vessel_best)."""
    return max(aligned_corr(q, r), aligned_corr(np.ascontiguousarray(q[:, ::-1]), r))


def pair_dist(a: np.ndarray, a_flip: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance of one drawn pair, closer orientation (as nearest_mirrored)."""
    return max(0.0, float(nearest(np.stack([a, a_flip]), b[None], 1)[0].min()))


def real_features(arr: np.ndarray, ids: np.ndarray, rows: np.ndarray, detector, device, kind: str) -> np.ndarray:
    path = CACHE / f"{kind}_trainval.npy"
    ids_path = path.with_suffix(".ids.txt")
    want = "\n".join(ids) + "\n"
    if path.exists() and ids_path.exists() and ids_path.read_text(encoding="utf-8") == want:
        return np.load(path)
    f = np.concatenate([features(arr[rows[i:i + 256]], detector, device) for i in range(0, len(rows), 256)])
    ids_path.unlink(missing_ok=True)
    np.save(path, f)
    ids_path.write_text(want, encoding="utf-8")
    return f


def batch_features(imgs: list[np.ndarray], detector, device) -> np.ndarray:
    return np.concatenate([features(imgs[i:i + 256], detector, device) for i in range(0, len(imgs), 256)])


def pair_figure(name: str, pairs: list, arr: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(2, len(pairs), figsize=(1.3 * len(pairs), 3.3), squeeze=False)
    for k, (f, j, d, v, g, how) in enumerate(pairs):
        axes[0, k].imshow(np.asarray(add_visible_label(Image.open(f).convert("RGB"))))
        axes[0, k].set_title(f"g{g} d={max(d, 0.0):.3f}\nvessel {v:.2f}\n{how}", fontsize=5)
        axes[1, k].imshow(np.ascontiguousarray(arr[j]))
        for a in axes[:, k]:
            a.set_xticks([])
            a.set_yticks([])
            a.spines[:].set_visible(False)
    axes[0, 0].set_ylabel("synthetic", fontsize=7)
    axes[1, 0].set_ylabel("real train match", fontsize=7)
    fig.suptitle(f"{name}: 5 feature-closest synthetic images per grade + every vessel match "
                 "(d: Inception cosine distance, vessel: dedup correlation, either orientation)", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


@click.command()
@click.option("--sets", default="all", help="comma list like f010,f100_psi1, or 'all' synth/f<ppp>[_psi1] folders")
@click.option("--detector", "detector_kind", type=click.Choice(["inception", "stub"]), default="inception", hidden=True)
def main(sets, detector_kind):
    found = sorted(d.name for d in SYNTH.glob("f*") if d.is_dir() and SET_RE.fullmatch(d.name)) if SYNTH.exists() else []
    todo = found if sets == "all" else sets.split(",")
    if unknown := sorted(set(todo) - set(found)):
        raise SystemExit(f"unknown synthetic sets {unknown}; found {found}")
    if not todo:
        raise SystemExit(f"no synthetic sets under {SYNTH}: import them first (S9)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(device, detector_kind)
    df = load_splits()
    check_rows(CACHE / "aptos256.npy", df.image_id.tolist())
    arr = np.load(CACHE / "aptos256.npy", mmap_mode="r")
    ids, grades = df.image_id.to_numpy(), df.grade.to_numpy()
    tv = np.where(df.split.isin(["train", "val"]))[0]
    if not all((APTOS256 / f"{i}.png").exists() for i in ids[tv]):
        raise SystemExit(f"{APTOS256} lacks train/val PNGs (S3): the vessel check reads them")
    pos = np.full(len(df), -1)
    pos[tv] = np.arange(len(tv))
    feats = real_features(arr, ids[tv], tv, detector, device, detector_kind)
    fields = np.stack([vessel_field(APTOS256 / f"{i}.png") for i in tqdm(ids[tv], desc="vessel fields", leave=False)])
    val = np.where(df.split == "val")[0]
    val_flip = batch_features([np.ascontiguousarray(arr[i][:, ::-1]) for i in val], detector, device)
    out_t, out_f = RESULTS / "tables", RESULTS / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    refs, summary = {}, []
    for name in todo:
        p = int(SET_RE.fullmatch(name).group(1)) / 100
        train = np.where(fraction_mask(df, p))[0]
        ft, vt = feats[pos[train]], fields[pos[train]]
        if p not in refs:
            d, _, _, cand = nearest_mirrored(feats[pos[val]], val_flip, ft)
            refs[p] = d[:, 0], vessel_best(fields[pos[val]], vt, cand)[0]
        ref, ref_v = refs[p]
        p1, p5 = np.percentile(ref, [1, 5])
        rows, pairs = [], []
        for g in RARE_GRADES:
            files = sorted((SYNTH / name / f"g{g}").glob("*.png"))
            if not files:
                continue
            imgs = [np.asarray(Image.open(f).convert("RGB")) for f in files]
            fs = batch_features(imgs, detector, device)
            fs_flip = batch_features([np.ascontiguousarray(a[:, ::-1]) for a in imgs], detector, device)
            d, i, mir, cand = nearest_mirrored(fs, fs_flip, ft)
            fq = np.stack([vessel_field(f) for f in files])
            v, vi, vmir = vessel_best(fq, vt, cand)
            nn, dist = train[i[:, 0]], d[:, 0]
            rows += [dict(file=f.name, grade=g, nn_image_id=ids[a], nn_dist=round(float(b), 5), nn_mirrored=bool(m),
                          below_p1=bool(b < p1), below_p5=bool(b < p5), vessel_corr=round(float(c), 4),
                          vessel_image_id=ids[train[e]], vessel_mirrored=bool(vm), vessel_copy=bool(c >= VESSEL_COPY))
                     for f, a, b, m, c, e, vm in zip(files, nn, dist, mir, v, vi, vmir)]
            summary.append(dict(set=name, fraction=p, grade=g, n_synth=len(files), n_train=len(train), n_ref=len(ref),
                                synth_median_dist=round(float(np.median(dist)), 5), ref_median_dist=round(float(np.median(ref)), 5),
                                ref_median_grade=round(float(np.median(ref[grades[val] == g])), 5), ref_p1=round(float(p1), 5),
                                ref_p5=round(float(p5), 5), n_below_p1=int((dist < p1).sum()), n_below_p5=int((dist < p5).sum()),
                                n_vessel_copies=int((v >= VESSEL_COPY).sum()), n_vessel_above_ref=int((v > ref_v.max()).sum()),
                                max_vessel=round(float(v.max()), 4), ref_max_vessel=round(float(ref_v.max()), 4),
                                n_mirrored_nn=int(mir.sum())))
            closest = list(np.argsort(dist)[:5])
            pairs += [(files[k], nn[k], dist[k], pair_vessel(fq[k], vt[i[k, 0]]), g, "nearest" + (" (mirrored)" if mir[k] else ""))
                      for k in closest]
            pairs += [(files[k], train[vi[k]], pair_dist(fs[k], fs_flip[k], ft[vi[k]]), v[k], g,
                       "vessel match" + (" (mirrored)" if vmir[k] else "")) for k in np.where((v >= VESSEL_COPY) | (v > ref_v.max()))[0]]
        pd.DataFrame(rows).to_csv(out_t / f"memorization_{name}.csv", index=False)
        if pairs:
            pd.DataFrame([dict(file=f.name, grade=g, panel=how, real_image_id=ids[j], pair_dist=round(float(d), 5),
                               pair_vessel=round(float(v), 4)) for f, j, d, v, g, how in pairs]
                         ).to_csv(out_t / f"memorization_{name}_pairs.csv", index=False)
            pair_figure(name, pairs, arr, out_f / f"memorization_{name}.png")
    new = pd.DataFrame(summary)
    path = out_t / "memorization_summary.csv"
    if sets != "all" and path.exists():
        old = pd.read_csv(path)
        if "set" in old.columns:
            new = pd.concat([old[~old["set"].isin(todo)], new], ignore_index=True)
    new = new.sort_values(["set", "grade"]).reset_index(drop=True)
    new.to_csv(path, index=False)
    click.echo(new.to_string(index=False))


if __name__ == "__main__":
    main()

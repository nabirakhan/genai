"""Datasets, per-condition augmentation pipelines and loaders.

Conditions (locked design):
 1 none            plain cross-entropy
 2 class_weights   CE weighted by N / (K * n_k) on the fraction's actual train set
 3 oversample      WeightedRandomSampler (1 / n_k), same epoch length as the train set
 4 classical       H/V flip, rotation up to 180 deg (bilinear), colour jitter
 5 strong          classical + FundusRandAugment(2 ops, magnitude ra_magnitude) + mixup(alpha 0.4, applied in
                   the train loop). torchvision spreads each op's range over 31 bins, so magnitude 9 is 0.3 of
                   the max, about M=3 on the RandAugment paper / timm 0-10 scale (15 is about M=5)
 6 strong_gan      strong + synthetic grade-3/4 images at ratio x real rare-grade count, from synth_dir()
ra_magnitude and synth_variant default to config.json (set by the pre-registered checks, src/analysis/checks.py).
Val / test / IDRiD are never augmented and never contain synthetic images.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from src.data.cache import check_rows
from src.data.split import fraction_mask, load_splits
from src.genai.paths import CACHE, GRADES, RARE_GRADES, SPLITS, SYNTH, frac_tag

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONDITIONS = {1: "none", 2: "class_weights", 3: "oversample", 4: "classical", 5: "strong", 6: "strong_gan"}
NUM_CLASSES = len(GRADES)
BILINEAR = T.InterpolationMode.BILINEAR


def config_path() -> Path:
    return Path(os.environ.get("GENAI_CLASSIFY_CONFIG", Path(__file__).with_name("config.json")))


def load_config() -> dict:
    return json.loads(config_path().read_text(encoding="utf-8"))


class FundusRandAugment(T.RandAugment):
    """RandAugment without Rotate (RandomRotation covers it) and the Posterize/Solarize/Equalize intensity remaps."""
    _AUGMENTATION_SPACE = {k: v for k, v in T.RandAugment._AUGMENTATION_SPACE.items()
                           if k not in ("Rotate", "Posterize", "Solarize", "Equalize")}


def build_transform(condition: int, train: bool, ra_magnitude: int | None = None) -> T.Compose:
    ops: list = []
    if train and condition >= 4:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180, interpolation=BILINEAR),
                T.ColorJitter(0.2, 0.2, 0.1)]
    if train and condition >= 5:
        m = load_config()["ra_magnitude"] if ra_magnitude is None else ra_magnitude
        ops += [FundusRandAugment(num_ops=2, magnitude=m, num_magnitude_bins=31, interpolation=BILINEAR, fill=0)]
    ops += [T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


class FundusDataset(Dataset):
    """Rows of a uint8 (N,H,W,3) .npy (opened lazily with mmap, so it is spawn-safe) plus
    optional synthetic PNG files appended at the end."""

    def __init__(self, npy_path, indices, labels, transform, synth_paths=(), synth_labels=()):
        self.npy_path = str(npy_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.synth_paths = [str(p) for p in synth_paths]
        self.synth_labels = np.asarray(synth_labels, dtype=np.int64)
        self.transform = transform
        self._arr = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_arr"] = None
        return d

    def __len__(self) -> int:
        return len(self.indices) + len(self.synth_paths)

    @property
    def targets(self) -> np.ndarray:
        return np.concatenate([self.labels, self.synth_labels]) if len(self.synth_paths) else self.labels

    def __getitem__(self, i: int):
        if i < len(self.indices):
            if self._arr is None:
                self._arr = np.load(self.npy_path, mmap_mode="r")
            img = np.ascontiguousarray(self._arr[self.indices[i]])
            y = int(self.labels[i])
        else:
            j = i - len(self.indices)
            img = np.asarray(Image.open(self.synth_paths[j]).convert("RGB"))
            y = int(self.synth_labels[j])
        x = torch.from_numpy(img).permute(2, 0, 1)  # uint8 CHW
        return self.transform(x), y


def synth_dir(fraction: float, variant: str | None = None) -> Path:
    """synth/f<ppp> for variant "", synth/f<ppp>_<variant> otherwise (None: config.json)."""
    v = load_config()["synth_variant"] if variant is None else variant
    return SYNTH / (frac_tag(fraction) + (f"_{v}" if v else ""))


def synth_files(fraction: float, ratio: float, n_real: dict[int, int], variant: str | None = None) -> tuple[list[Path], list[int]]:
    """Deterministic prefix of the generated images for this fraction's GAN."""
    root = synth_dir(fraction, variant)
    paths, labels = [], []
    for g in RARE_GRADES:
        files = sorted((root / f"g{g}").glob("*.png"))
        k = int(round(ratio * n_real[g]))
        if len(files) < k:
            raise FileNotFoundError(f"need {k} synthetic grade-{g} images in {root / f'g{g}'}, found {len(files)}")
        paths += files[:k]
        labels += [g] * k
    return paths, labels


def make_loaders(condition: int, fraction: float, ratio: float, batch_size: int = 64, workers: int = 6,
                 seed: int = 0, eval_batch_size: int = 128, ra_magnitude: int | None = None,
                 synth_variant: str | None = None) -> tuple[dict, dict]:
    df = load_splits()
    arr_path = CACHE / "aptos256.npy"
    check_rows(arr_path, df.image_id.tolist())
    tr_idx = np.where(fraction_mask(df, fraction))[0]
    tr_y = df.grade.to_numpy()[tr_idx]
    n_real = {g: int((tr_y == g).sum()) for g in RARE_GRADES}
    synth_p, synth_y = synth_files(fraction, ratio, n_real, synth_variant) if condition == 6 else ([], [])

    train_ds = FundusDataset(arr_path, tr_idx, tr_y, build_transform(condition, True, ra_magnitude), synth_p, synth_y)
    eval_tf = build_transform(1, False)
    val_idx = np.where(df.split == "val")[0]
    test_idx = np.where(df.split == "test")[0]
    val_ds = FundusDataset(arr_path, val_idx, df.grade.to_numpy()[val_idx], eval_tf)
    test_ds = FundusDataset(arr_path, test_idx, df.grade.to_numpy()[test_idx], eval_tf)
    idrid_ds, idrid_ids = None, None
    if (CACHE / "idrid256.npy").exists() and (SPLITS / "idrid_test.csv").exists():
        idf = pd.read_csv(SPLITS / "idrid_test.csv")
        check_rows(CACHE / "idrid256.npy", idf.image_id.tolist())
        idrid_ds = FundusDataset(CACHE / "idrid256.npy", np.arange(len(idf)), idf.grade.to_numpy(), eval_tf)
        idrid_ids = idf.image_id.tolist()

    targets = train_ds.targets
    counts = np.bincount(targets, minlength=NUM_CLASSES)
    gen = torch.Generator().manual_seed(seed)
    sampler, shuffle = None, True
    if condition == 3:
        w = torch.as_tensor(1.0 / counts[targets], dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(train_ds), replacement=True, generator=gen)
        shuffle = False
    class_weights = None
    if condition == 2:
        class_weights = torch.tensor(len(targets) / (NUM_CLASSES * np.maximum(counts, 1)), dtype=torch.float32)

    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    once = dict(num_workers=0, pin_memory=torch.cuda.is_available())  # iterated once per run: 6 spawned workers cost more than they save
    loaders = {
        "train": DataLoader(train_ds, batch_size, shuffle=shuffle, sampler=sampler, generator=gen,
                            drop_last=len(train_ds) >= 2 * batch_size, **common),
        "val": DataLoader(val_ds, eval_batch_size, shuffle=False, **common),
        "aptos_test": DataLoader(test_ds, eval_batch_size, shuffle=False, **once),
    }
    if idrid_ds is not None:
        loaders["idrid"] = DataLoader(idrid_ds, eval_batch_size, shuffle=False, **once)
    info = dict(n_train=int(len(tr_idx)), n_synth=int(len(synth_p)), counts=counts.tolist(), n_real_rare=n_real,
                class_weights=class_weights, ids={"aptos_test": df.image_id.to_numpy()[test_idx].tolist(),
                                                  "val": df.image_id.to_numpy()[val_idx].tolist(), "idrid": idrid_ids})
    return loaders, info

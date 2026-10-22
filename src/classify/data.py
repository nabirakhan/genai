"""Datasets, per-condition augmentation pipelines and loaders.

Conditions (locked design):
 1 none            plain cross-entropy
 2 class_weights   CE weighted by N / (K * n_k) on the fraction's actual train set
 3 oversample      WeightedRandomSampler (1 / n_k), same epoch length as the train set
 4 classical       H/V flip, rotation up to 180 deg, colour jitter
 5 strong          classical + RandAugment(2, 9) + mixup(alpha 0.4, applied in the train loop)
 6 strong_gan      strong + synthetic grade-3/4 images at ratio x real rare-grade count
Val / test / IDRiD are never augmented and never contain synthetic images.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from src.data.split import fraction_mask, load_splits
from src.genai.paths import CACHE, GRADES, RARE_GRADES, SPLITS, SYNTH, frac_tag

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONDITIONS = {1: "none", 2: "class_weights", 3: "oversample", 4: "classical", 5: "strong", 6: "strong_gan"}
NUM_CLASSES = len(GRADES)


def build_transform(condition: int, train: bool) -> T.Compose:
    ops: list = []
    if train and condition >= 4:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180), T.ColorJitter(0.2, 0.2, 0.1)]
    if train and condition >= 5:
        ops += [T.RandAugment(num_ops=2, magnitude=9)]
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


def synth_files(fraction: float, ratio: float, n_real: dict[int, int]) -> tuple[list[Path], list[int]]:
    """Deterministic prefix of the generated images for this fraction's GAN."""
    root = SYNTH / frac_tag(fraction)
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
                 seed: int = 0, eval_batch_size: int = 128) -> tuple[dict, dict]:
    df = load_splits()
    arr_path = CACHE / "aptos256.npy"
    tr_idx = np.where(fraction_mask(df, fraction))[0]
    tr_y = df.grade.to_numpy()[tr_idx]
    n_real = {g: int((tr_y == g).sum()) for g in RARE_GRADES}
    synth_p, synth_y = synth_files(fraction, ratio, n_real) if condition == 6 else ([], [])

    train_ds = FundusDataset(arr_path, tr_idx, tr_y, build_transform(condition, True), synth_p, synth_y)
    eval_tf = build_transform(1, False)
    val_idx = np.where(df.split == "val")[0]
    test_idx = np.where(df.split == "test")[0]
    val_ds = FundusDataset(arr_path, val_idx, df.grade.to_numpy()[val_idx], eval_tf)
    test_ds = FundusDataset(arr_path, test_idx, df.grade.to_numpy()[test_idx], eval_tf)
    idrid_ds, idrid_ids = None, None
    if (CACHE / "idrid256.npy").exists() and (SPLITS / "idrid_test.csv").exists():
        idf = pd.read_csv(SPLITS / "idrid_test.csv")
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
    loaders = {
        "train": DataLoader(train_ds, batch_size, shuffle=shuffle, sampler=sampler, generator=gen,
                            drop_last=len(train_ds) >= 2 * batch_size, **common),
        "val": DataLoader(val_ds, eval_batch_size, shuffle=False, **common),
        "aptos_test": DataLoader(test_ds, eval_batch_size, shuffle=False, **common),
    }
    if idrid_ds is not None:
        loaders["idrid"] = DataLoader(idrid_ds, eval_batch_size, shuffle=False, **common)
    info = dict(n_train=int(len(tr_idx)), n_synth=int(len(synth_p)), counts=counts.tolist(), n_real_rare=n_real,
                class_weights=class_weights, ids={"aptos_test": df.image_id.to_numpy()[test_idx].tolist(),
                                                  "val": df.image_id.to_numpy()[val_idx].tolist(), "idrid": idrid_ids})
    return loaders, info

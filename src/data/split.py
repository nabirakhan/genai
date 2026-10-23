"""S4b: stratified 70/15/15 split + nested rare-grade fractions.

splits/aptos_splits.csv (committed) columns: image_id, split (train/val/test), frac_rank.
It holds no grades: APTOS labels may not be redistributed and the repo is public, so
load_splits() joins `grade` from the local raw/aptos/train.csv on every machine.
frac_rank: position of a TRAIN image in one seeded permutation of all train images (-1 for
val/test), so it reveals nothing about the grade. Fraction p keeps, per rare grade g, the
ceil(p * n_g) grade-g train images with the lowest frac_rank: nested by construction
(10% in 25% in 50% in 100%).

The committed CSV is the source of truth (written by the Kaggle data-prep notebook, which built
the GAN zips from it). Without --force this script only verifies it against train.csv.
"""
import hashlib
import math

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.genai.paths import FRACTIONS, RARE_GRADES, RAW_APTOS, SPLIT_SEED, SPLITS

PUBLIC_COLS = ["image_id", "split", "frac_rank"]


def aptos_labels() -> pd.DataFrame:
    path = RAW_APTOS / "train.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing: grades are not committed (APTOS terms); download APTOS first (S1)")
    return pd.read_csv(path).rename(columns={"id_code": "image_id", "diagnosis": "grade"})[["image_id", "grade"]]


def make_splits(labels: pd.DataFrame, dropped_ids: set[str], seed: int = SPLIT_SEED) -> pd.DataFrame:
    df = labels[~labels.image_id.isin(dropped_ids)].sort_values("image_id").reset_index(drop=True)
    trainval, test = train_test_split(df, test_size=0.15, stratify=df.grade, random_state=seed)
    train, val = train_test_split(trainval, test_size=0.15 / 0.85, stratify=trainval.grade, random_state=seed)
    df["split"] = "train"
    df.loc[df.image_id.isin(val.image_id), "split"] = "val"
    df.loc[df.image_id.isin(test.image_id), "split"] = "test"
    df["frac_rank"] = -1
    idx = df.index[df.split == "train"].to_numpy()
    df.loc[idx, "frac_rank"] = np.random.default_rng(seed).permutation(len(idx))
    return df[["image_id", "grade", "split", "frac_rank"]]


def fraction_mask(df: pd.DataFrame, p: float) -> pd.Series:
    """Boolean mask of TRAIN rows used at fraction p (grades 0-2 full, rare grades reduced)."""
    train = df.split == "train"
    mask = train.copy()
    for g in RARE_GRADES:
        sel = train & (df.grade == g)
        keep = math.ceil(p * int(sel.sum()))
        mask &= ~df.index.isin(df.loc[sel, "frac_rank"].sort_values().index[keep:])
    return mask


def load_splits() -> pd.DataFrame:
    df = pd.read_csv(SPLITS / "aptos_splits.csv")
    if "grade" not in df.columns:
        df = df.merge(aptos_labels(), on="image_id", how="left", validate="one_to_one")
        missing = df.grade.isna().sum()
        assert not missing, f"{missing} split ids are not in train.csv"
        df["grade"] = df.grade.astype(int)
    return df[["image_id", "grade", "split", "frac_rank"]]


def digest(df: pd.DataFrame) -> str:
    rows = df[PUBLIC_COLS].sort_values("image_id").astype(str).agg(",".join, axis=1)
    return hashlib.sha1("\n".join(rows).encode()).hexdigest()[:12]


def check(df: pd.DataFrame) -> None:
    assert set(df.split) == {"train", "val", "test"}
    assert df.image_id.is_unique
    tr = df.split == "train"
    assert sorted(df.loc[tr, "frac_rank"]) == list(range(int(tr.sum()))) and (df.loc[~tr, "frac_rank"] == -1).all()
    for g in sorted(df.grade.unique()):
        share = (df[df.grade == g].split == "train").mean()
        assert 0.6 <= share <= 0.8, f"grade {g} train share {share:.2f}"
    prev = None
    for p in FRACTIONS:
        m = fraction_mask(df, p)
        if prev is not None:
            assert not (prev & ~m).any(), f"fraction {p} not nested"
        prev = m


def report(df: pd.DataFrame) -> None:
    click.echo(pd.crosstab(df.grade, df.split).to_string())
    for p in FRACTIONS:
        counts = df[fraction_mask(df, p)].grade.value_counts().sort_index().to_dict()
        click.echo(f"fraction {p:.2f}: train counts {counts}")


@click.command()
@click.option("--seed", type=int, default=SPLIT_SEED, show_default=True)
@click.option("--force", is_flag=True, help="regenerate even if splits/aptos_splits.csv exists")
def main(seed: int, force: bool) -> None:
    out = SPLITS / "aptos_splits.csv"
    dropped_path = SPLITS / "aptos_dedup_dropped.csv"
    if out.exists() and not force:
        for f in (out, dropped_path):
            if f.exists() and any("grade" in c for c in pd.read_csv(f, nrows=0).columns):
                raise SystemExit(f"{f} carries APTOS grades (old format): never commit it; use the grade-free "
                                 "CSVs from a current genai-data-prep run")
        df = load_splits()
        check(df)
        click.echo(f"{out} exists ({len(df)} rows, digest {digest(df)}): verified against train.csv, not regenerating")
        report(df)
        return
    if not dropped_path.exists():
        raise SystemExit(f"{dropped_path} missing: the split is generated only right after dedup (Kaggle "
                         "genai-data-prep); on other machines pull the committed splits/*.csv instead")
    dropped = set(pd.read_csv(dropped_path).dropped_id)
    df = make_splits(aptos_labels(), dropped, seed)
    check(df)
    SPLITS.mkdir(exist_ok=True)
    df[PUBLIC_COLS].to_csv(out, index=False)
    click.echo(f"wrote {out} without grades ({len(df)} rows, digest {digest(df)})")
    report(df)


if __name__ == "__main__":
    main()

"""S4b: stratified 70/15/15 split + nested rare-grade fractions.

splits/aptos_splits.csv columns: image_id, grade, split (train/val/test), frac_rank.
frac_rank: for TRAIN images of grades 3 and 4, the position in a seeded permutation within
that grade (0..n_g-1); -1 for everything else. Fraction p keeps frac_rank < ceil(p * n_g),
so fractions are nested by construction (10% in 25% in 50% in 100%).

The committed CSV is the source of truth; this script only regenerates it.
"""
import math

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.genai.paths import FRACTIONS, RARE_GRADES, RAW_APTOS, SPLIT_SEED, SPLITS


def make_splits(labels: pd.DataFrame, dropped_ids: set[str], seed: int = SPLIT_SEED) -> pd.DataFrame:
    df = labels[~labels.image_id.isin(dropped_ids)].sort_values("image_id").reset_index(drop=True)
    trainval, test = train_test_split(df, test_size=0.15, stratify=df.grade, random_state=seed)
    train, val = train_test_split(trainval, test_size=0.15 / 0.85, stratify=trainval.grade, random_state=seed)
    df["split"] = "train"
    df.loc[df.image_id.isin(val.image_id), "split"] = "val"
    df.loc[df.image_id.isin(test.image_id), "split"] = "test"
    df["frac_rank"] = -1
    for g in RARE_GRADES:
        idx = df.index[(df.split == "train") & (df.grade == g)].to_numpy()
        df.loc[idx, "frac_rank"] = np.random.default_rng(seed + g).permutation(len(idx))
    return df[["image_id", "grade", "split", "frac_rank"]]


def fraction_mask(df: pd.DataFrame, p: float) -> pd.Series:
    """Boolean mask of TRAIN rows used at fraction p (grades 0-2 full, rare grades reduced)."""
    mask = df.split == "train"
    for g in RARE_GRADES:
        n_g = int(((df.split == "train") & (df.grade == g)).sum())
        keep = math.ceil(p * n_g)
        mask &= ~((df.grade == g) & (df.frac_rank >= keep))
    return mask


def load_splits() -> pd.DataFrame:
    return pd.read_csv(SPLITS / "aptos_splits.csv")


def check(df: pd.DataFrame) -> None:
    assert set(df.split) == {"train", "val", "test"}
    assert df.image_id.is_unique
    for g in sorted(df.grade.unique()):
        share = (df[df.grade == g].split == "train").mean()
        assert 0.6 <= share <= 0.8, f"grade {g} train share {share:.2f}"
    prev = None
    for p in FRACTIONS:
        m = fraction_mask(df, p)
        if prev is not None:
            assert not (prev & ~m).any(), f"fraction {p} not nested"
        prev = m


@click.command()
@click.option("--seed", type=int, default=SPLIT_SEED, show_default=True)
@click.option("--force", is_flag=True, help="regenerate even if splits/aptos_splits.csv exists")
def main(seed: int, force: bool) -> None:
    out = SPLITS / "aptos_splits.csv"
    if out.exists() and not force:
        df = pd.read_csv(out)
        check(df)
        click.echo(f"{out} exists ({len(df)} rows) and is the source of truth; not regenerating (use --force)")
        return
    labels = pd.read_csv(RAW_APTOS / "train.csv").rename(columns={"id_code": "image_id", "diagnosis": "grade"})
    dropped_path = SPLITS / "aptos_dedup_dropped.csv"
    dropped = set(pd.read_csv(dropped_path).dropped_id) if dropped_path.exists() else set()
    if not dropped:
        click.echo("WARNING: no dedup file found; splitting without dedup")
    df = make_splits(labels, dropped, seed)
    check(df)
    SPLITS.mkdir(exist_ok=True)
    df.to_csv(SPLITS / "aptos_splits.csv", index=False)
    click.echo(pd.crosstab(df.grade, df.split).to_string())
    for p in FRACTIONS:
        counts = df[fraction_mask(df, p)].grade.value_counts().sort_index().to_dict()
        click.echo(f"fraction {p:.2f}: train counts {counts}")


if __name__ == "__main__":
    main()

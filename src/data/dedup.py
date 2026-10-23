"""S4a: perceptual-hash near-duplicate detection on the 256px APTOS images.

Writes splits/aptos_dedup_dropped.csv (dropped_id, kept_id, distance, dropped_grade, kept_grade).
Clusters are connected components of pairs with pHash Hamming distance <= threshold; the
lowest image_id in each cluster is kept. Runs BEFORE splitting so near-duplicates never
straddle train/val/test.
"""
import click
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.genai.paths import APTOS256, RAW_APTOS, SPLITS

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash_u64(path) -> np.uint64:
    h = imagehash.phash(Image.open(path), hash_size=8)
    bits = np.packbits(h.hash.flatten())
    return np.frombuffer(bits.tobytes(), dtype=">u8")[0]


def hamming_rows(hashes: np.ndarray, i: int) -> np.ndarray:
    x = np.bitwise_xor(hashes, hashes[i]).view(np.uint8).reshape(len(hashes), 8)
    return _POPCOUNT[x].sum(axis=1)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def find_duplicates(ids: list[str], grades: list[int], hashes: np.ndarray, threshold: int) -> pd.DataFrame:
    n = len(ids)
    uf = UnionFind(n)
    pair_dist: dict[tuple[int, int], int] = {}
    for i in range(n):
        d = hamming_rows(hashes, i)
        for j in np.where(d[i + 1:] <= threshold)[0] + i + 1:
            j = int(j)
            uf.union(i, j)
            pair_dist[(i, j)] = int(d[j])
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)
    rows = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda k: ids[k])
        keep = members[0]
        for m in members[1:]:
            dist = pair_dist.get((min(keep, m), max(keep, m)))
            if dist is None:
                dist = int(hamming_rows(hashes[[keep, m]], 0)[1])
            rows.append({"dropped_id": ids[m], "kept_id": ids[keep], "distance": dist,
                         "dropped_grade": grades[m], "kept_grade": grades[keep]})
    return pd.DataFrame(rows, columns=["dropped_id", "kept_id", "distance", "dropped_grade", "kept_grade"])


@click.command()
@click.option("--threshold", type=int, default=4, show_default=True, help="max pHash Hamming distance")
def main(threshold: int) -> None:
    df = pd.read_csv(RAW_APTOS / "train.csv").sort_values("id_code").reset_index(drop=True)
    ids = df["id_code"].tolist()
    grades = df["diagnosis"].astype(int).tolist()
    hashes = np.array([phash_u64(APTOS256 / f"{i}.png") for i in tqdm(ids, desc="phash")], dtype=np.uint64)
    dropped = find_duplicates(ids, grades, hashes, threshold)
    SPLITS.mkdir(exist_ok=True)
    dropped.sort_values("dropped_id").to_csv(SPLITS / "aptos_dedup_dropped.csv", index=False)
    conflicts = int((dropped.dropped_grade != dropped.kept_grade).sum())
    click.echo(f"{len(dropped)} near-duplicates dropped (threshold {threshold}); "
               f"{conflicts} with conflicting labels; {len(ids) - len(dropped)} images remain.")


if __name__ == "__main__":
    main()

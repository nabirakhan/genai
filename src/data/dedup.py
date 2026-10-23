"""S4a: near-duplicate detection on the 256px APTOS crops (runs before splitting).

Default method "vessel": each image becomes a vessel-pattern field (green channel -> light blur
-> CLAHE -> 128x128). Candidate pairs are each image's top-k correlation neighbours (one matrix
product over all images); every candidate is re-scored as the best correlation over integer
shifts within +-3 px, which tolerates small crop differences between copies of the same photo.
Pairs at or above --threshold are duplicates. Images are visited in id order and an image is
dropped only if it duplicates an image already kept (no transitive chaining).

Copies whose grades disagree are dropped together (--conflicts drop): the label is unreliable.
--method phash keeps the earlier 64-bit pHash rule (Hamming <= --hamming, union-find) for
comparison only.

Evidence (dedup_audit on all 3,662 APTOS train images, Kaggle, Oct 2026): pHash<=4 flagged 2,914
pairs and chained them into clusters of up to 569 images, dropping 1,242 images; 85% of those
pairs score < 0.70 here and are visibly different eyes. The shift-tolerant score is bimodal:
149 candidate pairs >= 0.95 (pixel-identical copies), 1 in [0.90, 0.95), 45 in [0.85, 0.90),
unrelated pairs' 99.99th percentile 0.78. The default 0.90 sits in that gap: ~141 copies dropped,
40 of them disagreeing with their twin's grade.

Writes splits/aptos_dedup_dropped.csv: dropped_id, kept_id, score, dropped_grade, kept_grade, reason.
"""
import click
import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.genai.paths import APTOS256, RAW_APTOS, SPLITS

FIELD = 128
MAX_SHIFT = 3
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------- pHash (legacy rule, kept for the audit) ------------------------------------------

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
    """Legacy pHash rule: union-find clusters of pairs within `threshold` bits; keep lowest id."""
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
            rows.append({"dropped_id": ids[m], "kept_id": ids[keep], "score": dist,
                         "dropped_grade": grades[m], "kept_grade": grades[keep], "reason": "phash"})
    return pd.DataFrame(rows, columns=["dropped_id", "kept_id", "score", "dropped_grade", "kept_grade", "reason"])


# ---------- vessel-pattern rule ---------------------------------------------------------------

def disc_mask(size: int = FIELD, frac: float = 0.42) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    return (xx - c) ** 2 + (yy - c) ** 2 <= (frac * size) ** 2


MASK = disc_mask()
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def vessel_field(path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    g = cv2.GaussianBlur(bgr[:, :, 1], (0, 0), 1.0)
    g = _CLAHE.apply(g)
    return cv2.resize(g, (FIELD, FIELD), interpolation=cv2.INTER_AREA).astype(np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    v = v - v.mean(axis=-1, keepdims=True)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)


def aligned_corr(fa: np.ndarray, fb: np.ndarray, max_shift: int = MAX_SHIFT) -> float:
    a = unit(fa[MASK])
    shifted = np.stack([np.roll(fb, (dy, dx), axis=(0, 1))[MASK]
                        for dy in range(-max_shift, max_shift + 1) for dx in range(-max_shift, max_shift + 1)])
    return float((unit(shifted) @ a).max())


def candidate_scores(fields: np.ndarray, topk: int, extra=()) -> dict[tuple[int, int], float]:
    """Shift-tolerant score for each image's top-k correlation neighbours (+ any extra pairs)."""
    V = unit(fields[:, MASK])
    sim = V @ V.T
    np.fill_diagonal(sim, -1.0)
    cand = set(extra)
    for i in range(len(V)):
        for j in np.argsort(-sim[i])[:topk]:
            cand.add((min(i, int(j)), max(i, int(j))))
    return {p: aligned_corr(fields[p[0]], fields[p[1]]) for p in tqdm(sorted(cand), desc="aligned corr")}


def greedy_drop(n: int, pairs: dict[tuple[int, int], float], thr: float) -> np.ndarray:
    """Keep images in index order; drop one if it matches an already-kept image (no chaining)."""
    nbrs: dict[int, list[tuple[int, float]]] = {}
    for (i, j), s in pairs.items():
        if s >= thr:
            nbrs.setdefault(i, []).append((j, s))
            nbrs.setdefault(j, []).append((i, s))
    kept = np.zeros(n, bool)
    dup_of = np.full(n, -1)
    for i in range(n):
        m = [(j, s) for j, s in nbrs.get(i, []) if kept[j]]
        if m:
            dup_of[i] = max(m, key=lambda t: t[1])[0]
        else:
            kept[i] = True
    return dup_of


def vessel_duplicates(ids: list[str], grades: list[int], fields: np.ndarray, threshold: float, topk: int,
                      conflicts: str) -> pd.DataFrame:
    scores = candidate_scores(fields, topk)
    dup_of = greedy_drop(len(ids), scores, threshold)
    rows = []
    conflict_keepers = set()
    for i in np.where(dup_of >= 0)[0]:
        k = int(dup_of[i])
        s = scores[(min(i, k), max(i, k))]
        rows.append({"dropped_id": ids[i], "kept_id": ids[k], "score": round(s, 4),
                     "dropped_grade": grades[i], "kept_grade": grades[k], "reason": "duplicate"})
        if grades[i] != grades[k]:
            conflict_keepers.add(k)
    if conflicts == "drop":
        for k in sorted(conflict_keepers):
            rows.append({"dropped_id": ids[k], "kept_id": "", "score": float("nan"),
                         "dropped_grade": grades[k], "kept_grade": -1, "reason": "grade conflict among copies"})
    return pd.DataFrame(rows, columns=["dropped_id", "kept_id", "score", "dropped_grade", "kept_grade", "reason"])


@click.command()
@click.option("--method", type=click.Choice(["vessel", "phash"]), default="vessel", show_default=True)
@click.option("--threshold", type=float, default=0.9, show_default=True, help="vessel: min shift-tolerant correlation")
@click.option("--hamming", type=int, default=4, show_default=True, help="phash: max Hamming distance")
@click.option("--topk", type=int, default=5, show_default=True, help="vessel: correlation neighbours checked per image")
@click.option("--conflicts", type=click.Choice(["drop", "keep-first"]), default="drop", show_default=True)
def main(method: str, threshold: float, hamming: int, topk: int, conflicts: str) -> None:
    df = pd.read_csv(RAW_APTOS / "train.csv").sort_values("id_code").reset_index(drop=True)
    ids = df["id_code"].tolist()
    grades = df["diagnosis"].astype(int).tolist()
    if method == "phash":
        hashes = np.array([phash_u64(APTOS256 / f"{i}.png") for i in tqdm(ids, desc="phash")], dtype=np.uint64)
        dropped = find_duplicates(ids, grades, hashes, hamming)
    else:
        fields = np.stack([vessel_field(APTOS256 / f"{i}.png") for i in tqdm(ids, desc="vessel fields")])
        dropped = vessel_duplicates(ids, grades, fields, threshold, topk, conflicts)
    SPLITS.mkdir(exist_ok=True)
    dropped.sort_values("dropped_id").to_csv(SPLITS / "aptos_dedup_dropped.csv", index=False)
    dup = dropped[dropped.reason != "grade conflict among copies"]
    n_conf = int((dup.dropped_grade != dup.kept_grade).sum())
    click.echo(f"method={method}: {len(dup)} duplicate copies dropped, {n_conf} of them with a grade conflict; "
               f"{int((dropped.reason == 'grade conflict among copies').sum())} conflicting originals also dropped; "
               f"{len(ids) - dropped.dropped_id.nunique()} images remain.")


if __name__ == "__main__":
    main()

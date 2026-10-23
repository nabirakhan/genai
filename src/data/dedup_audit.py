"""Audit near-duplicate detection on the 256px APTOS crops.

  python -m src.data.dedup_audit --out <dir>

Compares the pHash rule (64-bit, Hamming <= 4, union-find) with a vessel-pattern score:
green channel -> light blur -> CLAHE -> 128x128 field. Two numbers per pair:
  corr      Pearson correlation of the masked fields (all pairs at once via one matrix product)
  aligned   best correlation over integer shifts within +-3 px (candidate pairs only), which
            tolerates the small crop differences two copies of the same photo can have
Candidates = pHash<=4 pairs + each image's top-3 corr neighbours. Writes summary.txt and contact
sheets so a human can see what each rule calls a duplicate.
"""
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.data.dedup import (MASK, UnionFind, aligned_corr, greedy_drop, hamming_rows, phash_u64, unit,
                            vessel_field)
from src.genai.paths import APTOS256, RAW_APTOS

BINS = [(0.95, 1.01), (0.9, 0.95), (0.85, 0.9), (0.8, 0.85), (0.7, 0.8), (-1, 0.7)]


def thumb(path, px: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((px, px), Image.LANCZOS)


def contact_sheet(rows, ids, grades, out: Path, title: str, px: int = 150, cols: int = 3) -> None:
    """rows: list of (i, j, ham, corr, aligned), drawn as a grid of pairs with a caption under each."""
    cell_w, cell_h = 2 * px + 10, px + 46
    nrows = max(1, -(-len(rows) // cols))
    sheet = Image.new("RGB", (cols * (cell_w + 14), 30 + nrows * (cell_h + 10)), (16, 12, 20))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), title, fill=(255, 220, 230))
    for k, (i, j, ham, corr, al) in enumerate(rows):
        x = (k % cols) * (cell_w + 14) + 6
        y = 30 + (k // cols) * (cell_h + 10)
        sheet.paste(thumb(APTOS256 / f"{ids[i]}.png", px), (x, y))
        sheet.paste(thumb(APTOS256 / f"{ids[j]}.png", px), (x + px + 10, y))
        conflict = grades[i] != grades[j]
        d.text((x, y + px + 4), f"aligned {al:.3f}  corr {corr:.3f}  ham {ham}", fill=(255, 200, 120))
        d.text((x, y + px + 22), f"grades {grades[i]}/{grades[j]}  {ids[i][:8]}/{ids[j][:8]}" + ("  CONFLICT" if conflict else ""),
               fill=(255, 120, 140) if conflict else (190, 190, 200))
    sheet.save(out)


def sample_bins(rows, key, rng, per_bin):
    out = []
    for lo, hi in BINS:
        cand = [r for r in rows if lo <= key(r) < hi]
        if cand:
            pick = rng.choice(len(cand), min(per_bin, len(cand)), replace=False)
            out += [cand[k] for k in sorted(pick)]
    return out


@click.command()
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--per-bin", type=int, default=6)
@click.option("--topk", type=int, default=3, help="corr neighbours per image added as candidates")
@click.option("--seed", type=int, default=0)
def main(out: Path, per_bin: int, topk: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_APTOS / "train.csv").sort_values("id_code").reset_index(drop=True)
    ids = df["id_code"].tolist()
    grades = df["diagnosis"].astype(int).tolist()
    n = len(ids)
    hashes = np.array([phash_u64(APTOS256 / f"{i}.png") for i in tqdm(ids, desc="phash")], dtype=np.uint64)
    F = np.stack([vessel_field(APTOS256 / f"{i}.png") for i in tqdm(ids, desc="vessel fields")])
    V = unit(F[:, MASK])
    sim = V @ V.T
    np.fill_diagonal(sim, -1.0)
    lines = [f"images: {n}"]

    # current rule
    uf = UnionFind(n)
    ham: dict[tuple[int, int], int] = {}
    for i in range(n):
        d = hamming_rows(hashes, i)
        for j in np.where(d[i + 1:] <= 4)[0] + i + 1:
            ham[(i, int(j))] = int(d[j])
            uf.union(i, int(j))
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)
    sizes = sorted((len(c) for c in clusters.values() if len(c) > 1), reverse=True)
    conflict_clusters = sum(1 for c in clusters.values() if len(c) > 1 and len({grades[k] for k in c}) > 1)
    lines += [f"pHash<=4 pairs: {len(ham)}",
              f"pHash clusters (size>1): {len(sizes)}; largest: {sizes[:12]}",
              f"images dropped by the pHash rule: {sum(s - 1 for s in sizes)}; clusters with grade conflicts: {conflict_clusters}"]

    # candidates and shift-tolerant score
    cand = set(ham)
    for i in range(n):
        for j in np.argsort(-sim[i])[:topk]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            cand.add((a, b))
    aligned = {p: aligned_corr(F[p[0]], F[p[1]]) for p in tqdm(sorted(cand), desc="aligned corr")}
    rows = [(i, j, ham.get((i, j), int(hamming_rows(hashes[[i, j]], 0)[1])), float(sim[i, j]), aligned[(i, j)]) for (i, j) in sorted(cand)]
    ph_rows = [r for r in rows if (r[0], r[1]) in ham]
    lines.append(f"candidate pairs scored: {len(rows)}")
    for lo, hi in BINS:
        lines.append(f"  pHash pairs with aligned corr in [{lo:.2f},{hi:.2f}): {sum(lo <= r[4] < hi for r in ph_rows)}"
                     f"   | all candidates: {sum(lo <= r[4] < hi for r in rows)}")
    for thr in (0.98, 0.95, 0.92, 0.9, 0.85, 0.8):
        dup_of = greedy_drop(n, aligned, thr)
        dropped = np.where(dup_of >= 0)[0]
        conf = int(sum(grades[i] != grades[dup_of[i]] for i in dropped))
        lines.append(f"aligned >= {thr:.2f}: greedy drop {len(dropped)} images, {conf} with a grade conflict")
    iu = np.triu_indices(n, 1)
    q = np.quantile(sim[iu], [0.5, 0.9, 0.99, 0.999, 0.9999])
    lines.append("all-pairs corr quantiles 50/90/99/99.9/99.99%: " + " ".join(f"{x:.3f}" for x in q))

    rng = np.random.default_rng(seed)
    contact_sheet(sample_bins(ph_rows, lambda r: r[4], rng, per_bin), ids, grades, out / "sheet_phash_pairs.png",
                  "pHash<=4 pairs (what the current rule drops), sampled by aligned corr, high to low")
    non_ph = [r for r in rows if (r[0], r[1]) not in ham]
    contact_sheet(sample_bins(non_ph, lambda r: r[4], rng, per_bin), ids, grades, out / "sheet_other_candidates.png",
                  "Top-corr neighbours NOT flagged by pHash, sampled by aligned corr, high to low")
    (out / "summary.txt").write_text("\n".join(lines))
    pd.DataFrame(rows, columns=["i", "j", "ham", "corr", "aligned"]).assign(
        id_i=lambda d: [ids[k] for k in d.i], id_j=lambda d: [ids[k] for k in d.j],
        grade_i=lambda d: [grades[k] for k in d.i], grade_j=lambda d: [grades[k] for k in d.j],
    ).to_csv(out / "candidate_pairs.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

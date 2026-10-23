"""S11b: bootstrap confidence intervals from per-image predictions (results/preds/*.csv).

results/tables/bootstrap_ci.csv     run_id, dataset, metric, point, lo, hi        (95% percentile CI)
results/tables/bootstrap_delta.csv  fraction, ratio, seed, dataset, metric, delta, lo, hi  for cond6 - cond5,
                                    resampling the SAME images for both runs (paired).
Metrics are computed from confusion matrices in numpy (fast enough for 1000 resamples x 168 runs).
"""
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd

from src.classify.data import NUM_CLASSES
from src.data.split import load_splits
from src.genai.paths import RESULTS, SPLITS

K = NUM_CLASSES
_W = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2 / (K - 1) ** 2
RUN_RE = re.compile(r"c(\d)_f(\d{3})_r([\d.]+)_s(\d+)")


def metrics_from_cm(cm: np.ndarray) -> dict[str, float]:
    tp = np.diag(cm).astype(float)
    row, col = cm.sum(1).astype(float), cm.sum(0).astype(float)
    rec = np.divide(tp, row, out=np.zeros(K), where=row > 0)
    prec = np.divide(tp, col, out=np.zeros(K), where=col > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    n = cm.sum()
    o = cm / n
    e = np.outer(row, col) / (n * n)
    denom = (_W * e).sum()
    qwk = 1 - (_W * o).sum() / denom if denom > 0 else float("nan")
    out = {"macro_f1": float(f1.mean()), "qwk": float(qwk), "acc": float(tp.sum() / n)}
    out.update({f"recall_{k}": float(rec[k]) for k in range(K)})
    return out


def _cm_batch(codes: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.bincount(codes[idx], minlength=K * K).reshape(K, K)


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, B: int, rng: np.random.Generator) -> pd.DataFrame:
    codes = y_true * K + y_pred
    n = len(codes)
    point = metrics_from_cm(_cm_batch(codes, np.arange(n)))
    samples = {m: np.empty(B) for m in point}
    for b in range(B):
        m = metrics_from_cm(_cm_batch(codes, rng.integers(0, n, n)))
        for k, v in m.items():
            samples[k][b] = v
    rows = [dict(metric=m, point=point[m], lo=float(np.nanpercentile(samples[m], 2.5)),
                 hi=float(np.nanpercentile(samples[m], 97.5))) for m in point]
    return pd.DataFrame(rows)


def bootstrap_delta(p6: pd.DataFrame, p5: pd.DataFrame, B: int, rng: np.random.Generator) -> pd.DataFrame:
    assert (p6.image_id.values == p5.image_id.values).all(), "paired runs must score the same images in the same order"
    yt = p6.y_true.to_numpy()
    c6, c5 = yt * K + p6.y_pred.to_numpy(), yt * K + p5.y_pred.to_numpy()
    n = len(yt)
    m6, m5 = metrics_from_cm(_cm_batch(c6, np.arange(n))), metrics_from_cm(_cm_batch(c5, np.arange(n)))
    deltas = {m: np.empty(B) for m in m6}
    for b in range(B):
        idx = rng.integers(0, n, n)
        a, c = metrics_from_cm(_cm_batch(c6, idx)), metrics_from_cm(_cm_batch(c5, idx))
        for k in deltas:
            deltas[k][b] = a[k] - c[k]
    return pd.DataFrame([dict(metric=m, delta=m6[m] - m5[m], lo=float(np.nanpercentile(deltas[m], 2.5)),
                              hi=float(np.nanpercentile(deltas[m], 97.5))) for m in m6])


@click.command()
@click.option("--preds-dir", type=click.Path(path_type=Path), default=RESULTS / "preds")
@click.option("--out-dir", type=click.Path(path_type=Path), default=RESULTS / "tables")
@click.option("-B", "--n-boot", type=int, default=1000)
@click.option("--seed", type=int, default=0)
def main(preds_dir, out_dir, n_boot, seed):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(preds_dir.glob("*.csv"))
    truth = {"aptos_test": load_splits().set_index("image_id").grade}  # preds files carry no APTOS labels
    if (SPLITS / "idrid_test.csv").exists():
        truth["idrid"] = pd.read_csv(SPLITS / "idrid_test.csv").set_index("image_id").grade
    ci_rows, delta_rows = [], []
    by_key = {}
    for f in files:
        m = RUN_RE.match(f.stem)
        if not m:
            continue
        cond, frac, ratio, s = int(m.group(1)), int(m.group(2)) / 100, float(m.group(3)), int(m.group(4))
        dataset = f.stem[m.end() + 1:]
        df = pd.read_csv(f)
        if "y_true" not in df.columns:
            df["y_true"] = df.image_id.map(truth[dataset])
            assert df.y_true.notna().all(), f"{f.name}: ids without a label"
            df["y_true"] = df.y_true.astype(int)
        ci = bootstrap_metrics(df.y_true.to_numpy(), df.y_pred.to_numpy(), n_boot, rng)
        ci.insert(0, "dataset", dataset)
        ci.insert(0, "run_id", f.stem[:m.end()])
        ci_rows.append(ci)
        by_key[(cond, frac, ratio, s, dataset)] = df
    for (cond, frac, ratio, s, dataset), p6 in by_key.items():
        if cond != 6:
            continue
        p5 = by_key.get((5, frac, 0.0, s, dataset))
        if p5 is None:
            continue
        d = bootstrap_delta(p6, p5, n_boot, rng)
        for c, v in (("dataset", dataset), ("seed", s), ("ratio", ratio), ("fraction", frac)):
            d.insert(0, c, v)
        delta_rows.append(d)
    if ci_rows:
        pd.concat(ci_rows).to_csv(out_dir / "bootstrap_ci.csv", index=False)
    if delta_rows:
        pd.concat(delta_rows).to_csv(out_dir / "bootstrap_delta.csv", index=False)
    click.echo(f"{len(ci_rows)} runs, {len(delta_rows)} paired deltas -> {out_dir}")


if __name__ == "__main__":
    main()

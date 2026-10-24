"""End-to-end CPU smoke test on fake fundus images. No real data, no network needed
(the classifier runs with --no-pretrained). Everything goes to a temp dir via GENAI_* env vars,
so the real splits/ and results/ are never touched.

  python tests/smoke_synthetic.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N_APTOS = 100
GRADES = [0] * 40 + [1] * 12 + [2] * 24 + [3] * 12 + [4] * 12


def fake_fundus(rng: np.random.Generator, grade: int, size=(360, 270)) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, r = w / 2 + rng.integers(-15, 15), h / 2, h * 0.47
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) < r * r
    img = np.zeros((h, w, 3), np.uint8)
    base = np.array([150 + 10 * grade, 60 + 5 * grade, 30], np.uint8)
    img[disc] = base
    noise = rng.integers(0, 40, (h, w, 3), dtype=np.uint8)
    img = np.where(disc[..., None], np.clip(img.astype(int) + noise, 0, 255), 0).astype(np.uint8)
    for _ in range(grade * 3):
        x, y = rng.integers(int(cx - r * 0.6), int(cx + r * 0.6)), rng.integers(int(cy - r * 0.6), int(cy + r * 0.6))
        img[y - 3:y + 3, x - 3:x + 3] = (255, 240, 200)
    return Image.fromarray(img)


def run(cmd: list[str], env: dict, check=True) -> subprocess.CompletedProcess:
    print("\n$", " ".join(cmd))
    p = subprocess.run(cmd, env=env, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
    print(p.stdout[-3000:])
    if p.returncode != 0:
        print(p.stderr[-4000:])
        if check:
            raise SystemExit(f"command failed: {cmd}")
    return p


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="genai_smoke_"))
    data, splits, results, ckpt = tmp / "data", tmp / "splits", tmp / "results", tmp / "ckpt"
    for d in (data / "raw" / "aptos" / "train_images", splits, results, ckpt):
        d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GENAI_DATA": str(data), "GENAI_SPLITS": str(splits), "GENAI_RESULTS": str(results),
           "GENAI_CHECKPOINTS": str(ckpt), "PYTHONIOENCODING": "utf-8", "OPENBLAS_NUM_THREADS": "2"}
    py = sys.executable
    rng = np.random.default_rng(0)

    # fake APTOS: 100 images, two exact duplicates
    ids = [f"img{i:03d}" for i in range(N_APTOS)]
    for i, g in zip(ids, GRADES):
        fake_fundus(rng, g).save(data / "raw" / "aptos" / "train_images" / f"{i}.png")
    shutil.copy(data / "raw/aptos/train_images/img000.png", data / "raw/aptos/train_images/img001.png")
    pd.DataFrame({"id_code": ids, "diagnosis": GRADES}).to_csv(data / "raw" / "aptos" / "train.csv", index=False)
    # fake IDRiD layout
    for split, n, label_file in (("Training", 7, "a. IDRiD_Disease Grading_Training Labels.csv"),
                                 ("Testing", 5, "b. IDRiD_Disease Grading_Testing Labels.csv")):
        d = data / "raw" / "idrid" / "B. Disease Grading" / "1. Original Images" / f"{'a' if split == 'Training' else 'b'}. {split} Set"
        d.mkdir(parents=True)
        rows = []
        for k in range(1, n + 1):
            g = k % 5
            fake_fundus(rng, g, (400, 266)).save(d / f"IDRiD_{k:03d}.jpg")
            rows.append({"Image name": f"IDRiD_{k:03d}", "Retinopathy grade": g, "Risk of macular edema ": 0})
        gt = data / "raw" / "idrid" / "B. Disease Grading" / "2. Groundtruths"
        gt.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(gt / label_file, index=False)

    # patch download_aptos expectations: use --skip-download path by calling the later stages directly
    run([py, "-m", "src.data.prepare_images", "--source", "both", "--workers", "2"], env)
    assert len(list((data / "aptos256").glob("*.png"))) == N_APTOS
    assert len(list((data / "idrid256").glob("*.png"))) == 12
    run([py, "-m", "src.data.dedup"], env)
    dropped = pd.read_csv(splits / "aptos_dedup_dropped.csv")
    assert "img001" in set(dropped.dropped_id), dropped
    run([py, "-m", "src.data.split"], env)
    sp = pd.read_csv(splits / "aptos_splits.csv")
    assert len(sp) == N_APTOS - len(dropped) and set(sp.split) == {"train", "val", "test"}
    assert not any("grade" in c for c in (*sp.columns, *dropped.columns)), "committed CSVs must not carry APTOS labels"
    p = run([py, "-m", "src.data.split"], env)
    assert "verified against train.csv" in p.stdout, p.stdout
    run([py, "-m", "src.data.cache"], env)
    assert np.load(data / "cache" / "aptos256.npy", mmap_mode="r").shape == (len(sp), 256, 256, 3)
    run([py, "-m", "src.data.make_sg3_zips"], env)
    with zipfile.ZipFile(data / "sg3" / "aptos256_f100.zip") as z:
        labels = json.loads(z.read("dataset.json"))["labels"]
        assert len(labels) == (sp.split == "train").sum() and all(n in z.namelist() for n, _ in labels)
    with zipfile.ZipFile(data / "sg3" / "aptos256_f010.zip") as z:
        n010 = len(json.loads(z.read("dataset.json"))["labels"])
    assert n010 < len(labels)

    # fake synthetic images for f100 (what Kaggle would produce)
    sys.path.insert(0, str(ROOT))
    os.environ.update(env)
    from src.gan.watermark import save_synthetic  # noqa: E402
    for g in (3, 4):
        for s in range(30):
            save_synthetic(fake_fundus(rng, g, (256, 256)).resize((256, 256)), data / "synth" / "f100" / f"g{g}" / f"g{g}_s{s:05d}.png",
                           visible=False, generator="fake", grade=g, seed=s)

    common = ["--epochs", "1", "--batch-size", "8", "--workers", "0", "--no-pretrained", "--allow-cpu", "--results-csv", str(results / "classifier_runs.csv")]
    run([py, "-m", "src.classify.train", "--condition", "3", "--fraction", "1.0", "--seed", "0", *common], env)
    run([py, "-m", "src.classify.train", "--condition", "5", "--fraction", "1.0", "--seed", "0", *common], env)
    run([py, "-m", "src.classify.train", "--condition", "6", "--fraction", "1.0", "--ratio", "1", "--seed", "0", *common, "--keep-weights"], env)
    p = run([py, "-m", "src.classify.train", "--condition", "5", "--fraction", "1.0", "--seed", "0", *common], env)
    assert "skipping" in p.stdout
    res = pd.read_csv(results / "classifier_runs.csv")
    assert len(res) == 6 and set(res.dataset) == {"aptos_test", "idrid"}, res
    assert (res[res.condition == 6].n_synth > 0).all()
    assert len(list((results / "preds").glob("*.csv"))) == 6
    assert all("y_true" not in pd.read_csv(f, nrows=1).columns for f in (results / "preds").glob("*.csv"))
    assert (ckpt / "c6_f100_r1_s0" / "best.pt").exists() and (ckpt / "c5_f100_r0_s0" / "best.pt").exists() and not (ckpt / "c3_f100_r0_s0").exists()

    run([py, "-m", "src.analysis.aggregate", "--runs-csv", str(results / "classifier_runs.csv"), "--out-dir", str(results / "tables"),
         "--web-dir", str(tmp / "web")], env)
    assert (results / "tables" / "summary_long.csv").exists() and (tmp / "web" / "tables.json").exists()
    run([py, "-m", "src.analysis.bootstrap", "--preds-dir", str(results / "preds"), "--out-dir", str(results / "tables"), "-B", "50"], env)
    assert (results / "tables" / "bootstrap_ci.csv").exists() and (results / "tables" / "bootstrap_delta.csv").exists()
    run([py, "-m", "src.analysis.plots", "--runs-csv", str(results / "classifier_runs.csv"), "--out-dir", str(results / "figures"),
         "--web-dir", str(tmp / "web")], env)
    assert list((results / "figures").glob("learning_curve_*.png"))
    p = run([py, "-m", "src.classify.queue", "--dry-run", "--results-csv", str(results / "classifier_runs.csv")], env)
    assert "84 selected" in p.stdout and "3 done" in p.stdout, p.stdout
    run([py, "-m", "src.pipeline", "status"], env)
    print(f"\nSMOKE_OK  ({tmp})")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

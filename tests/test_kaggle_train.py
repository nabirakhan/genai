"""CPU test of src/gan/kaggle_train.py with tests/fake_sg3_train.py (no GPU, ~1 min).

  python tests/test_kaggle_train.py
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

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FAKE = (ROOT / "tests" / "fake_sg3_train.py").as_posix()


def write_zip(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("dataset.json", json.dumps({"labels": [[f"{n}.png", 0] for n in names]}))


def prep_output(root: Path, split: pd.DataFrame) -> None:
    out = root / "notebooks" / "u" / "genai-data-prep" / "out"
    (out / "splits").mkdir(parents=True, exist_ok=True)
    split.to_csv(out / "splits" / "aptos_splits.csv", index=False)
    train = split[split.split == "train"].image_id.tolist()
    write_zip(out / "sg3" / "aptos256_f100.zip", train)
    for f in ("f010", "f025", "f050"):
        write_zip(out / "sg3" / f"aptos256_{f}.zip", train[: len(train) // 2])


def run(args: list[str], env: dict, expect_rc: int = 0) -> str:
    p = subprocess.run([PY, "-m", "src.gan.kaggle_train", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    assert p.returncode == expect_rc, f"rc {p.returncode} != {expect_rc}\n{out}"
    return out


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="kaggle_train_test_"))
    rng = np.random.default_rng(0)
    split = pd.DataFrame({"image_id": [f"{i:012x}" for i in rng.choice(2**40, 40, replace=False)],
                          "split": ["train"] * 28 + ["val"] * 6 + ["test"] * 6})
    split["frac_rank"] = -1
    split.loc[split.split == "train", "frac_rank"] = rng.permutation(28)
    committed = tmp / "committed"
    committed.mkdir()
    split.to_csv(committed / "aptos_splits.csv", index=False)
    env = {**os.environ, "GENAI_SPLITS": str(committed), "FAKE_SEC_PER_KIMG": "0.04", "PYTHONIOENCODING": "utf-8"}
    fast = ["--train-cmd", f'"{Path(PY).as_posix()}" "{FAKE}"', "--skip-compat", "--poll", "0.3", "--print-every", "5",
            "--min-start", "0.02", "--grace", "1"]

    # 1. nothing attached and not a first session -> refuse
    inp, work = tmp / "in1", tmp / "w1"
    prep_output(inp, split)
    out = run(["--input-root", str(inp), "--work", str(work), "--hours", "1", *fast], env, expect_rc=1)
    assert "REFUSED: no previous output" in out, out

    # 2. session 2 of f100: resume 720 -> 1000, carry forward, then start f010-ada from FFHQ and stop at the deadline
    inp, work = tmp / "in2", tmp / "w2"
    prep_output(inp, split)
    rd = inp / "notebooks" / "u" / "genai-gan-train" / "runs" / "00000-stylegan2-aptos256_f100-gpus2-batch32-gamma1-f100-ada-from0"
    rd.mkdir(parents=True)
    (rd / "network-snapshot-000680.pkl").write_bytes(b"x" * 1000 + b".")
    (rd / "network-snapshot-000720.pkl").write_bytes(b"x" * 1000 + b".")
    (rd / "network-snapshot-000760.pkl").write_bytes(b"x" * 1000)          # cut off mid-write: must be ignored
    out = run(["--input-root", str(inp), "--work", str(work), "--hours", str(40 / 3600), *fast], env)
    assert "TRAIN f100-ada: 720 -> 1000 kimg" in out and "network-snapshot-000720.pkl" in out, out
    assert "DONE f100-ada at 1000 kimg" in out, out
    assert (work / "final" / "f100-ada_kimg1000.pkl").exists()
    assert "TRAIN f010-ada: 0 -> 1000 kimg" in out and "time budget reached" in out and "NEXT_SESSION: f010-ada" in out, out
    ledger = json.loads((work / "ledger.json").read_text())
    assert ledger["f100-ada"] == 1000 and 0 < ledger["f010-ada"] < 1000, ledger
    assert not list((work / "runs").glob("*-f100-ada-from720/network-snapshot-*.pkl")), "finished run snapshots kept"
    f010_runs = list((work / "runs").glob("*-f010-ada-from0"))
    assert len(f010_runs) == 1 and len(list(f010_runs[0].glob("network-snapshot-*.pkl"))) <= 2

    # 3. session 3 attaches session 2's output only: f100 final is carried forward, f010 resumes
    inp3, work3 = tmp / "in3", tmp / "w3"
    prep_output(inp3, split)
    shutil.copytree(work, inp3 / "notebooks" / "u" / "genai-gan-train")
    out = run(["--input-root", str(inp3), "--work", str(work3), "--hours", "1", "--dry-run", *fast], env)
    k = ledger["f010-ada"]
    assert f"TRAIN f010-ada: {k} -> 1000 kimg" in out and (work3 / "final" / "f100-ada_kimg1000.pkl").exists(), out

    # 4. ledger says f010 got further than any attached pickle -> refuse instead of restarting it
    (inp3 / "notebooks" / "u" / "genai-gan-train" / "ledger.json").write_text(json.dumps({"f100-ada": 1000, "f010-ada": k + 400}))
    out = run(["--input-root", str(inp3), "--work", str(tmp / "w4"), "--hours", "1", "--dry-run", *fast], env, expect_rc=1)
    assert "REFUSED: ledger.json records f010-ada" in out, out

    # 5. f010 zip built from a different split (old per-grade frac_rank) -> refused; f100 alone still fine
    inp5 = tmp / "in5"
    other = split.copy()
    other.loc[other.split == "train", "frac_rank"] = rng.permutation(28)
    prep_output(inp5, other)
    out = run(["--input-root", str(inp5), "--work", str(tmp / "w5"), "--hours", "1", "--run", "f010-ada",
               "--first-session", "--dry-run", *fast], env)
    assert "REFUSED f010-ada: " in out and "built from split" in out, out
    out = run(["--input-root", str(inp5), "--work", str(tmp / "w6"), "--hours", "1", "--first-session", "--dry-run", *fast], env)
    assert "TRAIN f100-ada: 0 -> 1000 kimg" in out and "f100 zip == committed train split" in out, out

    shutil.rmtree(tmp, ignore_errors=True)
    print("KAGGLE_TRAIN_OK")


if __name__ == "__main__":
    main()

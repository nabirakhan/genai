"""Generates the two Kaggle notebooks (plain JSON, no nbformat needed):

  python notebooks/build_notebooks.py
  -> notebooks/kaggle_gan_train.ipynb, notebooks/kaggle_gan_eval.ipynb

Edit the CELLS below, re-run, upload the .ipynb to Kaggle (or paste cells). Both notebooks clone
this repo (public) at start, so code changes only need a git push.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/nabirakhan/genai"

SETUP = f'''import os, sys, glob, re, json, time, shutil, signal, subprocess
WORK = "/kaggle/working"; CODE = f"{{WORK}}/genai"
if not os.path.exists(CODE):
    subprocess.run(["git", "clone", "--recurse-submodules", REPO, CODE], check=True)
else:
    subprocess.run(["git", "-C", CODE, "pull", "--recurse-submodules"], check=False)
os.chdir(CODE)
subprocess.run([sys.executable, "scripts/apply_patches.py"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "click", "requests", "tqdm", "ninja",
                "psutil", "scipy"], check=True)
for opt in ("pyspng", "imageio-ffmpeg"):    # speed-ups only; stylegan3 falls back without them
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", opt]).returncode:
        print("optional package not installed:", opt)
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
import torch; print("torch", torch.__version__, "gpus", torch.cuda.device_count())
def sh(cmd, **kw):
    print("$", " ".join(cmd)); return subprocess.run(cmd, check=True, **kw)
'''

TRAIN_CELLS = [
    ("markdown", """# Conditional StyleGAN2-ADA fine-tunes on APTOS-256 (Kaggle T4 x2)

Each version is one session (~10 h) of the queue **f100-ada → f010-ada → f010-noaug (ablation) → f025-ada → f050-ada**,
every GAN trained to the same total kimg. The logic lives in the repo (`src/gan/kaggle_train.py`, settings in
`notebooks/kaggle_session.json`), so a git push changes the next session while this notebook stays the same.

Before every *Save Version → Save & Run All (Commit)*:
1. **Input: the genai-data-prep output** whose `splits/aptos_splits.csv` equals the repo's committed split (checked:
   the f010-f050 zips depend on it, f100 must hold exactly the committed train ids).
2. **Input: the newest output of this notebook** (snapshots, `final/`, `ledger.json`). Without it the session refuses
   to start, so nothing restarts from FFHQ by accident (`first_session` in the JSON is only for the very first one).
3. Settings: Accelerator **GPU T4 x2**, **Internet ON**.

The newest version's output always holds every finished GAN (`final/<tag>_kimg<N>.pkl`) plus the running one's
last two snapshots; the eval notebook attaches only that output."""),
    ("code", f'''REPO = "{REPO_URL}"
OVERRIDES = []        # e.g. ["--hours", "6"] or ["--run", "f010-noaug"]; empty = notebooks/kaggle_session.json'''),
    ("code", SETUP),
    ("code", '''r = subprocess.run([sys.executable, "-m", "src.gan.kaggle_train", *OVERRIDES])
print("runner exit code", r.returncode)'''),
    ("code", '''from PIL import Image as PImage
from IPython.display import display
for d in sorted(glob.glob(f"{WORK}/runs/*")):
    fakes = sorted(glob.glob(f"{d}/fakes[0-9]*.png"))       # fakes_init.png would sort last
    print(os.path.basename(d), "->", os.path.basename(fakes[-1]) if fakes else "no fakes yet")
    if fakes:
        im = PImage.open(fakes[-1]); im.thumbnail((1536, 1536)); display(im)   # full grid is 7680x4096
print(open(f"{WORK}/ledger.json").read() if os.path.exists(f"{WORK}/ledger.json") else "no ledger.json")
for root, _, files in os.walk(WORK):
    for f in files:
        p = os.path.join(root, f)
        if os.path.getsize(p) > 5e7: print(f"{os.path.getsize(p) / 1e6:8.0f} MB  {p}")
os.chdir(WORK)
shutil.rmtree(f"{WORK}/data", ignore_errors=True)
shutil.rmtree(CODE, ignore_errors=True)
if r.returncode == 4:
    raise RuntimeError("the runner trained nothing (refused or misconfigured inputs): see its log above")'''),
]

EVAL_CELLS = [
    ("markdown", """# GAN evaluation + image generation (Kaggle T4 x2)

Inputs: the train notebook's outputs (one per FRACTION/AUG) and the dataset `aptos-dr-256-genai`.
Per final pickle: 500 images per rare grade (seeds 0-499, psi 0.8) with `Synthetic=true` metadata,
overall FID/KID vs the val split (`calc_metrics.py`), per-class KID for grades 3/4
(`src/gan/metrics_per_class.py`). Outputs: `synth_{tag}.zip`, `gan_metrics.csv`, `samples_{tag}.png`.
Download on the lab PC with `kaggle kernels output <user>/<this-notebook> -p downloads/` then
`python -m src.gan.import_synth downloads/synth_f*.zip`."""),
    ("code", f'''REPO = "{REPO_URL}"
DATASET_DIR = "/kaggle/input/aptos-dr-256-genai"
N_PER_CLASS, PSI, CLASSES = 500, 0.8, "3,4"
ONLY_TAGS = None          # e.g. ["f100-ada"]; None = every tag found under /kaggle/input
GPUS = 2'''),
    ("code", SETUP),
    ("code", '''def total_kimg(path):
    m = re.search(r"-from(\\d+)[/\\\\].*snapshot-(\\d+)", path); return int(m.group(1)) + int(m.group(2)) if m else 0
best = {}
for s in glob.glob("/kaggle/input/**/network-snapshot-*.pkl", recursive=True):
    m = re.search(r"-(f\\d{3}-(?:ada|noaug))-from", s)
    if m and (ONLY_TAGS is None or m.group(1) in ONLY_TAGS):
        if m.group(1) not in best or total_kimg(s) > total_kimg(best[m.group(1)]):
            best[m.group(1)] = s
os.makedirs(f"{WORK}/pkls", exist_ok=True)
PKLS = {}
for tag, s in sorted(best.items()):
    dst = f"{WORK}/pkls/{tag}.pkl"           # copy: calc_metrics must not write next to a read-only input
    if not os.path.exists(dst): shutil.copy(s, dst)
    PKLS[tag] = dst; print(f"{tag}: {total_kimg(s)} kimg  <- {s}")
assert PKLS, "no snapshots found; attach the train notebook outputs"
vals = sorted(glob.glob("/kaggle/input/**/aptos256_val.zip", recursive=True)) + sorted(glob.glob("/kaggle/input/**/aptos256_val", recursive=True))
assert vals, "aptos256_val not found under /kaggle/input: add the data-prep notebook output or the dataset as input"
REAL_VAL = vals[0]
print("REAL_VAL =", REAL_VAL)'''),
    ("code", '''import pandas as pd
overall = []
for tag, pkl in PKLS.items():
    out = f"{WORK}/synth_{tag}"
    sh([sys.executable, "-m", "src.gan.generate", "--pkl", pkl, "--outdir", out, "--classes", CLASSES, "--n", str(N_PER_CLASS), "--psi", str(PSI), "--tag", tag])
    r = subprocess.run([sys.executable, "third_party/stylegan3/calc_metrics.py", "--metrics=fid50k_full,kid50k_full", f"--network={pkl}", f"--data={REAL_VAL}", f"--gpus={GPUS}"], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-800:])
    for line in r.stdout.splitlines():
        if line.startswith("{") and '"results"' in line:
            res = json.loads(line)["results"]
            overall.append(dict(gan=tag, grade="all", n_real=None, n_gen=50000, fid=res.get("fid50k_full"), kid=res.get("kid50k_full"), kid_std=None))
    sh([sys.executable, "-m", "src.gan.metrics_per_class", "--gan", tag, "--synth-dir", out, "--real", REAL_VAL, "--classes", CLASSES, "--out", f"{WORK}/gan_metrics.csv"])
    shutil.make_archive(out, "zip", out)
df = pd.read_csv(f"{WORK}/gan_metrics.csv")
df = pd.concat([df[df.grade.astype(str) != "all"], pd.DataFrame(overall)], ignore_index=True)
df.to_csv(f"{WORK}/gan_metrics.csv", index=False); print(df.to_string(index=False))'''),
    ("code", '''import numpy as np
from PIL import Image
sys.path.insert(0, CODE)
from src.gan.watermark import add_visible_label
for tag in PKLS:
    rows = []
    for g in CLASSES.split(","):
        files = sorted(glob.glob(f"{WORK}/synth_{tag}/g{g}/*.png"))[:8]
        rows.append(np.concatenate([np.asarray(add_visible_label(Image.open(f).convert("RGB"))) for f in files], axis=1))
    grid = Image.fromarray(np.concatenate(rows, axis=0)); grid.save(f"{WORK}/samples_{tag}.png"); display(grid.resize((grid.width // 2, grid.height // 2)))
for tag in PKLS: shutil.rmtree(f"{WORK}/synth_{tag}", ignore_errors=True)   # zips remain
os.chdir(WORK)
shutil.rmtree(CODE, ignore_errors=True)
print(sorted(os.listdir(WORK)))'''),
]


PREP_CELLS = [
    ("markdown", """# Data prep on Kaggle (CPU): splits + stylegan3 zips from the APTOS competition data

Use this when the lab PC has not produced the dataset yet. Before running:

1. **Add Input → Competitions → APTOS 2019 Blindness Detection** (accept the rules on the competition page first).
2. Optional: add an IDRiD dataset (part B, disease grading) and set `IDRID_INPUT` to its folder; otherwise IDRiD is skipped here and done on the lab PC.
3. Settings: **Internet ON**, CPU is enough. Run with *Save Version → Save & Run All (Commit)*.

Output (`/kaggle/working/out`): `splits/aptos_dedup_dropped.csv`, `splits/aptos_splits.csv`, `sg3/aptos256_f010..f100.zip`, `sg3/aptos256_val.zip`.
Then in the GAN notebooks use **Add Input → Your Work → this notebook** (its output is searched automatically).
Download `splits/*.csv` and commit them to the repo's `splits/` so every machine uses the split the GAN zips were built from.
They hold image ids, split and a grade-blind `frac_rank` only (no labels: APTOS terms, public repo); keep this notebook private."""),
    ("code", f'''REPO = "{REPO_URL}"
APTOS_INPUT = "auto"      # "auto" finds the folder holding train.csv + train_images under /kaggle/input
IDRID_INPUT = ""          # e.g. "/kaggle/input/idrid-disease-grading" (optional)
MAKE_CACHE = False        # also build the uint8 .npy caches (~800 MB) for the Kaggle classifier fallback'''),
    ("code", SETUP.replace('"ninja",\n                "psutil", "scipy"', '"imagehash", "opencv-python-headless", "scikit-learn", "pandas", "scipy", "pillow"')
                  .replace('for opt in ("pyspng", "imageio-ffmpeg"):', 'for opt in ():')),
    ("code", '''DATA = f"{WORK}/data"; OUT = f"{WORK}/out"
os.environ["GENAI_DATA"] = DATA
os.environ["GENAI_SPLITS"] = f"{OUT}/splits"
os.makedirs(f"{DATA}/raw/aptos", exist_ok=True); os.makedirs(f"{OUT}/splits", exist_ok=True)
if APTOS_INPUT == "auto":
    found = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train.csv", recursive=True)
             if os.path.isdir(os.path.join(os.path.dirname(p), "train_images"))]
    print("APTOS candidates:", found)
    assert found, "add the APTOS 2019 competition as input"
    APTOS_INPUT = found[0]
print("APTOS_INPUT =", APTOS_INPUT)
assert os.path.exists(f"{APTOS_INPUT}/train.csv"), "add the APTOS 2019 competition as input"
if not os.path.exists(f"{DATA}/raw/aptos/train_images"):
    os.symlink(f"{APTOS_INPUT}/train_images", f"{DATA}/raw/aptos/train_images")
shutil.copy(f"{APTOS_INPUT}/train.csv", f"{DATA}/raw/aptos/train.csv")
sh([sys.executable, "-m", "src.data.download_aptos", "--skip-download"])
if IDRID_INPUT:
    os.makedirs(f"{DATA}/raw/idrid", exist_ok=True)
    for name in os.listdir(IDRID_INPUT):
        dst = f"{DATA}/raw/idrid/{name}"
        if not os.path.exists(dst): os.symlink(f"{IDRID_INPUT}/{name}", dst)'''),
    ("code", '''sh([sys.executable, "-m", "src.data.prepare_images", "--source", "both" if IDRID_INPUT else "aptos"])
sh([sys.executable, "-m", "src.data.dedup"])
sh([sys.executable, "-m", "src.data.split"])
sh([sys.executable, "-m", "src.data.make_sg3_zips"])
if MAKE_CACHE:
    sh([sys.executable, "-m", "src.data.cache"])'''),
    ("code", '''shutil.copytree(f"{DATA}/sg3", f"{OUT}/sg3", dirs_exist_ok=True)
if MAKE_CACHE:
    shutil.copytree(f"{DATA}/cache", f"{OUT}/cache", dirs_exist_ok=True)
for root, _, files in os.walk(OUT):
    for f in files:
        p = os.path.join(root, f); print(f"{os.path.getsize(p) / 1e6:8.1f} MB  {p}")
os.chdir(WORK)
shutil.rmtree(DATA, ignore_errors=True)      # keep only /kaggle/working/out in the saved output
shutil.rmtree(CODE, ignore_errors=True)'''),
]


AUDIT_CELLS = [
    ("markdown", """# Duplicate audit (CPU): what does each rule call a near-duplicate?

Input: **APTOS 2019 Blindness Detection** competition. Internet ON. Crops all 3,662 images to 256 px,
then compares the pHash rule with a vessel-pattern correlation and shows contact sheets of flagged pairs."""),
    ("code", f'''REPO = "{REPO_URL}"'''),
    PREP_CELLS[2],
    ("code", '''DATA = f"{WORK}/data"; OUT = f"{WORK}/out"
os.environ["GENAI_DATA"] = DATA
os.makedirs(f"{DATA}/raw/aptos", exist_ok=True)
found = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train.csv", recursive=True)
         if os.path.isdir(os.path.join(os.path.dirname(p), "train_images"))]
assert found, "add the APTOS 2019 competition as input"
if not os.path.exists(f"{DATA}/raw/aptos/train_images"):
    os.symlink(f"{found[0]}/train_images", f"{DATA}/raw/aptos/train_images")
shutil.copy(f"{found[0]}/train.csv", f"{DATA}/raw/aptos/train.csv")
sh([sys.executable, "-m", "src.data.prepare_images", "--source", "aptos"])
sh([sys.executable, "-m", "src.data.dedup_audit", "--out", f"{OUT}/audit"])'''),
    ("code", '''from IPython.display import Image as IImage, display
print(open(f"{OUT}/audit/summary.txt").read())
display(IImage(f"{OUT}/audit/sheet_phash_pairs.png"))
display(IImage(f"{OUT}/audit/sheet_other_candidates.png"))
os.chdir(WORK)
shutil.rmtree(DATA, ignore_errors=True)
shutil.rmtree(CODE, ignore_errors=True)'''),
]


def nb(cells):
    out = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                     "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out["cells"].append(cell)
    return out


if __name__ == "__main__":
    (HERE / "kaggle_data_prep.ipynb").write_text(json.dumps(nb(PREP_CELLS), indent=1), encoding="utf-8")
    (HERE / "kaggle_dedup_audit.ipynb").write_text(json.dumps(nb(AUDIT_CELLS), indent=1), encoding="utf-8")
    (HERE / "kaggle_gan_train.ipynb").write_text(json.dumps(nb(TRAIN_CELLS), indent=1), encoding="utf-8")
    (HERE / "kaggle_gan_eval.ipynb").write_text(json.dumps(nb(EVAL_CELLS), indent=1), encoding="utf-8")
    print("wrote kaggle_data_prep.ipynb, kaggle_gan_train.ipynb, kaggle_gan_eval.ipynb")

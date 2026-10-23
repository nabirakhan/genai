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
FFHQ = ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/"
        "ffhq-res256-mirror-paper256-noaug.pkl")

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
    ("markdown", f"""# Conditional StyleGAN2-ADA fine-tune on APTOS-256 (Kaggle T4 x2)

One notebook, one configuration per version (`FRACTION`, `AUG`). Before running:

1. **Add data**: the private dataset `aptos-dr-256-genai` (built by the lab PC, S7).
2. **Resume**: add the previous version's *output* of this notebook as an input (`RESUME="auto"` finds the latest snapshot for the same `FRACTION-AUG`). First session uses the FFHQ-256 checkpoint.
3. Settings: Accelerator **GPU T4 x2**, **Internet ON**. Run with *Save Version → Save & Run All (Commit)* so `/kaggle/working/runs` persists.

Budget: `HOURS` of training per session (keep total < 12 h). Snapshots every `SNAP*4` kimg; only the last two are kept. `KIMG` is the **total** training length across sessions; the same value is used for every fraction once chosen from the f100 run.
Order of runs: f100-ada (sanity) → f010-ada → f010-noaug (ablation) → f025-ada → f050-ada."""),
    ("code", f'''FRACTION = "f100"      # f010 | f025 | f050 | f100
AUG = "ada"            # ada | noaug  (noaug only for the f010 ablation)
KIMG = 1000            # total kimg target across sessions (decide after f100 plateaus)
HOURS = 8.5            # training wall-clock budget for THIS session
RESUME = "auto"        # "ffhq" | "auto" (latest snapshot for FRACTION-AUG under /kaggle/input) | explicit path
REPO = "{REPO_URL}"
DATASET_DIR = "/kaggle/input/aptos-dr-256-genai"
GPUS, BATCH, BATCH_GPU, GAMMA, SNAP = 2, 32, 16, 1.0, 10
FFHQ = "{FFHQ}"'''),
    ("code", SETUP),
    ("code", '''# the zips come either from the data-prep notebook's output or from the aptos-dr-256-genai dataset
cands = sorted(glob.glob(f"/kaggle/input/**/aptos256_{FRACTION}.zip", recursive=True)) + \\
        sorted(glob.glob(f"/kaggle/input/**/aptos256_{FRACTION}", recursive=True))
if not cands:
    raise SystemExit(f"aptos256_{FRACTION} not found under /kaggle/input: add the data-prep notebook output or the dataset as input")
src = cands[0]
if os.path.isdir(src):                      # Kaggle extracted the zip -> re-zip locally (faster reads)
    DATA = f"{WORK}/data/aptos256_{FRACTION}.zip"
    if not os.path.exists(DATA):
        os.makedirs(f"{WORK}/data", exist_ok=True)
        shutil.make_archive(DATA[:-4], "zip", src)
else:
    DATA = src
print("DATA =", DATA, json.load(__import__("zipfile").ZipFile(DATA).open("dataset.json"))["labels"][:3])'''),
    ("code", '''def total_kimg(path):
    m = re.search(r"-from(\\d+)[/\\\\].*snapshot-(\\d+)", path)
    return int(m.group(1)) + int(m.group(2)) if m else 0
def latest_snapshot():
    snaps = [s for s in glob.glob("/kaggle/input/**/network-snapshot-*.pkl", recursive=True) if f"-{FRACTION}-{AUG}-from" in s]
    return max(snaps, key=total_kimg) if snaps else None
if RESUME == "ffhq":
    RESUME_PKL = FFHQ
elif RESUME == "auto":
    RESUME_PKL = latest_snapshot() or FFHQ
else:
    RESUME_PKL = RESUME
START = 0 if RESUME_PKL == FFHQ else total_kimg(RESUME_PKL)
REMAIN = max(0, KIMG - START)
print(f"resume from {RESUME_PKL}\\nalready trained {START} kimg, this session trains up to {REMAIN} kimg")
if RESUME_PKL == FFHQ:
    sh([sys.executable, "scripts/check_resume_compat.py", "--pkl", FFHQ])'''),
    ("code", '''desc = f"{FRACTION}-{AUG}-from{START}"
cmd = [sys.executable, "third_party/stylegan3/train.py", f"--outdir={WORK}/runs", "--cfg=stylegan2", f"--data={DATA}",
       "--cond=1", f"--gpus={GPUS}", f"--batch={BATCH}", f"--batch-gpu={BATCH_GPU}", f"--gamma={GAMMA}",
       "--cbase=16384", "--cmax=512", "--map-depth=8", "--mbstd-group=8", "--mirror=1", f"--aug={AUG}", "--target=0.6",
       f"--kimg={REMAIN}", "--tick=4", f"--snap={SNAP}", "--metrics=none", "--workers=2", f"--resume={RESUME_PKL}", f"--desc={desc}"]
print(" ".join(cmd))
def run_dir():
    ds = sorted(glob.glob(f"{WORK}/runs/*-{desc}")); return ds[-1] if ds else None
def snapshots():
    d = run_dir(); return sorted(glob.glob(f"{d}/network-snapshot-*.pkl")) if d else []
def prune(keep=2):
    for s in snapshots()[:-keep]:
        os.remove(s)
if REMAIN == 0:
    print("KIMG already reached; nothing to train")
else:
    logf = open(f"{WORK}/train_{desc}.log", "a")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time(); deadline = t0 + HOURS * 3600; stop_after = None; last_print = 0
    while p.poll() is None:
        time.sleep(60); prune()
        if time.time() - last_print > 600:
            last_print = time.time()
            tail = subprocess.run(["tail", "-n", "2", logf.name], capture_output=True, text=True).stdout.strip()
            print(f"[{(time.time()-t0)/3600:.2f} h] {tail[-300:]}")
        if stop_after is None and time.time() > deadline:
            stop_after = snapshots()[-1] if snapshots() else ""
            print("time budget reached; stopping after the next snapshot")
        if stop_after is not None and snapshots() and snapshots()[-1] != stop_after:
            time.sleep(45)   # let the pickle finish writing
            os.killpg(os.getpgid(p.pid), signal.SIGTERM); print("stopped after", snapshots()[-1]); break
        if stop_after is not None and time.time() > deadline + 50 * 60:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM); print("no new snapshot within 50 min; stopped"); break
    logf.close(); prune()
    print("exit code", p.poll())'''),
    ("code", '''from IPython.display import Image as IImage, display
d = run_dir()
if d:
    snaps = snapshots()
    print("run dir:", d); print("snapshots kept:", [os.path.basename(s) for s in snaps])
    if snaps: print(f"TOTAL kimg now = {total_kimg(snaps[-1])}  (use RESUME='auto' next session)")
    fakes = sorted(glob.glob(f"{d}/fakes*.png"))
    if fakes: display(IImage(fakes[-1], width=900))
    print(subprocess.run(["tail", "-n", "3", f"{d}/log.txt"], capture_output=True, text=True).stdout)
os.chdir(WORK)
shutil.rmtree(f"{WORK}/data", ignore_errors=True)      # keep the saved output small
shutil.rmtree(CODE, ignore_errors=True)'''),
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
Download `splits/*.csv` and commit them to the repo so every machine uses the same split."""),
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
    (HERE / "kaggle_gan_train.ipynb").write_text(json.dumps(nb(TRAIN_CELLS), indent=1), encoding="utf-8")
    (HERE / "kaggle_gan_eval.ipynb").write_text(json.dumps(nb(EVAL_CELLS), indent=1), encoding="utf-8")
    print("wrote kaggle_data_prep.ipynb, kaggle_gan_train.ipynb, kaggle_gan_eval.ipynb")

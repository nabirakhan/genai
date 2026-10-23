# Lab PC runbook (for Claude Code on the lab PC)

You are on the shared lab PC (Windows 11, RTX 4000 Ada 20 GB, i7-14700, 32 GB). Sessions are
~2 h a few times a week and the PC may be shut down afterwards. Every stage below is idempotent
and resumable. Work through the stages in order; `python -m src.pipeline status` tells you where
you are. Record anything that did not work as documented under **Field notes** at the end and commit it.

Rules
- Never commit data, `*.npy`, `*.pkl`, `*.pt`, `kaggle.json`. Only `splits/`, `results/`, `web/public/results/`, code and docs.
- Do not change the locked research design in CLAUDE.md. If a stage needs a design decision, stop and write it in Field notes.
- Never start a job that cannot checkpoint within 10 minutes. The classifier queue checkpoints every epoch.
- Run the end-of-session block before leaving, even mid-queue.

## Session start (every time)

```powershell
cd C:\genai                      # repo clone; first time: git clone --recurse-submodules https://github.com/nabirakhan/genai C:\genai
git pull --recurse-submodules
$env:GENAI_DATA = "C:\genai-data"     # fast SSD; raw zips are backed up to F:\genai-backup
.venv\Scripts\activate                # created in S0
python scripts/apply_patches.py
python -m src.pipeline status
nvidia-smi
```

Expected: a table of S0–S15 with DONE / PARTIAL / TODO. Continue at the first non-DONE stage.
`python -m src.pipeline cmd S4` prints a stage's exact command; `python -m src.pipeline run S4` runs it.

## S0 Environment (once, ~15 min)

```powershell
winget install --id Git.Git -e --silent
winget install --id astral-sh.uv -e --silent
# reopen the terminal so PATH updates
cd C:\genai
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements-lab.txt          # torch + torchvision, CUDA 12.8 wheels
uv pip install -e ".[api,dev]"
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True RTX 4000 Ada Generation`. If `cuda.is_available()` is False: update the NVIDIA driver (GeForce/Studio or RTX enterprise driver ≥ 550), then reinstall torch. No WSL and no CUDA toolkit are needed: GAN training runs on Kaggle, and stylegan3 inference falls back to reference ops.

Optional speed-up for the data stage: nothing; `prepare_images` uses all cores.

## S1 Download APTOS (~15 min, 9.5 GB)

1. On kaggle.com (your account): open the APTOS 2019 competition page and accept the rules once.
2. Account → API → *Create new token* → save `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json`. **Delete it at session end** (shared PC).
3. `python -m src.data.download_aptos`

Expected output ends with `APTOS extracted and verified.` (3,662 PNGs; counts 1805/370/999/193/295). Then back up the zip: `Copy-Item C:\genai-data\raw\aptos\aptos2019-blindness-detection.zip F:\genai-backup\`.

## S2 IDRiD (manual)

Download *B. Disease Grading* from IEEE DataPort (free login) on any machine and put the zip in
`C:\genai-data\raw\idrid\`. `prepare_images` extracts it. 516 images (413 train + 103 test), labels in the two CSVs.

## S3 Crop + resize (~5 min)

`python -m src.data.prepare_images --source both`
Expected: `aptos256` has 3,662 PNGs, `idrid256` has 516, `splits/idrid_test.csv` written, grade counts printed.

## S4 Dedup + split (~1 min)

```powershell
python -m src.data.dedup          # writes splits/aptos_dedup_dropped.csv, prints how many dropped
python -m src.data.split          # writes splits/aptos_splits.csv, prints grade x split table and fraction counts
git add splits && git commit -m "Add APTOS dedup list, splits and IDRiD labels" && git push
```

Expected: a few dozen duplicates dropped; train share per grade ≈ 0.70; fractions nested (asserted). These CSVs are the single source of truth for every machine; never regenerate them with a different seed.

## S5 Cache (~1 min)

`python -m src.data.cache` → `cache/aptos256.npy` (~710 MB) and `cache/idrid256.npy` (~100 MB).

## S6 stylegan3 zips (~2 min)

`python -m src.data.make_sg3_zips` → `sg3/aptos256_f010.zip … f100.zip, aptos256_val.zip`, each with `dataset.json`. Asserts no test image is included.

## S7 Kaggle dataset upload (~10 min, 1.5 GB)

`python -m src.data.kaggle_dataset` creates the private dataset `<user>/aptos-dr-256-genai` (or a new version). Expected: a URL printed and `C:\genai-data\kaggle_dataset\.uploaded` written. Kaggle notebooks attach this dataset.

## S8 Classifier runs, conditions 1–5 (60 runs, ~4 h total, resumable)

```powershell
python -m src.classify.queue --dry-run      # lists pending runs (84 total; condition 6 waits for synthetic images)
python -m src.classify.queue                # runs them; Ctrl+C stops after the current run
```

Expected: ~3–5 min per run on the RTX 4000 Ada; each run appends 2 rows (aptos_test, idrid) to `results/classifier_runs.csv` and a predictions file to `results/preds/`. Re-running the queue skips finished runs. If a run crashes (OOM, etc.) the queue continues; rerun the queue later. Watch the first run's epoch log: val QWK should climb above 0.8 within ~5 epochs at 100%.

Checks after a few runs: `python -m src.analysis.aggregate` prints the macro-F1 table so far.

## S9 Import synthetic images (after the Kaggle eval notebook ran)

```powershell
mkdir downloads; kaggle kernels output <user>/<eval-notebook-slug> -p downloads
python -m src.gan.import_synth downloads\synth_f010.zip downloads\synth_f025.zip downloads\synth_f050.zip downloads\synth_f100.zip
Copy-Item downloads\gan_metrics.csv results\gan_metrics.csv
```

Expected: `synth/f{ppp}/g3` and `g4` each with 500 PNGs carrying `Synthetic=true`.

## S10 Classifier runs, condition 6 (24 runs, ~1.5 h)

`python -m src.classify.queue` (it now finds the synthetic images). Keeps `best.pt` for `c5_f100_r0_s0` and `c6_f100_r1_s0` under `checkpoints/` (needed by S14/S15; do not delete).

## S11 Analysis (~5 min)

```powershell
python -m src.analysis.aggregate     # results/tables/*.csv|.md + web/public/results/tables.json
python -m src.analysis.bootstrap     # 95% CIs per run and paired deltas (cond 6 - cond 5)
python -m src.analysis.plots         # results/figures/*.png (+ copies in web/public/results)
git add results web/public/results && git commit -m "Results: tables, CIs, figures" && git push
```

## S12 Memorisation check (~10 min on GPU)

`python -m src.analysis.memorization` → `results/tables/memorization_summary.csv`, `memorization_{tag}.csv`, figures. Downloads the Inception detector once (internet).

## S13 Retinal Pigmentation Score

Needs Docker (`arajesh17/rps`, CPU, torch 1.7). Run it where Docker exists (laptop overnight is fine):
```powershell
python -m src.analysis.rps stage --per-grade 300     # copies PNGs + manifest to <GENAI_DATA>\rps
python -m src.analysis.rps run --dry-run             # shows the docker command; run without --dry-run (hours)
python -m src.analysis.rps merge                     # results/tables/rps_*.csv, figures/rps_hist.png
```
If the container layout differs from the README, adjust `RPS_CMD` in `src/analysis/rps.py` and note it below.

## S14 Grad-CAM figure (~1 min)

`python -m src.analysis.gradcam` → `results/figures/gradcam_examples.png`.

## S15 Demo assets (~10 min)

```powershell
# the final f100-ada pickle comes from the Kaggle train notebook output (download it to C:\genai-data\gan_pkls\)
$env:HF_TOKEN = "<token from huggingface.co/settings/tokens>"
python -m api.export_assets --pkl C:\genai-data\gan_pkls\f100-ada.pkl --upload <hf-user>/dr-gan-demo
uvicorn api.main:app --port 8000     # then open http://localhost:8000/docs and try /health, /generate
Remove-Item Env:HF_TOKEN
```

## Session end (every time, ~5 min)

```powershell
# 1. let the current classifier run finish its epoch (Ctrl+C in the queue window waits for the run to end)
scripts\end_of_session.ps1      # commits splits/ results/ web/public/results, pushes, removes kaggle.json, gh auth logout
```
Check `git status` is clean and `Test-Path $HOME\.kaggle\kaggle.json` is False before leaving.

## Optional: GAN training on the lab PC

The GPU can train a GAN at roughly 7–9 h per 1,000 kimg (about one run per lab week), so use it to
take one run off Kaggle (for example the f010 no-ADA ablation) or when the Kaggle quota is spent.
It needs stylegan3's CUDA kernels compiled once:

- **WSL2 route (reliable):** `wsl --install -d Ubuntu-22.04`, inside WSL install `build-essential` and the
  `cuda-toolkit-12-x` *wsl-ubuntu* package matching the torch build, create the venv there with the
  same `requirements-lab.txt`, and keep the repo and `GENAI_DATA` inside the WSL filesystem.
- **Native Windows route:** install Visual Studio Build Tools (C++ workload) and the CUDA toolkit
  matching torch; run from the "x64 Native Tools" prompt so `cl.exe` and `nvcc` are on PATH.

Then, with the venv active and the dataset zips from S6 present:

```powershell
python -m src.gan.train_local --fraction f010 --aug noaug --kimg 1000 --hours 2
```

It resumes from the latest snapshot of that fraction/aug automatically, stops after `--hours` at the
next snapshot, keeps the last two snapshots under `C:\genai-data\gan_runs\`, and prints the total
kimg reached. Run it again next session to continue. The classifier queue can run at the same time
(both fit in 20 GB; each slows the other down). The first run prints the custom-op build; if it
reports a fallback to reference ops, the compiler setup above is not working and training will be
far too slow to be useful.

## Fallbacks

- Lab time short: the classifier runs unchanged on Kaggle T4 (~8 min/run). Add `cache/*.npy` + `splits/` to the Kaggle dataset and run `python -m src.classify.queue` in a notebook with `GENAI_DATA=/kaggle/input/...` (read-only: set `GENAI_RESULTS`/`GENAI_CHECKPOINTS` to `/kaggle/working/...`).
- No CUDA on the lab PC: install the NVIDIA driver; if impossible, use the Kaggle fallback above.
- `prepare_images` slow: it is CPU-bound on PNG decode; ~3 min with 18 workers is normal.

## Field notes

(append dated notes here: what failed, what fixed it, timings observed)

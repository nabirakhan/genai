# Lab PC runbook (for Claude Code on the lab PC)

You are on the shared lab PC (Windows 11, RTX 4000 Ada 20 GB, i7-14700, 32 GB). Sessions are
~2 h a few times a week and the PC may be shut down afterwards. Every stage below is idempotent
and resumable. Work through the stages in order; `python -m src.pipeline status` tells you where
you are. Record anything that did not work as documented under **Field notes** at the end and commit it.

Rules
- Never commit data, `*.npy`, `*.pkl`, `*.pt`, `kaggle.json`. Only `splits/`, `results/`, `web/public/results/`, code and docs.
- The repo is public and APTOS may not be redistributed: never commit APTOS labels (grades per image id) or images of real APTOS eyes. The code already keeps grades out of `splits/` and `results/preds/`, and `.gitignore` holds back the `gradcam_*` and `memorization_*` figures, the per-image `memorization_f*.csv` tables and `rps_scores.csv` (only `memorization_summary.csv` is committed).
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

Expected: a table of S0–S15 (plus S8a, S10a) with DONE / PARTIAL / TODO. Continue at the first non-DONE stage.
`python -m src.pipeline cmd S4` prints a stage's exact command; `python -m src.pipeline run S4` runs it.
Claude Code: every shell call is a new process, so `activate` and `$env:` lines do not carry over; run `.venv\Scripts\python.exe ...` (or activate in the same call) and set `HF_TOKEN` in the call that uses it.

## S0 Environment (once, ~15 min)

```powershell
winget install --id Git.Git -e --silent
winget install --id astral-sh.uv -e --silent
# reopen the terminal so PATH updates
cd C:\genai
git config user.name "Nabira Khan"; git config user.email "nabirakhan.official@gmail.com"   # repo-local, not --global (shared PC)
git remote set-url origin https://nabirakhan@github.com/nabirakhan/genai.git   # per-user Credential Manager entry; other users' github.com logins stay untouched
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements-lab.txt          # torch + torchvision, CUDA 12.8 wheels
uv pip install -e ".[api,dev]"
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.11.0+cu128 True NVIDIA RTX 4000 Ada Generation`. A version without `+cu128` is the CPU wheel from the wrong index, not a driver problem (`uv pip install --reinstall -r requirements-lab.txt` after fixing the file). If it shows `+cu128` but `cuda.is_available()` is False: update the NVIDIA driver (GeForce/Studio or RTX enterprise driver ≥ 550), then reinstall torch. No WSL and no CUDA toolkit are needed: GAN training runs on Kaggle, and stylegan3 inference falls back to reference ops.

Optional speed-up for the data stage: nothing; `prepare_images` uses all cores.

## S1 Download APTOS (~15 min, 9.5 GB)

1. On kaggle.com (your account): open the APTOS 2019 competition page and accept the rules once.
2. Account → API → *Create new token*. Kaggle now issues a `KGAT_...` token: save it as plain text to `C:\Users\<you>\.kaggle\access_token` (an older `kaggle.json` also works). **Deleted at session end** by `scripts\end_of_session.ps1` (shared PC).
3. `python -m src.data.download_aptos`

Expected output ends with `APTOS extracted and verified.` (3,662 PNGs; counts 1805/370/999/193/295). Then back up the zip: `Copy-Item C:\genai-data\raw\aptos\aptos2019-blindness-detection.zip F:\genai-backup\`.

## S2 IDRiD (manual)

Download *B. Disease Grading* from IEEE DataPort (free login) on any machine and put the zip in
`C:\genai-data\raw\idrid\`. `prepare_images` extracts it. 516 images (413 train + 103 test), labels in the two CSVs.

## S3 Crop + resize (~5 min)

`python -m src.data.prepare_images --source both`
Expected: `aptos256` has 3,662 PNGs, `idrid256` has 516, `splits/idrid_test.csv` written, grade counts printed.

## S4 Verify the committed split (~10 s)

The split was made on Kaggle by the `genai-data-prep` notebook (vessel-correlation dedup, then the
stratified split); the GAN zips were built from it, so the classifier must use exactly the same file.
`splits/aptos_splits.csv` and `splits/aptos_dedup_dropped.csv` are already in git, without grades.

```powershell
python -m src.data.split          # joins grades from raw\aptos\train.csv and verifies; never pass --force
git add splits\idrid_test.csv; git commit -m "Add IDRiD test labels"; git push   # written by S3; CC BY 4.0, fine to commit
```

Expected: `verified against train.csv, not regenerating`, 3,484 rows, digest `017b5a96eb9b`, and these train counts:
f100 `{0: 1257, 1: 236, 2: 644, 3: 116, 4: 185}`; f010 keeps 12 / 19 of grades 3 / 4, f025 29 / 47, f050 58 / 93.
Do not run `src.data.dedup` or `split --force` here: a re-run on different crops could shift a
borderline pair and silently move test images into the GAN's training set.

## S5 Cache (~1 min)

`python -m src.data.cache` → `aptos256.npy: shape (3484, 256, 256, 3), 685 MB` and `idrid256.npy: shape (516, 256, 256, 3), 101 MB`.

## S6 / S7 stylegan3 zips + Kaggle dataset (optional, status SKIP)

Not needed: the Kaggle `genai-data-prep` notebook builds the zips and the GAN notebooks read its
output. Run `python -m src.data.make_sg3_zips` only for lab-PC GAN training (see the optional section below).

## S8a Pre-registered magnitude check (12 runs, ~1 h, before the rest of S8)

`report/prereg_checks.md` fixes the rule (RandAugment magnitude 9 vs 15, decided on validation data only). Do not change it.

```powershell
python -m src.classify.queue --checks magnitude --dry-run   # 6 baseline runs (c5 at 10%/100%, seeds 0-2) + 6 at magnitude 15
python -m src.classify.queue --checks magnitude             # resumable like S8; Ctrl+C stops after the current run
python -m src.analysis.checks magnitude --apply             # table + decision -> results\checks\decision_magnitude.json
git add results src\classify\config.json; git commit -m "Pre-registered magnitude check decided"; git push
```

Parallel windows: split the 12 runs by `--seed` as in S8 and apply after all windows have finished.

Expected: the 6 baseline runs are ordinary S8 runs (2 rows each in `results/classifier_runs.csv`); the 6 `*_m15_val` runs write one `dataset=val` row each to `results/checks/classifier_runs.csv` and are never scored on APTOS test or IDRiD. The queue prints no test/IDRiD metrics: until the decision is committed, do not open `results/classifier_runs.csv` or `results/preds/` and do not run `src.analysis.aggregate`. `checks` changes nothing while any of the 12 runs is missing (it lists them; rerun the queue). If 15 wins, `--apply` sets `ra_magnitude` 15 in `src/classify/config.json` and moves the 6 baseline runs to `results/checks/superseded_runs.csv` (+ `preds_superseded/`), so S8 re-runs them at 15 under the same ids; if 9 stays, only the decision is recorded. Then run S8 in full.

## S8 Classifier runs, conditions 1–5 (60 runs, ~9–12 h total, resumable, after S8a)

```powershell
python -m src.classify.queue --dry-run      # lists pending runs (84 total; all wait for S8a, condition 6 also for S10a and synthetic images)
python -m src.classify.queue                # runs them; Ctrl+C stops after the current run
```

Expected (measured 27 Oct): three seed windows keep the GPU at 100 % and finish ~5–7 runs/hour together (each run 20–30 min; condition 1 runs longest, early stop near epoch 15), ~16 GB peak; each run appends 2 rows (aptos_test, idrid) to `results/classifier_runs.csv` and a predictions file to `results/preds/`. Re-running the queue skips finished runs. If a run crashes (OOM, etc.) the queue continues; rerun the queue later. Watch the first run's epoch log: val QWK should climb above 0.8 within ~5 epochs at 100%.

Checks after a few runs: `python -m src.analysis.aggregate` prints the macro-F1 table so far.

**Three queue windows at once** (used on 27 Oct). The lab GPU is the bottleneck (100 % busy with three windows, ~16 GB), so extra windows add little throughput but keep the GPU fed between runs; on a Kaggle T4 one training step at batch 64 fp16 takes 164 ms with 3.9 GB peak. Run the queue in three PowerShell windows (venv active and `$env:GENAI_DATA` set in each), split by seed so that no run is in two windows:

```powershell
python -m src.classify.queue --seed 0 --workers 4    # window 1; windows 2 and 3: --seed 1, --seed 2
```

The same split works for `--checks magnitude` and `--checks psi` (4 runs per window); run `checks ... --apply` only after every window has finished. Never give two windows overlapping filters (a plain queue next to a `--seed 0` one would train the same runs twice). Per-run times in the CSV then include the sharing. An append that finds `results\classifier_runs.csv` open in another program (Excel) waits and says so until it is closed.

## S9 Import synthetic images (after a Kaggle eval run)

The Kaggle notebook `genai-gan-eval` runs twice. Version 1 (25 Oct) holds `synth_f010-ada.zip` and `synth_f100-ada.zip` (plus their `_psi1` zips for S10a); a later version (after the f025/f050 GANs finish, ~1 Nov) holds `synth_f025-ada.zip`, `synth_f050-ada.zip` and the f010-noaug metrics. Import what exists; S9 stays PARTIAL until all four sets are in, and the queue lists the missing condition-6 runs as waiting.

```powershell
Remove-Item downloads -Recurse -Force -ErrorAction SilentlyContinue; mkdir downloads
kaggle kernels output k230914nabirakhan/genai-gan-eval -p downloads      # newest version's output, ~1 GB with pkls\
python -m src.gan.import_synth downloads\synth_f010-ada.zip downloads\synth_f100-ada.zip   # second run: downloads\synth_f025-ada.zip downloads\synth_f050-ada.zip
python -c "import pandas as pd, pathlib as p; f = p.Path('results/gan_metrics.csv'); new = pd.read_csv('downloads/gan_metrics.csv', dtype={'grade': str}); old = pd.read_csv(f, dtype={'grade': str}) if f.exists() else new.iloc[:0]; pd.concat([old, new]).drop_duplicates(['gan', 'grade'], keep='last').to_csv(f, index=False)"
Copy-Item downloads\pkls\*.pkl C:\genai-data\gan_pkls\      # final GAN pickles, f100-ada.pkl is used by S15
```

The `python -c` line merges this run's metrics into `results\gan_metrics.csv` (rows of earlier runs are kept).

Expected: `synth\f{ppp}\g3` and `g4` each with 500 PNGs (ψ 0.8, conditional truncation: the condition-6 training images). Every PNG is checked before anything is written (256×256 RGB, `Synthetic=true`, grade = its folder, generator = `f<ppp>-ada`, seed = file name, `trunc_center` = class, ψ ≠ 1); `synth_f010-noaug.zip` is skipped; `_psi1` zips are imported in S10a. "holds images of another generation run" means `synth\<set>` already holds other images: stop and write a Field note.

`results\gan_metrics.csv`, per-class rows (grades 3, 4): `kid` = all val vs all 500 generated images of the grade, `kid_std` its standard error (`kid_se_real`: jackknife over the 25 / 39 val images; `kid_se_gen`: half-samples of the generated images, never with replacement). `<tag>_psi1` rows are the ψ-1 sets; `real-f100-train` is the real-vs-real floor (real val vs real train images of the grade: what "as good as real" looks like at this sample size). Per-class FID is empty on purpose (25 / 39 images cannot estimate a 2048-d covariance); grade `all` rows are calc_metrics' overall FID/KID.

## S10a Pre-registered ψ check (12 runs, ~1 h, after S9, before S10)

Same rule (`report/prereg_checks.md`): condition-6 images at ψ 0.8 (`synth\f<ppp>`) vs ψ 1 (`synth\f<ppp>_psi1`). Needs `synth_f010-ada_psi1.zip` and `synth_f100-ada_psi1.zip` from the Kaggle eval notebook output.

```powershell
python -m src.gan.import_synth downloads\synth_f010-ada_psi1.zip downloads\synth_f100-ada_psi1.zip   # -> synth\f010_psi1, synth\f100_psi1
python -m src.classify.queue --checks psi --dry-run     # 6 baseline runs (c6 1x at 10%/100%, seeds 0-2) + 6 on psi-1 images
python -m src.classify.queue --checks psi
python -m src.analysis.checks psi --apply
git add results src\classify\config.json; git commit -m "Pre-registered psi check decided"; git push
```

Parallel windows: split the 12 runs by `--seed` as in S8 and apply after all windows have finished.

As in S8a: no condition-6 test/IDRiD numbers are looked at before the decision is committed. If ψ 1 wins, `--apply` sets `synth_variant` "psi1" and supersedes the 6 baseline runs; S10 then trains every condition-6 run on `synth\f<ppp>_psi1`, so it also needs `synth_f025-ada_psi1.zip` and `synth_f050-ada_psi1.zip`: re-run the Kaggle eval notebook with `ONLY_TAGS = ["f025-ada", "f050-ada"]; PSI_CHECK_TAGS = ONLY_TAGS; PSI1_ONLY = True` and import only those two zips (the queue lists those runs as waiting until then). Then S10.

## S10 Classifier runs, condition 6 (24 runs, ~1.5 h)

`python -m src.classify.queue` (it now finds the synthetic images; parallel windows as in S8). Keeps `best.pt` for `c5_f100_r0_s0` and `c6_f100_r1_s0` under `checkpoints/` (needed by S14/S15; do not delete).

## S11 Analysis (~5 min)

```powershell
python -m src.analysis.aggregate     # results/tables/*.csv|.md + web/public/results/tables.json
python -m src.analysis.bootstrap     # 95% CIs per run and paired deltas (cond 6 - cond 5)
python -m src.analysis.plots         # results/figures/*.png (+ copies in web/public/results)
python -m src.analysis.fraction_table   # results/tables/fraction_label_shares.csv|.md
git add results web/public/results; git commit -m "Results: tables, CIs, figures"; git push   # Windows PowerShell 5.1 has no &&
```

`fraction_table` needs only the split and `raw\aptos\train.csv` (any time after S4): train images per grade per fraction and each grade's share of the GAN's conditioning labels (f010: grades 3 / 4 = 12 / 19 of 2,168 images, 0.55% / 0.88% of the labels G saw). Report it next to the GAN metrics.

## S12 Memorisation check (~10 min on GPU)

`python -m src.analysis.memorization` (every `synth\f<ppp>` and `synth\f<ppp>_psi1` set; `--sets f010,f100_psi1` for some; an unknown name is an error) → `results\tables\memorization_summary.csv` (aggregate counts per set and grade; commit it), per-image `memorization_<set>.csv` and `results\figures\memorization_<set>.png` (gitignored: nearest-neighbour ids, real APTOS eyes). Needs `cache\aptos256.npy` (S5) and `aptos256\` (S3); downloads the Inception detector once (internet) and caches train+val features in `<GENAI_DATA>\cache\inception_trainval.npy`. If the download fails on a long `tmp_...` path, set `$env:DNNLIB_CACHE_DIR = "C:\dnnlib"` in the same call. Every image is scored in both orientations (the GANs train with x-flips).

Expected and how to read it:
- `n_vessel_copies` 0 everywhere. This is the copy test: a shift-tolerant vessel correlation ≥ 0.90 (either orientation) means a training photograph was reproduced. Also check `n_vessel_above_ref` (closer to a training eye than any held-out real eye gets). Any hit: open the figure (it shows every vessel match) and write a Field note.
- `n_below_p1` / `n_below_p5` near 1% / 5% of `n_synth` would match new real eyes. More means "more typical than new real eyes", which truncation and grade-3/4-only sets cause without copying: compare with the same GAN's `_psi1` set before calling it memorisation.

## S12b GAN-test (~2 min on GPU; after S10a's decision is committed)

`python -m src.analysis.gan_test` → `results\tables\gan_test.csv` + `results\figures\gan_test.png` (aggregates only; commit both). Needs `checkpoints\c5_f100_r0_s0\best.pt` from the finished S8 run (refused if superseded or older than its results row) and the imported sets. The real-only classifier labels every synthetic set; per set and grade compare `share_as_grade` with `real_val_recall` (the same model on the real val images of that grade): far above it → over-typical or memorised images (see S12); mass on grades 0–2 (`pred_0`..`pred_2`) → the GAN leaks towards the common grades (see `fraction_label_shares.md`); `pred_3` vs `pred_4` shows whether it mixes the two rare grades. It reads val images, so run it only after the ψ decision is committed.

## S13 Retinal Pigmentation Score

Needs Docker (`arajesh17/rps`, CPU, torch 1.7). `stage` needs `aptos256\` and `raw\aptos\train.csv` (grades are joined from it), so run `stage` on the lab PC and copy `<GENAI_DATA>\rps` to the Docker machine (laptop overnight is fine); `run` and `merge` need neither:
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
# C:\genai-data\gan_pkls\f100-ada.pkl was copied there in S9 (from the genai-gan-eval output)
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
Check `git status` is clean, `Get-ChildItem $HOME\.kaggle` shows no `kaggle.json` / `access_token` / `credentials.json`, and `cmdkey /list:git:https://nabirakhan@github.com` lists nothing before leaving (the script erases only that per-user entry; leave other users' `git:https://github.com` logins alone).

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

- Lab time short: the classifier runs unchanged on Kaggle T4 (~8 min/run). Attach the data-prep output (run with `MAKE_CACHE = True`) and the APTOS competition, copy the competition's `train.csv` to `<GENAI_DATA>/raw/aptos/train.csv` (grades are joined from it; they are not in `splits/`), and run the S8a, S8 (later S10a, S10) commands in that order with `GENAI_DATA` on a writable copy (set `GENAI_RESULTS`/`GENAI_CHECKPOINTS` to `/kaggle/working/...`).
- No CUDA on the lab PC: install the NVIDIA driver; if impossible, use the Kaggle fallback above.
- `prepare_images` slow: it is CPU-bound on PNG decode; ~3 min with 18 workers is normal.

## Field notes

(append dated notes here: what failed, what fixed it, timings observed)

- **2026-10-27 (session 1).** S0–S5, S8a (magnitude 9 kept), S8 31/60, S9 f010/f100 + both `_psi1` sets imported.
  - Timings: APTOS download ~69 MB/s (~3 min); S3 APTOS 20 s, IDRiD 7 s; S5 ~20 s. Classifier runs are GPU-bound (100 % util, ~99 W, CPU ~50 %): with 3 seed windows each run takes ~20–30 min (c1 runs longest, early stop near epoch 15), i.e. ~5–7 runs/hour overall, about half the speed assumed in S8. Peak memory ~16 GB for 3 windows.
  - Kaggle now issues `KGAT_...` API tokens: save to `$HOME\.kaggle\access_token` (no JSON); the CLI picks it up.
  - Shared PC: Windows Credential Manager may already hold another user's `git:https://github.com` login. The remote is set to `https://nabirakhan@github.com/nabirakhan/genai.git` so GCM stores and uses a separate per-user entry; `end_of_session.ps1`'s generic `cmdkey /delete:git:https://github.com` would remove the other user's entry, so session 1 ended with the steps run by hand (commit, push, Kaggle token removal) instead of the script.
  - From Git Bash, `robocopy` needs `MSYS_NO_PATHCONV=1` or its `/E`-style switches are rewritten as paths (exit 16).

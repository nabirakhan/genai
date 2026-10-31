# GenAI Project — Context for Claude Code

AI4009 Generative AI course project, Fall 2026, FAST-NUCES Karachi. Solo: Nabira Khan (23K-0914). Instructor: Dr. Usman Ali.
Title: **When Does a GAN Help? Comparing GAN-Generated Retinal Images Against Simple Fixes for Rare Severe Diabetic Retinopathy Grades**

Deadline: ~13 Dec 2026. Deliverables: IEEE-style report (PDF), 10-min live demo, this repo + README, web demo app (React on Vercel + FastAPI inference API on a Hugging Face Space; replaces the proposal's Gradio wording).

## Research question (2 lines)
Severe DR (grades 3, 4) is rare in training data, so classifiers miss it. Test whether GAN-generated fundus images help more than cheap fixes, and at what amount of real data and synthetic ratio the GAN stops being worth it.

## Locked design (do not change without asking)
- **Generator:** class-conditional StyleGAN2-ADA on **all 5 grades** (rare grades borrow structure from grades 0–2). Code: `NVlabs/stylegan3` with `--cfg=stylegan2 --aug=ada --cond=1`, 256×256, fine-tuned from the released FFHQ-256 StyleGAN2 checkpoint (class-embedding layer newly initialised; `--cbase=16384 --cmax=512 --map-depth=8`). Not the old `stylegan2-ada-pytorch` (breaks on new PyTorch / Ada GPUs).
- **Data:** APTOS 2019 (3,662 imgs; grade counts 1805/370/999/193/295) for training. IDRiD disease-grading set (516 imgs, CC BY 4.0) as **external test only**, never trained or tuned on.
- **Split:** stratified 70/15/15 train/val/test, seed 42. Duplicate removal *before* splitting so copies never straddle splits: shift-tolerant vessel-pattern correlation ≥ 0.90, no transitive chaining, copies with conflicting grades dropped together (`src/data/dedup.py`; pHash was tried first and rejected, it chained 1,242 distinct eyes into "duplicates", see `src/data/dedup_audit.py`). Test split never touches the GAN. The committed `splits/*.csv` are the source of truth on every machine: written by the Kaggle `genai-data-prep` notebook (which built the GAN zips from them), they hold image ids, split and a grade-blind `frac_rank` only, never labels (APTOS terms, public repo); `load_splits()` joins grades from the local `raw/aptos/train.csv`.
- **Fractions:** 10/25/50/100% of grade 3+4 training images; grades 0–2 always kept in full. Subsets **nested** (10 ⊂ 25 ⊂ 50 ⊂ 100), fixed seed. One GAN per fraction, trained only on that fraction's train split, same kimg budget for all.
- **Classifier conditions (6):** (1) none, (2) class weights, (3) oversampling, (4) classical aug (flip/rotate/color jitter), (5) RandAugment + mixup, (6) = (5) + GAN images of grades 3/4 at 1× real rare-grade count.
- **Synthetic-ratio sweep:** condition (6) also at 0.5× and 2×, only at fractions 10% and 100%.
- **Seeds:** 3 per setting → 6×4×3 + 2×2×3 = **84 classifier runs**.
- **Metrics:** macro F1, quadratic weighted kappa, recall for grades 3 and 4 — on APTOS test and IDRiD, with bootstrap CIs from saved per-image predictions. GAN: FID/KID (rough indicators only), visual check.
- **Ablation:** StyleGAN2 with vs without ADA at 10%.
- **Responsible AI:** Retinal Pigmentation Score on real vs generated; nearest-neighbour memorisation check; all synthetic images carry `Synthetic=true` PNG metadata and a visible SYNTHETIC label wherever displayed (never on classifier training copies: a visible mark would be a shortcut feature).
- **Main question:** does (6) beat (5)? Negative result is valid.

## Status and next steps (updated 2026-10-25)
- **GANs** (Kaggle `genai-gan-train`, 1,000 kimg each): f100-ada, f010-ada and f010-noaug done; f025-ada at 400 kimg after version 6 (25 Oct); f050-ada not started. Version 7 (started 31 Oct 11:40 UTC, 10 h) resumes f025 from 400 kimg and then starts f050; version 8 finishes f050 (weekly GPU quota 30 h, resets Saturday 00:00 UTC). The notebook's `OVERRIDES` is back to `[]` (version 6 used `["--hours", "7.9"]` to fit the last quota hours; do the same for any session that would outrun the remaining quota).
- **Eval** (Kaggle `genai-gan-eval`): version 1 (25 Oct) holds `synth_f010-ada.zip`, `synth_f100-ada.zip`, their `_psi1` zips, `gan_metrics.csv` (overall FID f010 17.2, f100 18.3), sample grids and `pkls/`. Once f025/f050 finish, run it again with `ONLY_TAGS = ["f010-noaug", "f025-ada", "f050-ada"]`.
- **Lab PC** (sessions 27–30 Oct): S0–S5 done; S8a decided (magnitude 9 kept); S8 complete (60/60); S9 imported f010/f100 and both `_psi1` sets; S10a has 5 of 6 baseline runs (missing `c6_f010_r1_s2`) and 0 of 6 ψ-1 runs, no decision yet. Remaining: finish S10a (7 runs) and decide, then S10 (12 ratio runs at 10%/100%; the 6 runs at 25%/50% wait for the second eval run), then S11–S15. At the measured ~5–7 runs/hour that is ~4–5 lab hours. Run the queue in three windows split by `--seed` (runbook S8). Commit after S4, after each pre-registered decision, and at session end (`scripts\end_of_session.ps1`). Until a check's decision is committed, do not open `results/classifier_runs.csv` / `results/preds/` or run `src.analysis.aggregate` (`report/prereg_checks.md`, Data use).

## Where things run
| Machine | Use for |
|---|---|
| **Kaggle** (account k230914nabirakhan, private notebooks) | `genai-data-prep` (CPU): dedup, split, GAN zips. `genai-gan-train` (T4 x2, internet on): all GAN training (5 runs: 4 fractions + no-ADA ablation), resuming across sessions from its previous version's output. `genai-gan-eval`: FID/KID, image generation. |
| **Lab PC** (Windows 11, RTX 4000 Ada 20 GB) | APTOS/IDRiD download, crop, cache (the split comes from git), the 84 classifier runs, IDRiD eval, analysis, memorisation, Grad-CAM, demo-asset export. **Follow `LAB_RUNBOOK.md` stage by stage**; `python -m src.pipeline status` shows progress. |
| **Nabira's laptop** (no NVIDIA GPU) | Code, CPU smoke tests, notebooks, report, RPS via Docker, frontend design. |
| **HF Space** (Docker, free CPU) + **Vercel** | Demo: `api/` inference service, `web/` React frontend. |

### Lab PC constraints
- i7-14700 (20C/28T), 32 GB RAM, C: 512 GB SSD, F: 2 TB HDD. Admin rights. **Available only during lab hours (~8 h/week)**; PC may be shut down after. Every job must be resumable; never plan unattended multi-day runs.
- Shared machine: never leave tokens (`kaggle.json`, `gh` auth, `HF_TOKEN`) on it at session end. `scripts/end_of_session.ps1` removes them. Never commit credentials.

## Setup (any machine, native Windows, no WSL needed)
GAN training is on Kaggle, so nvcc is never needed locally; stylegan3 inference falls back to reference ops.
```powershell
git clone --recurse-submodules https://github.com/nabirakhan/genai; cd genai   # Windows PowerShell 5.1: no &&
uv venv .venv --python 3.11; .venv\Scripts\activate
uv pip install -r requirements-lab.txt            # CUDA 12.8 torch; CPU machines: --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[api,dev]"
python scripts/apply_patches.py                   # shape-tolerant resume + ADA p restore, see third_party/PATCHES.md
$env:GENAI_DATA = "C:\genai-data"                 # data root (laptop: E:\genai-data); never inside the repo
python tests/smoke_synthetic.py                   # CPU end-to-end on fake images
```

## Session workflow (lab hours)
1. `git pull --recurse-submodules`, activate venv, set `GENAI_DATA`, `python -m src.pipeline status`.
2. Continue at the first non-DONE stage of `LAB_RUNBOOK.md`. The classifier queue skips finished runs and resumes from `checkpoints/{run_id}/last.pt`.
3. Results append to `results/classifier_runs.csv` (2 rows per run: aptos_test, idrid) and `results/preds/`. Small files only in git.
4. ~10 min before leaving: let the current run finish, run `scripts\end_of_session.ps1` (commit + push + credential cleanup).

## Git and publishing
- Commit and push to `main` as **Nabira Khan** only. In the repo (not `--global` on the shared lab PC): `git config user.name "Nabira Khan"`, `git config user.email "nabirakhan.official@gmail.com"`. Never add `Co-Authored-By` or "Generated with Claude Code" lines to commit messages or PRs.
- Keep this project's work local: no artifacts, cloud documents or other publishing (web search is fine).

## Repo layout
```
src/genai/paths.py   all paths; GENAI_DATA / GENAI_SPLITS / GENAI_RESULTS / GENAI_CHECKPOINTS env overrides
src/pipeline.py      stage tracker (status | cmd | run)
src/data/            download, crop (fundus.py), dedup, split (+ fraction_mask), cache, sg3 zips, kaggle dataset
src/classify/        data.py (6 conditions), model.py, metrics.py, train.py (resumable), queue.py (84 runs + --checks), config.json (settings fixed by the pre-registered checks)
src/gan/             kaggle_train (Kaggle session runner), train_local (optional lab-PC training), generate, metrics_per_class (FID/KID), watermark, import_synth; __init__ adds stylegan3 to sys.path
src/analysis/        aggregate, bootstrap, plots, checks (pre-registered decisions), memorization, gan_test, fraction_table, rps, gradcam
notebooks/           build_notebooks.py -> kaggle_gan_train.ipynb, kaggle_gan_eval.ipynb
third_party/         stylegan3 submodule (pinned), patches/, PATCHES.md
api/                 FastAPI service (core.py, main.py, export_assets.py, Dockerfile, space/)
web/                 Vite + React + TS (api.ts, types.ts, pages/); public/results/ exported by analysis
splits/ results/     tracked CSVs, tables, figures
scripts/             apply_patches.py, check_resume_compat.py, end_of_session.ps1
tests/               smoke_synthetic.py (CPU end-to-end), test_checks, test_gan_eval, test_kaggle_train, test_csv_concurrency
report/ demo/        outline, figure manifest, prereg_checks.md (pre-registered rules), demo script
```
Never commit: datasets, generated images in bulk, `*.pkl` / `*.pt` / `*.npy`, `kaggle.json`, `.env`, `web/node_modules`.

## Classifier defaults
EfficientNet-B0 (timm, ImageNet weights), 256px, fp16 AMP, batch 64, AdamW 3e-4 + cosine, ≤20 epochs, early stop (patience 6) on val QWK, memmapped uint8 cache, 6 persistent DataLoader workers. ~3–5 min/run on the RTX 4000 Ada.

## GAN defaults (Kaggle)
`train.py --cfg=stylegan2 --cond=1 --gpus=2 --batch=32 --batch-gpu=16 --gamma=1 --cbase=16384 --cmax=512 --map-depth=8 --mbstd-group=8 --mirror=1 --aug=ada --target=0.6 --tick=4 --snap=10 --metrics=none`, resume from FFHQ-256; measured 40.5 s/kimg on T4 x2 (1,000 kimg ≈ 11.4 h, so about 57 GPU-h for all five GANs); order f100 → f010 → f010-noaug → f025 → f050. `src/gan/kaggle_train.py` runs this queue, one ~10 h session per notebook version (settings in `notebooks/kaggle_session.json`; attach the newest train output + the data-prep output). Generation: 500 per rare grade, psi 0.8, seeds 0–499, conditional truncation towards each grade's mean w (`--trunc-center class`; the global `w_avg` is ~52–58% grade 0 and would pull rare-grade samples ~19% towards other grades). RandAugment in condition 5 is torchvision magnitude 9 of 31 bins (≈ M=3 on the paper/timm 0–10 scale).

## Timeline (weeks from 19 Oct 2026)
1 code + smoke tests, lab session 1 (data) · 2 lab: 60 baseline runs; Kaggle: f100 GAN · 3 Kaggle: f100 done, f010 ×2 · 4 Kaggle: f025, f050, generation; api/web skeleton · 5 lab: condition-6 runs · 6 RPS, memorisation, demo assets · 7 analysis, figures, frontend design · 8 report, README, rehearsal.

## Known issues / decisions
- On Nabira's laptop `python` is the Microsoft Store alias: use `py` or `.venv\Scripts\python.exe`. On the lab PC the activated venv's `python` is fine.
- numpy's OpenBLAS can fail to allocate thread buffers when several processes start together; `src/__init__.py` caps `OPENBLAS_NUM_THREADS` before numpy loads.
- stylegan3 resume from an unconditional pickle needs the shape-tolerant copy patch (fc0 and D.b4.out mismatch); verified on CPU by `scripts/check_resume_compat.py`.
- Windows shell commands over ~8k characters fail (cmd.exe limit); write files with the editor tools, not giant heredocs.
- APTOS may not be redistributed and the repo is public: the demo returns nearest-neighbour thumbnails ≤96 px + image id only; committed CSVs and `results/preds/` carry no APTOS labels (joined at analysis time); figures showing real APTOS images (`gradcam_*`, `memorization_*`), the per-image `memorization_f*.csv` tables (nearest neighbours are same-grade, so ids imply grades) and `rps_scores.csv` are gitignored and used only in the report PDF.
- Kaggle runs Python 3.13 + PyTorch 2.11: stylegan3 needs patches 0002 (distutils/pkg_resources imports), 0003 (custom-op module loading), 0004 (InfiniteSampler calls `Sampler.__init__()` without data_source) and 0005 (SciPy >= 1.18 `sqrtm` without `disp`, FID only), see `third_party/PATCHES.md`.

## How to help me
I learn fast — explain once. Give the fix directly, then why. Flag security issues. Optimise for memory and speed. No unnecessary boilerplate or comments.

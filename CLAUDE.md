# GenAI Project — Lab PC Context

AI4009 Generative AI course project, Fall 2026, FAST-NUCES Karachi. Solo: Nabira Khan (23K-0914). Instructor: Dr. Usman Ali.
Title: **When Does a GAN Help? Comparing GAN-Generated Retinal Images Against Simple Fixes for Rare Severe Diabetic Retinopathy Grades**

Deadline: ~mid-Dec 2026. Deliverables: IEEE-style report (PDF), 10-min live demo, this repo + README, Gradio demo app.

## Research question (2 lines)
Severe DR (grades 3, 4) is rare in training data, so classifiers miss it. Test whether GAN-generated fundus images help more than cheap fixes, and at what amount of real data and synthetic ratio the GAN stops being worth it.

## Locked design (do not change without asking)
- **Generator:** class-conditional StyleGAN2-ADA on **all 5 grades** (rare grades borrow structure from grades 0–2). Code: `NVlabs/stylegan3` with `--cfg=stylegan2 --aug=ada --cond=1`, 256×256, fine-tuned from a released StyleGAN2 256px checkpoint (class-embedding layer newly initialised). Not the old `stylegan2-ada-pytorch` (breaks on new PyTorch / Ada GPUs).
- **Data:** APTOS 2019 (3,662 imgs; grade counts 1805/370/999/193/295) for training. IDRiD disease-grading set (516 imgs, CC BY 4.0) as **external test only**, never trained or tuned on.
- **Split:** stratified 70/15/15 train/val/test, fixed seed. Perceptual-hash dedup *before* splitting so near-duplicates never straddle splits. Test split never touches the GAN.
- **Fractions:** 10/25/50/100% of grade 3+4 training images; grades 0–2 always kept in full. Subsets **nested** (10 ⊂ 25 ⊂ 50 ⊂ 100), fixed seed. One GAN per fraction, trained only on that fraction's train split.
- **Classifier conditions (6):** (1) none, (2) class weights, (3) oversampling, (4) classical aug (flip/rotate/color jitter), (5) RandAugment + mixup, (6) = (5) + GAN images of grades 3/4 at 1× real rare-grade count.
- **Synthetic-ratio sweep:** condition (6) also at 0.5× and 2×, only at fractions 10% and 100%.
- **Seeds:** 3 per setting → 6×4×3 + 2×2×3 = **84 classifier runs**.
- **Metrics:** macro F1, quadratic weighted kappa, recall for grades 3 and 4 — on APTOS test and IDRiD. GAN: FID/KID (rough indicators only), visual check.
- **Ablation:** StyleGAN2 with vs without ADA at 10%.
- **Responsible AI:** Retinal Pigmentation Score on real vs generated; nearest-neighbour memorisation check; all synthetic images labelled/watermarked SYNTHETIC.
- **Main question:** does (6) beat (5)? Negative result is valid.

## Where things run
| Machine | Use for |
|---|---|
| **Kaggle T4** | All GAN training (5 runs: 4 fractions + no-ADA ablation), FID/KID, image generation. Runs short + checkpointed, resume across sessions. |
| **Lab PC** (this machine) | Data prep, the 84 classifier runs, IDRiD eval, RPS, Grad-CAM, Gradio app dev. |

The proposal only mentions Kaggle; the lab PC is extra capacity.

### Lab PC constraints
- Windows 11, RTX 4000 Ada **20 GB**, i7-14700 (20C/28T), 32 GB RAM. C: 512 GB SSD, F: 2 TB HDD.
- Admin rights. **Available only during lab hours (~8 h/week)** — PC may be shut down after. Every job must be resumable; never plan unattended multi-day runs.
- Shared machine: never leave tokens (`kaggle.json`, `gh` auth) on it at session end unless it's a personal account. Never commit credentials.

## Lab PC setup (one time)
1. **WSL2** (admin PowerShell): `wsl --install -d Ubuntu-22.04`, reboot, create user. The Windows NVIDIA driver already exposes CUDA to WSL — **do not install a Linux NVIDIA driver inside WSL**. Check: `nvidia-smi` inside WSL.
2. Inside WSL: `sudo apt update && sudo apt install -y build-essential git tmux unzip`.
3. CUDA toolkit (needed only for stylegan3 custom ops — `nvcc`): install the `cuda-toolkit-12-x` **wsl-ubuntu** package matching the PyTorch CUDA build.
4. Miniconda → `conda create -n genai python=3.11 -y && conda activate genai`.
5. PyTorch with CUDA 12.x wheels (pick from pytorch.org), then:
   `pip install click requests tqdm pyspng ninja imageio-ffmpeg scipy psutil pillow scikit-learn pandas matplotlib timm imagehash opencv-python-headless gradio grad-cam kaggle`
6. Verify: `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
7. Kaggle API: put `kaggle.json` in `~/.kaggle/`, `chmod 600`. Accept APTOS competition rules on the website first, else download fails.
8. Keep repo and cached data in the WSL filesystem (`~/genai`, `~/data`) — `/mnt/c` and `/mnt/f` are slow. Use F: only for raw-zip backups.

## Session workflow (lab hours)
1. `tmux new -s lab` (or `tmux a -t lab`), `conda activate genai`, `git pull`.
2. Run the classifier queue script; it skips runs whose result row already exists and resumes from the last checkpoint.
3. Results append to `results/*.csv` (one row per run: condition, fraction, ratio, seed, dataset, metrics). Small files only in git.
4. ~10 min before leaving: let current run checkpoint, `git add results/ && git commit && git push`.

## Planned repo layout
```
src/data/       dedup (pHash), stratified split, nested fraction subsets, 256px cache
src/classify/   train.py --condition --fraction --ratio --seed ; eval on APTOS test + IDRiD
src/analysis/   tables, learning curves, fraction × ratio map, RPS, memorisation NN
notebooks/      Kaggle GAN notebooks (stylegan3 fine-tune, FID/KID, generation)
app/            Gradio: Generate (+ nearest real neighbour), Classify (baseline vs GAN + Grad-CAM), Results
results/        CSV metrics
third_party/    stylegan3 (pinned commit, not modified in place)
```
Never commit: datasets, generated images in bulk, `*.pkl` / checkpoints, `kaggle.json`, `.env`.

## Classifier defaults (T4 + lab friendly)
EfficientNet-B0 (timm, ImageNet weights) or ResNet-18, 256px, AMP, batch 64, AdamW, ~20 epochs, early stop on val QWK, cached uint8 tensors. Target ~5 min/run → 84 runs ≈ one week of lab time.

## Timeline
1 data (APTOS + IDRiD, dedup, split, cache) · 2 classifier + 5 baselines, set up stylegan3 · 3 conditional GAN fine-tune, FID/KID · 4 GAN per fraction, ADA ablation, memorisation · 5 learning-curve + ratio runs, IDRiD eval · 6 RPS, collect results · 7 analysis, responsible-AI, Gradio app · 8 report, README, demo rehearsal.

## Open risks
- Unverified: stylegan3 fine-tuning with `--cond=1` from an unconditional pickle — test early in week 2. Fallback: train at 128px, or report the failure + ADA ablation as the finding.
- stylegan3 custom ops may need minor patches on new PyTorch; record any patch in `third_party/PATCHES.md`.

## How to help me
I learn fast — explain once. Give the fix directly, then why. Flag security issues. Optimise for memory and speed. No unnecessary boilerplate or comments.

---
title: When Does a GAN Help? demo API
emoji: 👁️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

FastAPI inference service for the AI4009 project *When Does a GAN Help?* (Nabira Khan, FAST-NUCES).
Serves `/generate` (class-conditional StyleGAN2-ADA, CPU), `/classify` (EfficientNet-B0 baseline vs
GAN-augmented, with Grad-CAM) and `/bank`. The React frontend on Vercel calls it.

Setup: create a Docker Space, add `Dockerfile` and this `README.md`, then set the secrets
`HF_MODEL_REPO`, `HF_TOKEN` (required: the model repo must stay private, it holds APTOS thumbnails), `CORS_ORIGINS` (comma-separated Vercel URLs), `PUBLIC_MODE=1`.
Assets are produced by `python -m api.export_assets --upload <HF_MODEL_REPO>` on the lab PC.
Generated images are experimental, labelled SYNTHETIC and not for clinical use.

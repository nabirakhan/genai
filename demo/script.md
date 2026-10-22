# 10-minute live demo script

Before: warm the HF Space (`/health`), open the Vercel app, have the local fallback ready
(`uvicorn api.main:app --port 8000` + `VITE_API_URL=http://localhost:8000`). Have `results/figures/` open as backup.

| min | what | where |
|---|---|---|
| 0–1 | Problem: grades 3/4 are 13% of APTOS; classifiers miss referable cases. The question: GAN vs cheap fixes, and when. | slide |
| 1–2 | Data pipeline in one picture: crop → dedup → split → nested fractions; IDRiD as external test. | slide (table of counts) |
| 2–4 | **Generate** tab: pick grade 4, a seed, psi 0.8 → synthetic image with SYNTHETIC label; show nearest real neighbour thumbnail and distance (memorisation check). Change grade to 0 to show conditioning. | web app |
| 4–6 | **Classify** tab: upload a held-out grade-3 image → baseline vs GAN-augmented probabilities and Grad-CAM side by side. One sentence on what the heat maps attend to. | web app |
| 6–8 | **Results** tab: main table at 100%, learning curve (macro F1 and recall-3 vs fraction), ratio map. State the paired Δ(6−5) with CIs. | web app / figures |
| 8–9 | External test: APTOS vs IDRiD; ADA ablation at 10%; RPS real vs synthetic. | figures |
| 9–10 | Answer the title question in two sentences; limitations; what I would do with 10x compute. | slide |

Fallbacks: Space cold → local API; upload fails → use `web/public/results/` figures; questions on cost →
GAN hours on Kaggle vs minutes for RandAugment + mixup.

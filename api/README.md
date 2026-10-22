# Demo API

FastAPI service in `api/`, inference code in `api/core.py` (no web imports), deployed as a Docker
Hugging Face Space (free CPU) via `api/space/`. The Vercel frontend (`web/`) calls it.

## Run locally

```powershell
$env:DEMO_ASSETS = "C:\genai-data\demo_assets"     # produced by python -m api.export_assets
.venv\Scripts\uvicorn api.main:app --reload --port 8000
# http://localhost:8000/docs
```

## Endpoints (contract for `web/src/types.ts`)

| method | path | body | response |
|---|---|---|---|
| GET | `/health` | | `{status, device, assets:{file:bool}, bank:int, loaded:{G,classifiers}}` |
| POST | `/generate` | `{grade:0-4, seed:int, psi:0.3-1.0, nearest:bool}` | `{image_png_b64, grade, seed, psi, synthetic:true, nearest_real:{image_id, distance, thumb_png_b64}, seconds}` |
| POST | `/classify` | multipart `file` (≤5 MB image) | `{probs:{baseline:[5], gan:[5]}, pred:{baseline, gan}, gradcam_png_b64:{baseline, gan}, preprocessed_png_b64, grade_names, seconds}` |
| GET | `/bank?grade=` | | `{grade, files:[...], base_url:"/static/bank/"}` |

Rules: generated images always carry the visible SYNTHETIC label and PNG metadata. Nearest real
neighbours are returned only as ≤96 px thumbnails plus the APTOS image id (APTOS data may not be
redistributed). Uploads are size-checked, decoded with PIL and re-encoded before use. CORS is
restricted to `CORS_ORIGINS` and `*.vercel.app`; a per-IP limit of `RATE_PER_MIN` requests applies.

## Assets

`python -m api.export_assets --pkl <f100 pkl> [--upload <hf repo>]` writes `DATA_ROOT/demo_assets/`:
`G_ema_f100.pkl` (generator only), `effnet_c5_f100.pt`, `effnet_c6_f100.pt`, `nn_index.npz`,
`nn_thumbs.zip`, `bank/`. The Space downloads them from `HF_MODEL_REPO` on first start.
Cold start of a sleeping Space is 1–2 minutes; open `/health` before a live demo.

"""FastAPI inference service for the demo (deployed as a Docker Hugging Face Space, port 7860).

  uvicorn api.main:app --reload --port 8000        # local, uses DEMO_ASSETS or DATA_ROOT/demo_assets

Env: DEMO_ASSETS, HF_MODEL_REPO, HF_TOKEN (private model repo), CORS_ORIGINS (comma list),
     CORS_ORIGIN_REGEX (default allows *.vercel.app), RATE_PER_MIN (default 30), PUBLIC_MODE (1 = thumbnails only).
OpenAPI at /docs is the contract for web/src/types.ts.
"""
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import numpy as np

from api.core import Backend, to_b64_png

MAX_UPLOAD = 5 * 1024 * 1024
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "30"))
_hits: dict[str, deque] = defaultdict(deque)
backend: Backend | None = None


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "?"
    q = _hits[ip]
    now = time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        raise HTTPException(429, "rate limit: try again in a minute")
    q.append(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global backend
    backend = Backend()
    bank = backend.assets / "bank"
    if bank.exists():
        app.mount("/static/bank", StaticFiles(directory=str(bank)), name="bank")
    yield


app = FastAPI(title="When Does a GAN Help? demo API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o],
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app"),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class GenerateIn(BaseModel):
    grade: int = Field(ge=0, le=4)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    psi: float = Field(default=0.8, ge=0.3, le=1.0)
    nearest: bool = True


@app.get("/health")
def health():
    return {"status": "ok", **backend.status()}


@app.post("/generate")
def generate(body: GenerateIn, request: Request):
    rate_limit(request)
    t0 = time.time()
    try:
        img = backend.generate(body.grade, body.seed, body.psi)
    except FileNotFoundError as e:
        raise HTTPException(503, f"generator not available: {e}")
    out = {"grade": body.grade, "seed": body.seed, "psi": body.psi, "synthetic": True,
           "image_png_b64": to_b64_png(backend.watermarked(img))}
    if body.nearest:
        try:
            out["nearest_real"] = backend.nearest_real(np.asarray(img))
        except FileNotFoundError:
            out["nearest_real"] = None
    out["seconds"] = round(time.time() - t0, 2)
    return out


@app.post("/classify")
async def classify(request: Request, file: UploadFile = File(...)):
    rate_limit(request)
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "max upload 5 MB")
    try:
        rgb = backend.preprocess_upload(data)
    except Exception:
        raise HTTPException(400, "not a readable image")
    t0 = time.time()
    try:
        out = backend.classify(rgb, with_cam=True)
    except FileNotFoundError as e:
        raise HTTPException(503, f"classifiers not available: {e}")
    out["seconds"] = round(time.time() - t0, 2)
    return out


@app.get("/bank")
def bank(grade: int | None = None):
    items = [p.name for p in backend.bank if grade is None or p.name.startswith(f"g{grade}_")]
    return {"grade": grade, "files": items, "base_url": "/static/bank/"}

"""Inference backend for the demo API. CPU-friendly; no FastAPI imports here.

Assets (DEMO_ASSETS dir, produced by api/export_assets.py, optionally pulled from HF Hub):
  G_ema_f100.pkl        conditional StyleGAN2 generator (D stripped)
  effnet_c5_f100.pt     strong-aug baseline classifier
  effnet_c6_f100.pt     strong-aug + GAN classifier
  nn_index.npz          feats (float16, N x 1280), ids (str) of the f100 real train set
  nn_thumbs.zip         {image_id}.jpg 96px thumbnails (public mode never serves more than this)
  bank/g{grade}_s{seed}.png   pre-generated, watermarked samples (fallback / gallery)
"""
import io
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import src.gan  # noqa: F401  (stylegan3 on sys.path)
from src.analysis.gradcam import cam_overlay
from src.classify.data import MEAN, STD
from src.classify.model import create_model, load_weights
from src.data.fundus import crop_fundus
from src.gan.watermark import add_visible_label
from src.genai.paths import DATA_ROOT

ASSETS = Path(os.environ.get("DEMO_ASSETS", DATA_ROOT / "demo_assets"))
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
GRADE_NAMES = ["0 no DR", "1 mild", "2 moderate", "3 severe", "4 proliferative"]


def ensure_assets(assets: Path = ASSETS) -> Path:
    """Download from HF Hub if the folder is missing/empty and HF_MODEL_REPO is set."""
    if any(assets.glob("*")) or not HF_MODEL_REPO:
        return assets
    from huggingface_hub import snapshot_download
    snapshot_download(HF_MODEL_REPO, local_dir=str(assets), token=os.environ.get("HF_TOKEN"))
    return assets


def load_G_ema(path: Path):
    import legacy  # stylegan3
    with open(path, "rb") as f:
        data = legacy._LegacyUnpickler(f).load()
    G = data["G_ema"] if isinstance(data, dict) else data
    return G.eval().requires_grad_(False)


def to_b64_png(img: Image.Image) -> str:
    import base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Backend:
    def __init__(self, assets: Path = ASSETS, device: str | None = None):
        self.assets = ensure_assets(assets)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.G = self.models = self.nn_feats = self.nn_ids = None
        self.centers: dict[int, torch.Tensor] = {}  # per-grade truncation centres, as for the training images
        self.bank = sorted((self.assets / "bank").glob("*.png")) if (self.assets / "bank").exists() else []

    # ---- lazy loaders -------------------------------------------------------------------
    def _g(self):
        if self.G is None:
            self.G = load_G_ema(self.assets / "G_ema_f100.pkl").to(self.device)
        return self.G

    def _models(self):
        if self.models is None:
            self.models = {}
            for key, fn in (("baseline", "effnet_c5_f100.pt"), ("gan", "effnet_c6_f100.pt")):
                self.models[key] = load_weights(create_model(pretrained=False), self.assets / fn, self.device)
        return self.models

    def _index(self):
        if self.nn_feats is None:
            z = np.load(self.assets / "nn_index.npz", allow_pickle=False)
            f = z["feats"].astype(np.float32)
            self.nn_feats = f / np.linalg.norm(f, axis=1, keepdims=True)
            self.nn_ids = z["ids"].astype(str)
        return self.nn_feats, self.nn_ids

    def status(self) -> dict:
        return {"device": str(self.device), "assets_dir": str(self.assets),
                "assets": {p.name: p.exists() for p in [self.assets / n for n in
                           ("G_ema_f100.pkl", "effnet_c5_f100.pt", "effnet_c6_f100.pt", "nn_index.npz", "nn_thumbs.zip")]},
                "bank": len(self.bank), "loaded": {"G": self.G is not None, "classifiers": self.models is not None}}

    # ---- features --------------------------------------------------------------------------
    @torch.no_grad()
    def embed(self, rgb: np.ndarray) -> np.ndarray:
        m = self._models()["baseline"]
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
        x = ((x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None])[None].to(self.device)
        f = m.forward_head(m.forward_features(x), pre_logits=True)[0].cpu().numpy()
        return f / (np.linalg.norm(f) + 1e-8)

    # ---- public ops ------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, grade: int, seed: int, psi: float = 0.8) -> Image.Image:
        from src.gan.generate import class_w_avg
        G = self._g()
        if grade not in self.centers:
            self.centers[grade] = class_w_avg(G, grade, self.device)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(self.device)
        c = torch.zeros([1, G.c_dim], device=self.device)
        c[0, grade] = 1
        ws = G.mapping(z, c, truncation_psi=1)
        center = self.centers[grade]
        img = G.synthesis(center + psi * (ws - center), noise_mode="const", force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        return Image.fromarray(img, "RGB")

    def nearest_real(self, rgb: np.ndarray, thumb_px: int = 96) -> dict:
        feats, ids = self._index()
        q = self.embed(rgb)
        d = 1 - feats @ q
        j = int(d.argmin())
        thumb = None
        zpath = self.assets / "nn_thumbs.zip"
        if zpath.exists():
            with zipfile.ZipFile(zpath) as z:
                name = f"{ids[j]}.jpg"
                if name in z.namelist():
                    im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                    im.thumbnail((thumb_px, thumb_px))
                    thumb = to_b64_png(im)
        return {"image_id": str(ids[j]), "distance": float(d[j]), "thumb_png_b64": thumb}

    def preprocess_upload(self, data: bytes) -> np.ndarray:
        im = Image.open(io.BytesIO(data))
        im.verify()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        return crop_fundus(np.asarray(im))

    def classify(self, rgb: np.ndarray, with_cam: bool = True) -> dict:
        out = {"probs": {}, "pred": {}, "gradcam_png_b64": {}}
        for key, m in self._models().items():
            if with_cam:
                overlay, probs = cam_overlay(m, rgb)
                out["gradcam_png_b64"][key] = to_b64_png(Image.fromarray(overlay))
            else:
                with torch.no_grad():
                    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
                    x = ((x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None])[None].to(self.device)
                    probs = m(x).softmax(1)[0].cpu().numpy()
            out["probs"][key] = [round(float(p), 4) for p in probs]
            out["pred"][key] = int(np.argmax(probs))
        out["preprocessed_png_b64"] = to_b64_png(Image.fromarray(rgb))
        out["grade_names"] = GRADE_NAMES
        return out

    def watermarked(self, img: Image.Image) -> Image.Image:
        return add_visible_label(img)

"""Shared fundus preprocessing: crop the retina disc, pad to square, resize.

Used by the data pipeline (APTOS + IDRiD) and by the demo API for uploads, so
training and inference see the same geometry.
"""
from pathlib import Path

import cv2
import numpy as np

from src.genai.paths import IMG_SIZE


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_fundus(rgb: np.ndarray, size: int = IMG_SIZE, tol: int = 10, min_frac: float = 0.01) -> np.ndarray:
    """Bounding box of rows/cols where >= min_frac of pixels are brighter than tol,
    centred on a black square canvas, resized with INTER_AREA. Robust to corner noise."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows = np.where(mask.mean(axis=1) > min_frac)[0]
    cols = np.where(mask.mean(axis=0) > min_frac)[0]
    if len(rows) and len(cols):
        rgb = rgb[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def save_png(rgb: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"could not write {path}")

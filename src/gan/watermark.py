"""Synthetic-image labelling.

Every generated PNG carries text metadata (Synthetic=true, generator, grade, seed, psi).
A VISIBLE "SYNTHETIC" corner label is added only to images that are displayed or exported
(demo app, report figures), never to the copies used for classifier training, because a
visible mark would be a trivial shortcut feature.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


def png_info(**meta) -> PngImagePlugin.PngInfo:
    info = PngImagePlugin.PngInfo()
    info.add_text("Synthetic", "true")
    for k, v in meta.items():
        info.add_text(str(k), str(v))
    return info


def add_visible_label(img: Image.Image, text: str = "SYNTHETIC") -> Image.Image:
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    size = max(10, out.width // 18)
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    w, h = x1 - x0, y1 - y0
    pad = size // 3
    box = (out.width - w - 3 * pad, out.height - h - 3 * pad, out.width - pad, out.height - pad)
    draw.rectangle(box, fill=(0, 0, 0, 170))
    draw.text((box[0] + pad, box[1] + pad - y0), text, fill=(255, 220, 0, 255), font=font)
    return Image.alpha_composite(out, overlay).convert("RGB")


def save_synthetic(img: Image.Image, path: str | Path, visible: bool = False, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    (add_visible_label(img) if visible else img).save(path, pnginfo=png_info(**meta))


def is_synthetic(path: str | Path) -> bool:
    with Image.open(path) as im:
        return im.text.get("Synthetic", "").lower() == "true"

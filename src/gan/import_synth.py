"""S9: import the Kaggle eval notebook's synth_*.zip files into <GENAI_DATA>/synth/.

  kaggle kernels output <user>/<eval-notebook> -p downloads/
  python -m src.gan.import_synth downloads/synth_f*.zip

synth_f<ppp>-ada.zip       -> synth/f<ppp>/g{3,4}/       psi 0.8: the condition-6 training images
synth_f<ppp>-ada_psi1.zip  -> synth/f<ppp>_psi1/g{3,4}/  psi 1.0 (untruncated), same seeds: the pre-registered psi
                                                        check (S10a), GAN-test and memorisation; condition 6 trains
                                                        on them only if that check adopts psi 1 (synth_variant)
The no-ADA ablation (synth_f010-noaug*.zip) is skipped: it is evaluated with FID/KID only and never
becomes classifier training data. Every PNG is verified before anything is written: 256x256 RGB,
Synthetic=true, grade metadata equal to its g3/g4 folder, psi 1.0 exactly for the _psi1 sets. A file
that already exists with other bytes stops the import (images of another generation run).
"""
import glob
import io
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

import click
from PIL import Image

from src.genai.paths import IMG_SIZE, SYNTH

ZIP_RE = re.compile(r"synth_(f\d{3})(?:-ada)?(_psi1)?")
PNG_RE = re.compile(r"(?:^|/)(g[34])/[^/]+\.png$")


def check_png(name: str, data: bytes, grade: str, psi1: bool, frac: str) -> None:
    with Image.open(io.BytesIO(data)) as im:
        size, mode, meta = im.size, im.mode, dict(im.text)
    psi, seed = meta.get("psi"), re.search(r"_s(\d+)\.png$", name)
    for bad, msg in ((size != (IMG_SIZE, IMG_SIZE) or mode != "RGB", f"{size} {mode}, expected {IMG_SIZE}x{IMG_SIZE} RGB"),
                     (meta.get("Synthetic") != "true", "lacks Synthetic=true metadata"),
                     (meta.get("grade") != grade[1:], f"grade metadata {meta.get('grade')} but folder {grade}"),
                     (meta.get("generator") != f"{frac}-ada", f"generator {meta.get('generator')!r}, expected {frac}-ada"),
                     (seed is None or meta.get("seed") != str(int(seed.group(1))), f"seed {meta.get('seed')!r} vs file name"),
                     (not psi1 and meta.get("trunc_center") != "class",
                      f"trunc_center {meta.get('trunc_center')!r}: the study samples with conditional truncation (class)"),
                     (psi is None or (float(psi) == 1.0) != psi1,
                      f"psi {psi}: {'a _psi1 set must have psi 1.0' if psi1 else 'psi 1.0 images belong in synth_<tag>_psi1.zip'}")):
        if bad:
            raise SystemExit(f"{name}: {msg}")


def import_zip(zpath: Path) -> dict[str, int]:
    if "-noaug" in zpath.stem:
        click.echo(f"skip {zpath.name}: no-ADA ablation (FID/KID only, never classifier training data)")
        return {}
    m = ZIP_RE.fullmatch(zpath.stem)
    if not m:
        raise SystemExit(f"cannot infer the set from {zpath.name}; expected synth_f010-ada.zip or synth_f010-ada_psi1.zip")
    tag, psi1 = m.group(1) + (m.group(2) or ""), bool(m.group(2))
    dest = SYNTH / tag
    with zipfile.ZipFile(zpath) as z:
        pngs = [(n, g.group(1)) for n in z.namelist() if (g := PNG_RE.search(n.replace("\\", "/")))]
        if not pngs:
            raise SystemExit(f"{zpath.name} holds no g3/ or g4/ PNGs")
        for name, g in pngs:
            data = z.read(name)
            check_png(f"{zpath.name}:{name}", data, g, psi1, m.group(1))
            out = dest / g / Path(name).name
            if out.exists() and out.read_bytes() != data:
                raise SystemExit(f"{out} differs from {zpath.name}:{name}: {dest} holds images of another generation run. "
                                 f"Move {dest} away to import these; runs already trained on {dest} used the old ones")
        for name, g in pngs:
            out = dest / g / Path(name).name
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.name + ".tmp")
                tmp.write_bytes(z.read(name))
                os.replace(tmp, out)
    counts = dict(sorted(Counter(g for _, g in pngs).items()))
    click.echo(f"{zpath.name} -> {dest}: {counts}")
    return counts


@click.command()
@click.argument("zips", nargs=-1)
def main(zips):
    """ZIPS: files or patterns like downloads/synth_f*.zip (expanded here: Windows PowerShell passes them literally)."""
    paths = [Path(p) for z in zips for p in (sorted(glob.glob(z)) if any(ch in z for ch in "*?[") else [z])]
    if not paths:
        raise SystemExit(f"no synth_f*.zip files match {list(zips)}")
    if missing := [str(p) for p in paths if not p.exists()]:
        raise SystemExit(f"not found: {missing}")
    for z in paths:
        import_zip(z)


if __name__ == "__main__":
    main()

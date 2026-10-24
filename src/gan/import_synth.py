"""S9: import synth_f{ppp}-ada.zip files produced on Kaggle into DATA_ROOT/synth/f{ppp}/g{3,4}/.

  kaggle kernels output <user>/<eval-notebook> -p downloads/
  python -m src.gan.import_synth downloads/synth_f*.zip

The no-ADA ablation (synth_f010-noaug.zip) is skipped: it is evaluated with FID/KID only and never
becomes classifier training data. Verifies: every PNG is 256x256 RGB with Synthetic=true metadata.
"""
import re
import zipfile
from pathlib import Path

import click
from PIL import Image

from src.genai.paths import IMG_SIZE, SYNTH


def import_zip(zpath: Path) -> dict[str, int]:
    if zpath.stem.endswith("-noaug"):
        click.echo(f"skip {zpath.name}: no-ADA ablation (FID/KID only, never classifier training data)")
        return {}
    m = re.fullmatch(r"synth_(f\d{3})(?:-ada)?", zpath.stem)
    if not m:
        raise SystemExit(f"cannot infer fraction tag from {zpath.name}; expected synth_f010-ada.zip etc.")
    tag = m.group(1)
    dest = SYNTH / tag
    counts: dict[str, int] = {}
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.lower().endswith(".png"):
                continue
            gm = re.search(r"(g[34])/", name.replace("\\", "/"))
            if not gm:
                continue
            out = dest / gm.group(1) / Path(name).name
            data = z.read(name)
            if out.exists() and out.read_bytes() != data:
                raise SystemExit(f"{out} differs from {zpath.name}:{name}: {dest} holds images of another generation run. "
                                 f"Move {dest} away to import these; {tag} condition-6 runs already done used the old ones")
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
            counts[gm.group(1)] = counts.get(gm.group(1), 0) + 1
    for g, n in counts.items():
        sample = sorted((dest / g).glob("*.png"))[:5]
        for p in sample:
            with Image.open(p) as im:
                assert im.size == (IMG_SIZE, IMG_SIZE) and im.mode == "RGB", (p, im.size, im.mode)
                assert im.text.get("Synthetic") == "true", f"{p} lacks Synthetic metadata"
    click.echo(f"{zpath.name} -> {dest}: {counts}")
    return counts


@click.command()
@click.argument("zips", nargs=-1, type=click.Path(exists=True, path_type=Path))
def main(zips):
    if not zips:
        raise SystemExit("give one or more synth_f*.zip files")
    for z in zips:
        import_zip(z)


if __name__ == "__main__":
    main()

"""Stage tracker for the lab-PC runbook.

  python -m src.pipeline status        # DONE / PARTIAL / TODO for every stage, from on-disk evidence
  python -m src.pipeline run S4        # run one stage's command
  python -m src.pipeline cmd S4        # just print the command

Stages map 1:1 to LAB_RUNBOOK.md. Manual stages (S2 IDRiD download, S9 Kaggle fetch) only report.
"""
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

import click
import pandas as pd

from src.genai.paths import (APTOS256, CACHE, CHECKPOINTS, DATA_ROOT, FRACTIONS, IDRID256, RAW_APTOS, RAW_IDRID,
                             RESULTS, SG3, SPLITS, SYNTH, frac_tag)

PY = sys.executable
SPLIT_DIGEST = "017b5a96eb9b"  # committed splits/aptos_splits.csv (LAB_RUNBOOK S4)


def _count(path, pattern) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def _done_runs() -> set[str]:
    f = RESULTS / "classifier_runs.csv"
    if not f.exists():
        return set()
    df = pd.read_csv(f, usecols=["run_id", "dataset"])
    return set(df[df.dataset == "aptos_test"].run_id)


def _grid_ids(cond6: bool) -> set[str]:
    from src.classify.queue import grid
    return {r["run_id"] for r in grid() if (r["condition"] == 6) == cond6}


@dataclass
class Stage:
    id: str
    name: str
    check: Callable[[], tuple[str, str]]  # -> (DONE|PARTIAL|TODO, detail)
    cmd: str | None


def s0():
    try:
        import torch
        ok = torch.cuda.is_available()
        return ("DONE" if ok else "PARTIAL", f"torch {torch.__version__}, cuda={ok}" + ("" if ok else " (CPU only)"))
    except Exception as e:
        return "TODO", f"torch import failed: {e}"


def s1():
    n = _count(RAW_APTOS / "train_images", "*.png")
    return ("DONE" if n == 3662 and (RAW_APTOS / "train.csv").exists() else "PARTIAL" if n else "TODO", f"{n}/3662 raw PNGs")


def s2():
    n = len([p for p in RAW_IDRID.rglob("IDRiD_*.jpg")]) if RAW_IDRID.exists() else 0
    z = _count(RAW_IDRID, "*.zip")
    return ("DONE" if n >= 516 else "PARTIAL" if (n or z) else "TODO", f"{n} IDRiD jpgs, {z} zips in {RAW_IDRID}")


def s3():
    a, i = _count(APTOS256, "*.png"), _count(IDRID256, "*.png")
    st = "DONE" if a == 3662 and i >= 516 else "PARTIAL" if (a or i) else "TODO"
    return st, f"aptos256 {a}/3662, idrid256 {i}/516"


def s4():
    from src.data.split import digest
    sp = SPLITS / "aptos_splits.csv"
    if not sp.exists():
        return "TODO", "no split CSVs"
    dg, lab, idr = digest(pd.read_csv(sp)), (RAW_APTOS / "train.csv").exists(), (SPLITS / "idrid_test.csv").exists()
    ok = dg == SPLIT_DIGEST
    st = "DONE" if ok and lab and idr else "PARTIAL"
    return st, f"digest {dg} ({'ok' if ok else 'expected ' + SPLIT_DIGEST}), train.csv={lab}, idrid_test.csv={idr}"


def s5():
    a, i = (CACHE / "aptos256.npy").exists(), (CACHE / "idrid256.npy").exists()
    return ("DONE" if a and i else "PARTIAL" if a else "TODO"), f"aptos256.npy={a} idrid256.npy={i}"


def s6():
    have = [p for p in FRACTIONS if (SG3 / f"aptos256_{frac_tag(p)}.zip").exists()]
    val = (SG3 / "aptos256_val.zip").exists()
    if not have and not val:
        return "SKIP", "zips are built on Kaggle by genai-data-prep (only needed for lab-PC GAN training)"
    return ("DONE" if len(have) == 4 and val else "PARTIAL"), f"{len(have)}/4 fraction zips, val={val}"


def s7():
    m = DATA_ROOT / "kaggle_dataset" / ".uploaded"
    return ("DONE", m.read_text().splitlines()[0]) if m.exists() else ("SKIP", "GAN notebooks read genai-data-prep's output instead")


def s8():
    done = _done_runs() & _grid_ids(False)
    return ("DONE" if len(done) == 60 else "PARTIAL" if done else "TODO"), f"{len(done)}/60 baseline runs"


def s9():
    have = {frac_tag(p): (_count(SYNTH / frac_tag(p) / "g3", "*.png"), _count(SYNTH / frac_tag(p) / "g4", "*.png")) for p in FRACTIONS}
    ok = [t for t, (a, b) in have.items() if a and b]
    return ("DONE" if len(ok) == 4 else "PARTIAL" if ok else "TODO"), " ".join(f"{t}:{a}/{b}" for t, (a, b) in have.items())


def s10():
    done = _done_runs() & _grid_ids(True)
    return ("DONE" if len(done) == 24 else "PARTIAL" if done else "TODO"), f"{len(done)}/24 GAN-condition runs"


def s11():
    t = (RESULTS / "tables" / "summary_long.csv").exists()
    b = (RESULTS / "tables" / "bootstrap_ci.csv").exists()
    f = _count(RESULTS / "figures", "learning_curve_*.png")
    return ("DONE" if t and b and f else "PARTIAL" if t or f else "TODO"), f"tables={t} bootstrap={b} figures={f}"


def s12():
    p = RESULTS / "tables" / "memorization_summary.csv"
    return ("DONE" if p.exists() else "TODO"), str(p.name)


def s13():
    p = RESULTS / "tables" / "rps_summary.csv"
    return ("DONE" if p.exists() else "TODO"), str(p.name)


def s14():
    p = RESULTS / "figures" / "gradcam_examples.png"
    return ("DONE" if p.exists() else "TODO"), str(p.name)


def s15():
    d = DATA_ROOT / "demo_assets"
    need = ["G_ema_f100.pkl", "effnet_c5_f100.pt", "effnet_c6_f100.pt", "nn_index.npz"]
    have = [n for n in need if (d / n).exists()]
    return ("DONE" if len(have) == len(need) else "PARTIAL" if have else "TODO"), f"{len(have)}/{len(need)} demo assets"


STAGES = [
    Stage("S0", "environment (torch + CUDA)", s0, f'"{PY}" -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'cpu\')"'),
    Stage("S1", "download APTOS", s1, f'"{PY}" -m src.data.download_aptos'),
    Stage("S2", "place IDRiD zip (manual, IEEE DataPort)", s2, None),
    Stage("S3", "crop/resize to 256px", s3, f'"{PY}" -m src.data.prepare_images --source both'),
    Stage("S4", "verify committed split vs train.csv", s4, f'"{PY}" -m src.data.split'),
    Stage("S5", "uint8 cache arrays", s5, f'"{PY}" -m src.data.cache'),
    Stage("S6", "stylegan3 zips (optional)", s6, f'"{PY}" -m src.data.make_sg3_zips'),
    Stage("S7", "upload private Kaggle dataset (optional)", s7, f'"{PY}" -m src.data.kaggle_dataset'),
    Stage("S8", "classifier runs, conditions 1-5 (60)", s8, f'"{PY}" -m src.classify.queue'),
    Stage("S9", "import synthetic images from Kaggle (manual fetch)", s9, f'"{PY}" -m src.gan.import_synth <downloads/synth_f*.zip>'),
    Stage("S10", "classifier runs, condition 6 (24)", s10, f'"{PY}" -m src.classify.queue --condition 6'),
    Stage("S11", "aggregate + bootstrap + plots", s11, f'"{PY}" -m src.analysis.aggregate && "{PY}" -m src.analysis.bootstrap && "{PY}" -m src.analysis.plots'),
    Stage("S12", "memorisation NN check", s12, f'"{PY}" -m src.analysis.memorization'),
    Stage("S13", "RPS merge (after docker run)", s13, f'"{PY}" -m src.analysis.rps merge'),
    Stage("S14", "Grad-CAM figure", s14, f'"{PY}" -m src.analysis.gradcam'),
    Stage("S15", "export demo assets", s15,
          f'"{PY}" -m api.export_assets --pkl <GENAI_DATA>/gan_pkls/f100-ada.pkl --upload <hf-user>/dr-gan-demo'),
]


@click.group()
def cli():
    pass


@cli.command()
def status():
    click.echo(f"DATA_ROOT = {DATA_ROOT}   checkpoints = {CHECKPOINTS}")
    for s in STAGES:
        try:
            st, detail = s.check()
        except Exception as e:
            st, detail = "ERR", f"{type(e).__name__}: {e}"
        click.echo(f"{s.id:>4} {st:<8} {s.name:<48} {detail}")


@cli.command()
@click.argument("stage_id")
def cmd(stage_id):
    s = next(x for x in STAGES if x.id == stage_id.upper())
    click.echo(s.cmd or "(manual stage)")


@cli.command()
@click.argument("stage_id")
def run(stage_id):
    s = next(x for x in STAGES if x.id == stage_id.upper())
    if not s.cmd or "<" in s.cmd:
        raise SystemExit(f"{s.id} needs manual arguments: {s.cmd}")
    click.echo(s.cmd)
    raise SystemExit(subprocess.run(s.cmd, shell=True).returncode)


if __name__ == "__main__":
    cli()

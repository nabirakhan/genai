"""Single source of truth for on-disk locations.

Data lives outside the repo under GENAI_DATA (env var). Defaults:
  Windows: C:\genai-data   (lab PC: fast SSD)   -- laptop: set GENAI_DATA=E:\genai-data
  Linux/Kaggle: ~/genai-data
Everything under DATA_ROOT is ignored by git. Only splits/ and results/ (small CSVs) are tracked.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_default = r"C:\genai-data" if os.name == "nt" else str(Path.home() / "genai-data")
DATA_ROOT = Path(os.environ.get("GENAI_DATA", _default))

RAW_APTOS = DATA_ROOT / "raw" / "aptos"
RAW_IDRID = DATA_ROOT / "raw" / "idrid"
APTOS256 = DATA_ROOT / "aptos256"          # cropped 256px PNGs, one per APTOS train image
IDRID256 = DATA_ROOT / "idrid256"          # cropped 256px PNGs, all 516 IDRiD grading images
CACHE = DATA_ROOT / "cache"                # uint8 .npy arrays for the classifier
SG3 = DATA_ROOT / "sg3"                    # stylegan3 dataset zips (per fraction + val)
SYNTH = DATA_ROOT / "synth"                # synth/f{ppp}/g{3,4}/*.png imported from Kaggle
GAN_PKLS = DATA_ROOT / "gan_pkls"          # final G pickles per fraction (+ FFHQ source)

# Tracked outputs live in the repo; GENAI_SPLITS / GENAI_RESULTS / GENAI_CHECKPOINTS override them
# (the smoke test points them at a temp dir so it never touches the real CSVs).
CHECKPOINTS = Path(os.environ.get("GENAI_CHECKPOINTS", REPO_ROOT / "checkpoints"))  # ignored by git
SPLITS = Path(os.environ.get("GENAI_SPLITS", REPO_ROOT / "splits"))                  # tracked
RESULTS = Path(os.environ.get("GENAI_RESULTS", REPO_ROOT / "results"))               # tracked
WEB_RESULTS = REPO_ROOT / "web" / "public" / "results"                               # static copies for the Results page
STYLEGAN3 = REPO_ROOT / "third_party" / "stylegan3"

IMG_SIZE = 256
GRADES = (0, 1, 2, 3, 4)
RARE_GRADES = (3, 4)
FRACTIONS = (0.10, 0.25, 0.50, 1.00)
SPLIT_SEED = 42


def frac_tag(p: float) -> str:
    """0.1 -> 'f010', 1.0 -> 'f100'."""
    return f"f{int(round(p * 100)):03d}"


def ensure_dirs() -> None:
    for d in (RAW_APTOS, RAW_IDRID, APTOS256, IDRID256, CACHE, SG3, SYNTH, GAN_PKLS,
              CHECKPOINTS, SPLITS, RESULTS / "preds", RESULTS / "tables", RESULTS / "figures"):
        d.mkdir(parents=True, exist_ok=True)

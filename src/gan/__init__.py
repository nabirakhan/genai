"""Makes the pinned stylegan3 submodule importable (dnnlib, legacy, torch_utils, training)."""
import sys

from src.genai.paths import STYLEGAN3

_p = str(STYLEGAN3)
if _p not in sys.path:
    sys.path.insert(0, _p)

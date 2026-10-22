"""Apply third_party/patches/*.patch to the stylegan3 submodule. Idempotent.

  python scripts/apply_patches.py          # apply what is missing
  python scripts/apply_patches.py --check  # report only
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG3 = ROOT / "third_party" / "stylegan3"
PATCHES = sorted((ROOT / "third_party" / "patches").glob("*.patch"))


def git(*args, check=False):
    return subprocess.run(["git", "-C", str(SG3), *args], capture_output=True, text=True, check=check)


def main() -> int:
    check_only = "--check" in sys.argv
    if not (SG3 / "train.py").exists():
        print("stylegan3 submodule missing: run `git submodule update --init --recursive`")
        return 1
    rc = 0
    for p in PATCHES:
        if git("apply", "--check", "--reverse", str(p)).returncode == 0:
            print(f"[applied] {p.name}")
            continue
        if git("apply", "--check", str(p)).returncode != 0:
            print(f"[CONFLICT] {p.name} applies neither forward nor reverse; submodule commit changed?")
            rc = 1
            continue
        if check_only:
            print(f"[pending] {p.name}")
            rc = 1
        else:
            git("apply", str(p), check=True)
            print(f"[applied now] {p.name}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

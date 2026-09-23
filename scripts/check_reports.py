"""Check that the committed reports reproduce, after `ltv validate` has regenerated them.

Text is compared byte for byte. Charts are compared **pixel for pixel**, not byte for byte, and
the difference matters. Pillow's Windows wheel bundles zlib-ng, and the Linux wheel bundles stock
zlib. The same pixel stream therefore compresses to different PNG bytes on the two platforms, even
at identical matplotlib and Pillow versions. A byte diff failed CI on charts whose decoded pixels
were identical. So "reproducible" is tested at the level the claim is actually about: what the
chart shows. The tolerance is still zero. One changed pixel fails, and so does the halved decile
bar left behind by a mutation run.

Run after `ltv validate`:

    uv run python scripts/check_reports.py
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = "reports/"


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True).stdout


def decoded(png: bytes) -> tuple[str, tuple[int, int], bytes]:
    """What a PNG shows, independent of how it was compressed."""
    with Image.open(io.BytesIO(png)) as image:
        return image.mode, image.size, image.tobytes()


def same_picture(committed: bytes, regenerated: bytes) -> bool:
    return decoded(committed) == decoded(regenerated)


def problems() -> list[str]:
    found = []

    # A report the code now writes but nobody committed is as much a reproducibility failure as a
    # changed one: the committed set no longer describes what the pipeline produces.
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", REPORTS).decode().split()
    found.extend(f"{path}: generated but not committed" for path in untracked)

    changed = _git("diff", "--name-only", "--", REPORTS).decode().split()
    for path in changed:
        current = REPO_ROOT / path
        if not current.exists():
            found.append(f"{path}: committed but no longer generated")
            continue
        if path.endswith(".png") and same_picture(
            _git("show", f"HEAD:{path}"), current.read_bytes()
        ):
            print(f"{path}: identical pixels, different compression (ignored)")
            continue
        found.append(f"{path}: differs from the committed version")
    return found


def main() -> int:
    found = problems()
    for problem in found:
        print(f"NOT REPRODUCIBLE  {problem}")
    if found:
        # The text diff shows what moved. For a chart, open both copies: a pixel diff has no useful
        # text form. Flush first, or git's output overtakes ours in a piped CI log.
        sys.stdout.flush()
        subprocess.run(["git", "--no-pager", "diff", "--stat", "--", REPORTS], cwd=REPO_ROOT)
        return 1
    print("reports reproduce")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The report reproducibility check must ignore compression and nothing else.

Both directions are tested. A check that only proved re-encoding passes could be satisfied by one
that compares nothing at all.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

from PIL import Image

_SPEC = importlib.util.spec_from_file_location(
    "check_reports", Path(__file__).resolve().parents[1] / "scripts" / "check_reports.py"
)
check_reports = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_reports)


def _png(image: Image.Image, compress_level: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=compress_level)
    return buffer.getvalue()


def _chart() -> Image.Image:
    # Enough structure that different compression levels really do produce different bytes.
    image = Image.new("RGBA", (64, 32), "white")
    for x in range(64):
        image.putpixel((x, x % 32), (17, 17, 17, 255))
    return image


def test_same_pixels_compressed_differently_are_the_same_picture():
    fast, small = _png(_chart(), 1), _png(_chart(), 9)
    assert fast != small, "premise: the two encodings must differ as bytes"
    assert check_reports.same_picture(fast, small)


def test_one_changed_pixel_is_a_different_picture():
    changed = _chart()
    changed.putpixel((0, 31), (255, 0, 0, 255))
    assert not check_reports.same_picture(_png(_chart(), 6), _png(changed, 6))


def test_a_resized_chart_is_a_different_picture():
    assert not check_reports.same_picture(_png(_chart(), 6), _png(_chart().resize((65, 32)), 6))

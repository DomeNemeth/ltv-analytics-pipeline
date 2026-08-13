"""Shared fixtures for building synthetic CDNOW source files."""

from __future__ import annotations

from pathlib import Path


def cdnow_line(
    customer: int = 1, date: str = "19970101", qty: int = 1, amount: float = 11.77
) -> str:
    """Build one correctly formatted 26-character CDNOW record."""
    return f" {customer:05d} {date} {qty:2d} {amount:7.2f}"


def write_lines(path: Path, lines: list[str]) -> Path:
    # Per-line join, not "\n".join(...) + "\n" -- the latter turns an empty list into a file
    # containing one blank line, which is a different failure mode than an empty file.
    path.write_text("".join(f"{line}\n" for line in lines), encoding="ascii")
    return path

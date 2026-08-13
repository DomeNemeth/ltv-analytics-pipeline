"""The seam between the CLI flag and the fetcher.

Both ends of `--force-download` are tested elsewhere; this covers the join between them. That join
has already silently broken once, and a test at each end cannot see it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import cdnow_line, write_lines
from ltv.config import Settings
from ltv.ingest import cdnow as cdnow_module
from ltv.ingest.cdnow import ingest_cdnow


@pytest.fixture
def tiny_source(tmp_path: Path) -> Path:
    return write_lines(tmp_path / "CDNOW_master.txt", [cdnow_line(), cdnow_line(customer=2)])


@pytest.mark.parametrize("requested", [True, False])
def test_force_download_reaches_the_fetcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_source: Path, requested: bool
) -> None:
    seen: dict[str, object] = {}

    def fake_ensure(archive: object, raw_dir: Path, *, force: bool = False) -> Path:
        seen["force"] = force
        return tiny_source

    monkeypatch.setattr(cdnow_module, "ensure_local_copy", fake_ensure)

    ingest_cdnow(Settings(repo_root=tmp_path), force_download=requested)

    assert seen["force"] is requested


def test_explicit_source_path_skips_the_fetcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_source: Path
) -> None:
    """Passing a file directly must not reach out to the network at all."""

    def explode(*args: object, **kwargs: object) -> Path:
        raise AssertionError("ensure_local_copy must not be called when source_path is given")

    monkeypatch.setattr(cdnow_module, "ensure_local_copy", explode)

    result = ingest_cdnow(Settings(repo_root=tmp_path), source_path=tiny_source)

    assert result.rows == 2
    assert result.customers == 2

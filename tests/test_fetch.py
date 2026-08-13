"""Checksum, caching, and extraction behaviour for source downloads.

Fully hermetic: the network layer is monkeypatched, so these run offline and deterministically. That
matters because the branches here are the ones that decide whether a bad or tampered file gets
silently accepted, and a test that needs the internet to prove that is a test that gets skipped.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from ltv.ingest import fetch
from ltv.ingest.fetch import RemoteArchive, SourceDataError, ensure_local_copy, sha256_of

PAYLOAD = b"customer,amount\n1,11.77\n"
MEMBER = "fixture.txt"


def archive_for(payload: bytes = PAYLOAD, member: str = MEMBER) -> RemoteArchive:
    return RemoteArchive(
        name="fixture",
        # Deliberately unreachable. Any test that reaches the real network instead of the patched
        # downloader fails loudly rather than quietly fetching something.
        url="https://example.invalid/fixture.zip",
        member=member,
        sha256=hashlib.sha256(payload).hexdigest(),
        citation="test fixture",
    )


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Replace the network layer with one that writes a chosen archive to disk.

    Returns a list that records each URL "downloaded", so tests can assert that a cache hit did not
    perform a download.
    """
    calls: list[str] = []

    def install(*, member: str = MEMBER, payload: bytes = PAYLOAD, valid_zip: bool = True) -> None:
        def fake_download(url: str, destination: Path, name: str) -> None:
            calls.append(url)
            if not valid_zip:
                destination.write_bytes(b"<html>404 not found</html>")
                return
            with zipfile.ZipFile(destination, "w") as bundle:
                bundle.writestr(member, payload)

        monkeypatch.setattr(fetch, "_download", fake_download)

    install.calls = calls  # type: ignore[attr-defined]
    return install


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(PAYLOAD)

    assert sha256_of(target) == hashlib.sha256(PAYLOAD).hexdigest()


def test_downloads_extracts_and_caches(tmp_path: Path, serve) -> None:
    serve()

    result = ensure_local_copy(archive_for(), tmp_path)

    assert result == tmp_path / MEMBER
    assert result.read_bytes() == PAYLOAD
    assert serve.calls == ["https://example.invalid/fixture.zip"]


def test_no_partial_file_is_left_behind(tmp_path: Path, serve) -> None:
    """The write is staged then renamed; a leftover .partial would be rejected as corrupt later."""
    serve()

    ensure_local_copy(archive_for(), tmp_path)

    assert [p.name for p in tmp_path.iterdir()] == [MEMBER]


def test_a_failed_publish_leaves_no_usable_file(
    tmp_path: Path, serve, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the atomic rename itself.

    Asserting only that no `.partial` survives is satisfied by code that never staged one. Here the
    publish step fails outright: the cache must end up with no file at that path, so the next run
    re-downloads instead of finding a half-written one and refusing to proceed.
    """
    serve()

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(fetch.os, "replace", refuse)

    with pytest.raises(OSError, match="no space left"):
        ensure_local_copy(archive_for(), tmp_path)

    assert not (tmp_path / MEMBER).exists()


def test_valid_cached_copy_is_reused_without_downloading(tmp_path: Path, serve) -> None:
    serve()
    (tmp_path / MEMBER).write_bytes(PAYLOAD)

    ensure_local_copy(archive_for(), tmp_path)

    assert serve.calls == [], "a checksum-valid cached file must not trigger a download"


def test_force_redownloads_even_when_cache_is_valid(tmp_path: Path, serve) -> None:
    serve()
    (tmp_path / MEMBER).write_bytes(PAYLOAD)

    ensure_local_copy(archive_for(), tmp_path, force=True)

    assert len(serve.calls) == 1


def test_corrupt_cached_copy_is_rejected_not_silently_reused(tmp_path: Path, serve) -> None:
    serve()
    (tmp_path / MEMBER).write_bytes(b"tampered")

    with pytest.raises(SourceDataError, match="does not match its expected checksum"):
        ensure_local_copy(archive_for(), tmp_path)


def test_freshly_downloaded_file_failing_the_pin_is_rejected(tmp_path: Path, serve) -> None:
    """The supply-chain pin. If upstream content changes, the load must stop, not adapt."""
    serve(payload=b"something else entirely")

    with pytest.raises(SourceDataError, match="Checksum mismatch"):
        ensure_local_copy(archive_for(), tmp_path)

    assert not (tmp_path / MEMBER).exists(), "a file failing verification must not be cached"


def test_missing_member_names_what_the_archive_actually_contains(tmp_path: Path, serve) -> None:
    serve(member="unexpected_name.txt")

    with pytest.raises(SourceDataError, match="does not contain 'fixture.txt'"):
        ensure_local_copy(archive_for(), tmp_path)


def test_non_zip_response_is_reported_as_such(tmp_path: Path, serve) -> None:
    """Servers answer with an HTML error page far more often than they answer with a broken zip."""
    serve(valid_zip=False)

    with pytest.raises(SourceDataError, match="not a valid zip archive"):
        ensure_local_copy(archive_for(), tmp_path)


def test_network_failure_gives_an_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(fetch.httpx, "stream", refuse)

    with pytest.raises(SourceDataError, match="Could not download"):
        ensure_local_copy(archive_for(), tmp_path)

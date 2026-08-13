"""Download and verify third-party source archives.

Source data is never committed to this repository. It belongs to its authors, and a public repo
that redistributes someone else's dataset without an explicit licence grant is a fair thing for a
reviewer to object to. Instead each archive is fetched on demand and pinned by checksum.

The checksum pins the **extracted member**, not the zip container. Zip files re-compress to
different bytes with identical contents (timestamps, compression level, archiver version), so
hashing the container produces false alarms on a file that has not actually changed.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import truststore

_DOWNLOAD_TIMEOUT_SECONDS = 60.0
_CHUNK_BYTES = 1 << 20

#: brucehardie.com rejects the default `python-httpx` agent with a non-standard HTTP 465. We
#: identify the project honestly rather than impersonating a browser: a host that wants to block or
#: contact automated clients should be able to, and this URL tells them who we are.
_USER_AGENT = "ltv-analytics-pipeline/0.1.0 (+https://github.com/DomeNemeth/ltv-analytics-pipeline)"


class SourceDataError(RuntimeError):
    """Raised when source data cannot be fetched, extracted, or verified."""


@dataclass(frozen=True)
class RemoteArchive:
    """A zipped source dataset pinned by the checksum of one member file."""

    name: str
    url: str
    member: str
    sha256: str
    citation: str


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks so large files stay off the heap."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_local_copy(archive: RemoteArchive, raw_dir: Path, *, force: bool = False) -> Path:
    """Return a local, checksum-verified copy of ``archive.member``, downloading it if needed.

    A cached file whose checksum still matches is reused, so repeated pipeline runs do not hammer a
    personal academic web server.

    Args:
        archive: The pinned remote archive to materialise.
        raw_dir: Directory to cache the extracted member in.
        force: Re-download even if a valid cached copy exists.

    Raises:
        SourceDataError: On network failure, a missing member, or a checksum mismatch.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / archive.member

    if target.exists() and not force:
        actual = sha256_of(target)
        if actual == archive.sha256:
            return target
        raise SourceDataError(
            f"Cached {archive.member} in {raw_dir} does not match its expected checksum "
            f"(expected {archive.sha256}, got {actual}). The file is corrupt or was edited. "
            f"Delete it and re-run to download a fresh copy."
        )

    with TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / f"{archive.name}.zip"
        _download(archive.url, archive_path, archive.name)
        extracted = _extract_member(archive_path, archive.member, Path(tmp))

        actual = sha256_of(extracted)
        if actual != archive.sha256:
            raise SourceDataError(
                f"Checksum mismatch for {archive.member} downloaded from {archive.url}.\n"
                f"  expected {archive.sha256}\n"
                f"  actual   {actual}\n"
                f"The upstream file has changed. Verify the new file is what you expect, then "
                f"update the pinned checksum in the archive definition. Do not silently accept it."
            )

        # Publish atomically. A crash part-way through the write would otherwise leave a truncated
        # file that every later run rejects as corrupt instead of simply re-downloading it.
        staged = target.with_name(target.name + ".partial")
        staged.write_bytes(extracted.read_bytes())
        os.replace(staged, target)

    return target


def _ssl_context() -> ssl.SSLContext:
    """Verify TLS against the operating system trust store instead of a bundled CA list.

    TLS-inspecting middleboxes -- corporate proxies, and consumer antivirus doing the same thing --
    present certificates signed by a root installed locally but absent from certifi's bundle.
    Using the OS store makes downloads work in those environments without ever weakening
    verification. Verification stays on; only the source of trusted roots changes.
    """
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def _download(url: str, destination: Path, name: str) -> None:
    try:
        with httpx.stream(
            "GET",
            url,
            timeout=_DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=True,
            verify=_ssl_context(),
            headers={"User-Agent": _USER_AGENT},
        ) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        raise SourceDataError(
            f"Could not download {name} from {url} ({exc}). "
            f"The dataset is hosted on a personal academic site, so it may be temporarily "
            f"unavailable. Check your connection, or download the archive manually and place its "
            f"contents in the raw data directory."
        ) from exc


def _extract_member(archive_path: Path, member: str, into: Path) -> Path:
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            names = bundle.namelist()
            if member not in names:
                raise SourceDataError(
                    f"{archive_path.name} does not contain '{member}'. It contains: {names}. "
                    f"The upstream archive layout has changed."
                )
            bundle.extract(member, path=into)
    except zipfile.BadZipFile as exc:
        raise SourceDataError(
            f"{archive_path.name} is not a valid zip archive ({exc}). The server may have returned "
            f"an error page instead of the dataset."
        ) from exc

    return into / member

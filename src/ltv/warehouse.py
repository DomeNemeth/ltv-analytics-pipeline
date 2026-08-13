"""DuckDB connection handling.

The warehouse is a build artifact, not a source of truth: it is gitignored and every pipeline stage
must be able to rebuild it from scratch. Nothing here caches a connection globally, because DuckDB
takes an exclusive write lock on the file and a stray open handle makes the next stage fail with an
error that looks nothing like its actual cause.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import duckdb

from ltv.config import Settings, get_settings


@contextmanager
def connect(
    settings: Settings | None = None, *, read_only: bool = False
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the project warehouse, creating parent directories if needed.

    Args:
        settings: Configuration to use. Defaults to the process-wide settings.
        read_only: Open without taking a write lock. Use for queries so that a reader cannot block
            a concurrently running pipeline stage.
    """
    settings = settings or get_settings()

    if read_only and not settings.duckdb_path.exists():
        raise FileNotFoundError(
            f"No warehouse at {settings.duckdb_path}. "
            f"Build it first, e.g. `uv run ltv ingest cdnow`."
        )

    if not read_only:
        # Only a writer should have filesystem side effects. A read-only open that silently creates
        # directories is the kind of surprise that makes a later bug hard to locate.
        settings.ensure_dirs()

    connection = duckdb.connect(str(settings.duckdb_path), read_only=read_only)
    try:
        yield connection
    finally:
        connection.close()

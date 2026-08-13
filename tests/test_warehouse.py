"""Warehouse connection behaviour."""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest

from ltv.config import Settings
from ltv.warehouse import connect


def test_read_only_on_a_missing_warehouse_says_how_to_build_it(tmp_path: Path) -> None:
    settings = Settings(repo_root=tmp_path)

    with (
        pytest.raises(FileNotFoundError, match="ltv ingest cdnow"),
        connect(settings, read_only=True),
    ):
        pass


def test_read_only_open_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    """A reader must not create directories -- silent side effects make later bugs hard to place.

    The warehouse must already exist, otherwise the missing-file guard short-circuits before the
    directory creation and the test passes whether or not the guard is there.
    """
    settings = Settings(repo_root=tmp_path)
    with connect(settings) as connection:
        connection.execute("CREATE TABLE t AS SELECT 1 AS n")

    shutil.rmtree(settings.reports_dir)
    shutil.rmtree(settings.raw_dir)

    with connect(settings, read_only=True):
        pass

    assert not settings.reports_dir.exists(), "read-only open recreated the reports directory"
    assert not settings.raw_dir.exists(), "read-only open recreated the raw data directory"


def test_write_connection_creates_the_warehouse_and_round_trips(tmp_path: Path) -> None:
    settings = Settings(repo_root=tmp_path)

    with connect(settings) as connection:
        connection.execute("CREATE TABLE t AS SELECT 1 AS n")

    assert settings.duckdb_path.exists()

    with connect(settings, read_only=True) as connection:
        assert connection.execute("SELECT n FROM t").fetchone() == (1,)


def test_connection_is_closed_on_exit(tmp_path: Path) -> None:
    settings = Settings(repo_root=tmp_path)

    with connect(settings) as connection:
        connection.execute("CREATE TABLE t AS SELECT 1 AS n")

    # A leaked handle keeps DuckDB's exclusive write lock and breaks the next pipeline stage with an
    # error that looks nothing like its actual cause.
    with pytest.raises(duckdb.ConnectionException, match="already closed"):
        connection.execute("SELECT 1")

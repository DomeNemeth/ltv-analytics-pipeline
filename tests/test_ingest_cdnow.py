"""Phase 1 checkpoint: the real CDNOW file loads with its documented shape.

Marked `integration` because it needs the actual source file. It writes to a throwaway warehouse
rather than the development one, so running the suite never clobbers work in progress.

Skipped (not failed) when the source file has not been downloaded, so `uv run pytest` still works on
a clean clone with no network. CI runs `ltv ingest cdnow` first, so there it genuinely executes.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from ltv.config import Settings, get_settings
from ltv.ingest.cdnow import (
    CDNOW_MASTER,
    EXPECTED_CUSTOMERS,
    EXPECTED_ROWS,
    RAW_TABLE,
    ingest_cdnow,
)
from ltv.warehouse import connect

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def source_file() -> Path:
    path = get_settings().raw_dir / CDNOW_MASTER.member
    if not path.exists():
        message = f"{CDNOW_MASTER.member} not downloaded. Run `uv run ltv ingest cdnow` first."
        if os.getenv("CI"):
            # CI runs ingest before pytest. Skipping here would leave the build green while these
            # assertions -- the only ones that would catch a truncated download -- never ran.
            pytest.fail(f"{message} CI must not skip the integration tests.")
        pytest.skip(message)
    return path


@pytest.fixture(scope="module")
def loaded(source_file: Path, tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Ingest the real file into a disposable warehouse once for this module."""
    settings = Settings(repo_root=tmp_path_factory.mktemp("warehouse"))
    ingest_cdnow(settings, source_path=source_file)
    return settings


def test_loads_the_counts_published_in_the_dataset_readme(
    source_file: Path, tmp_path: Path
) -> None:
    settings = Settings(repo_root=tmp_path)

    result = ingest_cdnow(settings, source_path=source_file)

    assert result.rows == EXPECTED_ROWS
    assert result.customers == EXPECTED_CUSTOMERS


def test_ingest_is_idempotent(source_file: Path, tmp_path: Path) -> None:
    """Full refresh, so a second run must not double the table."""
    settings = Settings(repo_root=tmp_path)

    first = ingest_cdnow(settings, source_path=source_file)
    second = ingest_cdnow(settings, source_path=source_file)

    assert first.rows == second.rows == EXPECTED_ROWS


def test_raw_layer_is_typed_not_stringly(loaded: Settings) -> None:
    with connect(loaded, read_only=True) as connection:
        types = dict(
            connection.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'raw' AND table_name = 'cdnow_transactions'"
            ).fetchall()
        )

    assert types["customer_id"] == "INTEGER"
    assert types["order_date"] == "DATE"
    assert types["quantity"] == "INTEGER"
    # Money is DECIMAL. A float money column is a defect, not a style preference.
    assert types["gross_amount"] == "DECIMAL(10,2)"


def test_audit_columns_are_populated(loaded: Settings) -> None:
    with connect(loaded, read_only=True) as connection:
        nulls, distinct_files, min_line, max_line = connection.execute(
            f"""
            SELECT
                count(*) FILTER (WHERE _source_file IS NULL OR _ingested_at IS NULL),
                count(DISTINCT _source_file),
                min(_line_number),
                max(_line_number)
            FROM {RAW_TABLE}
            """
        ).fetchone()

    assert nulls == 0
    assert distinct_files == 1
    assert (min_line, max_line) == (1, EXPECTED_ROWS)


def test_raw_preserves_source_fidelity(loaded: Settings) -> None:
    """The duplicate and zero-value rows survive into raw. Staging decides what to do with them."""
    with connect(loaded, read_only=True) as connection:
        duplicates, zero_value, negative = connection.execute(
            f"""
            SELECT
                (SELECT count(*) - count(DISTINCT (customer_id, order_date, quantity, gross_amount))
                   FROM {RAW_TABLE}),
                (SELECT count(*) FROM {RAW_TABLE} WHERE gross_amount = 0),
                (SELECT count(*) FROM {RAW_TABLE} WHERE gross_amount < 0 OR quantity <= 0)
            """
        ).fetchone()

    assert duplicates == 255
    assert zero_value == 80
    assert negative == 0


def test_cohort_window_matches_the_documented_dataset(loaded: Settings) -> None:
    """CDNOW is a Q1-1997 acquisition cohort observed to June 1998: 78 weeks, two 39-week halves."""
    with connect(loaded, read_only=True) as connection:
        first_order, last_order, latest_first_purchase = connection.execute(
            f"""
            WITH first_purchases AS (
                SELECT min(order_date) AS first_order
                FROM {RAW_TABLE}
                GROUP BY customer_id
            )
            SELECT
                (SELECT min(order_date) FROM {RAW_TABLE}),
                (SELECT max(order_date) FROM {RAW_TABLE}),
                (SELECT max(first_order) FROM first_purchases)
            """
        ).fetchone()

    assert first_order == date(1997, 1, 1)
    assert last_order == date(1998, 6, 30)
    assert latest_first_purchase <= date(1997, 3, 31)

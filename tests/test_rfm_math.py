"""Phase 2 checkpoint: the RFM models agree with hand-computed values for three known customers.

The expected values are computed here with pandas, straight from the raw table, by logic written
independently of the SQL under test. That independence is the whole point: asserting the models
against numbers extracted from the same models would prove only that DuckDB is deterministic.

The fixture builds its own warehouse rather than reading the development one, so the test can never
pass or fail on the basis of a stale `ltv transform` someone ran days ago.

Marked `integration` because it needs the real source file.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ltv.config import Settings, get_settings
from ltv.ingest.cdnow import CDNOW_MASTER, RAW_TABLE, ingest_cdnow
from ltv.transform import run_dbt
from ltv.warehouse import connect

pytestmark = pytest.mark.integration

#: The canonical Fader & Hardie cutoff for CDNOW, asserted independently in dbt by
#: assert_calibration_window_matches_benchmark. Restated here so this module does not depend on the
#: model it is checking to tell it where the window ends.
CALIBRATION_END = date(1997, 9, 30)

#: Chosen to cover the three shapes that break differently, not at random:
#:
#: * 1     -- a one-time buyer. frequency 0, recency 0, monetary_value 0. The ~60% majority case,
#:            and the one where the "average repeat spend" denominator is zero.
#: * 40    -- a mid-frequency buyer with a three-line day, so the occasion collapse changes his
#:            numbers. Without the collapse his frequency is wrong.
#: * 19339 -- a heavy buyer with an eight-line day and 56 line items across the file. The loudest
#:            possible failure if the collapse ever stops happening.
SAMPLE_CUSTOMERS = (1, 40, 19339)


@pytest.fixture(scope="module")
def source_file() -> Path:
    path = get_settings().raw_dir / CDNOW_MASTER.member
    if not path.exists():
        message = f"{CDNOW_MASTER.member} not downloaded. Run `uv run ltv ingest cdnow` first."
        if os.getenv("CI"):
            pytest.fail(f"{message} CI must not skip the integration tests.")
        pytest.skip(message)
    return path


@pytest.fixture(scope="module")
def built(source_file: Path) -> Iterator[Settings]:
    """Ingest and transform into a warehouse of this module's own, then remove it.

    Only ``warehouse_filename`` is overridden, not ``repo_root``: the dbt project directory is
    derived from the repo root, so relocating that would point dbt at a directory containing no
    models. This is the one setting that isolates the data without moving the code.
    """
    settings = Settings(warehouse_filename="pytest-rfm.duckdb")
    try:
        ingest_cdnow(settings, source_path=source_file)
        # `build`, not `run`, so a failing dbt test fails this fixture. It also means a clean clone
        # running only `uv run pytest` still verifies the whole transformation layer.
        run_dbt(["build"], settings)
        yield settings
    finally:
        settings.duckdb_path.unlink(missing_ok=True)
        settings.duckdb_path.with_suffix(".duckdb.wal").unlink(missing_ok=True)


@pytest.fixture(scope="module")
def raw_transactions(built: Settings) -> pd.DataFrame:
    with connect(built, read_only=True) as connection:
        return connection.execute(
            f"SELECT customer_id, order_date, gross_amount FROM {RAW_TABLE}"
        ).fetch_df()


def expected_rfm(raw: pd.DataFrame, customer_id: int, window_end: date) -> dict[str, float]:
    """Compute one customer's RFM summary from raw line items, without touching the dbt models."""
    rows = raw[(raw["customer_id"] == customer_id) & (raw["order_date"].dt.date <= window_end)]

    # The collapse, done in pandas: one occasion per day the customer bought anything.
    daily = rows.groupby(rows["order_date"].dt.date)["gross_amount"].sum().sort_index()

    occasions = len(daily)
    frequency = occasions - 1
    first_order, last_order = daily.index[0], daily.index[-1]
    total_spend = float(daily.sum())
    first_occasion_spend = float(daily.iloc[0])

    return {
        "occasions": occasions,
        "frequency": frequency,
        "recency": (last_order - first_order).days,
        "customer_age": (window_end - first_order).days,
        "total_spend": total_spend,
        "monetary_value": ((total_spend - first_occasion_spend) / frequency if frequency else 0.0),
    }


def actual_rfm(settings: Settings, customer_id: int) -> dict[str, float]:
    with connect(settings, read_only=True) as connection:
        row = connection.execute(
            """
            SELECT occasions, frequency, recency, customer_age, total_spend, monetary_value
            FROM main.int_customers__rfm_calibration
            WHERE source = 'cdnow' AND customer_id = ?
            """,
            [customer_id],
        ).fetchone()

    assert row is not None, f"customer {customer_id} is missing from the calibration summary"
    names = ("occasions", "frequency", "recency", "customer_age", "total_spend", "monetary_value")
    return {name: float(value) for name, value in zip(names, row, strict=True)}


@pytest.mark.parametrize("customer_id", SAMPLE_CUSTOMERS)
def test_calibration_rfm_matches_hand_computed_values(
    built: Settings, raw_transactions: pd.DataFrame, customer_id: int
) -> None:
    expected = expected_rfm(raw_transactions, customer_id, CALIBRATION_END)
    actual = actual_rfm(built, customer_id)

    for field, want in expected.items():
        # Money is compared with a tolerance because monetary_value is a mean stored as a fixed
        # scale decimal; the counts and day differences are exact and compare exactly.
        assert actual[field] == pytest.approx(want, abs=1e-4), (
            f"customer {customer_id}: {field} was {actual[field]}, hand-computed {want}"
        )


def test_the_sample_actually_exercises_the_occasion_collapse(
    built: Settings, raw_transactions: pd.DataFrame
) -> None:
    """Guard the premise of the test above, not the code under test.

    Two of the sampled customers were chosen because they have days holding several line items, so
    their frequency is only correct if the collapse happened. If someone swaps the sample for
    customers who never bought twice in a day, every assertion above would still pass while no
    longer covering the collapse at all. This fails in that case.
    """
    collapsing_customers = []
    for customer_id in SAMPLE_CUSTOMERS:
        rows = raw_transactions[raw_transactions["customer_id"] == customer_id]
        line_items = len(rows)
        occasions = rows["order_date"].nunique()
        if line_items > occasions:
            collapsing_customers.append(customer_id)

    assert len(collapsing_customers) >= 2, (
        "the sampled customers no longer include multi-line-item days, so these tests would pass "
        f"even with the occasion collapse removed. Collapsing customers: {collapsing_customers}"
    )


def test_the_warehouse_is_readable_after_a_dbt_run(built: Settings) -> None:
    """dbt must not leave its DuckDB handle open when it finishes.

    dbt runs in-process, and its adapter caches a connection past the end of the invocation. DuckDB
    then refuses any connection to the same file under a different configuration, so the next stage
    to open the warehouse read-only fails with an error naming neither dbt nor the real cause.

    This is the shape of the Phase 6 flow -- dbt, then Python, then dbt, in one process -- so it is
    asserted directly rather than left to be discovered there.
    """
    with connect(built, read_only=True) as connection:
        (models,) = connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name LIKE 'int_%'"
        ).fetchone()

    assert models > 0


def test_one_time_buyers_carry_no_repeat_spend(built: Settings) -> None:
    """The Gamma-Gamma denominator is zero for these customers, so they must be excluded, not zero.

    A one-time buyer with a non-zero monetary_value means the first purchase leaked into the mean.
    """
    with connect(built, read_only=True) as connection:
        one_time, with_spend, eligible = connection.execute(
            """
            SELECT
                count(*),
                count(*) FILTER (WHERE monetary_value != 0),
                count(*) FILTER (WHERE is_gamma_gamma_eligible)
            FROM main.int_customers__rfm_calibration
            WHERE frequency = 0
            """
        ).fetchone()

    assert one_time == 14_119
    assert with_spend == 0
    assert eligible == 0

"""CDNOW master dataset: parse the fixed-width file and load it into `raw`.

The CDNOW master file is the canonical CLV benchmark dataset: the complete purchase history through
June 1998 of the 23,570 customers who made their first purchase at CDNOW in Q1 1997.

The raw layer is a faithful, unfiltered copy of the source. The 255 byte-identical duplicate rows
and 80 zero-value rows in this file are deliberately preserved here and dealt with in dbt staging,
where the decision is visible, tested, and reviewable rather than buried in a loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ltv.config import Settings, get_settings
from ltv.ingest.fetch import RemoteArchive, ensure_local_copy
from ltv.warehouse import connect

CDNOW_MASTER = RemoteArchive(
    name="cdnow_master",
    url="https://www.brucehardie.com/datasets/CDNOW_master.zip",
    member="CDNOW_master.txt",
    sha256="61fd6f7bf3497187a7ec0d9ff6a99b7568f1fe911a9f3665e02efecd6b34ed07",
    citation=(
        "Fader, Peter S. and Bruce G. S. Hardie (2001), 'Forecasting Repeat Sales at CDNOW: "
        "A Case Study', Interfaces, 31 (May-June), Part 2 of 2, S94-S107."
    ),
)

RAW_TABLE = "raw.cdnow_transactions"

#: Counts published in the dataset's own read_me. Asserted by tests, not enforced by the loader --
#: a loader that refuses to load anything but one exact file is not a loader.
EXPECTED_ROWS = 69_659
EXPECTED_CUSTOMERS = 23_570

#: Every line in the file is exactly this wide. Truncation is therefore detectable, not silent.
LINE_WIDTH = 26

#: Columns that are blank on every one of the 69,659 lines, verified empirically. These are the
#: field separators, and checking them is what turns a shifted-column file into a loud failure
#: instead of plausible-looking garbage.
SEPARATOR_COLUMNS = (0, 6, 15, 18)

#: (name, start, stop) as half-open, 0-based bounds, derived from the separator positions above.
#: Note the field at 16:18 -- a naive whitespace split would absorb column 18 into it.
#: These drive the parser directly; changing them changes how the file is read.
FIELDS = (
    ("customer_id", 1, 6),
    ("order_date", 7, 15),
    ("quantity", 16, 18),
    ("gross_amount", 19, 26),
)

_SLICES = {name: slice(start, stop) for name, start, stop in FIELDS}


class CDNOWFormatError(ValueError):
    """Raised when the CDNOW source file does not match its documented fixed-width layout."""


@dataclass(frozen=True)
class IngestResult:
    """Outcome of a raw load, for the CLI to report and tests to assert on."""

    table: str
    rows: int
    customers: int
    source_file: Path


def parse_cdnow_master(path: Path) -> pd.DataFrame:
    """Parse the CDNOW fixed-width master file strictly.

    Strict means every deviation from the documented layout raises, naming the line number and its
    contents. A lenient parser on a fixed-width file is how a column shift becomes a silently wrong
    dataset that still loads, still passes row counts, and produces a plausible but wrong model.

    Returns:
        One row per source line, with a 1-based ``line_number`` so any row can be traced back to a
        specific line of the source file.

    Raises:
        CDNOWFormatError: On any unexpected width, separator, or unparseable field.
    """
    try:
        text = path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise CDNOWFormatError(
            f"{path} contains non-ASCII bytes at position {exc.start}, but the CDNOW master file "
            f"is plain ASCII. The file is corrupt or is not the CDNOW master file."
        ) from exc

    lines = text.splitlines()
    if not lines:
        raise CDNOWFormatError(f"{path} is empty.")

    customer_ids: list[int] = []
    order_dates: list[str] = []
    quantities: list[int] = []
    amounts: list[float] = []

    for number, line in enumerate(lines, start=1):
        if len(line) != LINE_WIDTH:
            raise CDNOWFormatError(
                f"{path} line {number}: expected {LINE_WIDTH} characters, found {len(line)}. "
                f"Line was {line!r}."
            )
        for column in SEPARATOR_COLUMNS:
            if line[column] != " ":
                raise CDNOWFormatError(
                    f"{path} line {number}: expected a blank field separator at column {column}, "
                    f"found {line[column]!r}. Columns have shifted. Line was {line!r}."
                )

        try:
            customer_ids.append(int(line[_SLICES["customer_id"]]))
            quantities.append(int(line[_SLICES["quantity"]]))
            amounts.append(float(line[_SLICES["gross_amount"]]))
        except ValueError as exc:
            raise CDNOWFormatError(
                f"{path} line {number}: could not parse a numeric field ({exc}). Line was {line!r}."
            ) from exc
        order_dates.append(line[_SLICES["order_date"]])

    try:
        parsed_dates = pd.to_datetime(pd.Series(order_dates), format="%Y%m%d", errors="raise")
    except ValueError as exc:
        raise CDNOWFormatError(
            f"{path}: one or more dates are not valid YYYYMMDD values ({exc})."
        ) from exc

    return pd.DataFrame(
        {
            "customer_id": pd.array(customer_ids, dtype="int32"),
            "order_date": parsed_dates,
            "quantity": pd.array(quantities, dtype="int32"),
            # Transient float only. Money is stored as DECIMAL in the warehouse; the round trip is
            # exact here because the source has at most two decimal places and small magnitudes.
            "gross_amount": amounts,
            "line_number": pd.array(range(1, len(lines) + 1), dtype="int32"),
        }
    )


_CREATE_TABLE = f"""
CREATE OR REPLACE TABLE {RAW_TABLE} (
    customer_id   INTEGER       NOT NULL,
    order_date    DATE          NOT NULL,
    quantity      INTEGER       NOT NULL,
    gross_amount  DECIMAL(10,2) NOT NULL,
    _source_file  VARCHAR       NOT NULL,
    _line_number  INTEGER       NOT NULL,
    _ingested_at  TIMESTAMPTZ   NOT NULL
)
"""

_INSERT = f"""
INSERT INTO {RAW_TABLE}
SELECT
    customer_id,
    order_date,
    quantity,
    CAST(gross_amount AS DECIMAL(10,2)),
    $source_file,
    line_number,
    $ingested_at
FROM parsed
"""


def ingest_cdnow(
    settings: Settings | None = None,
    *,
    source_path: Path | None = None,
    force_download: bool = False,
) -> IngestResult:
    """Load the CDNOW master file into ``raw.cdnow_transactions``.

    The table is replaced wholesale on every run. These datasets are small and static, so a full
    refresh is honest and idempotent; incremental loading here would be complexity for its own sake.

    Args:
        settings: Configuration to use. Defaults to the process-wide settings.
        source_path: Read this file instead of fetching the pinned archive. Used by tests and by the
            deliberately-corrupted-load demonstration.
        force_download: Re-fetch the archive even if a checksum-valid copy is already cached.
            Ignored when ``source_path`` is given.
    """
    settings = settings or get_settings()
    resolved = source_path or ensure_local_copy(
        CDNOW_MASTER, settings.raw_dir, force=force_download
    )

    parsed = parse_cdnow_master(resolved)

    with connect(settings) as connection:
        connection.execute("CREATE SCHEMA IF NOT EXISTS raw")
        connection.execute(_CREATE_TABLE)
        connection.register("parsed", parsed)
        connection.execute(
            _INSERT,
            {"source_file": resolved.name, "ingested_at": datetime.now(tz=UTC)},
        )
        rows, customers = connection.execute(
            f"SELECT count(*), count(DISTINCT customer_id) FROM {RAW_TABLE}"
        ).fetchone()

    return IngestResult(table=RAW_TABLE, rows=rows, customers=customers, source_file=resolved)

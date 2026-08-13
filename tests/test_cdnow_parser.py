"""Parser tests.

These use inline fixtures and never touch the network, so the fast suite stays runnable on a clean
clone with no data downloaded. Each malformed-input test represents a real way a fixed-width file
degrades: truncation, column drift, encoding damage, and junk in a numeric field.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers import cdnow_line, write_lines
from ltv.ingest.cdnow import (
    FIELDS,
    LINE_WIDTH,
    SEPARATOR_COLUMNS,
    CDNOWFormatError,
    parse_cdnow_master,
)


def test_fixture_helper_matches_the_real_layout() -> None:
    """Guard the guard: if this helper drifts, every other test here becomes meaningless."""
    assert len(cdnow_line()) == LINE_WIDTH
    assert cdnow_line() == " 00001 19970101  1   11.77"


def test_field_bounds_extract_the_documented_values() -> None:
    """FIELDS is load-bearing -- the parser slices with it, so its bounds must be right."""
    extracted = {name: cdnow_line()[start:stop].strip() for name, start, stop in FIELDS}

    assert extracted == {
        "customer_id": "00001",
        "order_date": "19970101",
        "quantity": "1",
        "gross_amount": "11.77",
    }


def test_field_ranges_and_separators_tile_the_line_exactly() -> None:
    """FIELDS and SEPARATOR_COLUMNS must stay consistent with each other.

    Widening a field to swallow a separator would still parse -- int() and float() tolerate leading
    blanks -- so nothing else in the suite would notice. This pins the layout as a whole.
    """
    covered = [column for _, start, stop in FIELDS for column in range(start, stop)]

    assert len(covered) == len(set(covered)), "field ranges overlap each other"
    assert not set(covered) & set(SEPARATOR_COLUMNS), "a field range swallows a separator column"
    assert set(covered) | set(SEPARATOR_COLUMNS) == set(range(LINE_WIDTH)), (
        "fields plus separators must account for every column in the line"
    )


def test_parsed_columns_follow_the_field_definitions(tmp_path: Path) -> None:
    source = write_lines(tmp_path / "cdnow.txt", [cdnow_line()])

    frame = parse_cdnow_master(source)

    assert [name for name, _, _ in FIELDS] == list(frame.columns)[:-1]


def test_parses_fields_and_types(tmp_path: Path) -> None:
    source = write_lines(
        tmp_path / "cdnow.txt",
        [cdnow_line(1, "19970101", 1, 11.77), cdnow_line(2, "19980630", 12, 1286.01)],
    )

    frame = parse_cdnow_master(source)

    assert list(frame.columns) == [
        "customer_id",
        "order_date",
        "quantity",
        "gross_amount",
        "line_number",
    ]
    assert frame.customer_id.tolist() == [1, 2]
    assert frame.quantity.tolist() == [1, 12]
    assert frame.gross_amount.tolist() == [11.77, 1286.01]
    assert [str(d.date()) for d in frame.order_date] == ["1997-01-01", "1998-06-30"]


def test_line_number_is_one_based_and_traces_to_the_source(tmp_path: Path) -> None:
    """Audit columns are only useful if they point at the right line."""
    source = write_lines(tmp_path / "cdnow.txt", [cdnow_line(customer=c) for c in (7, 8, 9)])

    frame = parse_cdnow_master(source)

    assert frame.line_number.tolist() == [1, 2, 3]
    assert frame.loc[frame.line_number == 2, "customer_id"].item() == 8


def test_duplicate_and_zero_value_rows_are_preserved(tmp_path: Path) -> None:
    """Raw is a faithful copy. Filtering these is dbt staging's job, not the loader's."""
    duplicate = cdnow_line(3, "19970102", 2, 20.76)
    source = write_lines(
        tmp_path / "cdnow.txt", [duplicate, duplicate, cdnow_line(4, "19970103", 1, 0.00)]
    )

    frame = parse_cdnow_master(source)

    assert len(frame) == 3
    assert (frame.gross_amount == 0.0).sum() == 1


def test_rejects_truncated_line(tmp_path: Path) -> None:
    source = write_lines(tmp_path / "cdnow.txt", [cdnow_line(), cdnow_line()[:20]])

    with pytest.raises(CDNOWFormatError, match="line 2: expected 26 characters, found 20"):
        parse_cdnow_master(source)


def test_rejects_shifted_columns(tmp_path: Path) -> None:
    """A separator column filled with a digit means fields have drifted -- the dangerous case,
    because the row is still 26 characters and would parse into plausible nonsense."""
    shifted = "X" + cdnow_line()[1:]
    source = write_lines(tmp_path / "cdnow.txt", [shifted])

    with pytest.raises(CDNOWFormatError, match="blank field separator at column 0"):
        parse_cdnow_master(source)


def test_rejects_non_numeric_field(tmp_path: Path) -> None:
    corrupt = " 0000A 19970101  1   11.77"
    source = write_lines(tmp_path / "cdnow.txt", [corrupt])

    with pytest.raises(CDNOWFormatError, match="line 1: could not parse a numeric field"):
        parse_cdnow_master(source)


def test_rejects_impossible_date(tmp_path: Path) -> None:
    source = write_lines(tmp_path / "cdnow.txt", [cdnow_line(date="19971332")])

    with pytest.raises(CDNOWFormatError, match="not valid YYYYMMDD"):
        parse_cdnow_master(source)


def test_rejects_non_ascii_file(tmp_path: Path) -> None:
    source = tmp_path / "cdnow.txt"
    source.write_bytes(cdnow_line().encode("ascii") + "€".encode() + b"\n")

    with pytest.raises(CDNOWFormatError, match="non-ASCII"):
        parse_cdnow_master(source)


def test_rejects_empty_file(tmp_path: Path) -> None:
    source = write_lines(tmp_path / "cdnow.txt", [])

    with pytest.raises(CDNOWFormatError, match="is empty"):
        parse_cdnow_master(source)

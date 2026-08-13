"""CLI behaviour.

The exit code is the contract CI depends on: a failed ingest must exit non-zero, and the reason must
be readable rather than buried in a traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ltv import cli
from ltv.ingest.cdnow import CDNOWFormatError, IngestResult
from ltv.ingest.fetch import SourceDataError

runner = CliRunner()


@pytest.fixture
def fake_result() -> IngestResult:
    return IngestResult(
        table="raw.cdnow_transactions",
        rows=69_659,
        customers=23_570,
        source_file=Path("data/raw/CDNOW_master.txt"),
    )


def test_bare_invocation_lists_commands() -> None:
    result = runner.invoke(cli.app, [])

    assert "info" in result.output
    assert "ingest" in result.output


def test_info_reports_configuration() -> None:
    result = runner.invoke(cli.app, ["info"])

    assert result.exit_code == 0
    assert "calibration split" in result.output


def test_ingest_reports_counts_and_attribution(
    monkeypatch: pytest.MonkeyPatch, fake_result: IngestResult
) -> None:
    monkeypatch.setattr(cli, "ingest_cdnow", lambda *a, **k: fake_result)

    result = runner.invoke(cli.app, ["ingest", "cdnow"])

    assert result.exit_code == 0
    assert "69,659" in result.output
    assert "23,570" in result.output
    assert "Fader" in result.output, "the dataset's authors should be credited on load"


def test_force_download_flag_reaches_the_loader(
    monkeypatch: pytest.MonkeyPatch, fake_result: IngestResult
) -> None:
    """Guards the wiring: the flag previously deleted a hardcoded filename instead."""
    seen: dict[str, object] = {}

    def capture(settings: object, **kwargs: object) -> IngestResult:
        seen.update(kwargs)
        return fake_result

    monkeypatch.setattr(cli, "ingest_cdnow", capture)

    runner.invoke(cli.app, ["ingest", "cdnow", "--force-download"])

    assert seen == {"force_download": True}


def test_default_invocation_does_not_force_download(
    monkeypatch: pytest.MonkeyPatch, fake_result: IngestResult
) -> None:
    seen: dict[str, object] = {}

    def capture(settings: object, **kwargs: object) -> IngestResult:
        seen.update(kwargs)
        return fake_result

    monkeypatch.setattr(cli, "ingest_cdnow", capture)

    runner.invoke(cli.app, ["ingest", "cdnow"])

    assert seen == {"force_download": False}


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (
            SourceDataError("Could not download cdnow_master from https://example.invalid"),
            "Could not download",
        ),
        (CDNOWFormatError("line 42: expected 26 characters, found 20"), "line 42"),
    ],
)
def test_failures_exit_non_zero_with_a_readable_message(
    monkeypatch: pytest.MonkeyPatch, error: Exception, fragment: str
) -> None:
    def fail(*args: object, **kwargs: object) -> IngestResult:
        raise error

    monkeypatch.setattr(cli, "ingest_cdnow", fail)

    result = runner.invoke(cli.app, ["ingest", "cdnow"])

    assert result.exit_code == 1
    assert fragment in result.output
    assert "Traceback" not in result.output

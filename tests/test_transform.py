"""The dbt runner's wiring: configuration flow, preconditions, and argument handling.

These run offline against throwaway warehouses -- nothing here invokes dbt for real. The real dbt
build is exercised by tests/test_rfm_math.py, which is marked `integration`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml  # dbt depends on PyYAML, so it is always present alongside this project's dbt runner.

from ltv.config import Settings, get_settings
from ltv.transform import TransformError, _dbt_environment, run_dbt
from ltv.warehouse import connect


def test_dbt_project_defaults_match_settings_defaults() -> None:
    """The analysis window is declared in two places; this stops them drifting apart.

    dbt_project.yml carries defaults so `dbt build` works standalone for someone running dbt
    directly, while `ltv transform` overrides them from Settings. If the two disagree, the same repo
    produces different calibration windows depending on how it was invoked -- and the numbers would
    still look entirely reasonable.
    """
    project = yaml.safe_load((get_settings().dbt_dir / "dbt_project.yml").read_text())
    defaults = Settings()

    assert project["vars"]["calibration_weeks"] == defaults.calibration_weeks
    assert project["vars"]["holdout_weeks"] == defaults.holdout_weeks


def test_refuses_to_run_against_a_warehouse_with_no_raw_data(tmp_path: Path) -> None:
    settings = Settings(repo_root=tmp_path)
    with connect(settings):
        pass  # Create the file, but load nothing into it.

    with pytest.raises(TransformError, match="no `raw` tables"):
        run_dbt(["build"], settings)


def test_refuses_to_run_when_the_warehouse_does_not_exist(tmp_path: Path) -> None:
    """The precondition check reuses the read-only connection guard, so the message names ingest."""
    settings = Settings(repo_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="ltv ingest cdnow"):
        run_dbt(["build"], settings)


def test_dbt_environment_restores_the_previous_value(tmp_path: Path) -> None:
    """dbt runs in this process, so a leaked env var would follow every later stage in the flow."""
    settings = Settings(repo_root=tmp_path)
    sentinel = "/somewhere/else.duckdb"
    os.environ["LTV_DUCKDB_PATH"] = sentinel
    try:
        with _dbt_environment(settings):
            assert os.environ["LTV_DUCKDB_PATH"] == str(settings.duckdb_path)
        assert os.environ["LTV_DUCKDB_PATH"] == sentinel
    finally:
        os.environ.pop("LTV_DUCKDB_PATH", None)


def test_dbt_environment_unsets_a_variable_it_introduced(tmp_path: Path) -> None:
    settings = Settings(repo_root=tmp_path)
    os.environ.pop("LTV_DUCKDB_PATH", None)

    with _dbt_environment(settings):
        assert "LTV_DUCKDB_PATH" in os.environ

    assert "LTV_DUCKDB_PATH" not in os.environ


class _RecordingRunner:
    """Stands in for dbtRunner so the invocation can be inspected without running dbt."""

    invocations: list[list[str]] = []

    def invoke(self, args: list[str]) -> _RecordingRunner:
        type(self).invocations.append(list(args))
        return self

    success = True
    exception = None


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Make run_dbt record its dbt invocation instead of performing it."""
    import dbt.cli.main

    _RecordingRunner.invocations = []
    monkeypatch.setattr(dbt.cli.main, "dbtRunner", _RecordingRunner)

    settings = Settings(repo_root=tmp_path)
    with connect(settings) as connection:
        connection.execute("CREATE SCHEMA raw")
        connection.execute("CREATE TABLE raw.placeholder (id INTEGER)")
    return settings


def test_a_leading_flag_applies_to_the_default_command(recorded: Settings) -> None:
    """`ltv transform --select staging` must filter a build, not replace the command with a flag."""
    run_dbt(["--select", "staging"], recorded)

    assert _RecordingRunner.invocations[0][:3] == ["build", "--select", "staging"]


def test_a_leading_word_selects_a_different_command(recorded: Settings) -> None:
    run_dbt(["test", "--select", "staging"], recorded)

    invocation = _RecordingRunner.invocations[0]
    assert invocation[:3] == ["test", "--select", "staging"]
    assert "build" not in invocation


def test_defaults_to_build_with_no_arguments(recorded: Settings) -> None:
    run_dbt(None, recorded)

    assert _RecordingRunner.invocations[0][0] == "build"


def test_passes_the_analysis_window_from_settings(recorded: Settings) -> None:
    """Settings is the runtime source of truth, so a non-default window must reach dbt."""
    settings = Settings(repo_root=recorded.repo_root, calibration_weeks=26, holdout_weeks=13)

    run_dbt(["build"], settings)

    invocation = _RecordingRunner.invocations[0]
    variables = invocation[invocation.index("--vars") + 1]
    assert '"calibration_weeks": 26' in variables
    assert '"holdout_weeks": 13' in variables


def test_reports_a_dbt_failure_rather_than_returning_quietly(recorded: Settings) -> None:
    class _FailingRunner(_RecordingRunner):
        success = False

    import dbt.cli.main

    original = dbt.cli.main.dbtRunner
    dbt.cli.main.dbtRunner = _FailingRunner
    try:
        with pytest.raises(TransformError, match="dbt build failed"):
            run_dbt(["build"], recorded)
    finally:
        dbt.cli.main.dbtRunner = original

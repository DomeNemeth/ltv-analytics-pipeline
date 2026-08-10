"""Config is the single source of truth for paths, so it gets real tests, not a smoke test."""

from __future__ import annotations

from pathlib import Path

from ltv.config import Settings


def test_repo_root_default_contains_the_project() -> None:
    """The default root must be the actual repo, not the package directory."""
    settings = Settings()
    assert (settings.repo_root / "pyproject.toml").is_file()


def test_all_paths_are_derived_from_repo_root(tmp_path: Path) -> None:
    """Overriding the root must move every derived path -- no half-relocated tree in Docker."""
    settings = Settings(repo_root=tmp_path)

    derived = [
        settings.data_dir,
        settings.raw_dir,
        settings.duckdb_path,
        settings.dbt_dir,
        settings.dashboard_dir,
        settings.reports_dir,
    ]
    for path in derived:
        assert tmp_path in path.parents, f"{path} does not live under the configured repo root"


def test_warehouse_filename_is_configurable(tmp_path: Path) -> None:
    settings = Settings(repo_root=tmp_path, warehouse_filename="scratch.duckdb")
    assert settings.duckdb_path == tmp_path / "data" / "scratch.duckdb"


def test_ensure_dirs_is_idempotent(tmp_path: Path) -> None:
    settings = Settings(repo_root=tmp_path)
    settings.ensure_dirs()
    settings.ensure_dirs()

    assert settings.raw_dir.is_dir()
    assert settings.reports_dir.is_dir()


def test_calibration_split_defaults_to_the_fader_hardie_window() -> None:
    """39/39 weeks is what makes our CDNOW metrics comparable to published results."""
    settings = Settings()
    assert settings.calibration_weeks == 39
    assert settings.holdout_weeks == 39

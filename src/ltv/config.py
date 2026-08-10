"""Central configuration.

Every path in this project is derived from :attr:`Settings.repo_root`. Nothing anywhere else in the
codebase may hardcode a filesystem path -- that is what makes the repo clonable and what lets Docker
relocate the whole tree by setting a single environment variable (``LTV_REPO_ROOT``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root, resolved from this file's location (src/ltv/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, overridable via ``LTV_``-prefixed environment variables or a ``.env`` file.

    Only genuinely free parameters are fields. Everything else is a derived property, so that
    overriding ``repo_root`` moves the entire tree consistently rather than leaving half the paths
    pointing at the old location.
    """

    model_config = SettingsConfigDict(
        env_prefix="LTV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    repo_root: Path = REPO_ROOT
    warehouse_filename: str = "warehouse.duckdb"

    # Calibration/holdout split. 39/39 weeks is the canonical Fader & Hardie CDNOW split, which is
    # what makes our error metrics comparable to published benchmarks.
    calibration_weeks: int = 39
    holdout_weeks: int = 39

    random_seed: int = 42

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def raw_dir(self) -> Path:
        """Downloaded source files. Gitignored -- rebuilt by ``ltv ingest``."""
        return self.data_dir / "raw"

    @property
    def duckdb_path(self) -> Path:
        """The warehouse. Gitignored -- a build artifact, never a source of truth."""
        return self.data_dir / self.warehouse_filename

    @property
    def dbt_dir(self) -> Path:
        return self.repo_root / "dbt"

    @property
    def dashboard_dir(self) -> Path:
        return self.repo_root / "dashboard"

    @property
    def reports_dir(self) -> Path:
        """Generated validation metrics and charts. Committed -- they are project output."""
        return self.repo_root / "reports"

    def ensure_dirs(self) -> None:
        """Create the directories the pipeline writes to. Safe to call repeatedly."""
        for path in (self.data_dir, self.raw_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()

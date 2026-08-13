"""Run the dbt transformation layer.

dbt is invoked through dbt's own Python entry point rather than a shell script. That choice does
three things a wrapper script could not:

* It works identically on Windows, macOS, and Linux, which the shell-script alternative does not
  without maintaining two scripts that drift apart.
* It resolves no executable from ``PATH``. Inside ``uv run`` the ``dbt`` binary is on the path, but
  inside a Prefect worker or a Docker entrypoint it may not be, and the resulting error names the
  wrong problem.
* ``env_var()`` in ``profiles.yml`` reads the environment of *this* process, so configuration flows
  straight from :class:`~ltv.config.Settings` with nothing to plumb through a subprocess.

The single source of truth for the analysis window is ``Settings``. ``dbt_project.yml`` declares
matching defaults so that a plain ``dbt build`` still works for someone who wants to run dbt
directly, and :mod:`tests.test_transform` asserts the two cannot drift apart.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from ltv.config import Settings, get_settings
from ltv.warehouse import connect

#: What ``ltv transform`` runs when given no arguments. ``build`` rather than ``run`` so that a
#: failing test fails the command -- running models without their tests is how a broken model gets
#: shipped looking green.
DEFAULT_ARGS: tuple[str, ...] = ("build",)


class TransformError(RuntimeError):
    """Raised when dbt fails, or when the warehouse is not in a state dbt can run against."""


@contextmanager
def _dbt_environment(settings: Settings) -> Iterator[None]:
    """Export the settings dbt needs, and restore the previous environment afterwards.

    Restoring matters because dbt runs in this process: leaving ``LTV_DUCKDB_PATH`` set would let a
    later stage in the same process (the Prefect flow, or a test) silently inherit it.
    """
    exported = {"LTV_DUCKDB_PATH": str(settings.duckdb_path)}
    previous = {key: os.environ.get(key) for key in exported}

    os.environ.update(exported)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _require_raw_data(settings: Settings) -> None:
    """Fail early and actionably if there is nothing for dbt to read.

    Without this, dbt opens the warehouse, DuckDB happily creates an empty database file, and the
    run fails deep inside a compiled model with a "table does not exist" error that names a dbt
    relation rather than the missing ingest step.
    """
    with connect(settings, read_only=True) as connection:
        (raw_tables,) = connection.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'raw'"
        ).fetchone()

    if not raw_tables:
        raise TransformError(
            f"The warehouse at {settings.duckdb_path} has no `raw` tables, so there is nothing to "
            f"transform. Load a source first, e.g. `uv run ltv ingest cdnow`."
        )


def _release_warehouse() -> None:
    """Close the DuckDB connections dbt opened.

    Running dbt in-process means its adapter keeps a connection to the warehouse alive after the
    invocation returns. DuckDB refuses a second connection to the same file under a different
    configuration, so the next stage that opens the warehouse read-only fails with

        Can't open a connection to same database file with a different configuration

    which names neither dbt nor the stage that actually left the handle open. This matters well
    beyond tests: the Phase 6 flow is dbt -> Python -> dbt in a single process, and the Python
    stage in the middle is exactly the read-only reader that would hit it.

    dbt-duckdb registers this same cleanup with ``atexit``, which is too late to help a process that
    keeps running.
    """
    import gc

    from dbt.adapters.duckdb.connections import DuckDBConnectionManager
    from dbt.adapters.factory import reset_adapters

    # Two separate references have to go: dbt's adapter registry, and dbt-duckdb's class-level
    # environment cache. Releasing either alone leaves the other holding the handle open.
    reset_adapters()
    DuckDBConnectionManager.close_all_connections()

    # close_all_connections only drops its reference; the file handle is released when the
    # connection object is collected. CPython would normally do that immediately on refcount zero,
    # but dbt's object graph contains cycles, so the collector has to be asked.
    gc.collect()


def run_dbt(args: Sequence[str] | None = None, settings: Settings | None = None) -> None:
    """Invoke dbt against the project warehouse.

    Args:
        args: dbt command and flags, e.g. ``("build",)`` or ``("test", "--select", "staging")``.
            Defaults to :data:`DEFAULT_ARGS`.
        settings: Configuration to use. Defaults to the process-wide settings.

    Raises:
        TransformError: If the warehouse has no raw data, or if dbt reports failure.
    """
    # Imported lazily: pulling in dbt costs a couple of seconds, and paying that on `ltv info` or
    # `ltv ingest` -- neither of which touches dbt -- would make the whole CLI feel broken.
    from dbt.cli.main import dbtRunner

    settings = settings or get_settings()
    _require_raw_data(settings)

    # A leading flag means the caller passed options for the default command rather than naming a
    # different one, so `ltv transform --select staging` does what it looks like it does while
    # `ltv transform test` still selects a different dbt command.
    requested = list(args or ())
    if not requested or requested[0].startswith("-"):
        requested = [*DEFAULT_ARGS, *requested]

    project_dir = str(settings.dbt_dir)
    invocation = [
        *requested,
        "--project-dir",
        project_dir,
        # profiles.yml lives beside dbt_project.yml and is committed, so the repo is self-contained
        # and nothing depends on a ~/.dbt directory that only exists on one machine.
        "--profiles-dir",
        project_dir,
        "--vars",
        json.dumps(
            {
                "calibration_weeks": settings.calibration_weeks,
                "holdout_weeks": settings.holdout_weeks,
            }
        ),
    ]

    with _dbt_environment(settings):
        try:
            result = dbtRunner().invoke(invocation)
        finally:
            _release_warehouse()

    if not result.success:
        raise TransformError(
            f"dbt {' '.join(requested)} failed. The dbt output above names the failing node; "
            f"full logs are in {settings.dbt_dir / 'logs'}."
        ) from result.exception

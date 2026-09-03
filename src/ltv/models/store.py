"""Persist fitted models, and write their predictions into the warehouse.

Fits are saved so that Phase 4 scores the model that produced the predictions table rather than
silently refitting a different one. That distinction is not pedantic: a ``--full-bayes`` run samples
a posterior, so a refit gives *different numbers*, and validation metrics computed against a fit
nobody kept are not reproducible.

The artifacts are gitignored. They are build output, rebuilt by ``ltv fit``, and a netCDF of a
posterior over 23,570 customers has no business in version control.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ltv.config import Settings, get_settings
from ltv.warehouse import connect

if TYPE_CHECKING:  # pragma: no cover - avoids importing the model layer to write a row
    from ltv.models.clv import CalibrationFingerprint

#: Schema for anything written by Python rather than dbt. Kept separate from `main` (dbt's models)
#: and `raw` (ingest) so the warehouse itself shows which stage produced a relation.
MODEL_SCHEMA = "model"

PREDICTIONS_TABLE = f"{MODEL_SCHEMA}.customer_predictions"

#: Provenance for the predictions table: what each fit trained on, so a later stage can tell whether
#: the warehouse has moved underneath it. See CalibrationFingerprint for why this is not optional.
FIT_RUNS_TABLE = f"{MODEL_SCHEMA}.fit_runs"

#: Holdout scoring results, written by `ltv validate`. In the warehouse rather than only in the
#: committed markdown report because Phase 5's dashboard has a model-validation page, and a
#: dashboard that re-derives its own error metrics is a second implementation waiting to
#: disagree with the first.
VALIDATION_METRICS_TABLE = f"{MODEL_SCHEMA}.validation_metrics"


def artifact_path(source: str, method: str, settings: Settings | None = None) -> Path:
    """Where one fitted model's InferenceData lives.

    Keyed on the fit method as well as the source, so a fast MAP run cannot overwrite an expensive
    NUTS run that took minutes to produce.
    """
    settings = settings or get_settings()
    return settings.model_dir / f"{source}_{method}.nc"


def save_fit(model, source: str, method: str, settings: Settings | None = None) -> Path:
    """Save a fitted PyMC-Marketing model, returning where it went."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    path = artifact_path(source, method, settings)
    model.save(str(path))
    return path


def _replace_rows_for_sources(connection, table: str, frame: pd.DataFrame) -> int:
    """Replace one table's rows for exactly the sources present in ``frame``.

    Per source rather than a truncate, so refitting CDNOW does not silently delete Online Retail's
    rows once Phase 7 lands. Shared by every Python-written relation in the `model` schema, because
    all three want identical semantics and three copies of this would drift.

    The table is created from the incoming frame's own shape on first write. A frame whose columns
    have since changed will fail the insert rather than half-populate the table -- correct, and the
    fix is to drop the relation, which is a build artifact rebuilt by the command that wrote it.

    Returns:
        The number of rows the table now holds for those sources.
    """
    sources = sorted(frame["source"].unique())
    placeholders = ", ".join("?" for _ in sources)

    # Registered as a view over the DataFrame so the create/insert below reads it as a relation.
    connection.register("incoming_rows", frame)
    try:
        connection.execute(
            f"create table if not exists {table} as select * from incoming_rows where false"
        )
        connection.execute(f"delete from {table} where source in ({placeholders})", sources)
        connection.execute(f"insert into {table} select * from incoming_rows")
    finally:
        connection.unregister("incoming_rows")

    (written,) = connection.execute(
        f"select count(*) from {table} where source in ({placeholders})", sources
    ).fetchone()
    return int(written)


def load_fit(model_class, source: str, method: str, settings: Settings | None = None):
    """Load back the fitted model that produced the predictions table.

    Validation reads the saved fit rather than refitting, which is the reason :func:`save_fit`
    exists. Refitting would be defensible for MAP, which is deterministic, and wrong for
    ``--full-bayes``, which samples: a refit draws a different posterior, so metrics from it
    describe a model that never wrote a prediction anywhere.

    Args:
        model_class: The PyMC-Marketing class to reconstruct, e.g. ``BetaGeoModel``.
        source: Which dataset's fit to load.
        method: ``map_bgnbd``, ``mcmc_gammagamma``, and so on -- the same key ``save_fit`` used.

    Raises:
        FileNotFoundError: If no artifact exists, naming the command that writes one.
    """
    settings = settings or get_settings()
    path = artifact_path(source, method, settings)

    if not path.exists():
        raise FileNotFoundError(
            f"No fitted model at {path}. The artifacts are gitignored build output, so a fresh "
            f"clone has none. Run `uv run ltv fit` first."
        )

    return model_class.load(str(path))


def write_predictions(predictions: pd.DataFrame, settings: Settings | None = None) -> int:
    """Replace the predictions table for the sources present in ``predictions``.

    Returns:
        The number of rows written.
    """
    settings = settings or get_settings()

    with connect(settings) as connection:
        connection.execute(f"create schema if not exists {MODEL_SCHEMA}")
        return _replace_rows_for_sources(connection, PREDICTIONS_TABLE, predictions)


def write_fit_run(
    source: str,
    *,
    method: str,
    horizons: tuple[int, ...],
    holdout_days: int,
    fingerprint: CalibrationFingerprint,
    settings: Settings | None = None,
) -> None:
    """Record what this fit trained on, beside what it produced.

    Written as a separate relation rather than as columns on ``customer_predictions`` so the
    predictions grain stays ``(source, customer_id, horizon_days)`` and nothing about Phase 3's
    output shape changes.

    The horizons are stored as text rather than a list because this row exists to be read by a dbt
    test in plain SQL, and a comma-joined string compares the same way in every engine.
    """
    settings = settings or get_settings()

    run = pd.DataFrame(
        {
            "source": [source],
            "fit_method": [method],
            # The one non-deterministic value here, and deliberately kept out of the committed
            # validation report so that re-running on unchanged data produces an identical file.
            "fitted_at": [datetime.now(UTC).replace(tzinfo=None)],
            "horizons": [",".join(str(horizon) for horizon in horizons)],
            "holdout_days": [holdout_days],
            "calibration_rows": [fingerprint.rows],
            "sum_frequency": [fingerprint.sum_frequency],
            "sum_recency": [fingerprint.sum_recency],
            "sum_customer_age": [fingerprint.sum_customer_age],
            "sum_monetary_value": [fingerprint.sum_monetary_value],
        }
    )

    with connect(settings) as connection:
        connection.execute(f"create schema if not exists {MODEL_SCHEMA}")
        _replace_rows_for_sources(connection, FIT_RUNS_TABLE, run)


def write_validation_metrics(metrics: pd.DataFrame, settings: Settings | None = None) -> int:
    """Replace the validation metrics for the sources present in ``metrics``.

    Long format -- one row per (model family, quantity, metric) -- rather than a column per metric.
    Adding a metric is then a new row rather than a schema migration, and Phase 5's Evidence page
    can pivot whatever subset it wants to show. The alternative, a wide table, would have to be
    altered every time the report gains a line.

    Returns:
        The number of rows written.
    """
    settings = settings or get_settings()

    with connect(settings) as connection:
        connection.execute(f"create schema if not exists {MODEL_SCHEMA}")
        return _replace_rows_for_sources(connection, VALIDATION_METRICS_TABLE, metrics)

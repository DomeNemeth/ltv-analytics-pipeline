"""Persist fitted models, and write their predictions into the warehouse.

Fits are saved so that Phase 4 scores the model that produced the predictions table rather than
silently refitting a different one. That distinction is not pedantic: a ``--full-bayes`` run samples
a posterior, so a refit gives *different numbers*, and validation metrics computed against a fit
nobody kept are not reproducible.

The artifacts are gitignored. They are build output, rebuilt by ``ltv fit``, and a netCDF of a
posterior over 23,570 customers has no business in version control.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ltv.config import Settings, get_settings
from ltv.warehouse import connect

#: Schema for anything written by Python rather than dbt. Kept separate from `main` (dbt's models)
#: and `raw` (ingest) so the warehouse itself shows which stage produced a relation.
MODEL_SCHEMA = "model"

PREDICTIONS_TABLE = f"{MODEL_SCHEMA}.customer_predictions"


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


def write_predictions(predictions: pd.DataFrame, settings: Settings | None = None) -> int:
    """Replace the predictions table for the sources present in ``predictions``.

    Replaces per source rather than truncating the whole table, so fitting CDNOW does not silently
    delete Online Retail's predictions once Phase 7 lands.

    Returns:
        The number of rows written.
    """
    settings = settings or get_settings()
    sources = sorted(predictions["source"].unique())

    with connect(settings) as connection:
        connection.execute(f"create schema if not exists {MODEL_SCHEMA}")

        # Registered as a view over the DataFrame so the create/insert below reads it as a relation.
        connection.register("incoming_predictions", predictions)
        connection.execute(
            f"create table if not exists {PREDICTIONS_TABLE} as "
            f"select * from incoming_predictions where false"
        )
        connection.execute(
            f"delete from {PREDICTIONS_TABLE} where source in ({', '.join('?' for _ in sources)})",
            sources,
        )
        connection.execute(f"insert into {PREDICTIONS_TABLE} select * from incoming_predictions")
        connection.unregister("incoming_predictions")

        (written,) = connection.execute(
            f"select count(*) from {PREDICTIONS_TABLE} where source in "
            f"({', '.join('?' for _ in sources)})",
            sources,
        ).fetchone()

    return int(written)

"""The model-fitting stage: calibration RFM in, predictions table out.

This mirrors :mod:`ltv.transform` -- a stage entry point that takes :class:`~ltv.config.Settings`
and does one thing -- so the Phase 6 Prefect flow wires stages together rather than reaching inside
them. The modelling itself lives in :mod:`ltv.models.clv`; what is here is sequencing and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ltv.config import Settings, get_settings
from ltv.models.clv import (
    CalibrationData,
    check_assumptions,
    fit_models,
    load_calibration,
    predict,
)
from ltv.models.store import save_fit, write_predictions


@dataclass(frozen=True)
class FitResult:
    """What a fit produced, in the terms a reader needs to judge it."""

    source: str
    method: str
    customers: int
    spend_customers: int
    excluded_customers: int
    horizons: tuple[int, ...]
    rows_written: int
    purchase_artifact: Path
    spend_artifact: Path


def resolve_horizons(data: CalibrationData, settings: Settings) -> tuple[int, ...]:
    """The horizons every fit predicts at: the holdout window, and the headline LTV horizon.

    The holdout horizon is read from the warehouse rather than configured, because it must equal the
    window Phase 4 scores against -- exactly, not approximately. A configured value would be free to
    drift away from the split and the comparison would quietly stop measuring anything.

    Deduplicated, so setting ``forward_horizon_days`` to the holdout length yields one horizon
    rather than two identical ones.
    """
    horizons = {data.holdout_days, settings.forward_horizon_days}
    return tuple(sorted(horizons))


def run_fit(
    source: str = "cdnow",
    *,
    full_bayes: bool = False,
    settings: Settings | None = None,
) -> FitResult:
    """Fit the CLV models for one source and write per-customer predictions.

    Args:
        source: Which dataset to fit. One fit per source; they are separate populations.
        full_bayes: Sample the posterior with NUTS rather than taking the MAP estimate. Slower, and
            the only run whose uncertainty intervals mean anything.
        settings: Configuration to use. Defaults to the process-wide settings.

    Raises:
        ModelError: If the calibration data is missing or violates a model assumption.
    """
    settings = settings or get_settings()

    data = load_calibration(source, settings)
    check_assumptions(data)

    models = fit_models(data, full_bayes=full_bayes, settings=settings)
    horizons = resolve_horizons(data, settings)
    predictions = predict(models, data, horizons)

    rows_written = write_predictions(predictions, settings)

    return FitResult(
        source=source,
        method=models.method,
        customers=models.customers,
        spend_customers=models.spend_customers,
        excluded_customers=models.excluded_customers,
        horizons=horizons,
        rows_written=rows_written,
        purchase_artifact=save_fit(
            models.purchase_model, source, f"{models.method}_bgnbd", settings
        ),
        spend_artifact=save_fit(
            models.spend_model, source, f"{models.method}_gammagamma", settings
        ),
    )

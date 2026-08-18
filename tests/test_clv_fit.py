"""Phase 3: the CLV fit reads the right window, renames the right column, and hides no customer.

The expensive part of a fit is the fit, so almost everything here works on a small synthetic
population built in this module rather than on CDNOW. That is not a shortcut: the properties being
asserted -- which relation is read, which column becomes ``T``, who the spend model is fitted on,
what happens to customers it cannot be fitted on -- are all independent of how much data there is,
and a test that needs 58 seconds of optimisation to check a rename is a test nobody will run.

The one thing synthetic data cannot check is that the column names match the real dbt models, so
`test_calibration_columns_exist_in_the_warehouse` does exactly that against a real build.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ltv.config import Settings, get_settings
from ltv.fit import resolve_horizons
from ltv.models.clv import (
    BG_NBD_COLUMNS,
    CALIBRATION_RELATION,
    CalibrationData,
    ModelError,
    check_assumptions,
    fit_models,
    predict,
)
from ltv.models.store import PREDICTIONS_TABLE, write_predictions
from ltv.warehouse import connect

CALIBRATION_COLUMNS = (
    "customer_id",
    "frequency",
    "recency",
    "customer_age",
    "monetary_value",
    "is_gamma_gamma_eligible",
)


def synthetic_customers(n: int = 150, seed: int = 7) -> pd.DataFrame:
    """A population with the three shapes that behave differently in a CLV fit.

    Deliberately includes one-time buyers (frequency 0) and zero-spend repeat buyers, because those
    are the customers Gamma-Gamma cannot be fitted on -- which is the branch most of this module is
    about.
    """
    rng = np.random.default_rng(seed)
    frequency = rng.poisson(2, n)
    customer_age = rng.integers(100, 300, n)
    recency = np.where(frequency > 0, rng.integers(0, 100, n), 0)

    monetary_value = np.where(frequency > 0, rng.gamma(3.0, 10.0, n), 0.0)
    # A handful of repeat buyers whose entire spend was zero -- CDNOW has exactly one, and it is the
    # case that separates "bought more than once" from "has spend to learn from".
    monetary_value[:3] = np.where(frequency[:3] > 0, 0.0, monetary_value[:3])

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1, dtype=np.int32),
            "frequency": frequency,
            "recency": recency,
            "customer_age": customer_age,
            "monetary_value": monetary_value,
            "is_gamma_gamma_eligible": (frequency > 0) & (monetary_value > 0),
        }
    )


def synthetic_data(**overrides) -> CalibrationData:
    customers = synthetic_customers()
    for column, value in overrides.items():
        customers[column] = value
    return CalibrationData(source="synthetic", customers=customers, holdout_days=273)


# --------------------------------------------------------------------------------------------
# The boundary between dbt's column names and PyMC-Marketing's
# --------------------------------------------------------------------------------------------


def test_customer_age_becomes_T_and_recency_is_left_alone() -> None:
    """The rename that would be silent and catastrophic if it went the other way.

    recency and customer_age are both day counts over the same range, so swapping them produces a
    model that fits cleanly, converges, and predicts nonsense. Nothing downstream would complain.
    """
    data = synthetic_data()
    frame = data.purchase_frame

    assert tuple(frame.columns) == BG_NBD_COLUMNS
    assert "customer_age" not in frame.columns
    pd.testing.assert_series_equal(frame["T"], data.customers["customer_age"], check_names=False)
    pd.testing.assert_series_equal(frame["recency"], data.customers["recency"], check_names=False)
    # The two must not be interchangeable in this fixture, or the assertion above proves nothing.
    assert not frame["T"].equals(frame["recency"])


def test_spend_model_sees_only_customers_with_repeat_spend() -> None:
    data = synthetic_data()
    frame = data.spend_frame

    assert len(frame) == int(data.customers["is_gamma_gamma_eligible"].sum())
    assert len(frame) < len(data.customers), "fixture must contain ineligible customers"
    assert (frame["frequency"] > 0).all()
    assert (frame["monetary_value"] > 0).all()


def test_the_fit_reads_the_calibration_window_not_the_full_period() -> None:
    """The leakage guard the dbt layer cannot provide.

    Both relations are valid models and both would fit without complaint. Choosing the full-period
    summary would train on the holdout window and produce error metrics that look outstanding and
    mean nothing -- the single most damaging mistake available in this project.
    """
    assert CALIBRATION_RELATION == "int_customers__rfm_calibration"
    assert "full" not in CALIBRATION_RELATION


# --------------------------------------------------------------------------------------------
# Assumption checks
# --------------------------------------------------------------------------------------------


def test_recency_beyond_customer_age_is_rejected() -> None:
    data = synthetic_data()
    data.customers.loc[0, "recency"] = int(data.customers.loc[0, "customer_age"]) + 1

    with pytest.raises(ModelError, match="recency greater than customer_age"):
        check_assumptions(data)


def test_eligibility_flag_disagreeing_with_the_data_is_rejected() -> None:
    """dbt owns the eligibility definition; this catches the two drifting apart."""
    data = synthetic_data()
    data.customers.loc[0, "is_gamma_gamma_eligible"] = True
    data.customers.loc[0, "frequency"] = 0

    with pytest.raises(ModelError, match="flagged Gamma-Gamma eligible"):
        check_assumptions(data)


def test_negative_frequency_is_rejected() -> None:
    data = synthetic_data()
    data.customers.loc[0, "frequency"] = -1

    with pytest.raises(ModelError, match="negative frequency"):
        check_assumptions(data)


def test_a_population_with_no_repeat_spend_is_rejected() -> None:
    data = synthetic_data(is_gamma_gamma_eligible=False)

    with pytest.raises(ModelError, match="no customer|No customer"):
        check_assumptions(data)


def test_a_clean_population_passes() -> None:
    check_assumptions(synthetic_data())


# --------------------------------------------------------------------------------------------
# Horizons
# --------------------------------------------------------------------------------------------


def test_horizons_take_the_holdout_length_from_the_data() -> None:
    """The scoring horizon is derived, never configured.

    Phase 4 compares predictions against the holdout window. If this horizon could drift away from
    that window by a config change, the comparison would silently stop measuring anything.
    """
    settings = Settings(forward_horizon_days=365)
    horizons = resolve_horizons(synthetic_data(), settings)

    assert horizons == (273, 365)


def test_identical_horizons_collapse_to_one() -> None:
    settings = Settings(forward_horizon_days=273)

    assert resolve_horizons(synthetic_data(), settings) == (273,)


# --------------------------------------------------------------------------------------------
# A real (small) fit
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted() -> tuple:
    """One MAP fit on the synthetic population, reused by every test below.

    Small enough to be part of the fast suite: it needs no source data and no warehouse.
    """
    data = synthetic_data()
    models = fit_models(data, full_bayes=False)
    return models, data, predict(models, data, (273, 365))


def test_every_customer_is_predicted_at_every_horizon(fitted: tuple) -> None:
    """The checkpoint: no customer is dropped for being hard to model."""
    _, data, predictions = fitted

    assert len(predictions) == len(data.customers) * 2
    for horizon in (273, 365):
        at_horizon = predictions[predictions["horizon_days"] == horizon]
        assert set(at_horizon["customer_id"]) == set(data.customers["customer_id"])


def test_customers_without_repeat_spend_get_the_population_estimate(fitted: tuple) -> None:
    """Flagged, not blended, and never NULL -- see _spend_posterior for why."""
    models, data, predictions = fitted

    at_horizon = predictions[predictions["horizon_days"] == 273]
    fitted_individually = at_horizon["spend_estimate_source"] == "conditional"

    assert fitted_individually.sum() == models.spend_customers
    assert (~fitted_individually).sum() == models.excluded_customers
    assert at_horizon["expected_avg_value"].notna().all()

    # The substituted value is one number for everyone in that group, which is what makes it a
    # population estimate rather than a per-customer one.
    substituted = at_horizon.loc[~fitted_individually, "expected_avg_value"]
    assert substituted.nunique() == 1

    # And the flag must agree with dbt's, not merely be internally consistent.
    eligibility = at_horizon.set_index("customer_id")["is_gamma_gamma_eligible"]
    expected = data.customers.set_index("customer_id")["is_gamma_gamma_eligible"]
    pd.testing.assert_series_equal(eligibility, expected, check_names=False)


def test_forward_revenue_is_purchases_times_value(fitted: tuple) -> None:
    _, _, predictions = fitted

    np.testing.assert_allclose(
        predictions["expected_forward_revenue"],
        predictions["expected_purchases"] * predictions["expected_avg_value"],
        rtol=1e-9,
    )


def test_a_longer_horizon_predicts_more_purchases(fitted: tuple) -> None:
    """Sanity on the horizon actually reaching the model rather than being carried as a label."""
    _, _, predictions = fitted

    shorter = predictions[predictions["horizon_days"] == 273].set_index("customer_id")
    longer = predictions[predictions["horizon_days"] == 365].set_index("customer_id")

    assert (longer["expected_purchases"] > shorter["expected_purchases"]).all()


def test_a_map_fit_reports_no_uncertainty_interval(fitted: tuple) -> None:
    """CLAUDE.md section 6: interval claims require a --full-bayes run.

    A MAP fit has one draw, so an interval computed from it would be zero-width and read as
    certainty. NULL plus the fit_method column is the honest answer.
    """
    models, _, predictions = fitted

    assert not models.is_bayesian
    assert predictions["fit_method"].eq("map").all()
    assert predictions["forward_revenue_hdi_low"].isna().all()
    assert predictions["forward_revenue_hdi_high"].isna().all()


def test_probability_alive_is_a_probability(fitted: tuple) -> None:
    _, _, predictions = fitted

    assert predictions["probability_alive"].between(0.0, 1.0).all()


# --------------------------------------------------------------------------------------------
# Writing to the warehouse
# --------------------------------------------------------------------------------------------


def prediction_rows(source: str, customer_ids: range) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": source,
            "customer_id": list(customer_ids),
            "horizon_days": 273,
            "expected_purchases": 1.0,
            "probability_alive": 0.5,
            "expected_avg_value": 10.0,
            "expected_forward_revenue": 10.0,
            "forward_revenue_hdi_low": np.nan,
            "forward_revenue_hdi_high": np.nan,
            "spend_estimate_source": "conditional",
            "is_gamma_gamma_eligible": True,
            "fit_method": "map",
        }
    )


@pytest.fixture()
def warehouse(tmp_path: Path) -> Iterator[Settings]:
    settings = Settings(repo_root=tmp_path)
    settings.ensure_dirs()
    yield settings


def test_refitting_replaces_a_sources_predictions_rather_than_appending(
    warehouse: Settings,
) -> None:
    write_predictions(prediction_rows("cdnow", range(1, 4)), warehouse)
    written = write_predictions(prediction_rows("cdnow", range(1, 4)), warehouse)

    assert written == 3


def test_fitting_one_source_leaves_another_sources_predictions_alone(
    warehouse: Settings,
) -> None:
    """Phase 7 lands a second source. Refitting CDNOW must not delete Online Retail."""
    write_predictions(prediction_rows("online_retail", range(1, 6)), warehouse)
    write_predictions(prediction_rows("cdnow", range(1, 4)), warehouse)

    with connect(warehouse, read_only=True) as connection:
        counts = dict(
            connection.execute(
                f"select source, count(*) from {PREDICTIONS_TABLE} group by source"
            ).fetchall()
        )

    assert counts == {"online_retail": 5, "cdnow": 3}


# --------------------------------------------------------------------------------------------
# Against a real build
# --------------------------------------------------------------------------------------------


@pytest.mark.integration
def test_calibration_columns_exist_in_the_warehouse() -> None:
    """The one thing synthetic data cannot check: that these column names are real.

    Every unit test above builds its own frame, so all of them would keep passing if the dbt layer
    renamed a column tomorrow. This reads the actual relation.
    """
    settings = get_settings()
    if not settings.duckdb_path.exists():
        message = "No warehouse. Run `uv run ltv ingest cdnow && uv run ltv transform` first."
        if os.getenv("CI"):
            pytest.fail(f"{message} CI must not skip the integration tests.")
        pytest.skip(message)

    with connect(settings, read_only=True) as connection:
        columns = {
            row[0]
            for row in connection.execute(
                "select column_name from information_schema.columns where table_name = ?",
                [CALIBRATION_RELATION],
            ).fetchall()
        }

    assert set(CALIBRATION_COLUMNS) <= columns

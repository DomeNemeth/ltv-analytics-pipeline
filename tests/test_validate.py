"""Phase 4: the validation stage scores the right population over the right window.

Most of this runs on a small synthetic scored frame. The properties being asserted -- who is
included in each metric, which horizon is scored, what happens when there is nothing to score -- do
not depend on how much data there is, and a test that needs a 30-second fit to check a population
filter is a test nobody runs.

Two things synthetic data cannot check, so they are integration tests against a real build:
that the column names in ``int_customers__scored`` are real, and that the relation holds the
whole calibration population rather than the subset that happened to return.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ltv.config import Settings, get_settings
from ltv.models.challengers import CHAMPION
from ltv.validate import (
    NO_MAPE,
    SCORED_RELATION,
    ValidationError,
    _require_predictions,
    bucket_label,
    bucket_table,
    independence_check,
    load_scored,
    metrics_frame,
    score_all,
)
from ltv.warehouse import connect

SCORED_COLUMNS = (
    "source",
    "customer_id",
    "frequency",
    "recency",
    "customer_age",
    "monetary_value",
    "is_gamma_gamma_eligible",
    "duration_holdout",
    "horizon_days",
    "holdout_frequency",
    "holdout_spend",
    "holdout_monetary_value",
    "expected_purchases",
    "probability_alive",
    "expected_avg_value",
    "expected_forward_revenue",
    "baseline_purchases",
    "baseline_purchases_flat",
    "baseline_avg_value",
    "baseline_revenue",
    "baseline_revenue_flat",
    "baseline_carry_forward",
    "baseline_revenue_carry_forward",
    "baseline_zero",
    "spend_estimate_source",
    "fit_method",
)


def synthetic_scored(n: int = 40, seed: int = 11) -> pd.DataFrame:
    """A scored frame with the shape that matters: most customers never came back.

    Deliberately lopsided, because that is what a real CLV holdout looks like and it is the property
    every population filter in this module has to cope with. A fixture where everyone purchases
    would let a metric silently average zeros back in and still pass.
    """
    rng = np.random.default_rng(seed)
    frequency = rng.poisson(1.5, n)
    holdout_frequency = np.where(rng.random(n) < 0.3, rng.poisson(2, n) + 1, 0)
    monetary_value = np.where(frequency > 0, rng.gamma(3.0, 10.0, n), 0.0)
    holdout_spend = holdout_frequency * rng.gamma(3.0, 10.0, n)

    return pd.DataFrame(
        {
            "source": "synthetic",
            "customer_id": np.arange(1, n + 1, dtype=np.int32),
            "frequency": frequency,
            "recency": rng.integers(0, 100, n),
            "customer_age": rng.integers(150, 300, n),
            "monetary_value": monetary_value,
            "is_gamma_gamma_eligible": (frequency > 0) & (monetary_value > 0),
            "duration_holdout": 273,
            "horizon_days": 273,
            "holdout_frequency": holdout_frequency,
            "holdout_spend": holdout_spend,
            "holdout_monetary_value": np.where(
                holdout_frequency > 0, holdout_spend / np.maximum(holdout_frequency, 1), 0.0
            ),
            "expected_purchases": rng.gamma(2.0, 0.4, n),
            "probability_alive": rng.random(n),
            "expected_avg_value": rng.gamma(3.0, 10.0, n),
            "expected_forward_revenue": rng.gamma(2.0, 20.0, n),
            "baseline_purchases": rng.gamma(2.0, 0.4, n),
            "baseline_purchases_flat": 0.75,
            "baseline_avg_value": rng.gamma(3.0, 10.0, n),
            "baseline_revenue": rng.gamma(2.0, 20.0, n),
            "baseline_revenue_flat": 26.0,
            # The same-period rule: no rescaling, because both windows are the same length. It is
            # the honest comparand -- the rate baselines above inflate by holdout/customer_age.
            "baseline_carry_forward": frequency.astype(float),
            "baseline_revenue_carry_forward": frequency * monetary_value,
            # The MAE floor on a mostly-zero quantity.
            "baseline_zero": 0.0,
            "spend_estimate_source": "conditional",
            "fit_method": "map",
        }
    )


# --------------------------------------------------------------------------------------------
# Which population each metric covers
# --------------------------------------------------------------------------------------------


def test_purchase_and_revenue_metrics_cover_every_customer() -> None:
    """Including the ones who never came back, who are the majority and the hardest to predict.

    Scoring only the returners is the single most flattering mistake available in this phase: it
    removes precisely the customers the model gets wrong. The dbt layer guards the join; this guards
    the filter on the Python side of it.
    """
    frame = synthetic_scored()
    scores = score_all(frame)

    for item in scores:
        if item.quantity in ("purchases", "revenue"):
            assert item.population == len(frame)


def test_average_order_value_is_scored_only_on_customers_who_purchased() -> None:
    """A holdout order value of zero means "did not buy", not "bought for nothing"."""
    frame = synthetic_scored()
    returners = int((frame["holdout_frequency"] > 0).sum())
    assert 0 < returners < len(frame), "fixture must contain both returners and non-returners"

    for item in score_all(frame):
        if item.quantity == "avg_order_value":
            assert item.population == returners


def test_purchase_counts_do_not_report_a_mape() -> None:
    """Most actuals are zero, so a percentage error would describe a minority of the population."""
    assert "purchases" in NO_MAPE

    for item in score_all(synthetic_scored()):
        if item.quantity == "purchases":
            assert item.mape is None
        else:
            assert item.mape is not None


def test_every_quantity_is_scored_against_at_least_one_naive_baseline() -> None:
    """An error metric with nothing to beat is not a result. CLAUDE.md section 6."""
    scores = score_all(synthetic_scored())

    for quantity in {item.quantity for item in scores}:
        families = {item.family for item in scores if item.quantity == quantity}
        assert CHAMPION in families
        assert families & {"baseline_carry_forward", "baseline_rate", "baseline_flat"}, quantity


def test_the_all_zero_floor_is_scored_so_mae_cannot_be_read_as_accuracy() -> None:
    """On a mostly-zero quantity, the all-zero rule is what MAE is really measured against.

    A validation audit found the report claiming the model "beats both naive rules" on MAE while an
    all-zero predictor scored within 2% of it. Both statements were true; together they mean
    something quite different from the first alone. The floor has to be in the table, and it has to
    be scored by the same code as everything else, or the claim comes back.
    """
    frame = synthetic_scored()
    scores = score_all(frame)

    zero = next(s for s in scores if s.family == "baseline_zero" and s.quantity == "purchases")

    assert zero.aggregate.predicted_total == 0.0
    # MAE against an all-zero prediction is the mean actual, by definition. If it is not, the
    # column is not what it claims to be.
    assert zero.mae == pytest.approx(frame["holdout_frequency"].mean())


def test_the_same_period_baseline_is_not_rescaled() -> None:
    """The fair comparand: both windows are the same length, so no scaling belongs in it.

    `baseline_purchases` divides by each customer's observation length and multiplies by the holdout
    length, which inflates it whenever mean customer age is below the holdout length -- on CDNOW by
    a factor of 1.19, which was worth roughly half the model's apparent margin. This one must stay
    equal to the raw calibration count.
    """
    frame = synthetic_scored()

    np.testing.assert_allclose(frame["baseline_carry_forward"], frame["frequency"].astype(float))

    scores = score_all(frame)
    carry = next(
        s for s in scores if s.family == "baseline_carry_forward" and s.quantity == "purchases"
    )
    assert carry.aggregate.predicted_total == pytest.approx(frame["frequency"].sum())


def test_challenger_predictions_are_scored_on_counts_only() -> None:
    """Pairing a challenger's purchase model with the champion's spend model misattributes it."""
    frame = synthetic_scored()
    frame["pareto_nbd"] = frame["expected_purchases"] * 1.1

    scores = score_all(frame, {"pareto_nbd": "pareto_nbd"})
    quantities = {item.quantity for item in scores if item.family == "pareto_nbd"}

    assert quantities == {"purchases"}


# --------------------------------------------------------------------------------------------
# Bucketing
# --------------------------------------------------------------------------------------------


def test_frequency_buckets_are_open_ended_at_the_top() -> None:
    """Matching the published CDNOW breakdown of 0, 1, ..., 7+ so the tables read together."""
    assert [bucket_label(f) for f in (0, 1, 6)] == ["0", "1", "6"]
    assert bucket_label(7) == "7+"
    assert bucket_label(40) == "7+"


def test_bucket_table_is_ordered_numerically_with_the_open_bucket_last() -> None:
    table = bucket_table(synthetic_scored())
    labels = table["bucket"].tolist()

    assert labels == sorted(labels, key=lambda b: 99 if b == "7+" else int(b))
    assert labels[-1] == "7+" or labels[-1] == max(labels, key=int)


def test_bucket_table_accounts_for_every_customer() -> None:
    """A bucketing that drops rows would understate the shortfall it exists to locate."""
    frame = synthetic_scored()

    assert bucket_table(frame)["customers"].sum() == len(frame)


def test_bucket_shortfall_sums_to_the_aggregate_miss() -> None:
    """The decomposition must reconcile with the total, or it is describing a different error."""
    frame = synthetic_scored()
    table = bucket_table(frame)
    aggregate = frame["expected_purchases"].sum() - frame["holdout_frequency"].sum()

    assert table["shortfall"].sum() == pytest.approx(aggregate)


# --------------------------------------------------------------------------------------------
# Assumption checks
# --------------------------------------------------------------------------------------------


def test_independence_is_measured_only_on_gamma_gamma_eligible_customers() -> None:
    """monetary_value is a structural zero for everyone else, not a small value."""
    frame = synthetic_scored()
    eligible = int(frame["is_gamma_gamma_eligible"].sum())
    assert 0 < eligible < len(frame), "fixture must contain ineligible customers"

    assert independence_check(frame).correlation.n == eligible


def test_independence_detects_a_relationship_that_is_there() -> None:
    """The positive control: if the check cannot see a planted violation, it guards nothing."""
    frame = synthetic_scored()
    frame["is_gamma_gamma_eligible"] = True
    frame["monetary_value"] = frame["frequency"] * 10.0 + 1.0

    result = independence_check(frame)

    assert result.correlation.rho > 0.9
    assert not result.holds


# --------------------------------------------------------------------------------------------
# Output shape
# --------------------------------------------------------------------------------------------


def test_metrics_frame_is_long_and_carries_its_population() -> None:
    """Long format so Phase 5 can add a metric without a schema migration."""
    frame = metrics_frame("synthetic", score_all(synthetic_scored()))

    assert set(frame.columns) == {
        "source",
        "model_family",
        "quantity",
        "metric",
        "value",
        "population",
    }
    assert (frame["source"] == "synthetic").all()
    assert "percent_error" in set(frame["metric"])
    # Every row must say how many customers it describes, or a reader cannot tell that the
    # average-order-value rows cover a different population from the purchase-count rows.
    assert frame["population"].notna().all()
    assert frame[frame["quantity"] == "avg_order_value"]["population"].nunique() == 1


# --------------------------------------------------------------------------------------------
# Preconditions
# --------------------------------------------------------------------------------------------


def test_a_warehouse_with_no_fit_is_refused_by_name(tmp_path) -> None:
    """The message must name `ltv fit`, not a missing relation.

    Without this the dbt build fails inside a compiled model, reporting that
    model.customer_predictions does not exist -- true, and it names the symptom rather than the step
    the reader skipped.
    """
    settings = Settings(repo_root=tmp_path)
    with connect(settings):
        pass

    with pytest.raises(ValidationError, match="ltv fit"):
        _require_predictions(settings)


def test_predictions_without_provenance_are_refused(tmp_path) -> None:
    """Half the contract is not the contract: scoring needs to know what the fit trained on."""
    settings = Settings(repo_root=tmp_path)
    with connect(settings) as connection:
        connection.execute("create schema model")
        connection.execute("create table model.customer_predictions (source varchar)")

    with pytest.raises(ValidationError, match="ltv fit"):
        _require_predictions(settings)


# --------------------------------------------------------------------------------------------
# Against a real build
# --------------------------------------------------------------------------------------------


def _skip_without_warehouse(settings: Settings) -> None:
    if settings.duckdb_path.exists():
        return
    message = "No warehouse. Run `uv run ltv ingest cdnow && uv run ltv transform` first."
    if os.getenv("CI"):
        pytest.fail(f"{message} CI must not skip the integration tests.")
    pytest.skip(message)


@pytest.mark.integration
def test_scored_columns_exist_in_the_warehouse() -> None:
    """The one thing synthetic data cannot check: that these column names are real.

    Every unit test above builds its own frame, so all of them would keep passing if the dbt model
    renamed a column tomorrow.
    """
    settings = get_settings()
    _skip_without_warehouse(settings)

    with connect(settings, read_only=True) as connection:
        columns = {
            row[0]
            for row in connection.execute(
                "select column_name from information_schema.columns where table_name = ?",
                [SCORED_RELATION],
            ).fetchall()
        }

    if not columns:
        pytest.skip(f"{SCORED_RELATION} not built. Run `uv run ltv validate` first.")

    assert set(SCORED_COLUMNS) <= columns


@pytest.mark.integration
def test_the_scored_relation_holds_the_whole_calibration_population() -> None:
    """Guarded in dbt too, and asserted here because it is the defect that flatters every metric.

    The equivalent inner join in int_customers__holdout_actuals dropped CDNOW's population from
    23,570 to 7,058 and passed all 62 tests at the time.
    """
    settings = get_settings()
    _skip_without_warehouse(settings)

    with connect(settings, read_only=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "select table_name from information_schema.tables"
            ).fetchall()
        }
        if SCORED_RELATION not in tables:
            pytest.skip(f"{SCORED_RELATION} not built. Run `uv run ltv validate` first.")

        scored, calibration = connection.execute(
            f"""
            select
                (select count(*) from {SCORED_RELATION}),
                (select count(*) from int_customers__rfm_calibration)
            """
        ).fetchone()

    assert scored == calibration


@pytest.mark.integration
def test_loading_an_unknown_source_fails_rather_than_returning_nothing() -> None:
    settings = get_settings()
    _skip_without_warehouse(settings)

    with pytest.raises((ValidationError, Exception)):
        load_scored("no_such_source", settings)

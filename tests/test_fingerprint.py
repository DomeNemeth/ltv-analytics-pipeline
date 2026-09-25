"""The fit's fingerprints must change under the two edits the Phase 4 audit made undetected.

Rotating features between customers left every plain sum unchanged. Scaling the predictions after
the fit was invisible because nothing recorded them. Each test reproduces one of those edits and
asserts the fingerprint moves. Each also asserts the edit leaves the old fingerprint untouched,
since that is what made the edit invisible to the previous guard.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ltv.models.clv import FINGERPRINTED_PREDICTIONS, CalibrationData, PredictionFingerprint


def _calibration(n: int = 50) -> CalibrationData:
    rng = np.random.default_rng(3)
    frequency = rng.poisson(1.5, n)
    return CalibrationData(
        source="synthetic",
        customers=pd.DataFrame(
            {
                "customer_id": np.arange(1, n + 1),
                "frequency": frequency,
                "recency": rng.integers(0, 200, n),
                "customer_age": rng.integers(200, 273, n),
                "monetary_value": np.where(frequency > 0, rng.gamma(3.0, 10.0, n), 0.0),
            }
        ),
        holdout_days=273,
    )


def test_rotating_features_between_customers_moves_the_weighted_sums() -> None:
    data = _calibration()
    rotated = data.customers.copy()
    for column in ("frequency", "recency", "customer_age", "monetary_value"):
        rotated[column] = np.roll(rotated[column].to_numpy(), 1)
    moved = CalibrationData(data.source, rotated, data.holdout_days).fingerprint
    before = data.fingerprint

    # Premise: the plain sums cannot see it. This is what the audit exploited.
    assert moved.sum_frequency == before.sum_frequency
    assert moved.sum_customer_age == before.sum_customer_age

    assert moved.weighted_frequency != before.weighted_frequency
    assert moved.weighted_recency != before.weighted_recency
    assert moved.weighted_customer_age != before.weighted_customer_age
    assert moved.weighted_monetary_value_e4 != before.weighted_monetary_value_e4


def test_the_weights_do_not_depend_on_row_order() -> None:
    """dbt ranks by customer_id, so Python must agree however the frame happens to be sorted."""
    data = _calibration()
    shuffled = data.customers.sample(frac=1.0, random_state=0)
    before = data.fingerprint
    after = CalibrationData(data.source, shuffled, data.holdout_days).fingerprint

    assert after.weighted_frequency == before.weighted_frequency
    assert after.weighted_recency == before.weighted_recency
    assert after.weighted_customer_age == before.weighted_customer_age
    # Exact, not approximate: the monetary sum is integer ten-thousandths, so row order cannot
    # change it even in the last bit.
    assert after.weighted_monetary_value_e4 == before.weighted_monetary_value_e4


def test_the_smallest_real_spend_change_moves_the_monetary_fingerprint() -> None:
    """0.0001 on one customer, the precision monetary_value is stored at.

    The first weighted guard compared with a 0.01 tolerance and would have passed this.
    """
    data = _calibration()
    changed = data.customers.copy()
    target = changed.index[changed["monetary_value"] > 0][0]
    changed.loc[target, "monetary_value"] += 0.0001

    before = data.fingerprint
    after = CalibrationData(data.source, changed, data.holdout_days).fingerprint

    assert after.weighted_monetary_value_e4 != before.weighted_monetary_value_e4


def _predictions() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    ids = np.arange(1, 21)
    rows = 2 * len(ids)
    return pd.DataFrame(
        {
            "customer_id": np.concatenate([ids, ids]),
            "horizon_days": np.repeat([273, 365], len(ids)),
            "expected_purchases": rng.gamma(2.0, 0.5, rows),
            "expected_forward_revenue": rng.gamma(2.0, 20.0, rows),
            "expected_avg_value": rng.gamma(3.0, 10.0, rows),
            "probability_alive": rng.random(rows),
        }
    )


def _moved(before: PredictionFingerprint, after: PredictionFingerprint) -> set[str]:
    return {
        name
        for name, value in before.values.items()
        if not np.isclose(after.values[name], value, rtol=1e-12, atol=0.0)
    }


def test_scaling_the_predictions_moves_the_fingerprint() -> None:
    predictions = _predictions()
    scaled = predictions.assign(
        expected_purchases=predictions["expected_purchases"] * 1.167,
        expected_forward_revenue=predictions["expected_forward_revenue"] * 1.167,
    )

    moved = _moved(PredictionFingerprint.of(predictions), PredictionFingerprint.of(scaled))

    assert {"sum_expected_purchases", "sum_expected_forward_revenue"} <= moved


def test_reassigning_revenue_between_customers_moves_only_the_rank_weighted_sum() -> None:
    """The re-audit's G5: revenue rotated across customers passed every guard."""
    predictions = _predictions()
    by_horizon = predictions.groupby("horizon_days")["expected_forward_revenue"]
    rotated = predictions.assign(
        expected_forward_revenue=by_horizon.transform(lambda v: np.roll(v.to_numpy(), 1))
    )

    moved = _moved(PredictionFingerprint.of(predictions), PredictionFingerprint.of(rotated))

    assert moved == {"weighted_expected_forward_revenue"}


def test_swapping_horizon_labels_moves_only_the_horizon_weighted_sums() -> None:
    """The re-audit's G4: swapping 273 and 365 made the model look 8.8% high, and nothing failed."""
    predictions = _predictions()
    swapped = predictions.assign(horizon_days=predictions["horizon_days"].map({273: 365, 365: 273}))

    # Premise: a label swap is only visible if the two horizons hold different totals, which is
    # true of real forecasts (365 days predicts more than 273) and of this fixture.
    totals = predictions.groupby("horizon_days")["expected_purchases"].sum()
    assert not np.isclose(totals.loc[273], totals.loc[365])

    moved = _moved(PredictionFingerprint.of(predictions), PredictionFingerprint.of(swapped))

    assert "horizon_weighted_expected_purchases" in moved
    assert all(name.startswith("horizon_weighted_") for name in moved)


def test_overwriting_probability_alive_or_order_value_moves_the_fingerprint() -> None:
    """The re-audit's G6: both were edited on a copy and every guard stayed green."""
    predictions = _predictions()
    edited = predictions.assign(
        probability_alive=1.0, expected_avg_value=predictions["expected_avg_value"] * 0.9
    )

    moved = _moved(PredictionFingerprint.of(predictions), PredictionFingerprint.of(edited))

    assert {"sum_probability_alive", "sum_expected_avg_value"} <= moved


def test_the_dbt_guard_checks_the_same_columns_the_fit_records() -> None:
    """dbt cannot import the Python list, so the two copies are compared here instead."""
    sql = (
        Path(__file__).resolve().parents[1]
        / "dbt"
        / "tests"
        / "assert_predictions_are_the_ones_the_fit_wrote.sql"
    ).read_text(encoding="utf-8")
    block = re.search(r"\{% set columns = \[(.*?)\] %\}", sql, re.DOTALL)
    assert block, "the dbt test no longer declares its column list where this test expects it"

    assert tuple(re.findall(r"'(\w+)'", block.group(1))) == FINGERPRINTED_PREDICTIONS

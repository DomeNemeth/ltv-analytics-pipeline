"""The fit's fingerprints must change under the two edits the Phase 4 audit made undetected.

Rotating features between customers left every plain sum unchanged. Scaling the predictions after
the fit was invisible because nothing recorded them. Each test reproduces one of those edits and
asserts the fingerprint moves. Each also asserts the edit leaves the old fingerprint untouched,
since that is what made the edit invisible to the previous guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ltv.models.clv import CalibrationData, PredictionFingerprint


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
    assert moved.weighted_monetary_value != before.weighted_monetary_value


def test_the_weights_do_not_depend_on_row_order() -> None:
    """dbt ranks by customer_id, so Python must agree however the frame happens to be sorted."""
    data = _calibration()
    shuffled = data.customers.sample(frac=1.0, random_state=0)
    before = data.fingerprint
    after = CalibrationData(data.source, shuffled, data.holdout_days).fingerprint

    assert after.weighted_frequency == before.weighted_frequency
    assert after.weighted_recency == before.weighted_recency
    assert after.weighted_customer_age == before.weighted_customer_age
    # Float sums in a different order differ in the last bits, which is why the dbt guard has a
    # tolerance. Anything beyond that means the weights followed row position, not customer_id.
    assert np.isclose(after.weighted_monetary_value, before.weighted_monetary_value, rtol=1e-12)


def _predictions() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    ids = np.arange(1, 21)
    return pd.DataFrame(
        {
            "customer_id": np.concatenate([ids, ids]),
            "horizon_days": np.repeat([273, 365], len(ids)),
            "expected_purchases": rng.gamma(2.0, 0.5, 2 * len(ids)),
            "expected_forward_revenue": rng.gamma(2.0, 20.0, 2 * len(ids)),
        }
    )


def test_scaling_the_predictions_moves_the_fingerprint() -> None:
    predictions = _predictions()
    scaled = predictions.assign(
        expected_purchases=predictions["expected_purchases"] * 1.167,
        expected_forward_revenue=predictions["expected_forward_revenue"] * 1.167,
    )

    before, after = PredictionFingerprint.of(predictions), PredictionFingerprint.of(scaled)

    assert after.sum_expected_purchases != before.sum_expected_purchases
    assert after.sum_expected_forward_revenue != before.sum_expected_forward_revenue


def test_reassigning_predictions_between_customers_moves_the_weighted_sum() -> None:
    predictions = _predictions()
    swapped = predictions.assign(
        expected_purchases=np.roll(predictions["expected_purchases"].to_numpy(), 1)
    )

    before, after = PredictionFingerprint.of(predictions), PredictionFingerprint.of(swapped)

    assert np.isclose(after.sum_expected_purchases, before.sum_expected_purchases)
    assert after.weighted_expected_purchases != before.weighted_expected_purchases

"""Phase 4: the metric functions compute what they claim to.

Every expected value here is worked out by hand in the test itself rather than taken from a library
or from a previous run of the code under test. That is the only way this file has any value: a wrong
metric does not raise, it returns a plausible number, and a test that compares the implementation
against itself would pass on every one of them.

The arithmetic is deliberately small enough to check mentally. Four customers, integers, and the
expected value written out as the sum it is.
"""

from __future__ import annotations

import numpy as np
import pytest

from ltv.models.metrics import (
    DECILES,
    aggregate_error,
    decile_table,
    mae,
    mape,
    rmse,
    spearman,
)

# actual    0  1  2  3
# predicted 1  1  4  0
# error     1  0  2 -3   -> absolute 1, 0, 2, 3
ACTUAL = [0, 1, 2, 3]
PREDICTED = [1, 1, 4, 0]


def test_mae_is_the_mean_absolute_error() -> None:
    assert mae(ACTUAL, PREDICTED) == pytest.approx((1 + 0 + 2 + 3) / 4)


def test_rmse_punishes_the_large_miss_harder_than_mae() -> None:
    expected = np.sqrt((1**2 + 0**2 + 2**2 + 3**2) / 4)

    assert rmse(ACTUAL, PREDICTED) == pytest.approx(expected)
    assert rmse(ACTUAL, PREDICTED) > mae(ACTUAL, PREDICTED)


def test_aggregate_error_is_signed_and_can_cancel() -> None:
    """Totals can agree exactly while every customer in them is wrong.

    This fixture is that case: the totals are both 6, so the aggregate error is zero even though the
    per-customer MAE is 1.5. It is the reason the report prints both -- a revenue forecast that
    totals correctly may still be useless for choosing which customers to target.
    """
    result = aggregate_error(ACTUAL, PREDICTED)

    assert result.actual_total == 6.0
    assert result.predicted_total == 6.0
    assert result.error == 0.0
    assert result.percent_error == pytest.approx(0.0)
    assert mae(ACTUAL, PREDICTED) > 0


def test_aggregate_error_keeps_the_direction_of_the_miss() -> None:
    under = aggregate_error([10, 10], [8, 8])
    over = aggregate_error([10, 10], [12, 12])

    assert under.percent_error == pytest.approx(-20.0)
    assert over.percent_error == pytest.approx(+20.0)


def test_aggregate_percent_error_is_undefined_against_a_zero_total() -> None:
    assert np.isnan(aggregate_error([0, 0], [1, 2]).percent_error)


def test_mape_excludes_zero_actuals_and_says_how_many() -> None:
    """The guard that keeps a MAPE from describing a different population than the reader assumes.

    Only three of the four rows have a defined percentage error: |1-1|/1 = 0, |4-2|/2 = 1, and
    |0-3|/3 = 1, so the mean is 2/3. Silently dropping the fourth would report 66.7% for a
    population of four, when it is 66.7% for a population of three.
    """
    result = mape(ACTUAL, PREDICTED)

    assert result.value == pytest.approx(100 * (0 + 1 + 1) / 3)
    assert result.included == 3
    assert result.excluded == 1
    assert result.total == 4
    assert result.excluded_share == pytest.approx(0.25)


def test_mape_is_undefined_when_every_actual_is_zero() -> None:
    result = mape([0, 0, 0], [1, 2, 3])

    assert np.isnan(result.value)
    assert result.included == 0
    assert result.excluded == 3


def test_spearman_recovers_a_monotonic_relationship() -> None:
    """Rank correlation, so a non-linear but strictly increasing relation is still 1.0."""
    result = spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])

    assert result.rho == pytest.approx(1.0)
    assert result.n == 5


def test_spearman_reports_a_reversed_ranking_as_negative() -> None:
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]).rho == pytest.approx(-1.0)


def test_spearman_on_a_constant_predictor_is_undefined_not_zero() -> None:
    """The population-mean baseline predicts one number for everyone, which is the whole point.

    A constant has no ranking, so there is nothing to correlate. NaN says "this predictor cannot
    discriminate"; a zero would say "it discriminates, and gets it exactly as wrong as chance",
    which is a different and false claim.
    """
    result = spearman([1, 2, 3, 4], [7, 7, 7, 7])

    assert np.isnan(result.rho)
    assert np.isnan(result.p_value)
    assert result.n == 4


def test_decile_table_splits_into_ten_equal_groups_ranked_by_prediction() -> None:
    predicted = list(range(20))  # 0..19, so decile 1 must hold the largest two
    actual = [value * 2 for value in predicted]

    table = decile_table(actual, predicted, keys=list(range(20)))

    assert len(table) == DECILES
    assert (table["customers"] == 2).all()
    assert table["decile"].tolist() == list(range(1, DECILES + 1))
    # Decile 1 is the highest-predicted pair, 18 and 19, whose actuals are 36 and 38.
    assert table.loc[0, "mean_predicted"] == pytest.approx(18.5)
    assert table.loc[0, "mean_actual"] == pytest.approx(37.0)
    # And the shares must account for all of the realised total, or the table misleads by omission.
    assert table["share_of_actual"].sum() == pytest.approx(1.0)
    assert table["total_actual"].sum() == pytest.approx(sum(actual))


def test_decile_table_is_stable_when_predictions_tie() -> None:
    """14,120 CDNOW customers share one population spend estimate, so ties are the normal case.

    Without a deterministic tiebreaker the committed report changes between runs on identical data,
    and a report that churns cannot be diffed to see whether anything actually moved.
    """
    predicted = [1.0] * 20
    actual = list(range(20))

    first = decile_table(actual, predicted, keys=list(range(20)))
    second = decile_table(actual, predicted, keys=list(range(20)))

    assert (first["customers"] == 2).all(), "ties must not pile into one bucket"
    assert first.equals(second)


def test_a_predictor_with_no_discrimination_produces_a_flat_decile_table() -> None:
    """The negative control: this is what "the ranking does not work" looks like."""
    actual = [5, 1, 4, 2, 3, 0, 5, 1, 4, 2]

    table = decile_table(actual, [1.0] * 10, keys=list(range(10)))

    assert table["mean_predicted"].nunique() == 1


def test_alpha_converts_towards_the_published_units() -> None:
    """Pin which way the day/week conversion runs.

    `alpha` is the rate parameter of a gamma over a per-unit-time rate, so it carries the time unit.
    This project fits in days and the published CDNOW figures are in weeks, so the conversion
    divides. Getting the direction backwards produces a number 49 times off that still looks like a
    plausible parameter, and the only other thing describing the direction is a docstring -- which
    is not a guard, and was itself written backwards until an audit caught it.
    """
    from ltv.models.benchmarks import CDNOW_BG_NBD, alpha_in_weeks

    fitted_in_days = 33.011

    assert alpha_in_weeks(fitted_in_days) == pytest.approx(4.716, abs=0.001)
    assert alpha_in_weeks(fitted_in_days) < fitted_in_days
    # And the whole point of the conversion: it lands near the published value. Multiplying instead
    # would give 231, which this bound rejects.
    assert alpha_in_weeks(fitted_in_days) == pytest.approx(
        CDNOW_BG_NBD.parameters["alpha"], rel=0.1
    )


def test_mismatched_lengths_are_rejected_rather_than_broadcast() -> None:
    """A length mismatch means the join upstream dropped rows, which must not be averaged over."""
    with pytest.raises(ValueError, match="same shape"):
        mae([1, 2, 3], [1, 2])


def test_an_empty_population_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty population"):
        mae([], [])

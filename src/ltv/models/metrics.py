"""Error metrics for holdout scoring.

Pure functions over arrays: no warehouse, no configuration, no I/O. That is what makes them
checkable against numbers worked out by hand, which is the only way to know a metric function is
right -- a wrong one produces a plausible number rather than an error, and every downstream
conclusion inherits it.

Two choices here are deliberate and are the reason this module exists at all rather than a handful
of one-line calls to a library:

* **MAPE reports what it excluded.** 16,512 of CDNOW's 23,570 customers make no holdout purchase, so
  a percentage error against their actual of zero is undefined. Silently dropping them turns "the
  model is wrong about 70% of the population" into a confident number computed from the other 30%.
  :func:`mape` therefore returns the count it dropped alongside the value, and every caller is
  obliged to carry it into the report.
* **Ranking is measured separately from error.** The project's business question is which segments
  are worth acquiring, which is a question about *order*, not about absolute error. A model can have
  excellent aggregate error and rank customers no better than chance, and the reverse. Reporting
  only one of the two is how a CLV model gets described as "accurate" without anyone establishing
  that it is useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Deciles in a decile table. Named because it appears in both the split and the label arithmetic,
#: and the two silently disagreeing would produce a table that looks fine and does not sum.
DECILES = 10


@dataclass(frozen=True)
class AggregateError:
    """Totals and the signed error between them.

    Signed, not absolute: the direction is the finding. This project's BG/NBD fit under-predicts by
    14% while both naive baselines over-predict by nearly 50%, and an absolute error would report
    those as the same kind of miss.
    """

    actual_total: float
    predicted_total: float

    @property
    def error(self) -> float:
        return self.predicted_total - self.actual_total

    @property
    def percent_error(self) -> float:
        """Signed error as a percentage of the actual total. NaN when the actual total is zero."""
        if self.actual_total == 0:
            return float("nan")
        return 100.0 * self.error / self.actual_total


@dataclass(frozen=True)
class Mape:
    """A mean absolute percentage error, and the rows it could not be computed on."""

    value: float
    included: int
    excluded: int

    @property
    def total(self) -> int:
        return self.included + self.excluded

    @property
    def excluded_share(self) -> float:
        """What fraction of the population this number does not describe."""
        return self.excluded / self.total if self.total else float("nan")


@dataclass(frozen=True)
class Correlation:
    """A rank correlation, with the sample size it was computed on."""

    rho: float
    p_value: float
    n: int


def _paired(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a pair of array-likes to float arrays, refusing mismatched lengths.

    A length mismatch here would otherwise broadcast or truncate rather than raise, and a metric
    computed over the wrong pairing is exactly as plausible as one computed over the right one.
    """
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    if actual_values.shape != predicted_values.shape:
        raise ValueError(
            f"actual and predicted must have the same shape, got {actual_values.shape} and "
            f"{predicted_values.shape}. These are per-customer arrays; a mismatch means the join "
            f"that produced them dropped rows."
        )
    if actual_values.size == 0:
        raise ValueError("Cannot compute a metric over an empty population.")

    return actual_values, predicted_values


def mae(actual, predicted) -> float:
    """Mean absolute error, per customer."""
    actual_values, predicted_values = _paired(actual, predicted)
    return float(np.abs(predicted_values - actual_values).mean())


def rmse(actual, predicted) -> float:
    """Root mean squared error, per customer. Punishes large individual misses harder than MAE."""
    actual_values, predicted_values = _paired(actual, predicted)
    return float(np.sqrt(np.square(predicted_values - actual_values).mean()))


def aggregate_error(actual, predicted) -> AggregateError:
    """Totals across the population, and the signed gap between them.

    Reported alongside the per-customer errors rather than instead of them: aggregate error is what
    a revenue forecast cares about, and individual errors of opposite sign cancel in it. A model can
    total correctly while being wrong about every customer in it.
    """
    actual_values, predicted_values = _paired(actual, predicted)
    return AggregateError(
        actual_total=float(actual_values.sum()),
        predicted_total=float(predicted_values.sum()),
    )


def mape(actual, predicted) -> Mape:
    """Mean absolute percentage error over the rows where it is defined.

    Rows whose actual is zero are excluded, because the percentage error against zero is undefined
    rather than large. The count of them is returned, not swallowed: on a CLV holdout the excluded
    rows are the customers who did not come back, which is both the majority of the population and
    the part the model finds hardest. A MAPE quoted without that count describes a different, easier
    population than the one the reader assumes.
    """
    actual_values, predicted_values = _paired(actual, predicted)

    defined = actual_values != 0
    included = int(defined.sum())
    if not included:
        return Mape(value=float("nan"), included=0, excluded=int(actual_values.size))

    errors = np.abs((predicted_values[defined] - actual_values[defined]) / actual_values[defined])
    return Mape(
        value=float(100.0 * errors.mean()),
        included=included,
        excluded=int(actual_values.size - included),
    )


def spearman(x, y) -> Correlation:
    """Spearman rank correlation between two arrays.

    Rank-based rather than Pearson because both the quantities this is used on -- predicted against
    actual revenue, and frequency against monetary value -- are heavily skewed, and a Pearson
    coefficient on them mostly reports what the largest few customers did.
    """
    from scipy.stats import spearmanr

    x_values, y_values = _paired(x, y)

    # A constant array has no ranking, so the correlation is undefined rather than zero. This is a
    # real case here, not a defensive check: the population-mean baseline predicts the same number
    # for every customer, which is exactly what makes it the floor. Returning NaN explicitly says
    # "this predictor cannot discriminate", where scipy would emit a ConstantInputWarning to stderr
    # and hand back a NaN that looks like a bug.
    if x_values.min() == x_values.max() or y_values.min() == y_values.max():
        return Correlation(rho=float("nan"), p_value=float("nan"), n=x_values.size)

    result = spearmanr(x_values, y_values)
    return Correlation(rho=float(result.statistic), p_value=float(result.pvalue), n=x_values.size)


def decile_table(actual, predicted, keys) -> pd.DataFrame:
    """Rank customers by prediction, then report what each tenth actually did.

    This is the discrimination view: does the model put the valuable customers at the top? It is the
    question the project's stated purpose -- which segments are worth acquiring -- actually asks,
    and it is not answered by an error metric. A model that predicts every customer's revenue as the
    population mean has a respectable MAE and a completely flat decile table.

    Args:
        actual: Realised holdout quantity per customer.
        predicted: The prediction to rank by, per customer.
        keys: Customer identifiers, used only to break ranking ties deterministically. Without a
            stable tiebreaker the committed report changes between runs on identical data, and a
            report that churns cannot be diffed to see whether anything moved.

    Returns:
        One row per decile, 1 being the highest-predicted, with the actual outcome in each and the
        share of the realised total it captured.
    """
    actual_values, predicted_values = _paired(actual, predicted)
    key_values = np.asarray(keys)
    if key_values.shape != actual_values.shape:
        raise ValueError("keys must have the same shape as actual and predicted.")

    frame = pd.DataFrame(
        {"key": key_values, "actual": actual_values, "predicted": predicted_values}
    ).sort_values(["predicted", "key"], ascending=[False, True], kind="mergesort")

    # Equal-count deciles by position in the sorted order, so ties cannot pile into one bucket and
    # leave another empty -- which is a real risk here, where 14,120 customers share one population
    # spend estimate.
    position = np.arange(len(frame))
    frame["decile"] = (position * DECILES // len(frame)) + 1

    actual_total = frame["actual"].sum()
    table = (
        frame.groupby("decile")
        .agg(
            customers=("actual", "size"),
            mean_predicted=("predicted", "mean"),
            mean_actual=("actual", "mean"),
            total_actual=("actual", "sum"),
        )
        .reset_index()
    )
    table["share_of_actual"] = (
        table["total_actual"] / actual_total if actual_total else float("nan")
    )
    return table

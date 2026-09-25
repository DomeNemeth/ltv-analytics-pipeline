"""Published CDNOW results, transcribed from the source papers.

Every number in this module was read out of the paper named beside it, not recalled. That
distinction is the whole point of the module: a benchmark comparison against half-remembered
figures is worse than no benchmark, because it looks like corroboration.

Held as constants rather than fetched at runtime. A validation command that needs a working internet
connection to produce a report is a validation command that stops working, and the figures are from
a 2005 journal article -- they are not going to change.

**The comparison is not like-for-like, and the report must say so.** Two differences matter:

* **Population.** The paper fits a 1/10th systematic sample of 2,357 customers. This project
  fits the full master file, 23,570. Different samples of the same cohort should give similar
  estimates, and showing that they do is the content of the check -- but it is a consistency
  check, not a reproduction.
* **Time unit.** The paper works in weeks; this project works in days. Under the BG/NBD
  parameterisation the transaction rate lambda is gamma-distributed with shape ``r`` and rate
  ``alpha``, so its mean is ``r / alpha``. A rate measured per day is a seventh of the same rate
  measured per week, so going from this project's units to the paper's **divides** ``alpha`` by
  seven -- 33.011 becomes 4.716, against a published 4.414. ``r``, ``a`` and ``b`` are untouched;
  they are shape parameters of distributions over a dimensionless probability or a rate ratio.

  The direction is easy to state backwards, and stating it backwards produces a number off by a
  factor of 49 that still looks like a plausible parameter. :func:`alpha_in_weeks` is the single
  place it happens, and ``test_alpha_converts_towards_the_published_units`` pins which way it goes
  -- without that test the only thing stopping someone "correcting" the function is this
  paragraph, which is not a guard.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Days per week. Named because it is a unit conversion rather than a magic number, and because
#: getting its direction wrong produces a benchmark comparison that is off by a factor of 49.
DAYS_PER_WEEK = 7


@dataclass(frozen=True)
class PublishedFit:
    """One published parameter set, with enough provenance to be checked by a reader."""

    model: str
    parameters: dict[str, float]
    log_likelihood: float
    customers: int
    citation: str
    location: str


#: Fader, Peter S., Bruce G. S. Hardie, and Ka Lok Lee (2005), "'Counting Your Customers' the Easy
#: Way: An Alternative to the Pareto/NBD," Marketing Science, 24 (2), 275-284.
#:
#: Section 7 (p. 280): "we take a 1/10th systematic sample of the customers. We calibrate the model
#: using the repeat transaction data for the 2,357 sampled customers over the first half of the
#: 78-week period and forecast their future purchasing over the remaining 39 weeks."
#:
#: Both parameter sets are Table 2, p. 281. Time unit is weeks throughout.
CDNOW_BG_NBD = PublishedFit(
    model="BG/NBD",
    parameters={"r": 0.243, "alpha": 4.414, "a": 0.793, "b": 2.426},
    log_likelihood=-9582.4,
    customers=2357,
    citation=(
        "Fader, Peter S., Bruce G. S. Hardie, and Ka Lok Lee (2005), "
        "\"'Counting Your Customers' the Easy Way: An Alternative to the Pareto/NBD,\" "
        "Marketing Science, 24 (2), 275-284."
    ),
    location="Table 2, p. 281",
)

CDNOW_PARETO_NBD = PublishedFit(
    model="Pareto/NBD",
    parameters={"r": 0.553, "alpha": 10.578, "s": 0.606, "beta": 11.669},
    log_likelihood=-9595.0,
    customers=2357,
    citation=CDNOW_BG_NBD.citation,
    location="Table 2, p. 281",
)

#: The paper's own out-of-sample result, quoted so this project's error can be placed beside it
#: (p. 281): "In the subsequent 39-week forecast period, both models track the actual (cumulative)
#: sales trajectory, with the Pareto/NBD performing slightly better than the BG/NBD (under-
#: forecasting by 2% versus 4%)."
#:
#: **This is a cumulative figure and must not be compared to a holdout-window figure.** It is the
#: error in total repeat transactions across all 78 weeks, so an excellent in-sample fit over the
#: first 39 weeks dilutes whatever happens in the second 39. Scored on the forecast window alone the
#: number would be considerably larger. The validation report computes both, on both bases, rather
#: than quoting one against the other.
CDNOW_CUMULATIVE_UNDERFORECAST_PERCENT = {"BG/NBD": -4.0, "Pareto/NBD": -2.0}

#: Also p. 281, and worth pinning because this project reproduces the same structure independently:
#: "the 1,411 people who made no repeat purchases in the first 39 weeks. This group makes a total of
#: 334 transactions in Weeks 40-78, which comprises 18% of all of the forecast period transactions."
#: 1,411 of 2,357 is 59.9%, against 14,119 of 23,570 -- also 59.9% -- on the full file.
CDNOW_SAMPLE_ZERO_CLASS = 1411

#: Correlation between actual forecast-period transactions and BG/NBD conditional expectations,
#: across all 2,357 customers (Table 3, p. 282, discussed on p. 281).
CDNOW_BG_NBD_CONDITIONAL_CORRELATION = 0.626


def alpha_in_weeks(alpha_days: float) -> float:
    """Convert a BG/NBD ``alpha`` fitted in days to the weeks the published figures use.

    ``alpha`` is the rate parameter of the gamma distribution over the per-unit-time transaction
    rate, so it carries the time unit and scales with it. ``r``, ``a`` and ``b`` do not and must be
    left alone -- rescaling them too is the natural mistake, and it produces a comparison that looks
    wrong in a way that invites "fixing" the model.
    """
    return alpha_days / DAYS_PER_WEEK

"""Fit BG/NBD and Gamma-Gamma to the calibration window, and predict forward from it.

Two models, deliberately separate, because they answer different questions:

* **BG/NBD** predicts *how many* purchases a customer makes next, and whether they are still active.
  It is fitted on every customer, because a one-time buyer's frequency of 0 is real information
  about the population's dropout rate rather than a missing value.
* **Gamma-Gamma** predicts *how much* each purchase is worth. It can only be fitted on customers who
  have repeat spend to learn from, which on CDNOW is 9,450 of 23,570.

Expected forward revenue is the product of the two. This project computes that product directly
rather than through PyMC-Marketing's ``customer_lifetime_value`` helper. The helper is not wrong,
but it discounts to present value and takes ``future_t`` in *months regardless of the time_unit
passed to it*, which is a trap worth stepping around: our stated definition is undiscounted expected
forward revenue over a horizon in days (CLAUDE.md section 6), and at a zero discount rate the two
agree anyway. A multiplication a reader can check beats a helper they have to go and read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from ltv.config import Settings, get_settings
from ltv.warehouse import connect

if TYPE_CHECKING:  # pragma: no cover - import cost is only worth paying at runtime
    from pymc_marketing.clv import BetaGeoModel, GammaGammaModel

#: The relation the fit reads. Named as a constant so the leakage guard is testable: fitting on
#: int_customers__rfm_full instead would train on the holdout window and produce excellent,
#: worthless metrics. The dbt layer cannot catch that -- both relations are valid models, and
#: choosing the wrong one is a Python-side mistake -- so tests/test_clv_fit.py asserts on this name.
CALIBRATION_RELATION = "int_customers__rfm_calibration"

#: Where the holdout length comes from. Derived, never hardcoded: it has to equal the window Phase 4
#: scores against, and a literal here would drift the moment the split changes.
WINDOWS_RELATION = "int_sources__analysis_windows"

#: BetaGeoModel requires a column literally named ``T`` (Fader & Hardie's notation for customer age
#: at the end of the observation window). The dbt layer spells it ``customer_age`` on purpose --
#: DuckDB folds unquoted identifiers to lowercase, so a column that must be quoted as "T" everywhere
#: is worse than one that says what it means. The rename happens here, at the boundary, exactly
#: once.
#:
#: Getting this wrong is silent and severe: recency and T are both day counts in the same range, so
#: swapping them fits cleanly and predicts nonsense. It has its own test.
BG_NBD_RENAMES = {"customer_age": "T"}

BG_NBD_COLUMNS = ("customer_id", "frequency", "recency", "T")
GAMMA_GAMMA_COLUMNS = ("customer_id", "monetary_value", "frequency")


class ModelError(RuntimeError):
    """Raised when the calibration data cannot support a fit, or a fit does not converge."""


@dataclass(frozen=True)
class CalibrationData:
    """Everything the models are fitted on, plus the window that defines it."""

    source: str
    customers: pd.DataFrame
    holdout_days: int

    @property
    def eligible(self) -> pd.DataFrame:
        """Customers Gamma-Gamma can learn from: at least one repeat purchase, and positive spend.

        The flag is computed in dbt (``is_gamma_gamma_eligible``) rather than recomputed here, so
        there is one definition of eligibility and the dashboard's customer counts agree with the
        model's.
        """
        return self.customers[self.customers["is_gamma_gamma_eligible"]]

    @property
    def purchase_frame(self) -> pd.DataFrame:
        return self.customers.rename(columns=BG_NBD_RENAMES)[list(BG_NBD_COLUMNS)]

    @property
    def spend_frame(self) -> pd.DataFrame:
        return self.eligible[list(GAMMA_GAMMA_COLUMNS)].reset_index(drop=True)


@dataclass(frozen=True)
class FittedModels:
    """A fitted pair, and the facts about the fit that any claim made from it depends on."""

    purchase_model: BetaGeoModel
    spend_model: GammaGammaModel
    method: str
    customers: int
    spend_customers: int

    @property
    def excluded_customers(self) -> int:
        """Customers who carry no Gamma-Gamma information. Reported, never hidden."""
        return self.customers - self.spend_customers

    @property
    def is_bayesian(self) -> bool:
        """Whether uncertainty intervals from this fit mean anything.

        MAP returns a single point, so its "posterior" has one draw and any interval computed from
        it would be a zero-width fiction. CLAUDE.md section 6 makes this a rule: interval claims
        require a --full-bayes run.
        """
        return self.method == "mcmc"


def load_calibration(source: str, settings: Settings | None = None) -> CalibrationData:
    """Read one source's calibration-window RFM summary out of the warehouse.

    Raises:
        ModelError: If the source has no rows, or the transformation layer has not been built.
    """
    settings = settings or get_settings()

    with connect(settings, read_only=True) as connection:
        try:
            customers = connection.execute(
                f"""
                select
                    customer_id,
                    frequency,
                    recency,
                    customer_age,
                    monetary_value,
                    is_gamma_gamma_eligible
                from {CALIBRATION_RELATION}
                where source = ?
                order by customer_id
                """,
                [source],
            ).df()

            window = connection.execute(
                f"select duration_holdout from {WINDOWS_RELATION} where source = ?",
                [source],
            ).fetchone()
        except Exception as exc:  # noqa: BLE001 - re-raised with the actionable message below
            raise ModelError(
                f"Could not read {CALIBRATION_RELATION} from {settings.duckdb_path}. "
                f"Build the transformation layer first: `uv run ltv transform`."
            ) from exc

    if customers.empty:
        raise ModelError(
            f"No calibration rows for source {source!r}. "
            f"Check `uv run ltv ingest {source}` ran, then `uv run ltv transform`."
        )

    # decimal(12,4) arrives as Decimal objects, which PyMC cannot broadcast. Cast once, here, rather
    # than letting a dtype error surface from inside a model graph.
    customers["monetary_value"] = customers["monetary_value"].astype(float)

    return CalibrationData(source=source, customers=customers, holdout_days=int(window[0]))


def check_assumptions(data: CalibrationData) -> None:
    """Fail loudly on data the models would silently mis-fit.

    Every check here corresponds to something the model assumes and does not verify. A violation
    does not raise inside PyMC -- it fits, converges, and returns parameters that mean nothing.

    Raises:
        ModelError: On the first violated assumption, naming how many rows violate it.
    """
    customers = data.customers

    # BetaGeoModel's own docstring states this requirement: a customer cannot have reached their
    # last purchase after the window ended. The dbt layer makes it true by construction, so a
    # violation means the two windows have come apart.
    impossible_age = customers["customer_age"] < customers["recency"]
    if impossible_age.any():
        raise ModelError(
            f"{impossible_age.sum():,} customers have recency greater than customer_age, which "
            f"BG/NBD requires to be impossible (a purchase cannot fall after the window closes). "
            f"The calibration window and the occasion dates have come apart."
        )

    negative = customers["frequency"] < 0
    if negative.any():
        raise ModelError(f"{negative.sum():,} customers have negative frequency.")

    # Eligibility is decided in dbt; this asserts the flag still means what the fit assumes it
    # means. If they ever disagree, the Gamma-Gamma fit would quietly include zero-spend customers.
    eligible = data.eligible
    inconsistent = (eligible["frequency"] <= 0) | (eligible["monetary_value"] <= 0)
    if inconsistent.any():
        raise ModelError(
            f"{inconsistent.sum():,} customers are flagged Gamma-Gamma eligible but have no repeat "
            f"purchases or no repeat spend. The dbt flag and the model's requirement disagree."
        )

    if eligible.empty:
        raise ModelError(
            f"No customer in {data.source!r} has repeat spend, so Gamma-Gamma cannot be fitted."
        )


def _use_numba_backend() -> None:
    """Compile model graphs with numba rather than PyTensor's Python fallback.

    This is not a micro-optimisation. With no C compiler present, PyTensor falls back to a pure
    Python backend and the BG/NBD MAP fit on CDNOW takes **265 seconds**; under numba it takes
    **25**, with parameter estimates identical to four decimal places. Measured on this project's
    own data, not assumed.

    Set explicitly in code rather than left to a PYTENSOR_FLAGS environment variable, so the command
    behaves the same for someone who just cloned the repo as it does in CI. numba needs no compiler
    and no admin rights, which is the same reason it and nutpie are pinned at all (CLAUDE.md
    section 4).
    """
    import pytensor

    pytensor.config.mode = "NUMBA"


def fit_models(
    data: CalibrationData,
    *,
    full_bayes: bool = False,
    settings: Settings | None = None,
) -> FittedModels:
    """Fit BG/NBD on every customer and Gamma-Gamma on those with repeat spend.

    Args:
        data: Calibration-window summary, already assumption-checked.
        full_bayes: Sample the posterior with NUTS instead of taking the MAP estimate. Slower by
            orders of magnitude, and the only way to get honest uncertainty intervals.
        settings: Configuration to use. Defaults to the process-wide settings.
    """
    from pymc_marketing.clv import BetaGeoModel, GammaGammaModel

    settings = settings or get_settings()
    _use_numba_backend()

    if full_bayes:
        # nutpie is the compiler-free NUTS sampler this project pins; seeding it is what makes a
        # re-run reproduce the intervals it reported last time.
        method = "mcmc"
        fit_kwargs = {"nuts_sampler": "nutpie", "random_seed": settings.random_seed}
    else:
        # find_MAP is a deterministic optimisation, so it takes no seed. Passing one would be
        # accepted and mean nothing, which is worse than not passing it.
        method = "map"
        fit_kwargs = {}

    # Data goes to fit(), not to the constructor: PyMC-Marketing deprecated the constructor form in
    # 0.19 and removes it in 1.0. Both work today and emit a warning; this is the one that survives.
    purchase_model = BetaGeoModel()
    purchase_model.fit(data.purchase_frame, method=method, **fit_kwargs)

    spend_model = GammaGammaModel()
    spend_model.fit(data.spend_frame, method=method, **fit_kwargs)

    return FittedModels(
        purchase_model=purchase_model,
        spend_model=spend_model,
        method=method,
        customers=len(data.customers),
        spend_customers=len(data.spend_frame),
    )


def _spend_posterior(models: FittedModels, data: CalibrationData):
    """Expected spend per purchase for *every* customer, as a full posterior.

    Customers with no repeat spend get the fitted *population* estimate rather than a NULL or a
    zero. This is the standard Fader & Hardie treatment, and on CDNOW it matters a great deal:
    14,120 of 23,570 customers are in this group, so NULLing them would leave every dashboard total
    silently covering 40% of the customer base, and zeroing them would understate it outright.

    The substitution is flagged in the output rather than blended in, because a population default
    and a fitted per-customer value are different kinds of number and a reader must be able to tell
    which they are looking at.

    Kept as a posterior rather than collapsed to a mean so that the Bayesian intervals downstream
    carry *both* sources of uncertainty -- how often someone buys and how much they spend. Taking
    the mean here first would quietly report purchase uncertainty alone as if it were the whole
    story.

    Returns:
        The aligned spend posterior, and a boolean mask of which customers were fitted individually.
    """
    conditional = models.spend_model.expected_customer_spend(data=data.spend_frame)
    population = models.spend_model.expected_new_customer_spend()

    customer_ids = data.customers["customer_id"].to_numpy()
    aligned = conditional.reindex(customer_id=customer_ids)

    # `population` has dims (chain, draw) and broadcasts across customer_id, so every unfitted
    # customer receives the population posterior rather than a single number.
    fitted_individually = aligned.isel(chain=0, draw=0).notnull().to_numpy()
    return aligned.fillna(population), fitted_individually


def _hdi_bounds(posterior, is_bayesian: bool):
    """Highest-density interval per customer, or NULLs when the fit cannot support one.

    A MAP fit has exactly one draw, so an interval computed from it would have zero width and read
    as certainty. CLAUDE.md section 6 forbids that claim, so this returns NULLs and the
    ``fit_method`` column tells the reader why they are NULL.
    """
    import numpy as np

    if not is_bayesian:
        empty = np.full(posterior.sizes["customer_id"], np.nan)
        return empty, empty

    import arviz as az

    interval = az.hdi(posterior.rename("value"), hdi_prob=0.94)["value"]
    return (
        interval.sel(hdi="lower").to_numpy(),
        interval.sel(hdi="higher").to_numpy(),
    )


def predict(
    models: FittedModels,
    data: CalibrationData,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Predict forward behaviour for every customer, at each horizon.

    Returns one row per customer per horizon. The grain carries ``horizon_days`` as a column rather
    than encoding the horizon in column names, so a revenue number can never be read without the
    window it applies to.
    """
    purchase_frame = data.purchase_frame
    spend, fitted_individually = _spend_posterior(models, data)

    # Being alive depends on observed history, not on how far ahead we look, so it is computed once
    # and repeated across horizons rather than recomputed per horizon.
    alive = (
        models.purchase_model.expected_probability_alive(data=purchase_frame)
        .mean(dim=("chain", "draw"))
        .to_numpy()
    )

    frames = []
    for horizon in horizons:
        purchases = models.purchase_model.expected_purchases(data=purchase_frame, future_t=horizon)

        # Undiscounted expected forward revenue: how many purchases, times what each is worth.
        # Multiplied draw by draw, so the product carries the uncertainty of both models. CLAUDE.md
        # section 6 -- revenue, not margin, because margin is a business input these datasets do
        # not contain.
        revenue = purchases * spend
        revenue_low, revenue_high = _hdi_bounds(revenue, models.is_bayesian)

        frames.append(
            pd.DataFrame(
                {
                    "source": data.source,
                    "customer_id": data.customers["customer_id"].to_numpy(),
                    "horizon_days": horizon,
                    "expected_purchases": purchases.mean(dim=("chain", "draw")).to_numpy(),
                    "probability_alive": alive,
                    "expected_avg_value": spend.mean(dim=("chain", "draw")).to_numpy(),
                    "expected_forward_revenue": revenue.mean(dim=("chain", "draw")).to_numpy(),
                    "forward_revenue_hdi_low": revenue_low,
                    "forward_revenue_hdi_high": revenue_high,
                    "spend_estimate_source": [
                        "conditional" if fitted else "population_mean"
                        for fitted in fitted_individually
                    ],
                    "is_gamma_gamma_eligible": data.customers["is_gamma_gamma_eligible"].to_numpy(),
                    "fit_method": models.method,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)

"""Fit alternative purchase models on the same frame, to find out whose fault the error is.

Phase 3's BG/NBD fit under-predicts holdout purchases by 14%, and that gap has already been
diagnosed: it is not MAP, not the priors, and not an apples-to-oranges comparison. Monthly repeat
occasions fall ~40% through the calibration window and then *plateau* through the holdout, and
BG/NBD can only explain a decline as dropout, so it extrapolates a decay the real cohort stops
doing. That is model misspecification.

The way to test that claim rather than assert it is to fit models that make different assumptions
about dropout and see whether the error moves:

* **MBG/NBD** removes BG/NBD's assumption that a customer who has never repeated cannot yet have
  churned. That assumption is the reason 14,119 CDNOW customers -- 59.9% of the base -- carry
  ``probability_alive`` of exactly 1.0 by construction while only 14.6% of them actually returned.
  If dropout timing is the problem, this is the model that shows it.
* **Pareto/NBD** replaces the "dropout only at a purchase" mechanism entirely: a customer can become
  inactive at any moment, governed by an exponential lifetime. It is the older and more expensive
  model that BG/NBD was proposed as a cheap alternative to.

Neither is promoted by running this. They are scored, reported, and left where they are; the
champion in ``model.customer_predictions`` stays BG/NBD. Swapping the model a dashboard reads is a
decision someone should make on the evidence, not a side effect of producing the evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ltv.config import Settings, get_settings
from ltv.models.clv import CalibrationData, ModelError, _use_numba_backend

#: The alternatives, keyed by the name they are reported under. Values are the PyMC-Marketing class
#: names rather than the classes themselves, so importing this module costs nothing -- PyMC is
#: seconds to import and `ltv validate` without --compare-models never needs it.
CHALLENGERS: dict[str, str] = {
    "mbg_nbd": "ModifiedBetaGeoModel",
    "pareto_nbd": "ParetoNBDModel",
}

#: What the champion is called wherever a model family is named alongside the challengers.
CHAMPION = "bg_nbd"


@dataclass(frozen=True)
class ChallengerFit:
    """One alternative model's predictions on the calibration frame, or why there are none.

    A failure is a result, not an omission. Pareto/NBD's likelihood involves a Gaussian
    hypergeometric function and has never been exercised on this project's compiler-less host, so
    "it did not converge in reasonable time on a machine with no C compiler" is a finding worth
    printing rather than an excuse for a silently shorter table.
    """

    family: str
    seconds: float
    expected_purchases: np.ndarray | None = None
    probability_alive: np.ndarray | None = None
    parameters: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def fit_challenger(
    family: str,
    data: CalibrationData,
    *,
    settings: Settings | None = None,
) -> ChallengerFit:
    """Fit one challenger at MAP and predict over the holdout window.

    The frame is :attr:`CalibrationData.purchase_frame` unchanged -- the same object BG/NBD was
    fitted on, not a re-derivation of it. A comparison between models fitted on frames built
    separately measures the difference between the frames as much as the models.

    The horizon is ``data.holdout_days``, read from the warehouse, for the same reason: a challenger
    scored over a different window than the champion is not a comparison.

    Never raises for a modelling failure. The failure is returned, so one uncooperative model cannot
    take the whole validation run with it.
    """
    settings = settings or get_settings()

    if family not in CHALLENGERS:
        raise ModelError(
            f"Unknown model family {family!r}. Known challengers: {', '.join(CHALLENGERS)}."
        )

    import pymc_marketing.clv as clv

    # Must happen before the model is constructed. ParetoNBDModel.fit wraps its graph in
    # `pytensor.config.change_flags(mode=get_default_mode())`, which resolves `config.mode` at call
    # time -- so it inherits numba only if numba was already selected. Left to PyTensor's Python
    # fallback, the simpler BG/NBD fit takes 265 seconds against 25 (CLAUDE.md section 4); the
    # Pareto/NBD likelihood is considerably heavier than that.
    _use_numba_backend()

    model_class = getattr(clv, CHALLENGERS[family])
    frame = data.purchase_frame

    started = time.perf_counter()
    try:
        model = model_class()
        model.fit(frame, method="map")

        purchases = model.expected_purchases(data=frame, future_t=data.holdout_days)
        alive = model.expected_probability_alive(data=frame)
        parameters = {
            name: float(values.mean(dim=("chain", "draw")).item())
            for name, values in model.idata.posterior.data_vars.items()
            if values.ndim == 2  # scalar parameters only; skip anything with a customer dimension
        }
    except Exception as exc:  # noqa: BLE001 - a failed challenger is reported, never fatal
        return ChallengerFit(
            family=family,
            seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    return ChallengerFit(
        family=family,
        seconds=time.perf_counter() - started,
        expected_purchases=purchases.mean(dim=("chain", "draw")).to_numpy(),
        probability_alive=alive.mean(dim=("chain", "draw")).to_numpy(),
        parameters=parameters,
    )

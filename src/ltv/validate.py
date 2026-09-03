"""The validation stage: predictions in, an honest account of how good they are out.

This is the stage the whole project is built around. CLAUDE.md section 1 claims the point of
difference is that this pipeline *validates* a CLV model rather than merely fitting one, and until
this command exists that sentence is unsupported.

Like :mod:`ltv.transform` and :mod:`ltv.fit`, it is a stage entry point that takes
:class:`~ltv.config.Settings` and does one thing, so the Phase 6 flow can wire stages together
without reaching inside them.

Three things about the design are deliberate:

* **The join is in dbt, not here.** ``int_customers__scored`` puts predictions, actuals and
  baselines on one row and is guarded by its own tests. The two worst defects this project has
  shipped were both joins, and both were invisible to the code that consumed them. A join inside a
  metrics function cannot be tested by the suite that tests joins.
* **Staleness is caught by that build, not re-implemented here.** ``run_validation`` builds the
  post-fit models before reading them, and the
  ``assert_predictions_match_the_current_calibration_inputs`` test fails if the warehouse has
  moved since the fit. A second definition of "fresh" written in Python
  would be a second thing to keep in agreement, which is the failure mode section 9 exists to warn
  about. What is checked here instead is the precondition that relation cannot express: whether
  there is anything to validate at all.
* **Every number is reported for the model and for two naive rules.** "The model beat nothing" is
  not a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ltv.config import Settings, get_settings
from ltv.models import benchmarks, metrics
from ltv.models.challengers import CHALLENGERS, CHAMPION, ChallengerFit, fit_challenger
from ltv.models.store import (
    FIT_RUNS_TABLE,
    PREDICTIONS_TABLE,
    write_validation_metrics,
)
from ltv.transform import POST_FIT_ARGS, run_dbt
from ltv.warehouse import connect

#: The relation every metric is computed from. Built by the post-fit dbt stage immediately before it
#: is read, so the numbers in a report always describe the warehouse as it stands.
SCORED_RELATION = "int_customers__scored"

#: Calibration repeat-count buckets for the canonical actual-versus-predicted table. The open-ended
#: top bucket follows the published CDNOW analysis, which reports "0, 1, ..., 7+", so the two tables
#: can be read side by side.
FREQUENCY_BUCKETS = (0, 1, 2, 3, 4, 5, 6)
TOP_BUCKET_LABEL = "7+"

#: What gets scored against what. Each entry is (quantity, actual column, {family: column}).
#: MAPE is reported only where a percentage error is defined for a meaningful share of the
#: population -- see :func:`score` for why purchase counts are not in that category.
SCORED_QUANTITIES: tuple[tuple[str, str, dict[str, str]], ...] = (
    (
        "purchases",
        "holdout_frequency",
        {
            CHAMPION: "expected_purchases",
            "baseline_carry_forward": "baseline_carry_forward",
            "baseline_rate": "baseline_purchases",
            "baseline_flat": "baseline_purchases_flat",
            "baseline_zero": "baseline_zero",
        },
    ),
    (
        "revenue",
        "holdout_spend",
        {
            CHAMPION: "expected_forward_revenue",
            "baseline_carry_forward": "baseline_revenue_carry_forward",
            "baseline_rate": "baseline_revenue",
            "baseline_flat": "baseline_revenue_flat",
            "baseline_zero": "baseline_zero",
        },
    ),
    (
        "avg_order_value",
        "holdout_monetary_value",
        {
            CHAMPION: "expected_avg_value",
            "baseline_rate": "baseline_avg_value",
        },
    ),
)

#: Quantities whose actuals are mostly zero, where a mean absolute percentage error would be
#: computed over a minority of the population and read as though it described all of it. The
#: clv-validator brief calls this out explicitly: "MAPE on counts that include zeros is undefined or
#: explosive." Aggregate signed error and per-customer MAE carry the same information without the
#: false precision.
NO_MAPE = frozenset({"purchases"})


class ValidationError(RuntimeError):
    """Raised when there is nothing to validate, or when what there is cannot be scored."""


@dataclass(frozen=True)
class Score:
    """Every metric for one predictor of one quantity."""

    family: str
    quantity: str
    mae: float
    rmse: float
    aggregate: metrics.AggregateError
    rank: metrics.Correlation
    population: int
    mape: metrics.Mape | None = None


@dataclass(frozen=True)
class Independence:
    """Gamma-Gamma's frequency/monetary independence assumption, measured rather than assumed."""

    correlation: metrics.Correlation
    by_frequency: pd.DataFrame

    @property
    def holds(self) -> bool:
        """Whether the assumption survives at the conventional 5% level.

        A convenience for the report's wording, not a decision procedure. With 9,450 customers a
        trivially small correlation is significant, which is exactly why the report prints the
        effect size and the spread of order value across frequency buckets rather than a verdict.
        """
        return self.correlation.p_value >= 0.05


@dataclass(frozen=True)
class BenchmarkComparison:
    """This project's BG/NBD parameters beside the published ones, in the same units."""

    fitted_days: dict[str, float]
    fitted_weeks: dict[str, float]
    published: benchmarks.PublishedFit
    fitted_customers: int


@dataclass(frozen=True)
class ValidationResult:
    """What a validation run established, in the terms a reader needs to judge it."""

    source: str
    fit_method: str
    customers: int
    #: Customers who made no holdout purchase at all. 16,512 of 23,570 on CDNOW -- the number
    #: that decides which metrics mean anything, because on a mostly-zero quantity a
    #: per-customer error is dominated by getting the zeros right.
    non_returners: int
    horizon_days: int
    scores: tuple[Score, ...]
    buckets: pd.DataFrame
    deciles: pd.DataFrame
    deciles_baseline: pd.DataFrame
    independence: Independence
    benchmark: BenchmarkComparison
    calibration_fit: metrics.AggregateError
    alive_tautology: tuple[int, float]
    report_path: Path
    chart_paths: tuple[Path, ...]
    metrics_written: int
    challengers: tuple[ChallengerFit, ...] = field(default_factory=tuple)

    def score(self, family: str, quantity: str) -> Score | None:
        for score in self.scores:
            if score.family == family and score.quantity == quantity:
                return score
        return None


def _require_predictions(settings: Settings) -> None:
    """Fail early and actionably when there is nothing to validate.

    Without this, the dbt build below fails inside a compiled model with a message naming a missing
    ``model.customer_predictions`` relation -- which is accurate and names the wrong problem. The
    reader needs to be told to run the fit, in the same way ``ltv transform`` tells them to run the
    ingest. Same reasoning as ``transform._require_raw_data``.
    """
    with connect(settings, read_only=True) as connection:
        (present,) = connection.execute(
            """
            select count(*)
            from information_schema.tables
            where table_schema = 'model' and table_name in ('customer_predictions', 'fit_runs')
            """
        ).fetchone()

    if present < 2:
        raise ValidationError(
            f"The warehouse at {settings.duckdb_path} has no {PREDICTIONS_TABLE} or no "
            f"{FIT_RUNS_TABLE}, so there are no predictions to score. Fit the models first: "
            f"`uv run ltv fit`."
        )


def load_scored(source: str, settings: Settings | None = None) -> pd.DataFrame:
    """Read one source's scored relation, ordered so it can be aligned with a model's own output.

    Sorted by ``customer_id`` because the challenger fits produce plain arrays in the order of the
    calibration frame, which ``load_calibration`` also orders by ``customer_id``. Alignment by
    position is only safe when both sides are ordered by the same key, and the ordering of a DuckDB
    table is not something to assume.
    """
    settings = settings or get_settings()

    with connect(settings, read_only=True) as connection:
        frame = connection.execute(
            f"select * from {SCORED_RELATION} where source = ? order by customer_id",
            [source],
        ).df()

    if frame.empty:
        raise ValidationError(
            f"No scored rows for source {source!r}. Check `uv run ltv fit` ran for this source."
        )

    # decimal columns arrive as Decimal objects, which numpy cannot do arithmetic on. Cast once here
    # rather than letting a dtype error surface from inside a metric.
    for column in ("monetary_value", "holdout_spend", "holdout_monetary_value"):
        frame[column] = frame[column].astype(float)

    return frame


def score(
    frame: pd.DataFrame, quantity: str, actual_column: str, family: str, column: str
) -> Score:
    """Compute every metric for one predictor of one quantity."""
    actual = frame[actual_column].to_numpy(dtype=float)
    predicted = frame[column].to_numpy(dtype=float)

    return Score(
        family=family,
        quantity=quantity,
        mae=metrics.mae(actual, predicted),
        rmse=metrics.rmse(actual, predicted),
        aggregate=metrics.aggregate_error(actual, predicted),
        rank=metrics.spearman(predicted, actual),
        population=len(frame),
        mape=None if quantity in NO_MAPE else metrics.mape(actual, predicted),
    )


def score_all(frame: pd.DataFrame, challengers: dict[str, str] | None = None) -> tuple[Score, ...]:
    """Score every predictor against every quantity it has a prediction for.

    ``avg_order_value`` is scored only on customers who actually purchased in the holdout window. A
    holdout average order value of zero means "did not buy", not "bought for nothing", and averaging
    those in would score the spend model against a quantity that does not exist for them -- while
    making it look far worse than it is, since it predicts a positive value for everybody.

    Args:
        frame: The scored relation, plus a column per challenger when there are any.
        challengers: Extra ``{family: column}`` predictors of purchase counts. They are scored on
            counts only: MBG/NBD and Pareto/NBD model *when* a customer buys, not how much they
            spend, so there is no challenger revenue figure to report and inventing one by reusing
            Gamma-Gamma would attribute the champion's spend model to a different purchase model.
    """
    scores = []
    for quantity, actual_column, families in SCORED_QUANTITIES:
        population = (
            frame[frame["holdout_frequency"] > 0] if quantity == "avg_order_value" else frame
        )
        predictors = dict(families)
        if quantity == "purchases":
            predictors.update(challengers or {})
        for family, column in predictors.items():
            scores.append(score(population, quantity, actual_column, family, column))
    return tuple(scores)


def bucket_label(frequency: int) -> str:
    """Bucket a calibration repeat count, matching the published CDNOW breakdown."""
    return str(frequency) if frequency in FREQUENCY_BUCKETS else TOP_BUCKET_LABEL


def bucket_table(frame: pd.DataFrame, extra: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Mean holdout purchases by calibration repeat count: actual, model, baselines, challengers.

    The canonical CLV validation table, and the one that says where the error actually lives. On
    CDNOW the aggregate shortfall of 2,817 purchases is not spread evenly: customers with exactly
    one or two calibration repeats account for roughly 69% of it, because BG/NBD writes off a
    one-repeat customer far more readily than the data supports.

    An aggregate error number would hide that completely, which is the reason a table that a
    reviewer can scan beats a metric they cannot decompose.
    """
    working = frame.copy()
    working["bucket"] = working["frequency"].map(bucket_label)
    for name, values in (extra or {}).items():
        working[name] = values

    columns = {
        "customers": ("holdout_frequency", "size"),
        "actual": ("holdout_frequency", "mean"),
        "actual_total": ("holdout_frequency", "sum"),
        CHAMPION: ("expected_purchases", "mean"),
        "baseline_rate": ("baseline_purchases", "mean"),
    }
    for name in extra or {}:
        columns[name] = (name, "mean")

    table = working.groupby("bucket").agg(**columns).reset_index()

    # Sort numerically with the open-ended bucket last, not lexicographically -- "10" before "2" in
    # a table that reads as an ordered progression is the kind of small wrongness that makes a
    # reviewer stop trusting the rest of the page.
    table["order"] = table["bucket"].map(
        lambda b: len(FREQUENCY_BUCKETS) if b == TOP_BUCKET_LABEL else int(b)
    )
    table = table.sort_values("order").drop(columns="order").reset_index(drop=True)

    table["shortfall"] = (table[CHAMPION] - table["actual"]) * table["customers"]
    return table


def independence_check(frame: pd.DataFrame) -> Independence:
    """Measure Gamma-Gamma's central assumption instead of assuming it.

    The model requires the monetary value of a purchase to be independent of how often the customer
    purchases. Where that fails, the model shrinks every customer toward a common population mean,
    so it under-values heavy buyers and over-values light ones -- and "which segments are worth
    acquiring" is a question asked along exactly that axis. A violation does not invalidate the
    model; it bounds what may be claimed from it, which is why it is measured and reported rather
    than tested for and ignored.

    Computed over Gamma-Gamma-eligible customers only, because monetary_value is defined as the mean
    over repeat purchases and is a structural zero for everyone else.
    """
    eligible = frame[frame["is_gamma_gamma_eligible"]]
    correlation = metrics.spearman(eligible["frequency"], eligible["monetary_value"])

    by_frequency = (
        eligible.assign(bucket=eligible["frequency"].map(bucket_label))
        .groupby("bucket")
        .agg(
            customers=("monetary_value", "size"),
            mean_order_value=("monetary_value", "mean"),
        )
        .reset_index()
    )
    by_frequency["order"] = by_frequency["bucket"].map(
        lambda b: len(FREQUENCY_BUCKETS) if b == TOP_BUCKET_LABEL else int(b)
    )
    by_frequency = by_frequency.sort_values("order").drop(columns="order").reset_index(drop=True)

    return Independence(correlation=correlation, by_frequency=by_frequency)


def benchmark_comparison(purchase_model, customers: int) -> BenchmarkComparison:
    """Place this project's BG/NBD parameters beside the published CDNOW estimates.

    The published work is in weeks and this project works in days, so ``alpha`` is converted before
    anything is compared. It is also fitted on a 1/10th sample rather than the full file, which is
    why this is a consistency check and not a reproduction -- see :mod:`ltv.models.benchmarks`.
    """
    fitted = {
        name: float(values.mean(dim=("chain", "draw")).item())
        for name, values in purchase_model.idata.posterior.data_vars.items()
        if values.ndim == 2
    }
    fitted_weeks = dict(fitted)
    fitted_weeks["alpha"] = benchmarks.alpha_in_weeks(fitted["alpha"])

    return BenchmarkComparison(
        fitted_days=fitted,
        fitted_weeks=fitted_weeks,
        published=benchmarks.CDNOW_BG_NBD,
        fitted_customers=customers,
    )


def calibration_fit(purchase_model, frame: pd.DataFrame) -> metrics.AggregateError:
    """How well the model describes the window it was fitted on. Never a predictive claim.

    Reported deliberately and labelled unambiguously. The clv-validator brief asks whether
    calibration-period fit is being presented anywhere as predictive performance, and the honest way
    to answer that is to print both and let the gap speak: an in-sample fit within a percent of the
    data, alongside an out-of-sample error an order of magnitude larger, is the clearest available
    statement of what "fitting is not predicting" means.

    Uses the unconditional expectation E[X(t)] with each customer's own observation length, which is
    the model's expected *repeat* transactions over the calibration window -- the comparand for the
    calibration frequency, which also counts repeats.
    """
    expected = purchase_model.expected_purchases_new_customer(
        data=frame.rename(columns={"customer_age": "T"}), t=frame["customer_age"].to_numpy()
    )
    return metrics.aggregate_error(
        frame["frequency"].to_numpy(dtype=float),
        expected.mean(dim=("chain", "draw")).to_numpy(),
    )


def alive_tautology(frame: pd.DataFrame) -> tuple[int, float]:
    """How many customers are "certainly alive" by construction, and how many actually returned.

    BG/NBD only lets a customer drop out immediately after a purchase, so a customer with zero
    repeats has never had an opportunity to. Their probability of being alive is exactly 1.0 as a
    property of the model's mechanics, not as a finding about them. This quantifies the gap so that
    no segment definition or dashboard tile can treat it as evidence of health.
    """
    certain = frame[frame["probability_alive"] >= 1.0]
    if certain.empty:
        return 0, float("nan")
    return len(certain), float((certain["holdout_frequency"] > 0).mean())


def metrics_frame(source: str, result_scores: tuple[Score, ...]) -> pd.DataFrame:
    """Flatten the scores into the long format the warehouse table holds."""
    rows = []
    for item in result_scores:
        values = {
            "mae": item.mae,
            "rmse": item.rmse,
            "actual_total": item.aggregate.actual_total,
            "predicted_total": item.aggregate.predicted_total,
            "percent_error": item.aggregate.percent_error,
            "spearman_rho": item.rank.rho,
        }
        if item.mape is not None:
            values["mape"] = item.mape.value
            values["mape_excluded"] = float(item.mape.excluded)

        rows.extend(
            {
                "source": source,
                "model_family": item.family,
                "quantity": item.quantity,
                "metric": metric,
                "value": value,
                "population": item.population,
            }
            for metric, value in values.items()
        )
    return pd.DataFrame(rows)


def run_validation(
    source: str = "cdnow",
    *,
    compare_models: bool = False,
    settings: Settings | None = None,
) -> ValidationResult:
    """Score one source's predictions against its holdout window and write the report.

    Args:
        source: Which dataset to validate. Sources are scored separately; they are separate
            populations fitted by separate models.
        compare_models: Also fit MBG/NBD and Pareto/NBD on the same calibration frame and put them
            in the bucket table. Slower, and the answer to "is the error the model's fault".
        settings: Configuration to use. Defaults to the process-wide settings.

    Raises:
        ValidationError: If there is nothing to validate.
        TransformError: If the post-fit build fails -- which includes the predictions being stale
            with respect to the warehouse they are about to be scored in.
    """
    from ltv.charts import write_charts
    from ltv.report import write_report

    settings = settings or get_settings()
    settings.ensure_dirs()

    _require_predictions(settings)

    # Build the post-fit models immediately before reading them, so a report can never describe a
    # scored relation built from an earlier state of the warehouse. This is also where the staleness
    # guard fires: assert_predictions_match_the_current_calibration_inputs fails the build if the
    # calibration relation has been rebuilt since the fit consumed it.
    run_dbt(list(POST_FIT_ARGS), settings)

    frame = load_scored(source, settings)

    challengers: tuple[ChallengerFit, ...] = ()
    extra: dict[str, np.ndarray] = {}
    if compare_models:
        challengers = _fit_challengers(source, frame, settings)
        extra = {fit.family: fit.expected_purchases for fit in challengers if fit.succeeded}
        # Attached to the frame rather than scored from loose arrays, so a challenger is subject to
        # exactly the same population and the same metric code as the champion.
        for family, predictions in extra.items():
            frame[family] = predictions

    scores = score_all(frame, {family: family for family in extra})

    from pymc_marketing.clv import BetaGeoModel

    from ltv.models.store import load_fit

    fit_method = str(frame["fit_method"].iloc[0])

    # The model that produced these predictions, not a fresh one. Needed for two things the
    # predictions table cannot answer: the fitted parameters, for the published-benchmark
    # comparison, and the unconditional expectation, for the in-sample calibration fit.
    purchase_model = load_fit(BetaGeoModel, source, f"{fit_method}_bgnbd", settings)

    written = write_validation_metrics(metrics_frame(source, scores), settings)

    result = ValidationResult(
        source=source,
        fit_method=fit_method,
        customers=len(frame),
        non_returners=int((frame["holdout_frequency"] == 0).sum()),
        horizon_days=int(frame["horizon_days"].iloc[0]),
        scores=scores,
        buckets=bucket_table(frame, extra),
        deciles=metrics.decile_table(
            frame["holdout_spend"], frame["expected_forward_revenue"], frame["customer_id"]
        ),
        # The same ranking exercise driven by the naive rule. Without it the decile table is
        # scored against nothing, which is the one place CLAUDE.md section 6's "report error
        # against a naive baseline" was not being applied -- and it is the section speaking
        # most directly to the project's stated business question.
        deciles_baseline=metrics.decile_table(
            frame["holdout_spend"], frame["baseline_revenue"], frame["customer_id"]
        ),
        independence=independence_check(frame),
        benchmark=benchmark_comparison(purchase_model, len(frame)),
        calibration_fit=calibration_fit(purchase_model, frame),
        alive_tautology=alive_tautology(frame),
        challengers=challengers,
        report_path=settings.reports_dir / f"validation_{source}.md",
        chart_paths=(),
        metrics_written=written,
    )

    # Charts first: the report embeds them, so a report that renders is a report whose figures
    # exist. Both are derived from `result`, which is why it is built before either and then
    # replaced with the paths they produced.
    charts = write_charts(result, frame, settings)
    return replace(result, chart_paths=charts, report_path=write_report(result, charts, settings))


def _fit_challengers(
    source: str, frame: pd.DataFrame, settings: Settings
) -> tuple[ChallengerFit, ...]:
    """Fit each challenger on the same calibration frame the champion used.

    Reads the calibration relation rather than reconstructing features from the scored frame: the
    comparison is only meaningful if every model sees byte-identical inputs, and rebuilding
    ``recency``/``T`` here would be a second derivation of quantities the project already defines
    once.
    """
    from ltv.models.clv import load_calibration

    data = load_calibration(source, settings)

    fitted_ids = data.customers["customer_id"].to_numpy()
    if not np.array_equal(fitted_ids, frame["customer_id"].to_numpy()):
        raise ValidationError(
            "The calibration frame and the scored frame describe different customers, or the same "
            "customers in a different order. Challenger predictions are aligned positionally, so "
            "this would silently attribute one customer's forecast to another."
        )

    return tuple(fit_challenger(family, data, settings=settings) for family in CHALLENGERS)

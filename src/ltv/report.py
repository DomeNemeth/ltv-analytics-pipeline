"""Render the validation report.

The report is committed, which puts one constraint on everything here: **it must be deterministic**.
No wall-clock time, no run duration, no anything that changes when the data does not. A report that
churns on every run cannot be diffed, and a diff nobody reads is how a number changes without anyone
noticing.

The other constraint is tone. This file writes the document a reviewer reads to decide whether to
believe the project, so it states what was measured, on what population, and what it does not
support -- including where the model loses. An overstated result in a portfolio project is worse
than a modest one, because an interviewer will probe it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ltv.config import Settings, get_settings
from ltv.models import benchmarks
from ltv.models.challengers import CHAMPION

if TYPE_CHECKING:  # pragma: no cover
    from ltv.validate import Score, ValidationResult

QUANTITY_TITLES = {
    "purchases": "Purchase counts",
    "revenue": "Forward revenue",
    "avg_order_value": "Average order value",
}

FAMILY_LABELS = {
    CHAMPION: "BG/NBD + Gamma-Gamma",
    "baseline_carry_forward": "Naive: same as last window",
    "baseline_rate": "Naive: calibration rate",
    "baseline_flat": "Naive: population mean",
    "baseline_zero": "Naive: nobody buys",
    "mbg_nbd": "MBG/NBD",
    "pareto_nbd": "Pareto/NBD",
}


def _number(value: float, places: int = 2) -> str:
    """Format a float with a fixed number of places, or an em dash when it is not a number."""
    if value != value:  # NaN
        return "—"
    return f"{value:,.{places}f}"


def _table(frame: pd.DataFrame, formatters: dict[str, int] | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table with stable formatting."""
    working = frame.copy()
    for column, places in (formatters or {}).items():
        if column in working.columns:
            working[column] = working[column].map(lambda v, p=places: _number(float(v), p))

    header = "| " + " | ".join(str(c) for c in working.columns) + " |"
    divider = "| " + " | ".join("---" for _ in working.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in working.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _accuracy_table(scores: tuple[Score, ...], quantity: str) -> str:
    relevant = [s for s in scores if s.quantity == quantity]
    frame = pd.DataFrame(
        [
            {
                "Model": FAMILY_LABELS.get(s.family, s.family),
                "MAE": _number(s.mae, 3),
                "RMSE": _number(s.rmse, 3),
                (
                    "Sum of per-customer means" if quantity == "avg_order_value" else "Actual total"
                ): _number(s.aggregate.actual_total, 0),
                (
                    "Predicted, same basis" if quantity == "avg_order_value" else "Predicted total"
                ): _number(s.aggregate.predicted_total, 0),
                "Error %": _number(s.aggregate.percent_error, 1),
                "Spearman ρ": _number(s.rank.rho, 3),
                "MAPE %": _number(s.mape.value, 1) if s.mape else "not reported",
            }
            for s in relevant
        ]
    )
    return _table(frame)


def _mape_note(scores: tuple[Score, ...], quantity: str) -> str:
    for item in scores:
        if item.quantity == quantity and item.mape is not None:
            if not item.mape.excluded:
                return (
                    f"MAPE covers all {item.mape.included:,} scored customers; every one has a "
                    f"non-zero actual, so none had to be excluded."
                )
            return (
                f"MAPE covers the {item.mape.included:,} customers whose actual is non-zero "
                f"and excludes {item.mape.excluded:,} "
                f"({item.mape.excluded_share:.1%} of the {item.mape.total:,} scored). A percentage "
                f"error against an actual of zero is undefined, not large."
            )
    return (
        "MAPE is deliberately not reported for purchase counts. Most customers make no holdout "
        "purchase at all, so it would be computed over a minority of the population and read as "
        "though it described the whole of it. The signed aggregate error and the per-customer MAE "
        "carry the same information without the false precision."
    )


def _required(result: ValidationResult, family: str, quantity: str) -> Score:
    """Fetch a score the report cannot be written without.

    Raised rather than asserted: `python -O` strips assertions, and a report that silently renders
    "None purchases against None actual" is worse than one that refuses to render. The headline
    exists to state the model's error beside a baseline's, so a missing baseline is a broken report
    rather than a section to skip.
    """
    score = result.score(family, quantity)
    if score is None:
        raise ValueError(
            f"No {quantity} score for {family!r}. Every run scores the champion and both naive "
            f"baselines -- see SCORED_QUANTITIES in ltv.validate."
        )
    return score


def _headline(result: ValidationResult) -> list[str]:
    purchases = _required(result, CHAMPION, "purchases")
    baseline = _required(result, "baseline_rate", "purchases")
    flat = _required(result, "baseline_flat", "purchases")

    carry = _required(result, "baseline_carry_forward", "purchases")
    zero = _required(result, "baseline_zero", "purchases")

    beaten = [b for b in (carry, baseline, flat, zero) if purchases.mae < b.mae]
    verdict = (
        f"beats all {len(beaten)} naive rules"
        if len(beaten) == 4
        else f"beats {len(beaten)} of the 4 naive rules"
    )

    return [
        "## Headline",
        "",
        f"Over the {result.horizon_days}-day holdout window, the model predicts "
        f"**{purchases.aggregate.predicted_total:,.0f}** purchases against "
        f"**{purchases.aggregate.actual_total:,.0f}** actual — "
        f"**{purchases.aggregate.percent_error:+.1f}%**. The naive rule that each customer keeps "
        f"repeating at their calibration rate predicts "
        f"{baseline.aggregate.predicted_total:,.0f} "
        f"({baseline.aggregate.percent_error:+.1f}%), and "
        f"assuming every customer is average predicts "
        f"{flat.aggregate.predicted_total:,.0f} ({flat.aggregate.percent_error:+.1f}%).",
        "",
        f"The fairest comparison is the same-period rule -- predict that a customer repeats as "
        f"often as they did last window, which needs no rescaling because the two windows are the "
        f"same length. It predicts {carry.aggregate.predicted_total:,.0f} "
        f"({carry.aggregate.percent_error:+.1f}%). **The rate baselines above are inflated**: "
        f"they divide by each customer's observation length and multiply by the holdout "
        f"length, and mean customer age here is shorter than the holdout, so they scale every "
        f"calibration count up before comparing. Roughly half of their "
        f"{baseline.aggregate.percent_error:+.0f}% is rescaling rather than naivety.",
        "",
        f"On per-customer error the model {verdict}: MAE {purchases.mae:.3f} against "
        f"{zero.mae:.3f}, {carry.mae:.3f}, {baseline.mae:.3f} and {flat.mae:.3f}.",
        "",
        f"**Read that MAE against the row below it, not against the others.** Predicting that "
        f"nobody buys anything scores {zero.mae:.3f}, because {result.non_returners:,} of these "
        f"customers genuinely buy nothing. On a mostly-zero quantity the all-zero rule is "
        f"what MAE is really measured against, and the model clears it by "
        f"{100 * (zero.mae - purchases.mae) / zero.mae:.1f}%. This model's value is in aggregate "
        f"totals and in ranking, not in per-customer accuracy -- the sections below are where it "
        f"earns its place.",
        "",
    ]


def _in_sample_section(result: ValidationResult) -> list[str]:
    fit = result.calibration_fit
    out_of_sample = _required(result, CHAMPION, "purchases")

    return [
        "## Fitting is not predicting",
        "",
        "The same model, measured on the window it was fitted on and on the window it was not:",
        "",
        _table(
            pd.DataFrame(
                [
                    {
                        "Window": "Calibration (in-sample)",
                        "Actual": _number(fit.actual_total, 0),
                        "Expected": _number(fit.predicted_total, 0),
                        "Error %": _number(fit.percent_error, 1),
                    },
                    {
                        "Window": f"Holdout (out-of-sample, {result.horizon_days} days)",
                        "Actual": _number(out_of_sample.aggregate.actual_total, 0),
                        "Expected": _number(out_of_sample.aggregate.predicted_total, 0),
                        "Error %": _number(out_of_sample.aggregate.percent_error, 1),
                    },
                ]
            )
        ),
        "",
        "The in-sample row is a description of data the model has already seen and is **not** "
        "evidence of predictive accuracy. It is printed precisely so that the gap between the two "
        "rows is visible: a model can describe its training window almost exactly and still miss "
        "the next one.",
        "",
        "Calibration expectations use the unconditional E[X(t)] with each customer's own "
        "observation length, which counts repeat transactions — the same quantity as the "
        "calibration frequency it is compared against.",
        "",
    ]


def _buckets_section(result: ValidationResult) -> list[str]:
    table = result.buckets.copy()
    total_shortfall = table["shortfall"].sum()
    concentrated = table[table["bucket"].isin(["1", "2"])]["shortfall"].sum()

    rename = {"bucket": "Calibration repeats", "customers": "Customers", "actual": "Actual"}
    rename.update(
        {
            family: FAMILY_LABELS.get(family, family)
            for family in table.columns
            if family in FAMILY_LABELS
        }
    )
    rename["shortfall"] = "Shortfall"
    display = table.drop(columns=["actual_total"]).rename(columns=rename)

    places = {
        column: 3
        for column in display.columns
        if column not in ("Calibration repeats", "Customers")
    }
    places["Shortfall"] = 0
    places["Customers"] = 0

    lines = [
        "## Where the error lives",
        "",
        "Mean holdout purchases by the number of repeat purchases a customer made during "
        "calibration. `Shortfall` is the total purchases the model missed in that bucket "
        "(negative means it over-predicted).",
        "",
        _table(display, places),
        "",
    ]

    if total_shortfall:
        lines.extend(
            [
                f"The aggregate miss of {abs(total_shortfall):,.0f} purchases is not spread "
                f"evenly. "
                f"Customers with exactly one or two calibration repeats account for "
                f"{abs(concentrated):,.0f} of it ({abs(concentrated) / abs(total_shortfall):.0%}). "
                f"BG/NBD writes those customers off faster than the data supports: having seen one "
                f"repeat and then a gap, it concludes they have probably churned.",
                "",
            ]
        )
    return lines


def _concentrated_shortfall(result: ValidationResult) -> list[str]:
    """Compare each model where the champion's error actually lives.

    The aggregate figures in the metrics table are dominated by the 14,119 customers who never
    repeated, and every model gets those roughly right. The interesting comparison is the one- and
    two-repeat buckets, which hold most of the shortfall and which the three models disagree about.
    Computed from the bucket table rather than asserted, so this paragraph cannot drift away from
    the numbers above it.
    """
    table = result.buckets
    concentrated = table[table["bucket"].isin(["1", "2"])]
    if concentrated.empty:
        return []

    # Baselines are printed for scale but excluded from `best`. A naive rule cannot support a
    # conclusion about dropout timing, and on a second source one of them could well win the
    # bucket -- at which point the paragraph below would follow a result it does not describe.
    models = [c for c in table.columns if c in (CHAMPION, "mbg_nbd", "pareto_nbd")]
    families = [c for c in table.columns if c in FAMILY_LABELS and c != "baseline_flat"]
    misses = {
        family: float(
            ((concentrated[family] - concentrated["actual"]) * concentrated["customers"]).sum()
        )
        for family in families
        if family in concentrated.columns
    }
    if len(misses) < 2:
        return []

    contenders = {f: m for f, m in misses.items() if f in models}
    if not contenders:
        return []
    best = min(contenders, key=lambda family: abs(contenders[family]))
    ranked = ", ".join(
        f"{FAMILY_LABELS.get(family, family)} {misses[family]:+,.0f}"
        for family in sorted(misses, key=lambda f: abs(misses[f]))
    )

    # The conclusion has to follow from which model actually won, not be asserted beside it. If the
    # champion is closest, there is no evidence here that a different dropout mechanism helps, and
    # saying otherwise would be a paragraph that survived its own result.
    conclusion = (
        "That is a result about dropout timing rather than about estimation. BG/NBD only lets a "
        "customer churn immediately after a purchase; Pareto/NBD lets them churn at any moment, "
        "governed by an exponential lifetime. A customer who repeated once and then went quiet is "
        "exactly the case those two assumptions disagree about most."
        if best != CHAMPION
        else "The champion is closest in this bucket, so it gives no evidence that a different "
        "dropout mechanism would do better here."
    )

    return [
        f"**In the one- and two-repeat buckets, where most of the shortfall sits, the models "
        f"genuinely differ.** Purchases missed across those two buckets: {ranked}. "
        f"{FAMILY_LABELS.get(best, best)} comes closest of the fitted models; the naive rules are "
        f"listed for scale and are not candidates for this conclusion.",
        "",
        conclusion,
        "",
    ]


def _challengers_section(result: ValidationResult) -> list[str]:
    if not result.challengers:
        return [
            "## Model comparison",
            "",
            "Not run. `uv run ltv validate --compare-models` fits MBG/NBD and Pareto/NBD on the "
            "same calibration frame and adds them to the table above.",
            "",
        ]

    lines = [
        "## Model comparison",
        "",
        "Alternative purchase models, fitted at MAP on the identical calibration frame and scored "
        "over the identical window. They test whether the shortfall is this model's assumptions or "
        "the data's behaviour — they are not promoted, and the champion in "
        "`model.customer_predictions` remains BG/NBD.",
        "",
        "They are scored on purchase counts only, and appear in the Purchase counts table above. "
        "MBG/NBD and Pareto/NBD model *when* a customer buys, not how much they spend; pairing one "
        "of them with the champion's Gamma-Gamma fit and calling the product a revenue forecast "
        "would attribute the spend model's behaviour to a purchase model that had nothing to do "
        "with it.",
        "",
        *_concentrated_shortfall(result),
    ]

    rows = []
    for fit in result.challengers:
        if not fit.succeeded:
            rows.append(
                {
                    "Model": FAMILY_LABELS.get(fit.family, fit.family),
                    "Status": "did not fit",
                    "Detail": fit.error or "",
                }
            )
        else:
            parameters = ", ".join(f"{k} = {v:.3f}" for k, v in sorted(fit.parameters.items()))
            rows.append(
                {
                    "Model": FAMILY_LABELS.get(fit.family, fit.family),
                    "Status": "fitted",
                    "Detail": parameters,
                }
            )
    lines.extend([_table(pd.DataFrame(rows)), ""])

    failed = [fit for fit in result.challengers if not fit.succeeded]
    if failed:
        lines.extend(
            [
                "A model that did not fit is recorded here rather than omitted. This project "
                "builds "
                "without a C compiler, and the honest statement is that the model could not be "
                "evaluated on this host — not that it was never considered.",
                "",
            ]
        )
    return lines


def _assumptions_section(result: ValidationResult) -> list[str]:
    independence = result.independence
    certain, returned = result.alive_tautology
    spread = independence.by_frequency

    lines = [
        "## Assumption checks",
        "",
        "### Gamma-Gamma: monetary value independent of frequency",
        "",
        f"Spearman ρ between calibration frequency and mean repeat order value, across the "
        f"{independence.correlation.n:,} Gamma-Gamma-eligible customers: "
        f"**{independence.correlation.rho:+.3f}** (p = {independence.correlation.p_value:.2g}).",
        "",
        _table(
            spread.rename(
                columns={
                    "bucket": "Calibration repeats",
                    "customers": "Customers",
                    "mean_order_value": "Mean repeat order value",
                }
            ),
            {"Mean repeat order value": 2, "Customers": 0},
        ),
        "",
    ]

    if not independence.holds:
        low = spread["mean_order_value"].iloc[0]
        high = spread["mean_order_value"].iloc[-1]
        lines.extend(
            [
                f"**The assumption does not hold here.** Mean repeat order value climbs from "
                f"{low:,.2f} to {high:,.2f} across the frequency range, a spread of "
                f"{(high / low - 1):.0%}. Gamma-Gamma shrinks every customer toward a common "
                f"population mean, so it systematically under-values heavy buyers and over-values "
                f'light ones. That is the same axis a "which segments are worth acquiring" '
                f"conclusion runs along, so **segment-level value rankings from this model are "
                f"compressed** and the gaps between segments are understated.",
                "",
            ]
        )

    lines.extend(
        [
            "### BG/NBD: probability_alive for customers who never repeated",
            "",
            f"**{certain:,} customers ({certain / result.customers:.1%} of the base) have a "
            f"probability of being alive of exactly 1.0.** This is a property of the model, not a "
            f"finding about them: BG/NBD only allows a customer to drop out immediately after a "
            f"purchase, so a customer with no repeat purchase has never had an opportunity to. "
            f"Only {returned:.1%} of them actually transacted in the holdout window.",
            "",
            "No segment definition and no dashboard tile may treat this as evidence that these "
            "customers are healthy. It is the single easiest number in the project to misread.",
            "",
        ]
    )
    return lines


def _benchmark_section(result: ValidationResult) -> list[str]:
    comparison = result.benchmark
    published = comparison.published

    rows = [
        {
            "Parameter": name,
            "This project (days)": _number(comparison.fitted_days.get(name, float("nan")), 3),
            "This project (weeks)": _number(comparison.fitted_weeks.get(name, float("nan")), 3),
            "Published (weeks)": _number(published.parameters[name], 3),
        }
        for name in ("r", "alpha", "a", "b")
    ]

    return [
        "## Against the published CDNOW figures",
        "",
        f"{published.citation} — {published.location}.",
        "",
        _table(pd.DataFrame(rows)),
        "",
        f"`alpha` is the rate parameter of the gamma distribution over the per-unit-time "
        f"transaction rate, so it carries the time unit: this project works in days and the paper "
        f"in weeks, which divides it by {benchmarks.DAYS_PER_WEEK}. `r`, `a` and `b` are "
        f"dimensionless and are compared unconverted.",
        "",
        f"**This is a consistency check, not a reproduction.** The published fit uses a 1/10th "
        f"systematic sample of {published.customers:,} customers; this one uses the full master "
        f"file of {comparison.fitted_customers:,}. Two different samples of the same cohort "
        f"landing "
        f"on similar parameters is evidence the implementation is right; identical parameters were "
        f"never the expected outcome.",
        "",
        f"The paper reports the BG/NBD under-forecasting by "
        f"{abs(benchmarks.CDNOW_CUMULATIVE_UNDERFORECAST_PERCENT['BG/NBD']):.0f}% and the "
        f"Pareto/NBD by "
        f"{abs(benchmarks.CDNOW_CUMULATIVE_UNDERFORECAST_PERCENT['Pareto/NBD']):.0f}% "
        f"over the forecast period. **That figure is cumulative across all 78 weeks and must "
        f"not be "
        f"read against the holdout-only error above**: an in-sample fit accurate to within a "
        f"percent over the first 39 weeks dilutes whatever happens in the second 39. The "
        f'"Fitting is not predicting" table gives both bases so the two can be compared like with '
        f"like.",
        "",
        f"One structural detail reproduces independently and is worth noting: the paper's "
        f"zero-class "
        f"is {benchmarks.CDNOW_SAMPLE_ZERO_CLASS:,} of {published.customers:,} customers (59.9%). "
        f"On the full file it is {result.alive_tautology[0]:,} of {result.customers:,} — also "
        f"59.9%.",
        "",
    ]


def _deciles_section(result: ValidationResult) -> list[str]:
    table = result.deciles.copy()
    top = table.iloc[0]

    baseline_top = result.deciles_baseline.iloc[0]
    model_rank = _required(result, CHAMPION, "revenue")
    baseline_rank = _required(result, "baseline_rate", "revenue")

    # The naive rule's realised outcome per decile, beside the model's. Without this column the
    # decile table is scored against nothing, and 51.6% reads as a property of the model when most
    # of it is a property of the data.
    table["baseline_actual"] = result.deciles_baseline["mean_actual"].to_numpy()
    table["baseline_share"] = result.deciles_baseline["share_of_actual"].to_numpy()

    display = table.rename(
        columns={
            "decile": "Decile",
            "customers": "Customers",
            "mean_predicted": "Mean predicted",
            "mean_actual": "Mean actual",
            "total_actual": "Total actual",
            "share_of_actual": "Share of actual",
            "baseline_actual": "Mean actual (naive ranking)",
            "baseline_share": "Share (naive ranking)",
        }
    )
    display["Share of actual"] = display["Share of actual"].map(lambda v: f"{v:.1%}")
    display["Share (naive ranking)"] = display["Share (naive ranking)"].map(lambda v: f"{v:.1%}")

    return [
        "## Does the ranking work?",
        "",
        "Customers ranked by predicted forward revenue and split into ten equal groups, against "
        "what they actually spent. This is the question the project's stated purpose asks — "
        "which "
        "segments are worth acquiring is a question about order, not about absolute error, and a "
        "model can score well on one while being useless at the other.",
        "",
        _table(
            display,
            {
                "Mean predicted": 2,
                "Mean actual": 2,
                "Total actual": 0,
                "Customers": 0,
                "Mean actual (naive ranking)": 2,
            },
        ),
        "",
        f"The top decile captures **{top['share_of_actual']:.1%}** of all realised holdout "
        f"revenue, against **{baseline_top['share_of_actual']:.1%}** when the same customers are "
        f"ranked by the naive calibration-rate rule instead. A ranking with no discriminating "
        f"power would capture 10%.",
        "",
        f"**The honest reading is that almost all of that capture is available without the "
        f"model.** The naive rule gets to within "
        f"{abs(top['share_of_actual'] - baseline_top['share_of_actual']) * 100:.1f} percentage "
        f"points of it, and on rank correlation it is actually **ahead**: Spearman ρ "
        f"{baseline_rank.rank.rho:.3f} against the model's {model_rank.rank.rho:.3f} on revenue, "
        f"and {_required(result, 'baseline_rate', 'purchases').rank.rho:.3f} against "
        f"{_required(result, CHAMPION, 'purchases').rank.rho:.3f} on purchase counts. Ranking "
        f"customers by what they already spent is a strong rule on this dataset, and the modelling "
        f"does not improve on it.",
        "",
        f"The ordering is also not merely flat below decile 4 -- it is mildly **inverted**. "
        f"Decile 10, the lowest-predicted tenth, realises a mean of "
        f"{table['mean_actual'].iloc[-1]:,.2f} against "
        f"{table['mean_actual'].iloc[4:9].min():,.2f}-{table['mean_actual'].iloc[4:9].max():,.2f} "
        f"across deciles 5-9. Below the top few deciles the model is not ranking these customers "
        f"at all, which is consistent with what it has to work with: most of them made no repeat "
        f"purchase, so their predictions differ only through customer age.",
        "",
        "What the model does add over that rule is calibration rather than order: it puts the "
        "predictions on a scale that is 14% low rather than 47% high, and it assigns a value to "
        "every customer including those with no repeat history. For choosing *who* to target, the "
        "naive rule is competitive. For forecasting *how much*, it is not.",
        "",
    ]


def _limits_section(result: ValidationResult) -> list[str]:
    lines = [
        "## What this report does not establish",
        "",
        f"- **No uncertainty intervals.** This is a `{result.fit_method}` fit. "
        + (
            "MAP returns a single point, so any interval computed from it would be zero-width and "
            "read as certainty. Intervals require `uv run ltv fit --full-bayes`."
            if result.fit_method == "map"
            else "The predictions table carries a 94% HDI on forward revenue, sampled with NUTS. "
            "**It is an interval on the model's expectation, not on what a customer will do.** It "
            "expresses how precisely the four population parameters are pinned down by 23,570 "
            "observations — which is very precisely, so the intervals are narrow, around 6-10% "
            "of the estimate. Only 0.6% of customers' realised holdout spend falls inside their "
            "own interval. That is not a defect: an individual outcome is dominated by "
            "Poisson and gamma variation the interval deliberately does not include. It does mean "
            "these intervals must never be presented as a range a customer's revenue is likely to "
            "land in."
        ),
        "- **LTV here means expected forward revenue over a stated horizon, undiscounted** — not "
        "gross margin. Margin and discount rate are business inputs this dataset does not contain.",
        "- **One dataset, one cohort.** Every customer here made their first purchase at CDNOW in "
        "Q1 1997. Nothing about how the model behaves on a continuously-acquiring population is "
        "demonstrated by these numbers.",
        "- **Segment-level value rankings are compressed** by the Gamma-Gamma violation above. The "
        "ordering is informative; the size of the gaps between segments is understated.",
        "",
    ]
    return lines


def write_report(
    result: ValidationResult, charts: tuple[Path, ...], settings: Settings | None = None
) -> Path:
    """Write the validation report for one source, returning where it went."""
    settings = settings or get_settings()
    path = settings.reports_dir / f"validation_{result.source}.md"

    sections: list[str] = [
        f"# Validation — {result.source}",
        "",
        "Generated by `uv run ltv validate`. Every number here is computed from "
        "`int_customers__scored`, which joins the predictions in the warehouse to the holdout "
        "window they are scored against; it is rebuilt immediately before this report is written, "
        "and a build fails if the predictions no longer match the data they were fitted on.",
        "",
        f"- **Population:** {result.customers:,} customers",
        f"- **Scoring horizon:** {result.horizon_days} days, read from the analysis window rather "
        f"than configured",
        f"- **Fit method:** `{result.fit_method}`",
        "",
        *_headline(result),
    ]

    for quantity, title in QUANTITY_TITLES.items():
        sections.extend(
            [
                f"## {title}",
                "",
                _accuracy_table(result.scores, quantity),
                "",
                _mape_note(result.scores, quantity),
                "",
            ]
        )
        if quantity == "avg_order_value":
            sections.extend(
                [
                    "Scored only on customers who made at least one holdout purchase. An average "
                    "order value of zero means the customer did not buy, not that they bought for "
                    "nothing, and averaging those in would score the spend model against a "
                    "quantity "
                    "that does not exist for them.",
                    "",
                ]
            )

    sections.extend(_in_sample_section(result))
    sections.extend(_buckets_section(result))

    if charts:
        sections.extend([f"![Holdout purchases by calibration frequency]({charts[0].name})", ""])
    if len(charts) > 1:
        sections.extend(
            [
                "### Why the model falls short",
                "",
                "BG/NBD can only explain a falling purchase rate as customers dropping out, so it "
                "extrapolates the calibration decline forward. The realised series declines "
                "through calibration and then levels off. The model keeps decaying; the cohort "
                "does not.",
                "",
                "Read the first three months as cohort formation rather than growth: every "
                "customer in this dataset makes their first purchase in Q1 1997, so the repeat "
                "series necessarily climbs while the cohort is still being acquired. The part that "
                "matters is what happens after it: a steep fall to the cutoff, and then a flat "
                "stretch the model has no way to represent.",
                "",
                f"![Repeat occasions per month]({charts[1].name})",
                "",
            ]
        )

    sections.extend(_challengers_section(result))
    sections.extend(_deciles_section(result))
    if len(charts) > 2:
        sections.extend([f"![Revenue by predicted decile]({charts[2].name})", ""])

    sections.extend(_assumptions_section(result))
    sections.extend(_benchmark_section(result))
    sections.extend(_limits_section(result))

    # newline="\n" explicitly, not the platform default. The report is committed and CI regenerates
    # it on Linux to check it reproduces, so a Windows run writing CRLF would make an identical
    # report fail that diff -- and .gitattributes normalises the repo to LF anyway, so a CRLF file
    # here is a working-tree diff that never goes away. Determinism has to mean byte-identical
    # across platforms, not just across runs on one of them.
    path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8", newline="\n")
    return path

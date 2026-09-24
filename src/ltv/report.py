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
    "baseline_recent": "Naive: recent rate",
    "baseline_carry_forward": "Naive: same as last window",
    "baseline_rate": "Naive: calibration rate",
    "baseline_flat": "Naive: population mean",
    "baseline_zero": "Naive: nobody buys",
    "mbg_nbd": "MBG/NBD",
    "pareto_nbd": "Pareto/NBD",
}

#: Where a family means something different for one quantity. The order-value baseline is "past
#: average order value", which the rate label above would misdescribe. The audit caught that table
#: calling it "calibration rate".
QUANTITY_LABELS = {("avg_order_value", "baseline_rate"): "Naive: past average order value"}

#: The naive rules that predict purchase counts, in the order the headline discusses them.
NAIVE_PURCHASE_RULES = (
    "baseline_recent",
    "baseline_carry_forward",
    "baseline_rate",
    "baseline_flat",
    "baseline_zero",
)

#: Models fitted to the calibration window, as opposed to naive rules.
FITTED_MODELS = (CHAMPION, "mbg_nbd", "pareto_nbd")


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


def _label(family: str, quantity: str) -> str:
    return QUANTITY_LABELS.get((quantity, family), FAMILY_LABELS.get(family, family))


def _accuracy_table(scores: tuple[Score, ...], quantity: str) -> str:
    rows = []
    for item in (s for s in scores if s.quantity == quantity):
        row = {
            "Model": _label(item.family, quantity),
            "MAE": _number(item.mae, 3),
            "RMSE": _number(item.rmse, 3),
        }
        # No total or aggregate error for order value. A sum of per-customer averages is not an
        # amount anyone spends, so a percentage error on it measures nothing. The audit caught the
        # earlier table reporting one.
        if quantity != "avg_order_value":
            row["Actual total"] = _number(item.aggregate.actual_total, 0)
            row["Predicted total"] = _number(item.aggregate.predicted_total, 0)
            row["Error %"] = _number(item.aggregate.percent_error, 1)
        row["Spearman ρ"] = _number(item.rank.rho, 3)
        row["MAPE %"] = _number(item.mape.value, 1) if item.mape else "not reported"
        rows.append(row)
    return _table(pd.DataFrame(rows))


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


def _sensitivity_sentence(result: ValidationResult, model_error: float) -> str:
    """Say how far the recent rule's result depends on its window, computed from the sensitivity.

    Two earlier versions of this sentence were wrong, and the Phase 4 re-audit caught both. One
    said the window was "fixed before looking at results". It was chosen after the first audit had
    reported 30, 61 and 91 days. The other said the conclusion "does not depend on the choice",
    which held only because the sensitivity stopped at three months. It now runs to the full
    calibration length, and the crossover is computed.
    """
    table = result.recent_sensitivity
    if table.empty:
        return ""
    listed = ", ".join(
        f"{int(row.window_days)} days {row.percent_error:+.1f}%" for row in table.itertuples()
    )
    beats = table["percent_error"].abs() < abs(model_error)
    if beats.all():
        verdict = "every window tested lands closer than the model."
    elif not beats.any():
        verdict = "no window tested lands closer than the model."
    elif beats.is_monotonic_decreasing:
        longest = int(table.loc[beats, "window_days"].max())
        shortest_losing = int(table.loc[~beats, "window_days"].min())
        verdict = (
            f"**the rule beats the model only with a short look-back.** It wins at {longest} "
            f"days and loses from {shortest_losing} days on, so the crossover lies between the "
            f"two. The result is about the most recent months of behaviour, not about "
            f"recent-rate rules in general."
        )
    else:
        verdict = "whether the rule beats the model depends on the window, with no clean cut-off."
    return _chosen_window_sentence(result) + f" Across look-backs of {listed}, " + verdict


#: The look-back windows the first Phase 4 audit reported before the window was chosen. Recorded
#: so the sentence about how the window was chosen stays true if the configured window changes.
AUDITED_WINDOWS = (30, 61, 91)


def _chosen_window_sentence(result: ValidationResult) -> str:
    """How the configured window came to be chosen, true for whichever window is configured.

    The first version was fixed text naming 91 days as the least favourable of three. That is true
    of 91 today, but it would have printed unchanged, and falsely, for any other setting.
    """
    window = result.recent_window_days
    if window not in AUDITED_WINDOWS:
        return f" The {window}-day window is set in dbt_project.yml (recent_window_days)."
    audited = result.recent_sensitivity[
        result.recent_sensitivity["window_days"].isin(AUDITED_WINDOWS)
    ]
    least_favourable = int(audited.loc[audited["percent_error"].abs().idxmax(), "window_days"])
    listed = ", ".join(str(w) for w in AUDITED_WINDOWS[:-1]) + f" and {AUDITED_WINDOWS[-1]}"
    return (
        f" The {window}-day window was chosen after an audit had reported {listed} days, so "
        f"it was not a blind choice"
        + (
            "; it is the least favourable of those to the rule."
            if least_favourable == window
            else "."
        )
    )


def _headline(result: ValidationResult) -> list[str]:
    """State the result against the strongest naive rule, not the most convenient one.

    The first version of this headline said the model's value was "in aggregate totals and in
    ranking". The ranking section below already showed a naive rule ahead on rank correlation, and
    the Phase 4 audit found a recent-rate rule closer on the total. So the verdicts here are
    computed from the scores, not written in advance of them.
    """
    purchases = _required(result, CHAMPION, "purchases")
    naive = {family: _required(result, family, "purchases") for family in NAIVE_PURCHASE_RULES}
    recent, carry = naive["baseline_recent"], naive["baseline_carry_forward"]
    rate, flat, zero = naive["baseline_rate"], naive["baseline_flat"], naive["baseline_zero"]
    model_error = purchases.aggregate.percent_error

    closer = [
        rule
        for family, rule in naive.items()
        if family != "baseline_zero" and abs(rule.aggregate.percent_error) < abs(model_error)
    ]
    if closer:
        best = min(closer, key=lambda rule: abs(rule.aggregate.percent_error))
        closest = (
            ""
            if best.family == "baseline_recent"
            else f" The closest is {FAMILY_LABELS[best.family].removeprefix('Naive: ')}, at "
            f"{best.aggregate.percent_error:+.1f}%."
        )
        aggregate = (
            f"**On the aggregate total, the model does not beat the best naive rule.** "
            f"{len(closer)} of the {len(naive) - 1} naive rules that forecast purchases "
            f"{'lands' if len(closer) == 1 else 'land'} closer.{closest} Predicting that each "
            f"customer keeps repeating at the rate of their last {result.recent_window_days} "
            f"days of calibration gives "
            f"{recent.aggregate.predicted_total:,.0f} ({recent.aggregate.percent_error:+.1f}%)."
            + _sensitivity_sentence(result, model_error)
        )
    else:
        aggregate = (
            f"On the aggregate total the model is closer than every naive rule tested, including "
            f"the rate of each customer's last {result.recent_window_days} days "
            f"({recent.aggregate.percent_error:+.1f}%)."
        )

    beaten = [rule for rule in naive.values() if purchases.mae < rule.mae]
    verdict = (
        f"has a lower MAE than all {len(naive)} naive rules"
        if len(beaten) == len(naive)
        else f"has a lower MAE than {len(beaten)} of the {len(naive)} naive rules"
    )

    model_rho = _required(result, CHAMPION, "revenue").rank.rho
    naive_rho = max(
        (_required(result, family, "revenue").rank.rho for family in NAIVE_PURCHASE_RULES),
        key=lambda rho: -1.0 if rho != rho else rho,
    )
    ranking = (
        "and *Does the ranking work?* shows it is not better at ranking either"
        if naive_rho >= model_rho
        else "though it does rank customers better than any naive rule"
    )

    return [
        "## Headline",
        "",
        f"Over the {result.horizon_days}-day holdout window, the model predicts "
        f"**{purchases.aggregate.predicted_total:,.0f}** purchases against "
        f"**{purchases.aggregate.actual_total:,.0f}** actual: **{model_error:+.1f}%**.",
        "",
        aggregate,
        "",
        f"The recent-rate rule works for the same reason the model misses. The purchase rate "
        f"falls through calibration (see *Why the model falls short*), and a rule that looks only "
        f"at the end of the window picks up the lower rate. Rules that average the whole window "
        f"overshoot: each customer's calibration rate gives {rate.aggregate.percent_error:+.1f}% "
        f"and the population rate {flat.aggregate.percent_error:+.1f}%. Both correctly adjust "
        f"for how long each customer was observed. Predicting that each customer repeats as "
        f"often as last window gives {carry.aggregate.percent_error:+.1f}%. That is closer, but "
        f"only because it under-counts exposure: customers averaged "
        f"{result.mean_customer_age:.0f} days of calibration against a {result.horizon_days}-day "
        f"holdout, and that bias partly cancels the fall in the rate.",
        "",
        f"On per-customer error the model {verdict}: MAE {purchases.mae:.3f}, against "
        f"{recent.mae:.3f} for the last-{result.recent_window_days}-day rule and "
        f"{carry.mae:.3f}, {rate.mae:.3f} and "
        f"{flat.mae:.3f} for the others. **Read it against predicting that nobody buys anything, "
        f"which scores {zero.mae:.3f}**, because {result.non_returners:,} of these customers "
        f"genuinely buy nothing. On a mostly-zero quantity that is the real floor, and the model "
        f"clears it by {100 * (zero.mae - purchases.mae) / zero.mae:.1f}%.",
        "",
        f"So the model's measurable advantage is narrow. Per customer it is modestly better than "
        f"the naive rules. {_zero_repeat_sentence(result)}It also gives a structural account of "
        f"why it misses. It is {'not ' if closer else ''}more accurate on the total, {ranking}.",
        "",
    ]


def _zero_repeat_sentence(result: ValidationResult) -> str:
    """What the model offers customers with no repeat history, measured on the zero bucket.

    An earlier version said the model "gives every customer an estimate". So does every naive
    rule. The real difference is that each naive rule gives these customers either zero or one
    figure shared by everybody, while the model gives them an individual, non-zero estimate. It
    is only worth saying if that estimate is close to what they did, so it is printed against the
    actual.
    """
    zero = result.buckets[result.buckets["bucket"] == "0"]
    if zero.empty:
        return ""
    row = zero.iloc[0]
    return (
        f"For the {int(row['customers']):,} customers with no repeat history, every naive rule "
        f"predicts either zero or one figure shared by everybody. The model gives each of them "
        f"an individual estimate, averaging {row[CHAMPION]:.3f} purchases against "
        f"{row['actual']:.3f} actual. "
    )


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


#: How many of the worst months the shortfall-concentration sentence names.
WORST_MONTHS = 3


def _monthly_lines(result: ValidationResult) -> list[str]:
    """Month-by-month actual against expected, and what the pattern does and does not show.

    Replaces a sentence that said the model "keeps decaying at the calibration pace". The re-audit
    measured it and that was wrong: the model's decline across the holdout is steeper than the
    realised one, but far shallower than the calibration fall. So the text states the two rates
    and where the shortfall bunches, computed from the table.
    """
    monthly = result.monthly
    if monthly.empty or len(monthly) < 2:
        return []

    # Per day, not per month. October has 31 days and June 30, and the third audit pass pointed out
    # that a raw month-to-month comparison mixes that difference into the trend.
    first, last = monthly.iloc[0], monthly.iloc[-1]
    model_change = (last["expected"] / last["days"]) / (first["expected"] / first["days"]) - 1
    actual_change = (last["actual"] / last["days"]) / (first["actual"] / first["days"]) - 1
    shortfall = monthly["shortfall"].sum()
    worst = monthly.nsmallest(WORST_MONTHS, "shortfall")
    worst_share = worst["shortfall"].sum() / shortfall if shortfall else float("nan")
    worst_names = ", ".join(month.strftime("%b %Y") for month in worst["month"].sort_values())

    display = monthly.assign(month=monthly["month"].dt.strftime("%b %Y")).rename(
        columns={
            "month": "Month",
            "actual": "Actual",
            "expected": "Expected",
            "shortfall": "Expected − actual",
        }
    )[["Month", "Actual", "Expected", "Expected − actual"]]

    return [
        "BG/NBD can only explain a falling purchase rate as customers dropping out, so it carries "
        "a decline forward. Holdout purchases by month, against the model's expectation for the "
        "same months:",
        "",
        _table(display, {"Actual": 0, "Expected": 0, "Expected − actual": 0}),
        "",
        f"From the first holdout month to the last, per day, the model's expectation changes by "
        f"{model_change:+.1%} and the realised rate by {actual_change:+.1%}. "
        + (
            "The model's decline is steeper than the cohort's, but that is not the whole story. "
            if model_change < actual_change
            else "The realised decline is at least as steep as the model's, so an extrapolated "
            "decline does not explain the shortfall. "
        )
        + f"The shortfall is uneven: {worst_names} hold "
        f"{worst_share:.0%} of it. None of the three models has a seasonal term, so a seasonal "
        f"pattern would show up exactly like this. A single year of holdout cannot confirm that it "
        f"is seasonal.",
        "",
    ]


def bucket_misses(buckets: pd.DataFrame) -> pd.DataFrame:
    """Purchases each fitted model missed in each calibration-frequency bucket.

    Predicted minus actual, so negative means under-predicted. One row per model, a column per
    bucket, plus the net total and the sum of absolute misses. The two totals are both needed: a
    model can have a small net miss because it is close everywhere, or because large under- and
    over-predictions in different buckets offset. Only the absolute sum tells those apart.
    """
    models = [family for family in FITTED_MODELS if family in buckets.columns]
    rows = []
    for family in models:
        misses = (buckets[family] - buckets["actual"]) * buckets["customers"]
        row = {"model": family}
        row.update(dict(zip(buckets["bucket"], misses, strict=True)))
        row["net"] = float(misses.sum())
        row["absolute"] = float(misses.abs().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _bucket_comparison(result: ValidationResult) -> list[str]:
    """Compare the fitted models in every bucket, not only the ones where a challenger wins.

    The first version compared the one- and two-repeat buckets only, on the claim that "every model
    gets [the zero-repeat customers] roughly right". The Phase 4 audit found that false. Pareto/NBD
    misses the zero-repeat bucket nearly five times worse than BG/NBD, and its better total comes
    from over-predicting heavy buyers, which offsets it. Every bucket is shown now, and the
    conclusion is computed from them.
    """
    misses = bucket_misses(result.buckets)
    if len(misses) < 2:
        return []

    buckets = [c for c in misses.columns if c not in ("model", "net", "absolute")]
    display = misses.rename(columns={"net": "Net", "absolute": "Sum of absolute misses"})
    display["model"] = display["model"].map(lambda family: FAMILY_LABELS.get(family, family))
    display = display.rename(columns={"model": "Model"})

    by_net = misses.loc[misses["net"].abs().idxmin()]
    by_absolute = misses.loc[misses["absolute"].idxmin()]
    champion = misses[misses["model"] == CHAMPION].iloc[0]
    under = sum(champion[bucket] < 0 for bucket in buckets)

    lines = [
        "Purchases each fitted model missed, per calibration-repeat bucket. Negative means "
        "under-predicted.",
        "",
        _table(display, {column: 0 for column in [*buckets, "Net", "Sum of absolute misses"]}),
        "",
    ]
    if by_net["model"] != by_absolute["model"]:
        lines.append(
            f"**The net total and the bucket-level misses rank the models differently.** "
            f"{FAMILY_LABELS[by_net['model']]} has the smallest net miss "
            f"({by_net['net']:+,.0f}), but its misses add up to {by_net['absolute']:,.0f} "
            f"bucket by bucket, against {by_absolute['absolute']:,.0f} for "
            f"{FAMILY_LABELS[by_absolute['model']]}. Its total is closer because its under- and "
            f"over-predictions offset, not because it is closer in each bucket. So these fits do "
            f"not show that a different dropout mechanism fixes the shortfall."
        )
    else:
        lines.append(
            f"{FAMILY_LABELS[by_net['model']]} is closest both net and bucket by bucket "
            f"({by_net['net']:+,.0f} net, {by_net['absolute']:,.0f} absolute)."
        )
    extent = "every" if under == len(buckets) else "nearly every"
    lines.extend(
        [
            "",
            f"BG/NBD under-predicts in {under} of {len(buckets)} buckets."
            + (
                f" A shortfall in {extent} bucket is consistent with the purchase rate itself "
                f"changing between the windows, which none of the three models represents."
                if under >= len(buckets) - 1
                else ""
            ),
            "",
        ]
    )
    return lines


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
        *_bucket_comparison(result),
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

    # The paper's figure is cumulative over both windows, so this computes the same basis: expected
    # repeat transactions across calibration and holdout together, against actual. The earlier text
    # said the report "gives both bases" and then never computed this one. Holdout purchases count
    # as repeats here because every customer's first purchase falls inside calibration.
    holdout = _required(result, CHAMPION, "purchases").aggregate
    cumulative_actual = result.calibration_fit.actual_total + holdout.actual_total
    cumulative_expected = result.calibration_fit.predicted_total + holdout.predicted_total
    cumulative_error = 100.0 * (cumulative_expected - cumulative_actual) / cumulative_actual

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
        f"percent over the first 39 weeks dilutes whatever happens in the second 39. On the "
        f"paper's cumulative basis, this project's BG/NBD expects "
        f"{cumulative_expected:,.0f} repeat transactions against {cumulative_actual:,.0f} "
        f"actual: **{cumulative_error:+.1f}%**, the figure to set beside the paper's. The two "
        f"still differ in sample, so they are not expected to match exactly.",
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
        f"{_required(result, CHAMPION, 'purchases').rank.rho:.3f} on purchase counts. The naive "
        f"ranking is each customer's calibration purchase rate times their past average order "
        f"value, and the modelling does not improve on it.",
        "",
        *_lowest_decile_lines(result, table),
    ]


def _lowest_decile_lines(result: ValidationResult, table: pd.DataFrame) -> list[str]:
    """Describe decile 10 from its measured composition.

    The first version explained decile 10 as mostly one-time buyers whose predictions differ only
    through customer age. The Phase 4 audit found that 45% of it are repeat buyers the model rates
    as probably lapsed, and that they out-spend the one-time buyers around them. The explanation
    now comes from that composition, and is only offered when the ordering actually inverts.
    """
    bottom = result.lowest_decile
    middle = table["mean_actual"].iloc[4:9]
    lowest = table["mean_actual"].iloc[-1]
    if bottom is None or lowest <= middle.min():
        return []
    return [
        f"The ordering is also mildly **inverted** at the bottom. Decile 10, the "
        f"lowest-predicted tenth, realises a mean of {lowest:,.2f} against "
        f"{middle.min():,.2f}-{middle.max():,.2f} across deciles 5-9. Of its "
        f"{bottom.customers:,} customers, {bottom.repeat_buyers:,} "
        f"({bottom.repeat_buyers / bottom.customers:.0%}) are repeat buyers, with a mean "
        f"probability alive of {bottom.repeat_mean_probability_alive:.2f}. They spent "
        f"{bottom.repeat_mean_actual:,.2f} on average in the holdout, against "
        f"{bottom.one_time_mean_actual:,.2f} for the one-time buyers in the same decile."
        + (
            " The model ranks these repeat buyers at the very bottom, yet they spend more in the "
            "holdout than the one-time buyers ranked alongside them. That is the same write-off "
            "the frequency buckets show."
            if bottom.repeat_mean_actual > bottom.one_time_mean_actual
            else ""
        ),
        "",
    ]


def _intervals_line(result: ValidationResult) -> str:
    """Describe the intervals from the table, or their absence.

    The full-Bayes branch used to quote "6-10%" and "0.6%" as fixed text from one past run. Any
    later run would have printed them unverified, and without the one-off label the sign-off
    conditions require. Every figure here is computed from the predictions being reported.
    """
    intervals = result.intervals
    if intervals is None:
        return (
            f"- **No uncertainty intervals.** This is a `{result.fit_method}` fit, and the "
            f"predictions table carries none. MAP returns a single point, so any interval "
            f"computed from it would be zero-width and read as certainty. Intervals require "
            f"`uv run ltv fit --full-bayes`."
        )
    return (
        f"- **The intervals are on the model's expectation, not on what a customer will do.** "
        f"This `{result.fit_method}` fit carries a 94% HDI on forward revenue for "
        f"{intervals.customers:,} customers. It expresses how precisely the population "
        f"parameters are pinned down, which here is very precisely: the median interval is "
        f"{intervals.median_relative_width:.1%} of its estimate (5th-95th percentile "
        f"{intervals.p5_relative_width:.1%}-{intervals.p95_relative_width:.1%}). Only "
        f"{intervals.coverage:.1%} of customers' realised holdout spend falls inside their own "
        f"interval. That is not a defect: an individual outcome is dominated by Poisson and "
        f"gamma variation the interval deliberately leaves out. It does mean these intervals "
        f"must never be presented as a range a customer's revenue is likely to land in."
    )


def _limits_section(result: ValidationResult) -> list[str]:
    lines = [
        "## What this report does not establish",
        "",
        _intervals_line(result),
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
                    f"{result.order_value_shared_estimate:,} of those customers had no repeat "
                    f"purchase in calibration. The spend model and the naive rule both give them "
                    f"the same population mean, so on those rows the two cannot differ, and the "
                    f"comparison above is diluted towards a tie.",
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
                *_monthly_lines(result),
                "Read the first three months of the chart as cohort formation rather than "
                "growth: every customer in this dataset makes their first purchase in Q1 1997, so "
                "the repeat series necessarily climbs while the cohort is still being acquired.",
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

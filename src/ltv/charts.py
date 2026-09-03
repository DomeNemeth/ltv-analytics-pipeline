"""Charts for the validation report.

Three figures, chosen because each answers a question a table cannot:

1. **Holdout purchases by calibration frequency** -- the canonical CLV validation chart. Where does
   the error live? An aggregate number says a model is 14% low; this says the shortfall is
   concentrated in customers who repeated once or twice.
2. **Monthly repeat occasions** -- why the error exists. The calibration window declines steeply and
   the holdout window flatly does not, which is a violated stationarity assumption rather than a
   parameter estimated badly. This is the figure that turns "our model is 14% off" into a diagnosis.
3. **Revenue by predicted decile** -- whether the ranking is usable. The project's business question
   is which segments to acquire, which is about order, not magnitude.

Matplotlib is imported inside the functions rather than at module scope. It costs seconds to import
and pulls a font cache build on a cold machine, and `ltv info` should not pay for that.

Everything here is deterministic: no timestamps, no random jitter, fixed figure sizes and a fixed
DPI. The PNGs are committed, so a re-run on unchanged data must produce the same bytes -- otherwise
every validation run shows up as a diff and nobody can tell which ones meant something.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ltv.config import Settings, get_settings
from ltv.models.challengers import CHAMPION
from ltv.warehouse import connect

if TYPE_CHECKING:  # pragma: no cover
    from ltv.validate import ValidationResult

#: Fixed so the committed PNGs are byte-stable across runs and machines.
FIGURE_DPI = 120
FIGURE_SIZE = (9.0, 5.0)

#: Readable in both the report and a README, and distinguishable in greyscale -- an interviewer may
#: well print it.
ACTUAL_STYLE = {"color": "#111111", "marker": "o", "linewidth": 2.2, "zorder": 3}
SERIES_STYLES = {
    CHAMPION: {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
    "baseline_rate": {"color": "#999999", "marker": "^", "linestyle": "--"},
    "mbg_nbd": {"color": "#d62728", "marker": "D", "linestyle": "-."},
    "pareto_nbd": {"color": "#2ca02c", "marker": "v", "linestyle": ":"},
}
SERIES_LABELS = {
    CHAMPION: "BG/NBD",
    "baseline_rate": "Naive: calibration rate",
    "mbg_nbd": "MBG/NBD",
    "pareto_nbd": "Pareto/NBD",
}


def _figure():
    import matplotlib

    # Non-interactive backend, set before pyplot is imported. Without it, a machine with no display
    # -- CI, or a Docker build in Phase 6 -- either fails outright or hangs waiting for a window.
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axes = pyplot.subplots(figsize=FIGURE_SIZE)
    return pyplot, figure, axes


def _save(pyplot, figure, path: Path) -> Path:
    figure.tight_layout()
    # metadata={"Date": None} strips the creation timestamp matplotlib otherwise writes into the PNG
    # header, which would make every re-run produce different bytes for an identical chart.
    figure.savefig(path, dpi=FIGURE_DPI, metadata={"Software": None, "Date": None})
    pyplot.close(figure)
    return path


def purchases_by_frequency(result: ValidationResult, path: Path) -> Path:
    """Actual against predicted holdout purchases, by calibration repeat count."""
    pyplot, figure, axes = _figure()
    table = result.buckets

    axes.plot(table["bucket"], table["actual"], label="Actual", **ACTUAL_STYLE)
    for family, style in SERIES_STYLES.items():
        if family in table.columns:
            axes.plot(table["bucket"], table[family], label=SERIES_LABELS[family], **style)

    axes.set_xlabel("Repeat purchases in the calibration window")
    axes.set_ylabel(f"Mean purchases in the {result.horizon_days}-day holdout")
    axes.set_title(
        f"Holdout purchases by calibration frequency ({result.source}, "
        f"{result.customers:,} customers)"
    )
    axes.legend(frameon=False)
    axes.grid(axis="y", alpha=0.3)
    return _save(pyplot, figure, path)


def monthly_occasions(source: str, path: Path, settings: Settings) -> Path:
    """Repeat purchase occasions per month, with the calibration cutoff marked.

    The diagnostic figure. BG/NBD explains a declining purchase rate as customers dropping out, and
    extrapolates the decline forward. If the real series declines and then levels off, the model
    keeps decaying and the forecast falls short -- which is a violated assumption, not a bad fit.

    Read straight from the occasion grain rather than from any summary, so it cannot inherit a
    windowing mistake from the models it is being used to explain.
    """
    with connect(settings, read_only=True) as connection:
        series = connection.execute(
            """
            with occasions as (
                select
                    occasions.order_date,
                    occasions.customer_id,
                    windows.calibration_end
                from int_customers__purchase_occasions as occasions
                inner join int_sources__analysis_windows as windows
                    on occasions.source = windows.source
                where occasions.source = ?
            ),

            -- Repeat occasions only: a customer's first ever purchase is acquisition, not repeat
            -- behaviour, and CDNOW's cohort all acquire in the first quarter. Leaving them in would
            -- put a spike at the start of the series that has nothing to do with retention.
            repeats as (
                select
                    order_date,
                    calibration_end
                from occasions
                qualify row_number() over (
                    partition by customer_id order by order_date
                ) > 1
            )

            select
                date_trunc('month', order_date) as month,
                count(*) as occasions,
                max(order_date <= calibration_end) as in_calibration
            from repeats
            group by 1
            order by 1
            """,
            [source],
        ).df()

    pyplot, figure, axes = _figure()
    calibration = series[series["in_calibration"]]
    holdout = series[~series["in_calibration"]]

    axes.plot(
        calibration["month"],
        calibration["occasions"],
        label="Calibration window (fitted)",
        **ACTUAL_STYLE,
    )
    axes.plot(
        holdout["month"],
        holdout["occasions"],
        label="Holdout window (predicted)",
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
    )
    if not holdout.empty and not calibration.empty:
        # Join the two series so the eye reads one trajectory rather than two unrelated lines.
        bridge = pd.concat([calibration.tail(1), holdout.head(1)])
        axes.plot(bridge["month"], bridge["occasions"], color="#1f77b4", linewidth=2.2)
        axes.axvline(holdout["month"].iloc[0], color="#d62728", linestyle="--", alpha=0.7)

    axes.set_xlabel("Month")
    axes.set_ylabel("Repeat purchase occasions")
    axes.set_title(f"Repeat occasions per month, {source} — the decline stops after the cutoff")
    axes.legend(frameon=False)
    axes.grid(axis="y", alpha=0.3)
    axes.set_ylim(bottom=0)
    return _save(pyplot, figure, path)


def revenue_by_decile(result: ValidationResult, path: Path) -> Path:
    """Predicted against actual holdout revenue, by predicted decile.

    The discrimination view. A flat actual line across deciles means the model orders customers no
    better than chance, whatever its error metrics say.
    """
    pyplot, figure, axes = _figure()
    table = result.deciles

    positions = range(len(table))
    width = 0.4
    axes.bar(
        [p - width / 2 for p in positions],
        table["mean_predicted"],
        width,
        label="Predicted",
        color="#1f77b4",
    )
    axes.bar(
        [p + width / 2 for p in positions],
        table["mean_actual"],
        width,
        label="Actual",
        color="#111111",
    )

    axes.set_xticks(list(positions))
    axes.set_xticklabels(table["decile"])
    axes.set_xlabel("Decile of predicted forward revenue (1 = highest)")
    axes.set_ylabel(f"Mean {result.horizon_days}-day revenue per customer ($)")
    axes.set_title(f"Predicted against actual revenue by decile ({result.source})")
    axes.legend(frameon=False)
    axes.grid(axis="y", alpha=0.3)
    return _save(pyplot, figure, path)


def write_charts(
    result: ValidationResult, frame: pd.DataFrame, settings: Settings | None = None
) -> tuple[Path, ...]:
    """Render every figure the report embeds, returning their paths in report order."""
    settings = settings or get_settings()
    reports = settings.reports_dir
    source = result.source

    return (
        purchases_by_frequency(result, reports / f"{source}_purchases_by_frequency.png"),
        monthly_occasions(source, reports / f"{source}_monthly_occasions.png", settings),
        revenue_by_decile(result, reports / f"{source}_revenue_by_decile.png"),
    )

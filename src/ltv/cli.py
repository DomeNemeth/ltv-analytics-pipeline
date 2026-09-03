"""Command line entry point.

Commands are added as each pipeline stage lands, so that ``ltv --help`` always reflects what the
repo can actually do rather than advertising stubs.
"""

from __future__ import annotations

import typer

from ltv import __version__
from ltv.config import get_settings
from ltv.ingest.cdnow import CDNOW_MASTER, CDNOWFormatError, ingest_cdnow
from ltv.ingest.fetch import SourceDataError
from ltv.transform import TransformError, run_dbt

app = typer.Typer(
    name="ltv",
    help="Probabilistic customer lifetime value pipeline.",
    no_args_is_help=True,
    add_completion=False,
)

ingest_app = typer.Typer(
    help="Load source data into the warehouse `raw` schema.",
    no_args_is_help=True,
)
app.add_typer(ingest_app, name="ingest")


@app.callback()
def main() -> None:
    """Keep Typer in subcommand mode.

    Without an explicit callback, Typer collapses a single-command app into a bare root command,
    which would break ``ltv info`` today and every command added in later phases.
    """


@app.command()
def info() -> None:
    """Show resolved configuration and whether the warehouse has been built."""
    settings = get_settings()
    warehouse = settings.duckdb_path
    built = "yes" if warehouse.exists() else "no (run the pipeline first)"

    typer.echo(f"ltv-analytics-pipeline {__version__}")
    typer.echo(f"  repo root         {settings.repo_root}")
    typer.echo(f"  warehouse         {warehouse}")
    typer.echo(f"  warehouse built   {built}")
    typer.echo(f"  raw data          {settings.raw_dir}")
    typer.echo(f"  reports           {settings.reports_dir}")
    typer.echo(f"  models            {settings.model_dir}")
    typer.echo(f"  calibration split {settings.calibration_weeks}w / {settings.holdout_weeks}w")
    typer.echo(f"  forward horizon   {settings.forward_horizon_days}d")


@ingest_app.command("cdnow")
def ingest_cdnow_command(
    force_download: bool = typer.Option(
        False, "--force-download", help="Re-download the source archive even if cached."
    ),
) -> None:
    """Load the CDNOW master dataset into raw.cdnow_transactions."""
    try:
        result = ingest_cdnow(get_settings(), force_download=force_download)
    except (SourceDataError, CDNOWFormatError) as exc:
        # Both carry messages written to be acted on. Showing them inside a Python traceback would
        # bury the one useful line under twenty useless ones.
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"loaded {result.table}")
    typer.echo(f"  source     {result.source_file}")
    typer.echo(f"  rows       {result.rows:,}")
    typer.echo(f"  customers  {result.customers:,}")
    typer.echo(f"  data from  {CDNOW_MASTER.citation}")


@app.command(
    # Anything after `transform` is handed to dbt untouched, so `ltv transform test --select
    # staging` works without this command having to mirror dbt's entire flag surface.
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def transform(ctx: typer.Context) -> None:
    """Build the dbt models and run their tests. Extra arguments are passed through to dbt."""
    try:
        run_dbt(ctx.args or None, get_settings())
    except (TransformError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def fit(
    source: str = typer.Option("cdnow", help="Which dataset to fit. Sources fit separately."),
    full_bayes: bool = typer.Option(
        False,
        "--full-bayes",
        help="Sample the posterior with NUTS instead of taking the MAP estimate. Much slower, and "
        "the only run whose uncertainty intervals mean anything.",
    ),
) -> None:
    """Fit BG/NBD + Gamma-Gamma on the calibration window and write customer predictions."""
    # Imported here rather than at module scope: PyMC pulls in a large dependency tree and compiles
    # nothing until asked, but the import alone is seconds. Paying that on `ltv info` would make the
    # whole CLI feel broken -- the same reasoning as the dbt import in transform.py.
    from ltv.fit import run_fit
    from ltv.models.clv import ModelError

    try:
        result = run_fit(source, full_bayes=full_bayes, settings=get_settings())
    except (ModelError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    horizons = ", ".join(f"{horizon}d" for horizon in result.horizons)
    typer.echo(f"fitted {result.source} ({result.method})")
    typer.echo(f"  customers        {result.customers:,}")
    typer.echo(
        f"  spend model on   {result.spend_customers:,} "
        f"({result.excluded_customers:,} excluded, no repeat spend)"
    )
    typer.echo(f"  horizons         {horizons}")
    typer.echo(f"  predictions      {result.rows_written:,} rows in model.customer_predictions")
    typer.echo(f"  saved            {result.purchase_artifact.name}, {result.spend_artifact.name}")

    if not full_bayes:
        # Said every run, not buried in docs. A MAP fit gives point estimates only, and the
        # temptation to quote an interval from one is exactly what CLAUDE.md section 6 forbids.
        typer.echo(
            "  note             MAP fit: no uncertainty intervals. Use --full-bayes for those."
        )


@app.command()
def validate(
    source: str = typer.Option(
        "cdnow", help="Which dataset to validate. Sources score separately."
    ),
    compare_models: bool = typer.Option(
        False,
        "--compare-models",
        help="Also fit MBG/NBD and Pareto/NBD on the same calibration frame and compare them. "
        "Slower, and the answer to whether the error is this model's assumptions or the data's.",
    ),
) -> None:
    """Score the fitted predictions against the holdout window and write the validation report."""
    # Lazy, for the same reason `fit` is: this pulls matplotlib and (with --compare-models) PyMC,
    # and paying seconds of import on `ltv info` would make the whole CLI feel broken.
    from ltv.transform import TransformError
    from ltv.validate import ValidationError, run_validation

    try:
        result = run_validation(source, compare_models=compare_models, settings=get_settings())
    except (ValidationError, TransformError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    purchases = result.score("bg_nbd", "purchases")
    # The same-period rule, not the calibration-rate one. The rate baselines divide by each
    # customer's observation length and multiply by the holdout length, which inflates them
    # whenever mean customer age is below the holdout -- echoing one of those beside the model
    # overstates the win, which is exactly what a validation audit caught this command doing.
    baseline = result.score("baseline_carry_forward", "purchases")
    floor = result.score("baseline_zero", "purchases")

    typer.echo(f"validated {result.source} ({result.fit_method})")
    typer.echo(f"  customers        {result.customers:,}")
    typer.echo(f"  horizon          {result.horizon_days}d (the holdout window)")

    if purchases and baseline:
        typer.echo(
            f"  purchases        {purchases.aggregate.predicted_total:,.0f} predicted vs "
            f"{purchases.aggregate.actual_total:,.0f} actual "
            f"({purchases.aggregate.percent_error:+.1f}%)"
        )
        # The baseline is echoed beside the model on purpose. An error number without something to
        # beat is not a result, and printing only the model's is how it becomes one by accident.
        floor_note = f", {floor.mae:.3f} predicting nobody buys" if floor else ""
        typer.echo(
            f"  MAE per customer {purchases.mae:.3f} model vs {baseline.mae:.3f} naive "
            f"(same as last window){floor_note}"
        )

    for fit in result.challengers:
        status = f"fitted in {fit.seconds:.0f}s" if fit.succeeded else f"FAILED -- {fit.error}"
        typer.echo(f"  {fit.family:<16} {status}")

    typer.echo(f"  metrics          {result.metrics_written:,} rows in model.validation_metrics")
    typer.echo(f"  report           {result.report_path.name}")
    typer.echo(f"  charts           {', '.join(path.name for path in result.chart_paths)}")

    if result.fit_method == "map":
        typer.echo("  note             MAP fit: no uncertainty intervals in these metrics.")


if __name__ == "__main__":
    app()

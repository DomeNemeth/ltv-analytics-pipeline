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
    typer.echo(f"  calibration split {settings.calibration_weeks}w / {settings.holdout_weeks}w")


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


if __name__ == "__main__":
    app()

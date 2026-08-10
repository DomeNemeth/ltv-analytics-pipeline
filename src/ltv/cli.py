"""Command line entry point.

Commands are added as each pipeline stage lands, so that ``ltv --help`` always reflects what the
repo can actually do rather than advertising stubs.
"""

from __future__ import annotations

import typer

from ltv import __version__
from ltv.config import get_settings

app = typer.Typer(
    name="ltv",
    help="Probabilistic customer lifetime value pipeline.",
    no_args_is_help=True,
    add_completion=False,
)


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


if __name__ == "__main__":
    app()

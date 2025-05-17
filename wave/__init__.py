"""
WaveSeer package
"""

from typing import List

import typer

# Create Typer app instance before importing commands that use it
app = typer.Typer()

# Import modules that depend on app
from .ingest import ingest as cli_ingest
from .scan import scan as cli_scan
from .cluster import app as cluster_app
from .api import app as api_app
from .seer import app as seer_app


@app.command("ingest")
def _ingest_cmd(
    all_: bool = typer.Option(False, "--all", "-a",
                           help="Ingest all symbols/timeframes from config"),
    symbols: List[str] = typer.Option([], "--symbol", "-s",
                                   help="Symbols to ingest (repeatable)"),
    tfs: List[str] = typer.Option([], "--tf", "-t",
                               help="Timeframes to ingest (repeatable)"),
):
    """CSV → Parquet ingestion"""
    return cli_ingest(all_, symbols, tfs)


app.command("scan")(cli_scan)
app.add_typer(cluster_app, name="cluster")
app.add_typer(api_app, name="api")
app.add_typer(seer_app, name="seer")


@app.command("ui")
def ui(
    host: str = typer.Option("127.0.0.1", help="UI host"),
    port: int = typer.Option(8050, help="UI port"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """Launch Dash UI"""
    from .ui.app import run as run_ui
    run_ui()

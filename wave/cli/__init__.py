"""Command-line interface for Waveseer pattern detection."""

import typer
from typing import Optional
import importlib.metadata

# Create the CLI app
app = typer.Typer(
    name="waveseer",
    help="Pattern detection in financial time series data using machine learning",
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    """Get the version of the package."""
    if value:
        try:
            version = importlib.metadata.version("waveseer")
            typer.echo(f"Waveseer version: {version}")
        except importlib.metadata.PackageNotFoundError:
            typer.echo("Waveseer package not installed")
        raise typer.Exit()

# Import commands from submodules to register them with the CLI
# Dynamically import modules, ignore failures (e.g., circular dependencies)
for _mod in ('train_cmd', 'detect_cmd', 'analyze_cmd', 'api_cmd', 'export_cmd', 'crypto_cmd'):
    try:
        __import__(f'wave.cli.{_mod}', fromlist=[_mod])
    except ImportError:
        pass

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=_version_callback, is_eager=True,
        help="Show version and exit."
    ),
) -> None:
    """Waveseer CLI for pattern detection in financial time series data."""
    pass

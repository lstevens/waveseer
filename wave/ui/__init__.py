import typer
from .app import run as _run_dash

app = typer.Typer()

@app.command("ui")
def ui_cmd():
    """Launch Dash UI"""
    _run_dash()

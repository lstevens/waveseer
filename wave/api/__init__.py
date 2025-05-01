import typer
import uvicorn
import yaml
from pathlib import Path

app = typer.Typer()

@app.command("api")
def api_cmd(
    reload: bool = typer.Option(False, "--reload", help="Enable hot reload")
):
    """Launch Pattern API"""
    cfg = yaml.safe_load(Path("config.yml").read_text())
    host = cfg["pattern_api"]["host"]
    port = cfg["pattern_api"]["port"]
    uvicorn.run("wave.api.app:app", host=host, port=port, reload=reload)

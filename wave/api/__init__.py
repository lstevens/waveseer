import typer
import uvicorn
import yaml
from pathlib import Path

app = typer.Typer()


def create_app():
    """
    Factory function that returns the FastAPI application instance.
    This function exists for testing purposes to allow test code to create
    app instances with specific configurations.
    
    Returns:
        FastAPI: The configured FastAPI application
    """
    # Import here to avoid circular imports
    from wave.api.app import app as fastapi_app
    return fastapi_app


@app.command("api")
def api_cmd(
    reload: bool = typer.Option(False, "--reload", help="Enable hot reload")
):
    """Launch Pattern API"""
    cfg = yaml.safe_load(Path("config.yml").read_text())
    host = cfg["pattern_api"]["host"]
    port = cfg["pattern_api"]["port"]
    uvicorn.run("wave.api.app:app", host=host, port=port, reload=reload)

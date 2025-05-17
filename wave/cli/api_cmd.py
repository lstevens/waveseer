"""Commands for running the Waveseer API server."""

import typer
from typing import Optional, List
import os
from pathlib import Path
import yaml

from wave.cli import app
import uvicorn

api_app = typer.Typer(name="api")
app.add_typer(api_app, name="api", help="Run and manage the Waveseer API server")


@api_app.command("serve")
def serve_api(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h", help="Host address to bind the server"
    ),
    port: int = typer.Option(
        9000, "--port", "-p", help="Port to bind the server"
    ),
    reload: bool = typer.Option(
        False, "--reload", help="Enable hot reload for development"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    models_dir: Optional[Path] = typer.Option(
        None, "--models-dir", "-m", help="Directory containing exported models"
    ),
    log_level: str = typer.Option(
        "info", "--log-level", "-l", help="Logging level (debug, info, warning, error)"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Number of worker processes (use >1 for production)"
    ),
    allow_origins: List[str] = typer.Option(
        ["*"], "--allow-origin", help="CORS allowed origins"
    ),
) -> None:
    """Run the Waveseer pattern detection API server."""
    # Load configuration from file if provided
    config = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.yml' or config_file.suffix.lower() == '.yaml':
                    config = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    import json
                    config = json.load(f)
                else:
                    typer.echo(f"Unsupported config file format: {config_file.suffix}")
        except Exception as e:
            typer.echo(f"Error loading config file: {str(e)}")

    # Override with command line args
    if 'pattern_api' not in config:
        config['pattern_api'] = {}

    config['pattern_api']['host'] = host
    config['pattern_api']['port'] = port

    # Set models directory environment variable if provided
    if models_dir:
        os.environ["WAVESEER_MODELS_DIR"] = str(models_dir)

    # Save config to temporary file if it doesn't exist
    if not os.path.exists('config.yml'):
        with open('config.yml', 'w') as f:
            yaml.dump(config, f)
        typer.echo("Created temporary config.yml file")

    # Run API server
    typer.echo(f"Starting Waveseer API server at http://{host}:{port}")
    typer.echo(f"API documentation will be available at http://{host}:{port}/docs")

    if models_dir:
        typer.echo(f"Using models from {models_dir}")

    # Run uvicorn server
    uvicorn.run(
        "wave.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        workers=workers
    )


@api_app.command("test")
def test_api(
    url: str = typer.Option(
        "http://127.0.0.1:9000", "--url", help="URL of the Waveseer API"
    ),
) -> None:
    """Test connection to the Waveseer API server."""
    import requests

    typer.echo(f"Testing connection to {url}...")

    try:
        # Test health endpoint
        health_response = requests.get(f"{url}/health")
        if health_response.status_code == 200:
            typer.echo(f"✅ Health check successful: {health_response.json()}")
        else:
            typer.echo(f"❌ Health check failed: {health_response.status_code}")

        # Test models endpoint
        models_response = requests.get(f"{url}/models")
        if models_response.status_code == 200:
            models = models_response.json()
            typer.echo(f"✅ Models endpoint successful: {len(models)} models available")
            for model in models:
                typer.echo(f"  - {model['name']} ({model['type']})")
        else:
            typer.echo(f"❌ Models endpoint failed: {models_response.status_code}")

        # Test root endpoint
        root_response = requests.get(url)
        if root_response.status_code == 200:
            typer.echo(f"✅ Root endpoint successful: {root_response.json().get('name')}")
            typer.echo(f"  API documentation: {url}/docs")
        else:
            typer.echo(f"❌ Root endpoint failed: {root_response.status_code}")

    except requests.RequestException as e:
        typer.echo(f"❌ Connection error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error testing API: {str(e)}")
        raise typer.Exit(1)

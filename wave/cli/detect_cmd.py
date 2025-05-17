"""Commands for detecting patterns in time series data."""

import typer
from typing import Optional, List
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wave.cli import app

detect_app = typer.Typer(name="detect")
app.add_typer(detect_app, name="detect", help="Detect patterns in time series data")


@detect_app.command("file")
def detect_from_file(
    input_file: Path = typer.Argument(
        ..., help="CSV or JSON file containing time series data"
    ),
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to exported model file"
    ),
    column: str = typer.Option(
        "close", "--column", "-c", help="Column name containing price data (for CSV)"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save detection results"
    ),
    visualize: bool = typer.Option(
        False, "--visualize/--no-visualize", help="Visualize detected patterns"
    ),
    threshold: float = typer.Option(
        0.6, "--threshold", "-t", help="Detection confidence threshold"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show verbose output"),
) -> None:
    """Detect patterns in time series data from a file."""
    from wave.ml.export.model_export import load_exported_model

    try:
        # Load data
        typer.echo(f"Loading data from {input_file}...")
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
            if column not in df.columns:
                typer.echo(f"Error: Column '{column}' not found in CSV file.")
                raise typer.Exit(1)
            series = df[column].values
        elif input_file.suffix.lower() == '.json':
            with open(input_file, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                series = np.array(data)
            elif isinstance(data, dict) and column in data:
                series = np.array(data[column])
            else:
                typer.echo(f"Error: Could not find '{column}' data in JSON file.")
                raise typer.Exit(1)
        else:
            typer.echo(f"Error: Unsupported file format: {input_file.suffix}")
            raise typer.Exit(1)

        # Normalize the series
        if len(series) < 10:
            typer.echo(f"Error: Time series too short ({len(series)} points). Need at least 10 points.")
            raise typer.Exit(1)

        typer.echo(f"Loaded {len(series)} data points.")

        # Load model
        typer.echo(f"Loading model from {model_path}...")
        model = load_exported_model(model_path)

        # Prepare input for the model
        # Min-max normalization
        series_min, series_max = series.min(), series.max()
        if series_max > series_min:
            norm_series = (series - series_min) / (series_max - series_min)
        else:
            norm_series = np.ones_like(series) * 0.5

        import torch
        input_tensor = torch.tensor(norm_series, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Perform inference
        typer.echo("Detecting patterns...")
        with torch.no_grad():
            outputs = model(input_tensor)

        # Process outputs based on model type
        if isinstance(outputs, tuple):
            # Some models return both class predictions and attention scores
            predictions, attentions = outputs
        else:
            predictions = outputs
            attentions = None

        # Extract results based on model output shape
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # Multi-class prediction
            confidence, pattern_idx = torch.max(torch.softmax(predictions, dim=1), dim=1)
            confidence = confidence.item()
            pattern_type = pattern_idx.item()
        else:
            # Binary or regression output
            confidence = torch.sigmoid(predictions).item() if predictions.size(1) == 1 else predictions.item()
            pattern_type = 1 if confidence > threshold else 0

        # Output results
        typer.echo(f"Detection results:")
        typer.echo(f"  Pattern type: {pattern_type}")
        typer.echo(f"  Confidence: {confidence:.4f}")

        # Create result dictionary
        result = {
            "pattern_type": int(pattern_type),
            "confidence": float(confidence),
            "threshold": threshold,
            "input_length": len(series),
        }

        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            typer.echo(f"Results saved to {output_file}")

        # Visualize if requested
        if visualize:
            plt.figure(figsize=(10, 6))
            plt.plot(series)
            plt.title(f"Detected Pattern (Type: {pattern_type}, Confidence: {confidence:.4f})")
            plt.xlabel("Time")
            plt.ylabel("Value")

            # If we have attention, add it as heatmap
            if attentions is not None:
                att = attentions.squeeze().cpu().numpy()
                ax2 = plt.gca().twinx()
                ax2.set_ylabel('Attention', color='r')
                ax2.plot(att, color='r', alpha=0.5)
                ax2.tick_params(axis='y', labelcolor='r')

            plt.tight_layout()
            plt.show()

    except Exception as e:
        typer.echo(f"Error during pattern detection: {str(e)}")
        if verbose:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(1)


@detect_app.command("api")
def detect_from_api(
    api_url: str = typer.Option(
        "http://127.0.0.1:9000", "--url", help="URL of the Waveseer API"
    ),
    sequence: List[float] = typer.Option(
        ..., "--seq", help="Sequence of values to analyze"
    ),
    timeframe: str = typer.Option(
        "1h", "--tf", help="Timeframe for the sequence"
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use for detection"
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Path to save detection results"
    ),
) -> None:
    """Detect patterns by calling the Waveseer API."""
    import requests

    try:
        # Prepare request data
        data = {
            "tf": timeframe,
            "seq": sequence,
            "use_ml": True,
        }
        if model_name:
            data["model_name"] = model_name

        # Make API request
        typer.echo(f"Sending request to {api_url}/match...")
        response = requests.post(f"{api_url}/match", json=data)

        # Check response
        if response.status_code != 200:
            typer.echo(f"Error: API returned status code {response.status_code}")
            typer.echo(response.text)
            raise typer.Exit(1)

        # Parse response
        result = response.json()
        typer.echo(f"Detection results:")
        typer.echo(f"  Pattern ID: {result['pattern_id']}")
        typer.echo(f"  Confidence: {result['score']:.4f}")
        typer.echo(f"  Distance: {result['dist']:.4f}")
        if 'pattern_type' in result and result['pattern_type']:
            typer.echo(f"  Pattern type: {result['pattern_type']}")
        if 'ml_model' in result and result['ml_model']:
            typer.echo(f"  Model used: {result['ml_model']}")

        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            typer.echo(f"Results saved to {output_file}")

    except requests.RequestException as e:
        typer.echo(f"Error connecting to API: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error during pattern detection: {str(e)}")
        raise typer.Exit(1)

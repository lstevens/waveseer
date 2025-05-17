"""Commands for training Waveseer pattern detection models."""

import typer
from typing import Optional
from pathlib import Path
import json
import os

from wave.cli import app
from wave.ml.train.train_patterns import run_training
from wave.ml.train.hyperparam_search import run_hyperparameter_search

train_app = typer.Typer(name="train")
app.add_typer(train_app, name="train", help="Train pattern detection models")


@train_app.command("model")
def train_model(
    model_type: str = typer.Option(
        "cnn", "--model-type", "-m", help="Model architecture type (cnn, lstm, transformer, hybrid)"
    ),
    data_path: Path = typer.Option(
        "data/processed", "--data-path", "-d", help="Path to training data directory"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to model configuration JSON file"
    ),
    output_dir: Path = typer.Option(
        "models", "--output-dir", "-o", help="Directory to save trained model"
    ),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate"),
    device: str = typer.Option("auto", "--device", help="Device to use (cpu, cuda, auto)"),
    tensorboard: bool = typer.Option(True, "--tensorboard/--no-tensorboard", help="Log to TensorBoard"),
    early_stopping: bool = typer.Option(True, "--early-stopping/--no-early-stopping", help="Use early stopping"),
    patience: int = typer.Option(10, "--patience", "-p", help="Early stopping patience"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show verbose output"),
) -> None:
    """Train a pattern detection model with specified configuration."""
    typer.echo(f"Training {model_type} model...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load config if specified
    config = {}
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Set config parameters from command line arguments if not in config
    if "model_type" not in config:
        config["model_type"] = model_type
    if "batch_size" not in config:
        config["batch_size"] = batch_size
    if "learning_rate" not in config:
        config["learning_rate"] = learning_rate
    if "epochs" not in config:
        config["epochs"] = epochs
    if "device" not in config:
        config["device"] = device
    if "tensorboard" not in config:
        config["tensorboard"] = tensorboard
    if "early_stopping" not in config:
        config["early_stopping"] = early_stopping
    if "patience" not in config:
        config["patience"] = patience

    # Run training
    model, metrics = run_training(
        data_path=data_path,
        config=config,
        output_dir=output_dir,
        verbose=verbose
    )

    typer.echo(f"Training complete. Model saved to {output_dir}")
    typer.echo(f"Final validation metrics: {metrics}")


@train_app.command("hyperparam")
def train_hyperparam(
    model_type: str = typer.Option(
        "cnn", "--model-type", "-m", help="Model architecture type (cnn, lstm, transformer, hybrid)"
    ),
    param_space_path: Path = typer.Option(
        ..., "--param-space", "-p", help="Path to parameter space JSON file"
    ),
    data_path: Path = typer.Option(
        "data/processed", "--data-path", "-d", help="Path to training data directory"
    ),
    output_dir: Path = typer.Option(
        "hyperparam_results", "--output-dir", "-o", help="Directory to save results"
    ),
    n_trials: int = typer.Option(10, "--n-trials", "-n", help="Number of trials"),
    search_method: str = typer.Option(
        "random", "--method", help="Search method (random, grid)"
    ),
    tensorboard: bool = typer.Option(True, "--tensorboard/--no-tensorboard", help="Log to TensorBoard"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Show verbose output"),
) -> None:
    """Run hyperparameter search for pattern detection models."""
    typer.echo(f"Running hyperparameter search for {model_type} model...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load parameter space
    with open(param_space_path, 'r') as f:
        param_space = json.load(f)

    # Run hyperparameter search
    best_params, best_metrics = run_hyperparameter_search(
        model_type=model_type,
        param_space=param_space,
        data_path=data_path,
        output_dir=output_dir,
        n_trials=n_trials,
        search_method=search_method,
        tensorboard=tensorboard,
        verbose=verbose
    )

    typer.echo(f"Hyperparameter search complete. Results saved to {output_dir}")
    typer.echo(f"Best parameters: {best_params}")
    typer.echo(f"Best metrics: {best_metrics}")

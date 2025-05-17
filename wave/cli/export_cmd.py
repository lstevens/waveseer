"""Commands for exporting and optimizing trained models."""

import typer
from typing import Optional, List
from pathlib import Path
import os
import torch

from wave.cli import app
from wave.ml.export.model_export import (
    export_to_torchscript,, 
    quantize_model,, 
    optimize_for_inference,, 
    export_model_config,, 
    export_for_production,, 
    get_model_size
)
from wave.ml.export.benchmark import benchmark_model_variants, generate_benchmark_report

export_app = typer.Typer(name="export")
app.add_typer(export_app, name="export", help="Export and optimize trained models")


@export_app.command("model")
def export_model(
    model_path: Path = typer.Argument(
        ..., help="Path to trained PyTorch model (.pt or .pth file)"
    ),
    output_dir: Path = typer.Option(
        "exported_models", "--output-dir", "-o", help="Directory to save exported model"
    ),
    model_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Name for exported model (default: derive from input)"
    ),
    quantize: bool = typer.Option(
        True, "--quantize/--no-quantize", help="Apply quantization to reduce model size"
    ),
    quantization_type: str = typer.Option(
        "dynamic", "--quant-type", help="Quantization type (dynamic or static)"
    ),
    optimize: bool = typer.Option(
        True, "--optimize/--no-optimize", help="Apply optimizations for inference"
    ),
    target_platforms: List[str] = typer.Option(
        ["default"], "--platform", "-p", help="Target platforms (default, mobile, cpu, gpu)"
    ),
    sequence_length: int = typer.Option(
        100, "--seq-length", "-s", help="Example sequence length for model input"
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", help="Show verbose output"
    ),
) -> None:
    """Export a trained model to optimized format for deployment."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Derive model name if not provided
        if model_name is None:
            model_name = Path(model_path).stem

        typer.echo(f"Exporting model {model_path} as {model_name}...")

        # Create example input
        example_input = torch.randn(1, 1, sequence_length)

        # Export the model
        export_paths = export_for_production(
            model_path=model_path,
            example_input=example_input,
            output_dir=output_dir,
            model_name=model_name,
            quantize=quantize,
            quantization_type=quantization_type,
            optimize=optimize,
            target_platforms=target_platforms,
            verbose=verbose
        )

        # Print results
        typer.echo(f"Export complete. Generated files:")
        for platform, path in export_paths.items():
            size = get_model_size(path)
            size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.2f} MB"
            typer.echo(f"  {platform}: {path} ({size_str})")

    except Exception as e:
        typer.echo(f"Error exporting model: {str(e)}")
        if verbose:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(1)


@export_app.command("benchmark")
def benchmark_models(
    model_dir: Path = typer.Argument(
        ..., help="Directory containing models to benchmark"
    ),
    output_file: Path = typer.Option(
        "benchmark_results.json", "--output", "-o", help="Path to save benchmark results"
    ),
    sequence_length: int = typer.Option(
        100, "--seq-length", "-s", help="Sequence length for benchmark inputs"
    ),
    num_runs: int = typer.Option(
        10, "--num-runs", "-n", help="Number of runs for each benchmark"
    ),
    model_filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="Filter models by name (e.g., 'cnn' or 'quantized')"
    ),
    include_plots: bool = typer.Option(
        True, "--plots/--no-plots", help="Include visualization plots in report"
    ),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", help="Show verbose output"
    ),
) -> None:
    """Benchmark multiple model variants and compare performance."""
    try:
        # Ensure model directory exists
        if not model_dir.exists() or not model_dir.is_dir():
            typer.echo(f"Error: Model directory {model_dir} does not exist.")
            raise typer.Exit(1)

        # Find model files
        model_files = list(model_dir.glob("*.pt"))
        if not model_files:
            typer.echo(f"Error: No model files (.pt) found in {model_dir}.")
            raise typer.Exit(1)

        # Filter models if specified
        if model_filter:
            model_files = [f for f in model_files if model_filter in f.name]
            if not model_files:
                typer.echo(f"Error: No models matching filter '{model_filter}' found.")
                raise typer.Exit(1)

        typer.echo(f"Found {len(model_files)} models to benchmark.")

        # Create model paths dictionary
        model_paths = {f.stem: str(f) for f in model_files}

        # Create example input data
        input_data = torch.randn(1, 1, sequence_length)

        # Run benchmarks
        typer.echo(f"Running benchmarks with {num_runs} iterations each...")
        results = benchmark_model_variants(
            model_paths=model_paths,
            input_data=input_data,
            num_runs=num_runs,
            verbose=verbose
        )

        # Generate report
        typer.echo(f"Generating benchmark report...")
        report_file = generate_benchmark_report(
            results=results,
            output_path=output_file,
            include_plots=include_plots
        )

        # Print summary
        typer.echo(f"Benchmark complete. Results saved to {output_file}")

        # Print basic comparison
        typer.echo("\nPerformance Summary:")
        print_format = "{:<25} {:<15} {:<15} {:<15}"
        typer.echo(print_format.format("Model", "Avg Time (ms)", "Size", "Rel. Size"))
        typer.echo("-" * 70)

        # Find the largest model for relative comparison
        largest_size = max(r['model_size'] for r in results.values())

        for model_name, result in results.items():
            avg_time = result['avg_inference_time'] * 1000  # Convert to ms
            size = result['model_size']
            size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.2f} MB"
            rel_size = result['model_size'] / largest_size
            rel_size_str = f"{rel_size:.2%}"

            typer.echo(print_format.format(model_name, f"{avg_time:.2f}", size_str, rel_size_str))

    except Exception as e:
        typer.echo(f"Error benchmarking models: {str(e)}")
        if verbose:
            import traceback
            typer.echo(traceback.format_exc())
        raise typer.Exit(1)

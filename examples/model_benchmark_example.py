"""
Example script for benchmarking optimized pattern detection models.

This script demonstrates how to use the benchmark utilities to compare
different optimization techniques and model variants.
"""

import os
import sys
import torch
import argparse
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

# Add the project root to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from wave.ml.models.cnn import CNNPatternModel
from wave.ml.models.lstm import LSTMPatternModel
from wave.ml.models.transformer import TransformerPatternModel
from wave.ml.export.model_export import (
    export_for_production,, 
    measure_inference_speed
)
from wave.ml.export.benchmark import (
    benchmark_model_variants,, 
    generate_benchmark_report,, 
    BenchmarkResult
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def generate_synthetic_dataset(num_samples: int, sequence_length: int, num_classes: int):
    """
    Generate a synthetic dataset for benchmarking.

    Args:
        num_samples: Number of samples to generate
        sequence_length: Length of each sequence
        num_classes: Number of classes

    Returns:
        Tuple of (inputs, labels)
    """
    # Generate synthetic input data
    inputs = torch.randn(num_samples, 1, sequence_length)

    # Generate synthetic labels
    labels = torch.randint(0, num_classes, (num_samples,))

    return inputs, labels


def main():
    """
    Main function for the model benchmark example.
    """
    parser = argparse.ArgumentParser(description="Benchmark pattern detection models")

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["cnn", "lstm", "transformer"],
        choices=["cnn", "lstm", "transformer"],
        help="Models to benchmark"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results"
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Sequence length for model input"
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of classes"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for benchmarking"
    )

    parser.add_argument(
        "--export-dir",
        type=str,
        default="exported_models",
        help="Directory to save exported models"
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.export_dir, exist_ok=True)

    # Generate synthetic dataset
    logger.info(f"Generating synthetic dataset with {args.num_samples} samples")
    inputs, labels = generate_synthetic_dataset(
        args.num_samples, args.sequence_length, args.num_classes)

    # Example input for model export (batch size 1)
    example_input = inputs[:1].clone()

    # Dictionary to store all benchmark results
    all_results = []

    # Benchmark each model
    for model_type in args.models:
        logger.info(f"Benchmarking {model_type} model...")

        # Create model
        if model_type == "cnn":
            model = CNNPatternModel(
                n_features=1,
                n_classes=args.num_classes,
                hidden_size=64,
                kernel_sizes=[3, 5, 7],
                channels=[32, 64, 128],
                dropout=0.2
            )
        elif model_type == "lstm":
            model = LSTMPatternModel(
                n_features=1,
                n_classes=args.num_classes,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
                bidirectional=True
            )
        elif model_type == "transformer":
            model = TransformerPatternModel(
                n_features=1,
                n_classes=args.num_classes,
                hidden_size=64,
                n_heads=4,
                num_layers=2,
                dropout=0.2,
                dim_feedforward=128
            )

        # Set model to evaluation mode
        model.eval()

        # Measure original model inference speed
        logger.info(f"Measuring original {model_type} model inference speed...")
        original_speed = measure_inference_speed(model, example_input)

        # Export model variants
        logger.info(f"Exporting {model_type} model variants...")
        exported_paths = export_for_production(
            model=model,
            example_input=example_input,
            output_dir=os.path.join(args.export_dir, model_type),
            model_name=f"{model_type}_pattern_model",
            quantize=True,
            optimize=True,
            target_platforms=["default", "mobile"]
        )

        # Benchmark model variants
        logger.info(f"Benchmarking {model_type} model variants...")
        benchmark_results = benchmark_model_variants(
            model_paths=exported_paths,
            input_data=example_input,
            model_name=model_type,
            num_runs=3,
            measure_memory=True
        )

        # Add results to the collection
        all_results.extend(benchmark_results)

        # Generate individual report for this model
        model_report_path = os.path.join(args.output_dir, f"{model_type}_benchmark_report.json")
        generate_benchmark_report(benchmark_results, model_report_path, include_plots=True)

    # Generate combined report for all models
    if all_results:
        combined_report_path = os.path.join(args.output_dir, "combined_benchmark_report.json")
        generate_benchmark_report(all_results, combined_report_path, include_plots=True)

        # Generate additional comparison plots
        plot_model_comparisons(all_results, args.output_dir)


def plot_model_comparisons(results: List[BenchmarkResult], output_dir: str):
    """
    Generate additional comparison plots across different models.

    Args:
        results: List of benchmark results
        output_dir: Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Extract data for plotting
    model_names = []
    variant_names = []
    inference_times = []
    sizes_kb = []

    for result in results:
        model_variant = f"{result.model_name}-{result.variant_name}"
        model_names.append(result.model_name)
        variant_names.append(model_variant)
        inference_times.append(result.inference_time_ms)
        sizes_kb.append(result.size_bytes / 1024)

    # Plot inference time comparison across all models and variants
    plt.figure(figsize=(12, 8))
    bars = plt.bar(variant_names, inference_times)

    # Color-code by model type
    unique_models = list(set(model_names))
    colors = plt.cm.tab10(range(len(unique_models)))
    color_map = {model: colors[i] for i, model in enumerate(unique_models)}

    for i, bar in enumerate(bars):
        bar.set_color(color_map[model_names[i]])

    plt.title("Inference Time Comparison Across Models")
    plt.xlabel("Model Variant")
    plt.ylabel("Inference Time (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', alpha=0.3)

    # Add legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[model]) for model in unique_models]
    plt.legend(legend_handles, unique_models, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "all_models_inference_time.png"))
    plt.close()

    # Plot model size comparison
    plt.figure(figsize=(12, 8))
    bars = plt.bar(variant_names, sizes_kb)

    # Color-code by model type
    for i, bar in enumerate(bars):
        bar.set_color(color_map[model_names[i]])

    plt.title("Model Size Comparison Across Models")
    plt.xlabel("Model Variant")
    plt.ylabel("Size (KB)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', alpha=0.3)

    # Add legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[model]) for model in unique_models]
    plt.legend(legend_handles, unique_models, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "all_models_size.png"))
    plt.close()


if __name__ == "__main__":
    main()

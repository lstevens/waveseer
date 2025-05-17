"""
Benchmarking utilities for pattern detection models.

This module provides tools for measuring and comparing model performance,
including inference speed, memory usage, and accuracy across different hardware.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from wave.ml.export.model_export import (
    load_exported_model,, 
    measure_inference_speed,, 
    get_model_size
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results of a model benchmark."""

    model_name: str
    variant_name: str
    size_bytes: int
    inference_time_ms: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    memory_usage_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "variant_name": self.variant_name,
            "size_bytes": self.size_bytes,
            "size_kb": self.size_bytes / 1024,
            "size_mb": self.size_bytes / (1024 * 1024),
            "inference_time_ms": self.inference_time_ms,
            "inferences_per_second": 1000 / self.inference_time_ms,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "memory_usage_mb": self.memory_usage_mb
        }


def benchmark_model_variants(
    model_paths: Dict[str, str],
    input_data: Union[torch.Tensor, DataLoader],
    true_labels: Optional[torch.Tensor] = None,
    model_name: str = "model",
    num_runs: int = 3,
    warmup_iterations: int = 10,
    measure_memory: bool = False
) -> List[BenchmarkResult]:
    """Benchmark different variants of a model.

    Args:
        model_paths: Dictionary mapping variant names to model paths
        input_data: Input tensor or DataLoader for inference
        true_labels: Optional tensor of true labels for accuracy metrics
        model_name: Name of the model being benchmarked
        num_runs: Number of benchmark runs per model
        warmup_iterations: Number of warm-up iterations before benchmarking
        measure_memory: Whether to measure memory usage (may not be accurate on all systems)

    Returns:
        List of BenchmarkResult objects for each variant

    Raises:
        FileNotFoundError: If any model file is not found
        RuntimeError: If benchmarking fails
    """
    results = []

    for variant_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            logger.info(f"Benchmarking {variant_name} variant...")

            # Get model size
            size_bytes = get_model_size(model_path)

            # Load the model
            model = load_exported_model(model_path)
            model.eval()

            # Prepare for inference
            if isinstance(input_data, DataLoader):
                # Use first batch for speed benchmarking
                for batch in input_data:
                    if isinstance(batch, (list, tuple)):
                        benchmark_input = batch[0]
                    else:
                        benchmark_input = batch
                    break
            else:
                benchmark_input = input_data

            # Measure inference time
            inference_times = []
            for run in range(num_runs):
                # Run warm-up iterations
                with torch.no_grad():
                    for _ in range(warmup_iterations):
                        _ = model(benchmark_input)

                # Measure inference time
                start_time = time.time()
                with torch.no_grad():
                    _ = model(benchmark_input)
                end_time = time.time()

                inference_time_ms = (end_time - start_time) * 1000
                inference_times.append(inference_time_ms)

            # Average inference time
            avg_inference_time = np.mean(inference_times)

            # Measure memory usage if requested
            memory_usage = None
            if measure_memory:
                try:
                    import psutil
                    import gc

                    # Force garbage collection
                    gc.collect()

                    # Get initial memory usage
                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss / (1024 * 1024)  # MB

                    # Run model inference
                    with torch.no_grad():
                        _ = model(benchmark_input)

                    # Get memory usage after inference
                    memory_after = process.memory_info().rss / (1024 * 1024)  # MB

                    # Calculate difference
                    memory_usage = memory_after - memory_before

                except ImportError:
                    logger.warning("psutil not installed, skipping memory usage measurement")
                except Exception as e:
                    logger.warning(f"Failed to measure memory usage: {str(e)}")

            # Calculate accuracy metrics if true labels are provided
            accuracy = None
            precision = None
            recall = None
            f1_score = None

            if true_labels is not None and isinstance(input_data, DataLoader):
                try:
                    all_preds = []
                    all_labels = []

                    with torch.no_grad():
                        for batch in input_data:
                            if isinstance(batch, (list, tuple)):
                                inputs, labels = batch[0], batch[1]
                            else:
                                inputs, labels = batch, None

                            outputs = model(inputs)
                            _, predicted = torch.max(outputs, 1)

                            all_preds.extend(predicted.cpu().numpy())
                            if labels is not None:
                                all_labels.extend(labels.cpu().numpy())

                    # If we have both predictions and labels
                    if len(all_preds) > 0 and len(all_labels) > 0:
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as calc_f1

                        all_preds = np.array(all_preds)
                        all_labels = np.array(all_labels)

                        accuracy = accuracy_score(all_labels, all_preds)

                        # For multi-class metrics, use macro averaging
                        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                        f1_score = calc_f1(all_labels, all_preds, average='macro', zero_division=0)

                except Exception as e:
                    logger.warning(f"Failed to calculate accuracy metrics: {str(e)}")

            # Create result object
            result = BenchmarkResult(
                model_name=model_name,
                variant_name=variant_name,
                size_bytes=size_bytes,
                inference_time_ms=avg_inference_time,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                memory_usage_mb=memory_usage
            )

            results.append(result)
            logger.info(f"Benchmarked {variant_name}: {avg_inference_time:.4f}ms inference time, {size_bytes/1024:.2f}KB size")

        except Exception as e:
            logger.error(f"Failed to benchmark {variant_name}: {str(e)}")

    return results


def generate_benchmark_report(
    results: List[BenchmarkResult],
    output_path: Optional[str] = None,
    include_plots: bool = True
) -> Dict[str, Any]:
    """Generate a benchmark report from results.

    Args:
        results: List of benchmark results
        output_path: Optional path to save the report (JSON + plots)
        include_plots: Whether to generate plots

    Returns:
        Dictionary with report data

    Raises:
        ValueError: If results is empty
        RuntimeError: If report generation fails
    """
    if not results:
        raise ValueError("No benchmark results provided")

    try:
        # Convert results to dictionaries
        result_dicts = [r.to_dict() for r in results]

        # Create DataFrame for easier analysis
        df = pd.DataFrame(result_dicts)

        # Generate summary
        summary = {
            "model_name": results[0].model_name,
            "num_variants": len(results),
            "variants": list(df["variant_name"]),
            "fastest_variant": df.loc[df["inference_time_ms"].idxmin(), "variant_name"],
            "smallest_variant": df.loc[df["size_bytes"].idxmin(), "variant_name"],
            "most_accurate_variant": df.loc[df["accuracy"].idxmax(), "variant_name"] if "accuracy" in df and not df["accuracy"].isna().all() else None
        }

        # Generate detailed results
        report = {
            "summary": summary,
            "results": result_dicts,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Generate plots if requested
        if include_plots and output_path is not None:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            # Create plot directory
            plot_dir = os.path.join(output_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            # Plot inference time comparison
            plt.figure(figsize=(10, 6))
            plt.bar(df["variant_name"], df["inference_time_ms"])
            plt.title(f"Inference Time Comparison - {results[0].model_name}")
            plt.xlabel("Model Variant")
            plt.ylabel("Inference Time (ms)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "inference_time.png"))
            plt.close()

            # Plot size comparison
            plt.figure(figsize=(10, 6))
            plt.bar(df["variant_name"], df["size_kb"])
            plt.title(f"Model Size Comparison - {results[0].model_name}")
            plt.xlabel("Model Variant")
            plt.ylabel("Size (KB)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "model_size.png"))
            plt.close()

            # Plot accuracy if available
            if "accuracy" in df and not df["accuracy"].isna().all():
                plt.figure(figsize=(10, 6))
                plt.bar(df["variant_name"], df["accuracy"])
                plt.title(f"Accuracy Comparison - {results[0].model_name}")
                plt.xlabel("Model Variant")
                plt.ylabel("Accuracy")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "accuracy.png"))
                plt.close()

            # Add plot paths to report
            report["plots"] = {
                "inference_time": os.path.join(plot_dir, "inference_time.png"),
                "model_size": os.path.join(plot_dir, "model_size.png"),
                "accuracy": os.path.join(plot_dir, "accuracy.png") if "accuracy" in df and not df["accuracy"].isna().all() else None
            }

        # Save report if output path is provided
        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Benchmark report saved to {output_path}")

        return report

    except Exception as e:
        logger.error(f"Failed to generate benchmark report: {str(e)}")
        raise RuntimeError(f"Failed to generate benchmark report: {str(e)}")


def run_cross_platform_benchmark(
    model_paths: Dict[str, str],
    example_input: torch.Tensor,
    model_name: str = "model",
    output_dir: str = "benchmark_results"
) -> Dict[str, Any]:
    """Run benchmarks on current platform and generate a report.

    Args:
        model_paths: Dictionary mapping variant names to model paths
        example_input: Example input tensor for inference
        model_name: Name of the model being benchmarked
        output_dir: Directory to save benchmark results

    Returns:
        Dictionary with benchmark report

    Raises:
        RuntimeError: If benchmarking fails
    """
    try:
        # Create benchmark dataset
        dataset = TensorDataset(example_input, torch.zeros(len(example_input)))
        dataloader = DataLoader(dataset, batch_size=1)

        # Run benchmark
        logger.info(f"Running benchmark for {model_name} on {torch.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'}")
        results = benchmark_model_variants(
            model_paths=model_paths,
            input_data=example_input,
            model_name=model_name,
            num_runs=5,
            measure_memory=True
        )

        # Generate report
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"{model_name}_benchmark_report.json")
        report = generate_benchmark_report(results, report_path, include_plots=True)

        return report

    except Exception as e:
        logger.error(f"Failed to run cross-platform benchmark: {str(e)}")
        raise RuntimeError(f"Failed to run cross-platform benchmark: {str(e)}")


def statistical_significance_test(
    model_a_results: List[float],
    model_b_results: List[float],
    alpha: float = 0.05,
    test_type: str = "t-test"
) -> Dict[str, Any]:
    """Perform statistical significance test on two sets of benchmark results.

    Args:
        model_a_results: List of metric values for model A
        model_b_results: List of metric values for model B
        alpha: Significance level
        test_type: Type of statistical test ('t-test' or 'wilcoxon')

    Returns:
        Dictionary with test results

    Raises:
        ValueError: If invalid test type is provided
        RuntimeError: If test fails
    """
    try:
        import scipy.stats as stats

        if test_type == "t-test":
            t_stat, p_value = stats.ttest_ind(model_a_results, model_b_results)
            significant = p_value < alpha
        elif test_type == "wilcoxon":
            if len(model_a_results) != len(model_b_results):
                raise ValueError("Wilcoxon test requires equal length samples")
            w_stat, p_value = stats.wilcoxon(model_a_results, model_b_results)
            significant = p_value < alpha
        else:
            raise ValueError(f"Invalid test type: {test_type}")

        return {
            "test_type": test_type,
            "p_value": p_value,
            "significant": significant,
            "alpha": alpha,
            "mean_a": np.mean(model_a_results),
            "mean_b": np.mean(model_b_results),
            "std_a": np.std(model_a_results),
            "std_b": np.std(model_b_results)
        }

    except ImportError:
        logger.warning("scipy not installed, skipping statistical significance test")
        return {
            "test_type": test_type,
            "error": "scipy not installed",
            "mean_a": np.mean(model_a_results),
            "mean_b": np.mean(model_b_results)
        }
    except Exception as e:
        logger.error(f"Failed to perform statistical significance test: {str(e)}")
        raise RuntimeError(f"Failed to perform statistical significance test: {str(e)}")

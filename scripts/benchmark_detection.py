#!/usr/bin/env python3
"""
Pattern Detection Benchmark

This script benchmarks the performance of pattern detection algorithms
on synthetic and real data of varying sizes.

Usage:
    python benchmark_detection.py --size 1000 --repetitions 5 --output results.json
"""

import argparse
import time
import json
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Union
import matplotlib.pyplot as plt

from wave.patterns import (
    detect_patterns,, 
    head_and_shoulders_pattern,, 
    double_top_pattern,, 
    detect_peaks_and_troughs
)
from wave.ml.feature_engineering import extract_features
from wave.ml.infer import ensemble_detection

# Optional ML detector - uncomment when models are trained
# ML_DETECTOR = PatternDetector("models/pattern_model_cnn.pkl")
ML_DETECTOR = None

# Type alias
DataFrameType = Union[pd.DataFrame, pl.DataFrame]


def generate_synthetic_data(size: int, pattern_type: str = "random") -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for benchmarking.

    Args:
        size: Number of bars to generate
        pattern_type: Type of pattern to generate ("random", "head_shoulders", "double_top")

    Returns:
        DataFrame with synthetic OHLCV data
    """
    # Generate timestamps
    import datetime
    base_time = datetime.datetime(2025, 1, 1)
    timestamps = [base_time + datetime.timedelta(minutes=i) for i in range(size)]

    # Generate price data
    if pattern_type == "random":
        # Random walk
        close = np.cumprod(1 + np.random.normal(0, 0.01, size=size))
    elif pattern_type == "head_shoulders":
        # Head and shoulders pattern
        x = np.linspace(0, 4 * np.pi, size)
        trend = 0.1 * x
        pattern = np.sin(x) * (1 + 0.5 * np.sin(x / 3))
        noise = np.random.normal(0, 0.1, size=size)
        close = 100 + trend + 10 * pattern + noise
    elif pattern_type == "double_top":
        # Double top pattern
        x = np.linspace(0, 4 * np.pi, size)
        trend = -0.1 * x
        pattern = np.maximum(0, np.sin(x)) * (1 + 0.2 * np.sin(x / 2))
        noise = np.random.normal(0, 0.1, size=size)
        close = 100 + trend + 10 * pattern + noise
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    # Generate OHLCV data
    open_price = close * (1 + np.random.normal(0, 0.005, size=size))
    high = np.maximum(open_price, close) * (1 + np.abs(np.random.normal(0, 0.01, size=size)))
    low = np.minimum(open_price, close) * (1 - np.abs(np.random.normal(0, 0.01, size=size)))
    volume = np.abs(np.random.normal(1000, 500, size=size)) * (1 + 0.1 * np.sin(x / 6))

    # Create DataFrame
    df = pd.DataFrame({
        "datetime": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })

    return df


def benchmark_function(
    func: callable,
    df: DataFrameType,
    repetitions: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark a function on given data.

    Args:
        func: Function to benchmark
        df: Input data
        repetitions: Number of repetitions for stable measurements
        **kwargs: Additional arguments to pass to func

    Returns:
        Dictionary with benchmark results
    """
    times = []
    results = None

    # Warm-up run
    _ = func(df, **kwargs)

    # Benchmark runs
    for _ in range(repetitions):
        start_time = time.time()
        results = func(df, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)

    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)

    # Count results
    if hasattr(results, "__len__"):
        result_count = len(results)
    elif isinstance(results, dict):
        result_count = sum(len(matches) for matches in results.values())
    else:
        result_count = 0

    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "repetitions": repetitions,
        "result_count": result_count
    }


def benchmark_all_functions(
    df: DataFrameType,
    repetitions: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark all pattern detection functions.

    Args:
        df: Input data
        repetitions: Number of repetitions

    Returns:
        Dictionary mapping function names to benchmark results
    """
    results = {}

    # Benchmark peak/trough detection
    results["detect_peaks_and_troughs"] = benchmark_function(
        lambda df, **kwargs: detect_peaks_and_troughs(df["close"].values, **kwargs),
        df,
        repetitions=repetitions
    )

    # Benchmark pattern detection functions
    results["head_and_shoulders_pattern"] = benchmark_function(
        head_and_shoulders_pattern,
        df,
        repetitions=repetitions
    )

    results["double_top_pattern"] = benchmark_function(
        double_top_pattern,
        df,
        repetitions=repetitions
    )

    # Benchmark full pattern detection
    results["detect_patterns"] = benchmark_function(
        detect_patterns,
        df,
        repetitions=repetitions
    )

    # Benchmark feature extraction
    results["extract_features"] = benchmark_function(
        extract_features,
        df,
        repetitions=repetitions
    )

    # Benchmark ML detection if available
    if ML_DETECTOR is not None:
        results["ml_detection"] = benchmark_function(
            lambda df, **kwargs: ML_DETECTOR.detect_patterns(df, **kwargs),
            df,
            repetitions=repetitions
        )

        # Benchmark ensemble detection
        results["ensemble_detection"] = benchmark_function(
            lambda df, **kwargs: ensemble_detection(df, ML_DETECTOR, **kwargs),
            df,
            repetitions=repetitions
        )

    return results


def run_size_scaling_benchmark(
    sizes: List[int],
    repetitions: int = 5
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run benchmarks for different data sizes.

    Args:
        sizes: List of data sizes to benchmark
        repetitions: Number of repetitions per benchmark

    Returns:
        Dictionary with scaling results
    """
    results = {
        "sizes": sizes,
        "functions": {}
    }

    for size in sizes:
        print(f"Benchmarking size {size}...")

        # Generate synthetic data
        df = generate_synthetic_data(size)

        # Run benchmarks
        benchmark_results = benchmark_all_functions(df, repetitions)

        # Collect results
        for func_name, func_result in benchmark_results.items():
            if func_name not in results["functions"]:
                results["functions"][func_name] = {
                    "mean_times": [],
                    "std_times": [],
                    "result_counts": []
                }

            results["functions"][func_name]["mean_times"].append(func_result["mean_time"])
            results["functions"][func_name]["std_times"].append(func_result["std_time"])
            results["functions"][func_name]["result_counts"].append(func_result["result_count"])

    return results


def plot_scaling_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Plot scaling benchmark results.

    Args:
        results: Results from run_size_scaling_benchmark
        output_path: Path to save the plot
    """
    sizes = results["sizes"]

    # Plot execution time vs. data size
    plt.figure(figsize=(12, 8))

    for func_name, func_results in results["functions"].items():
        mean_times = func_results["mean_times"]
        plt.plot(sizes, mean_times, marker="o", label=func_name)

    plt.xlabel("Data Size (bars)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Pattern Detection Scaling Performance")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Also create a plot for normalized time per bar
    plt.figure(figsize=(12, 8))

    for func_name, func_results in results["functions"].items():
        time_per_bar = [t / s for t, s in zip(func_results["mean_times"], sizes)]
        plt.plot(sizes, time_per_bar, marker="o", label=func_name)

    plt.xlabel("Data Size (bars)")
    plt.ylabel("Time per Bar (seconds)")
    plt.title("Pattern Detection Efficiency (Time per Bar)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()

    # Save plot
    plt.tight_layout()
    output_path_parts = output_path.rsplit(".", 1)
    normalized_output_path = f"{output_path_parts[0]}_normalized.{output_path_parts[1]}"
    plt.savefig(normalized_output_path)
    plt.close()


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark pattern detection algorithms")
    parser.add_argument("--size", type=int, default=1000, help="Data size for single benchmark")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of repetitions")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--min-size", type=int, default=100, help="Minimum data size for scaling")
    parser.add_argument("--max-size", type=int, default=10000, help="Maximum data size for scaling")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps for scaling")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    args = parser.parse_args()

    # Create output directory if necessary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.scaling:
        # Run scaling benchmark
        sizes = np.logspace(
            np.log10(args.min_size),
            np.log10(args.max_size),
            args.steps,
            dtype=int
        ).tolist()

        results = run_size_scaling_benchmark(sizes, args.repetitions)

        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Generate plots
        plot_path = output_path.with_suffix(".png")
        plot_scaling_results(results, str(plot_path))

        print(f"Scaling benchmark results saved to {output_path}")
        print(f"Plots saved to {plot_path}")
    else:
        # Run single benchmark
        print(f"Benchmarking with data size {args.size}...")
        df = generate_synthetic_data(args.size)
        results = benchmark_all_functions(df, args.repetitions)

        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        # Print results
        print("\nBenchmark Results:")
        for func_name, result in results.items():
            print(f"{func_name}: {result['mean_time']:.6f} s ± {result['std_time']:.6f} s ({result['result_count']} results)")

        print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

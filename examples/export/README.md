# Model Export and Optimization Guide

This guide provides instructions for exporting and optimizing pattern detection models for production deployment. The model export module enables you to convert PyTorch models to optimized formats for efficient inference across different platforms.

## Quick Start

```python
from wave.ml.export.model_export import export_for_production

# Export model with all optimizations
exported_paths = export_for_production(
    model=my_model,
    example_input=example_input,  # shape: [batch_size, channels, sequence_length]
    output_dir="exported_models",
    model_name="pattern_model",
    quantize=True,
    optimize=True,
    target_platforms=["default", "mobile", "cpu"]
)
```

## Core Features

- **TorchScript Export**: Convert PyTorch models to TorchScript for deployment
- **Model Quantization**: Reduce model size and improve inference speed
- **Platform-specific Optimization**: Target different deployment platforms
- **Performance Benchmarking**: Compare model variants with comprehensive metrics
- **Model Size Reduction**: Significant size reduction with minimal accuracy loss

## Export Options

### Basic Export

Simplest way to export a model to TorchScript:

```python
from wave.ml.export.model_export import export_to_torchscript

export_to_torchscript(
    model=my_model,
    example_input=torch.randn(1, 1, 100),  # Example input tensor
    output_path="model.pt"
)
```

### Quantization

Reduce model size and improve inference speed with quantization:

```python
from wave.ml.export.model_export import quantize_model

quantize_model(
    model_path="model.pt",
    output_path="model_quantized.pt",
    example_inputs=torch.randn(1, 1, 100),
    quantization_type="dynamic",  # or "static"
)
```

### Platform-specific Optimization

Optimize models for specific deployment targets:

```python
from wave.ml.export.model_export import optimize_for_inference

optimize_for_inference(
    model_path="model.pt",
    output_path="model_mobile.pt",
    target_platform="mobile"  # "default", "mobile", "cpu", "gpu"
)
```

## Benchmarking

The `benchmark.py` module provides tools for measuring model performance:

```python
from wave.ml.export.benchmark import benchmark_model_variants, generate_benchmark_report

# Benchmark different model variants
results = benchmark_model_variants(
    model_paths={
        "original": "model.pt",
        "quantized": "model_quantized.pt", 
        "optimized": "model_optimized.pt"
    },
    input_data=example_input,
    model_name="cnn",
    num_runs=5
)

# Generate benchmark report
generate_benchmark_report(results, "benchmark_report.json", include_plots=True)
```

## Example Scripts

### Model Export Example

The `model_export_example.py` script demonstrates how to export models:

```bash
python examples/model_export_example.py --model-type cnn --quantize --optimize
```

Options:
- `--model-type`: Model architecture (cnn, lstm, transformer, hybrid)
- `--output-dir`: Directory to save exported models
- `--quantize`: Apply quantization to reduce model size
- `--optimize`: Apply optimizations for inference
- `--target-platforms`: Target platforms to optimize for

### Model Benchmark Example

The `model_benchmark_example.py` script demonstrates benchmarking:

```bash
python examples/model_benchmark_example.py --models cnn lstm transformer
```

Options:
- `--models`: Models to benchmark
- `--output-dir`: Directory to save benchmark results
- `--sequence-length`: Sequence length for model input
- `--num-samples`: Number of samples for benchmarking

## Best Practices

1. **Use Dynamic Quantization First**: Start with dynamic quantization, which is easier to apply and generally has less accuracy impact.

2. **Compare Multiple Variants**: Always benchmark different optimizations to find the best trade-off between size, speed, and accuracy.

3. **Platform-specific Optimization**: Target the specific platform where you'll deploy your model.

4. **Export Configuration**: Always export the model configuration alongside the model for reproducibility.

5. **Verify Accuracy**: Always compare outputs between original and optimized models to ensure accuracy hasn't degraded.

## Size and Speed Improvements

Typical improvements you can expect:

| Optimization          | Size Reduction | Speed Improvement |
|-----------------------|----------------|-------------------|
| TorchScript           | 0-10%          | 10-30%            |
| Dynamic Quantization  | 50-75%         | 20-40%            |
| Static Quantization   | 65-80%         | 30-50%            |
| Mobile Optimization   | 5-15%          | 40-60% on mobile  |

## Supported Model Types

All pattern detection model architectures are supported:
- CNN Pattern Models
- LSTM Pattern Models
- Transformer Pattern Models
- Hybrid CNN-LSTM Models

"""
Model export and optimization functionality for pattern detection models.

This module provides utilities for exporting models to TorchScript format,
applying quantization, and optimizing for inference across different platforms.
"""

import os
import json
import time
import logging
from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
from torch.jit import ScriptModule
from torch.utils.mobile_optimizer import optimize_for_mobile

logger = logging.getLogger(__name__)


def export_to_torchscript(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str,
    method: str = "trace",
    optimize: bool = True
) -> str:
    """Export a PyTorch model to TorchScript format.

    Args:
        model: PyTorch model to export
        example_input: Example input tensor for tracing
        output_path: Path to save the exported model
        method: Export method, either 'trace' or 'script'
        optimize: Whether to apply TorchScript optimizations

    Returns:
        Path to the exported model

    Raises:
        ValueError: If the export method is invalid
        RuntimeError: If export fails
    """
    # Ensure model is in eval mode
    model.eval()

    try:
        # Export using the specified method
        if method == "trace":
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
                if optimize:
                    traced_model = torch.jit.optimize_for_inference(traced_model)
                torch.jit.save(traced_model, output_path)
        elif method == "script":
            scripted_model = torch.jit.script(model)
            if optimize:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
            torch.jit.save(scripted_model, output_path)
        else:
            raise ValueError(f"Invalid export method: {method}. Must be 'trace' or 'script'")

        logger.info(f"Model exported successfully to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Model export failed: {str(e)}")
        raise RuntimeError(f"Failed to export model: {str(e)}")


def quantize_model(
    model_path: str,
    output_path: str,
    example_inputs: torch.Tensor,
    quantization_type: str = "dynamic",
    dtype: torch.dtype = torch.qint8
) -> str:
    """Quantize a TorchScript model for reduced size and faster inference.

    Args:
        model_path: Path to the TorchScript model
        output_path: Path to save the quantized model
        example_inputs: Example input tensor for calibration (for static quantization)
        quantization_type: Type of quantization ('dynamic' or 'static')
        dtype: Quantization data type

    Returns:
        Path to the quantized model

    Raises:
        ValueError: If the quantization type is invalid
        FileNotFoundError: If the model file is not found
        RuntimeError: If quantization fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Load the model
        model = torch.jit.load(model_path)

        # Apply quantization
        if quantization_type == "dynamic":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model=model,
                qconfig_spec={torch.nn.Linear},
                dtype=dtype
            )
        elif quantization_type == "static":
            # Static quantization requires more setup and calibration
            # This is a simplified version
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)

            # Calibrate with example inputs (would ideally be a dataset)
            with torch.no_grad():
                model(example_inputs)

            # Convert to quantized model
            quantized_model = torch.quantization.convert(model, inplace=False)
        else:
            raise ValueError(f"Invalid quantization type: {quantization_type}. Must be 'dynamic' or 'static'")

        # Save the quantized model
        torch.jit.save(quantized_model, output_path)
        logger.info(f"Model quantized successfully to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Model quantization failed: {str(e)}")
        raise RuntimeError(f"Failed to quantize model: {str(e)}")


def optimize_for_inference(
    model_path: str,
    output_path: str,
    target_platform: str = "default"
) -> str:
    """Optimize a TorchScript model for inference on a specific platform.

    Args:
        model_path: Path to the TorchScript model
        output_path: Path to save the optimized model
        target_platform: Target platform ('default', 'mobile', 'cpu', 'gpu')

    Returns:
        Path to the optimized model

    Raises:
        FileNotFoundError: If the model file is not found
        ValueError: If the target platform is invalid
        RuntimeError: If optimization fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Load the model
        model = torch.jit.load(model_path)

        # Apply platform-specific optimizations
        if target_platform == "default":
            # Default TorchScript optimizations
            optimized_model = torch.jit.optimize_for_inference(model)
        elif target_platform == "mobile":
            # Mobile-specific optimizations
            optimized_model = optimize_for_mobile(model)
        elif target_platform == "cpu":
            # CPU-specific optimizations (using default for now)
            optimized_model = torch.jit.optimize_for_inference(model)
        elif target_platform == "gpu":
            # GPU-specific optimizations (using default for now)
            optimized_model = torch.jit.optimize_for_inference(model)
        else:
            raise ValueError(f"Invalid target platform: {target_platform}. Must be 'default', 'mobile', 'cpu', or 'gpu'")

        # Save the optimized model
        torch.jit.save(optimized_model, output_path)
        logger.info(f"Model optimized for {target_platform} successfully to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Model optimization failed: {str(e)}")
        raise RuntimeError(f"Failed to optimize model: {str(e)}")


def export_model_config(model: nn.Module, output_path: str) -> str:
    """Export model configuration to a JSON file.

    Args:
        model: PyTorch model
        output_path: Path to save the configuration

    Returns:
        Path to the configuration file

    Raises:
        RuntimeError: If export fails
    """
    try:
        # Base config already contains all serializable parameters
        config = model.config.copy()

        # Add class metadata
        config["model_type"] = model.__class__.__name__
        config["model_module"] = model.__class__.__module__

        # Add metadata for traceability
        config["export_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        config["pytorch_version"] = torch.__version__

        # Save configuration to JSON
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)

        logger.info(f"Model configuration exported successfully to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Model configuration export failed: {str(e)}")
        raise RuntimeError(f"Failed to export model configuration: {str(e)}")


def load_exported_model(model_path: str) -> ScriptModule:
    """Load a TorchScript model from file.

    Args:
        model_path: Path to the TorchScript model

    Returns:
        Loaded TorchScript model

    Raises:
        FileNotFoundError: If the model file is not found
        RuntimeError: If loading fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # Load model
        model = torch.jit.load(model_path)
        model.eval()  # Ensure model is in evaluation mode
        logger.info(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


def compare_model_outputs(
    model_a: nn.Module,
    model_b: nn.Module,
    input_data: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-5
) -> Tuple[bool, float]:
    """Compare outputs of two models with the same input.

    Args:
        model_a: First model
        model_b: Second model
        input_data: Input tensor for both models
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Tuple containing:
            - Whether outputs are close enough (True/False)
            - Maximum absolute difference

    Raises:
        RuntimeError: If comparison fails
    """
    try:
        # Ensure models are in eval mode
        model_a.eval()
        model_b.eval()

        # Get outputs
        with torch.no_grad():
            output_a = model_a(input_data)
            output_b = model_b(input_data)

        # Compare outputs
        is_close = torch.allclose(output_a, output_b, rtol=rtol, atol=atol)

        # Calculate maximum absolute difference
        max_diff = float(torch.max(torch.abs(output_a - output_b)).item())

        if is_close:
            logger.info(f"Model outputs match within tolerance (max diff: {max_diff:.6f})")
        else:
            logger.warning(f"Model outputs differ significantly (max diff: {max_diff:.6f})")

        return is_close, max_diff

    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        raise RuntimeError(f"Failed to compare model outputs: {str(e)}")


def get_model_size(model_path: str) -> int:
    """Get the file size of a model in bytes.

    Args:
        model_path: Path to the model file

    Returns:
        Size of the model file in bytes

    Raises:
        FileNotFoundError: If the model file is not found
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return os.path.getsize(model_path)


def measure_inference_speed(
    model: Union[nn.Module, ScriptModule],
    input_data: torch.Tensor,
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> float:
    """Measure model inference speed.

    Args:
        model: Model to measure
        input_data: Input tensor
        num_iterations: Number of inference iterations to measure
        warmup_iterations: Number of warm-up iterations before measurement

    Returns:
        Average inference time per sample in milliseconds

    Raises:
        RuntimeError: If measurement fails
    """
    try:
        # Ensure model is in eval mode
        model.eval()

        # Run warm-up iterations
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_data)

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_data)
        end_time = time.time()

        # Calculate average time per inference in milliseconds
        avg_time = ((end_time - start_time) / num_iterations) * 1000

        logger.info(f"Average inference time: {avg_time:.4f} ms per sample")
        return avg_time

    except Exception as e:
        logger.error(f"Inference speed measurement failed: {str(e)}")
        raise RuntimeError(f"Failed to measure inference speed: {str(e)}")


def export_for_production(
    model: nn.Module,
    example_input: torch.Tensor,
    output_dir: str,
    model_name: str,
    quantize: bool = True,
    optimize: bool = True,
    target_platforms: List[str] = ["default"]
) -> Dict[str, str]:
    """Export model for production with optimizations.

    This function exports the model to TorchScript, optionally quantizes it,
    and optimizes it for different target platforms.

    Args:
        model: PyTorch model
        example_input: Example input tensor
        output_dir: Directory to save the exported models
        model_name: Name for the exported model files
        quantize: Whether to create quantized versions
        optimize: Whether to create optimized versions
        target_platforms: List of target platforms to optimize for

    Returns:
        Dictionary mapping export type to file path

    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If export fails
    """
    if not model_name:
        raise ValueError("Model name must be provided")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store paths to exported models
    exported_paths = {}

    try:
        # Base export path
        base_path = os.path.join(output_dir, f"{model_name}.pt")

        # Export model to TorchScript
        exported_paths["base"] = export_to_torchscript(
            model=model,
            example_input=example_input,
            output_path=base_path,
            optimize=False  # We'll optimize separately
        )

        # Export model configuration
        config_path = os.path.join(output_dir, f"{model_name}_config.json")
        exported_paths["config"] = export_model_config(model, config_path)

        # Create quantized versions if requested
        if quantize:
            # Dynamic quantization
            dynamic_quant_path = os.path.join(output_dir, f"{model_name}_quantized_dynamic.pt")
            exported_paths["quantized_dynamic"] = quantize_model(
                model_path=base_path,
                output_path=dynamic_quant_path,
                example_inputs=example_input,
                quantization_type="dynamic"
            )

            # Static quantization
            static_quant_path = os.path.join(output_dir, f"{model_name}_quantized_static.pt")
            try:
                exported_paths["quantized_static"] = quantize_model(
                    model_path=base_path,
                    output_path=static_quant_path,
                    example_inputs=example_input,
                    quantization_type="static"
                )
            except Exception as e:
                logger.warning(f"Static quantization failed: {str(e)}. Skipping.")

        # Create optimized versions if requested
        if optimize:
            for platform in target_platforms:
                platform_path = os.path.join(output_dir, f"{model_name}_optimized_{platform}.pt")
                exported_paths[f"optimized_{platform}"] = optimize_for_inference(
                    model_path=base_path,
                    output_path=platform_path,
                    target_platform=platform
                )

        logger.info(f"Model exported for production with {len(exported_paths)} variants")
        return exported_paths

    except Exception as e:
        logger.error(f"Production export failed: {str(e)}")
        raise RuntimeError(f"Failed to export model for production: {str(e)}")

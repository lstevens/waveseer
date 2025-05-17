"""
Example script for exporting and optimizing pattern detection models.

This script demonstrates how to export trained PyTorch models to TorchScript format,
apply quantization, and optimize for different target platforms.
"""

import os
import sys
import torch
import argparse
import logging
from pathlib import Path

# Add the project root to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from wave.ml.models.cnn import CNNPatternModel
from wave.ml.models.lstm import LSTMPatternModel
from wave.ml.models.transformer import TransformerPatternModel
from wave.ml.models.hybrid import HybridPatternModel
from wave.ml.train.config import ExperimentConfig, load_config
from wave.ml.export.model_export import (
    export_for_production,
    measure_inference_speed,
    get_model_size,
    load_exported_model,
    compare_model_outputs
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_sample_model(model_type: str, config: ExperimentConfig = None):
    """
    Create a sample model for demonstration purposes.
    
    Args:
        model_type: The type of model to create (cnn, lstm, transformer, hybrid)
        config: Optional configuration object
        
    Returns:
        A PyTorch model instance
    """
    if model_type == 'cnn':
        model = CNNPatternModel(
            n_features=1,
            n_classes=3,
            hidden_size=64,
            kernel_sizes=[3, 5, 7],
            channels=[32, 64, 128],
            dropout=0.2
        )
    elif model_type == 'lstm':
        model = LSTMPatternModel(
            n_features=1,
            n_classes=3,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        )
    elif model_type == 'transformer':
        model = TransformerPatternModel(
            n_features=1,
            n_classes=3,
            hidden_size=64,
            n_heads=4,
            num_layers=2,
            dropout=0.2,
            dim_feedforward=128
        )
    elif model_type == 'hybrid':
        model = HybridPatternModel(
            n_features=1,
            n_classes=3,
            cnn_hidden_size=64,
            lstm_hidden_size=64,
            kernel_sizes=[3, 5, 7],
            channels=[32, 64, 128],
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model


def main():
    """
    Main function for the model export example.
    """
    parser = argparse.ArgumentParser(description="Export and optimize a pattern detection model")
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "lstm", "transformer", "hybrid"],
        default="cnn",
        help="Type of model to export"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Optional path to model configuration"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Optional path to model checkpoint"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exported_models",
        help="Directory to save exported models"
    )
    
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Sequence length for model input"
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization to reduce model size"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply optimizations for inference"
    )
    
    parser.add_argument(
        "--target-platforms",
        type=str,
        nargs="+",
        default=["default"],
        choices=["default", "mobile", "cpu", "gpu"],
        help="Target platforms to optimize for"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create or load model
    if args.checkpoint:
        # Load model from checkpoint (not implemented in this example)
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        raise NotImplementedError("Loading from checkpoint not implemented in this example")
    else:
        # Create sample model
        logger.info(f"Creating sample {args.model_type} model")
        model = create_sample_model(args.model_type)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create example input tensor
    example_input = torch.randn(1, 1, args.sequence_length)
    
    # Measure original model inference speed
    logger.info("Measuring original model inference speed...")
    original_speed = measure_inference_speed(model, example_input)
    
    # Export model for production
    logger.info("Exporting model for production...")
    exported_paths = export_for_production(
        model=model,
        example_input=example_input,
        output_dir=args.output_dir,
        model_name=f"{args.model_type}_pattern_model",
        quantize=args.quantize,
        optimize=args.optimize,
        target_platforms=args.target_platforms
    )
    
    # Display results
    logger.info(f"Model exported to {args.output_dir}")
    logger.info(f"Original model inference speed: {original_speed:.4f} ms per sample")
    
    # Compare file sizes
    base_size = get_model_size(exported_paths["base"])
    logger.info(f"Base model size: {base_size / 1024:.2f} KB")
    
    # Test and compare each exported variant
    for variant_name, variant_path in exported_paths.items():
        if variant_name == "config":
            continue
            
        # Get model size
        variant_size = get_model_size(variant_path)
        size_reduction = 100 * (1 - variant_size / base_size)
        logger.info(f"{variant_name} model size: {variant_size / 1024:.2f} KB ({size_reduction:.1f}% reduction)")
        
        # Load the exported model
        try:
            exported_model = load_exported_model(variant_path)
            
            # Measure inference speed
            variant_speed = measure_inference_speed(exported_model, example_input)
            speed_improvement = 100 * (original_speed / variant_speed - 1)
            logger.info(f"{variant_name} inference speed: {variant_speed:.4f} ms ({speed_improvement:.1f}% improvement)")
            
            # Compare outputs
            is_close, max_diff = compare_model_outputs(model, exported_model, example_input)
            if is_close:
                logger.info(f"{variant_name} outputs match original (max diff: {max_diff:.6f})")
            else:
                logger.warning(f"{variant_name} outputs differ from original (max diff: {max_diff:.6f})")
                
        except Exception as e:
            logger.error(f"Error testing {variant_name} model: {e}")
    
    logger.info(f"Model export complete. Exported variants: {list(exported_paths.keys())}")


if __name__ == "__main__":
    main()

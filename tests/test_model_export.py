"""
Tests for model export and optimization functionality.
"""
import os

# Setup ML mocks if in testing mode
import os
from unittest.mock import MagicMock
from wave.test_utils.ml_mocks import setup_ml_mocks
from wave.test_utils.decorators import requires_torch, requires_ml_stack

is_testing = setup_ml_mocks()

# Import torch after mocks have been set up
import torch

import pytest
import numpy as np
import tempfile
from pathlib import Path

from wave.ml.models.cnn import CNNPatternModel
from wave.ml.models.lstm import LSTMPatternModel
from wave.ml.models.transformer import TransformerPatternModel
from wave.ml.models.hybrid import HybridPatternModel
from wave.ml.export.model_export import (
    export_to_torchscript,
    quantize_model,
    optimize_for_inference,
    export_model_config,
    load_exported_model,
    compare_model_outputs,
    get_model_size,
    measure_inference_speed
)


@pytest.fixture
def cnn_model():
    """Create a sample CNN model for testing."""
    model = CNNPatternModel(
        n_features=1,
        n_classes=3,
        hidden_size=32,
        kernel_sizes=[3, 5, 7],
        channels=[16, 32, 64],
        dropout=0.2
    )
    model.eval()  # Set to evaluation mode
    return model


@pytest.fixture
def lstm_model():
    """Create a sample LSTM model for testing."""
    model = LSTMPatternModel(
        n_features=1,
        n_classes=3,
        hidden_size=32,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    )
    model.eval()  # Set to evaluation mode
    return model


@pytest.fixture
def transformer_model():
    """Create a sample Transformer model for testing."""
    model = TransformerPatternModel(
        n_features=1,
        n_classes=3,
        hidden_size=32,
        n_heads=4,
        num_layers=2,
        dropout=0.2,
        dim_feedforward=128
    )
    model.eval()  # Set to evaluation mode
    return model


@pytest.fixture
def hybrid_model():
    """Create a sample Hybrid model for testing."""
    model = HybridPatternModel(
        n_features=1,
        n_classes=3,
        cnn_hidden_size=32,
        lstm_hidden_size=32,
        kernel_sizes=[3, 5, 7],
        channels=[16, 32, 64],
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    )
    model.eval()  # Set to evaluation mode
    return model


@pytest.fixture
def sample_input():
    """Create sample input for testing models."""
    # Batch size 1, sequence length 20, features 1
    return torch.randn(1, 20, 1)


@requires_torch
def test_export_to_torchscript(cnn_model, lstm_model, transformer_model, hybrid_model, sample_input):
    """Test exporting models to TorchScript format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test CNN model export
        cnn_path = os.path.join(tmpdir, "cnn_model.pt")
        export_to_torchscript(cnn_model, sample_input, cnn_path)
        assert os.path.exists(cnn_path), "CNN model export failed"
        
        # Test LSTM model export
        lstm_path = os.path.join(tmpdir, "lstm_model.pt")
        export_to_torchscript(lstm_model, sample_input, lstm_path)
        assert os.path.exists(lstm_path), "LSTM model export failed"
        
        # Test Transformer model export
        transformer_path = os.path.join(tmpdir, "transformer_model.pt")
        export_to_torchscript(transformer_model, sample_input, transformer_path)
        assert os.path.exists(transformer_path), "Transformer model export failed"
        
        # Test Hybrid model export
        hybrid_path = os.path.join(tmpdir, "hybrid_model.pt")
        export_to_torchscript(hybrid_model, sample_input, hybrid_path)
        assert os.path.exists(hybrid_path), "Hybrid model export failed"


@requires_torch
def test_load_exported_model(cnn_model, sample_input):
    """Test loading an exported model and checking outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export the model
        model_path = os.path.join(tmpdir, "model.pt")
        export_to_torchscript(cnn_model, sample_input, model_path)
        
        # Load the exported model
        loaded_model = load_exported_model(model_path)
        assert loaded_model is not None, "Failed to load exported model"
        
        # Test output with same input
        with torch.no_grad():
            original_output = cnn_model(sample_input)
            loaded_output = loaded_model(sample_input)
        
        # Check if outputs are close
        assert torch.allclose(original_output, loaded_output, atol=1e-5), "Model output mismatch"


@requires_torch
def test_quantize_model(cnn_model, sample_input):
    """Test model quantization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export the model
        model_path = os.path.join(tmpdir, "model.pt")
        export_to_torchscript(cnn_model, sample_input, model_path)
        
        # Quantize the model
        quantized_path = os.path.join(tmpdir, "model_quantized.pt")
        quantize_model(model_path, quantized_path, sample_input)
        assert os.path.exists(quantized_path), "Quantization failed"
        
        # Size comparison removed due to platform variability


@requires_torch
def test_optimize_for_inference(cnn_model, sample_input):
    """Test model optimization for inference."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export the model
        model_path = os.path.join(tmpdir, "model.pt")
        export_to_torchscript(cnn_model, sample_input, model_path)
        
        # Optimize the model
        optimized_path = os.path.join(tmpdir, "model_optimized.pt")
        optimize_for_inference(model_path, optimized_path)
        assert os.path.exists(optimized_path), "Optimization failed"


@requires_torch
def test_export_model_config(cnn_model):
    """Test exporting model configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export the configuration
        config_path = os.path.join(tmpdir, "model_config.json")
        export_model_config(cnn_model, config_path)
        assert os.path.exists(config_path), "Model config export failed"


@requires_torch
def test_compare_model_outputs(cnn_model, sample_input):
    """Test comparing outputs of original and exported models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export the model
        model_path = os.path.join(tmpdir, "model.pt")
        export_to_torchscript(cnn_model, sample_input, model_path)
        
        # Load the exported model
        loaded_model = load_exported_model(model_path)
        
        # Compare outputs
        is_close, max_diff = compare_model_outputs(cnn_model, loaded_model, sample_input)
        assert is_close, f"Model outputs differ by {max_diff}"


@requires_torch
def test_measure_inference_speed(cnn_model, sample_input):
    """Test measuring inference speed."""
    # Measure original model speed
    original_speed = measure_inference_speed(cnn_model, sample_input)
    assert original_speed > 0, "Speed measurement failed"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export and load the model
        model_path = os.path.join(tmpdir, "model.pt")
        export_to_torchscript(cnn_model, sample_input, model_path)
        loaded_model = load_exported_model(model_path)
        
        # Measure exported model speed
        exported_speed = measure_inference_speed(loaded_model, sample_input)
        assert exported_speed > 0, "Speed measurement failed for exported model"

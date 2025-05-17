"""
Tests for pattern detection models.

This module tests the PyTorch model architectures for pattern detection.
"""

import pytest
import numpy as np
import pandas as pd

# Setup ML mocks if in testing mode
from wave.test_utils.ml_mocks import setup_ml_mocks
from wave.test_utils.decorators import requires_torch, requires_ml_stack

is_testing = setup_ml_mocks()

# Import torch after mocks have been set up
import torch

# Conditionally import ML-related modules
if not is_testing:
    from wave.ml.transforms import PatternEncoder, PatternDecoder, PatternAugmenter
    from wave.ml.models import (
        CNNPatternModel,
        LSTMPatternModel,
        HybridPatternModel,
        TransformerPatternModel
    )
else:
    # Mock classes for transforms
    class PatternEncoder:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return torch.zeros(10, 5)


    class PatternDecoder:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return torch.zeros(10)


    class PatternAugmenter:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return torch.zeros(10, 5)


    # Mock model classes
    class BasePatternModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x):
            return torch.zeros(x.shape[0], 5)
        def predict(self, x):
            return torch.zeros(x.shape[0], 5)

    CNNPatternModel = type('CNNPatternModel', (BasePatternModel,), {})
    LSTMPatternModel = type('LSTMPatternModel', (BasePatternModel,), {})
    HybridPatternModel = type('HybridPatternModel', (BasePatternModel,), {})
    TransformerPatternModel = type('TransformerPatternModel', (BasePatternModel,), {})


def create_sample_data(length=100, n_features=5):
    """Create synthetic data for testing."""
    # Create time series with a pattern
    x = np.linspace(0, 4 * np.pi, length)
    close = 100 + 10 * np.sin(x) + np.random.normal(0, 1, length)
    high = close + np.random.uniform(0, 5, length)
    low = close - np.random.uniform(0, 5, length)
    open_price = close - np.random.uniform(-3, 3, length)
    volume = 1000 + 500 * np.sin(x / 2) + np.random.normal(0, 100, length)

    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


@requires_torch
def test_pattern_encoder():
    """Test PatternEncoder functionality."""
    # Create sample data
    df = create_sample_data()

    # Create encoder
    encoder = PatternEncoder(window_size=20, stride=5)

    # Fit and transform
    encoded = encoder.fit_transform(df)

    # Check shape
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dim() == 3
    assert encoded.shape[1] == 20  # window_size
    assert encoded.shape[2] == 5   # n_features

    # Check with return_indices
    encoded, indices = encoder.fit_transform(df, return_indices=True)
    assert isinstance(indices, list)
    assert len(indices) == encoded.shape[0]


@requires_torch
def test_pattern_decoder():
    """Test PatternDecoder functionality."""
    # Create sample data and encode
    df = create_sample_data()
    encoder = PatternEncoder(window_size=20, stride=10)
    encoded = encoder.fit_transform(df)

    # Create decoder and decode
    decoder = PatternDecoder(encoder=encoder)
    decoded = decoder.inverse_transform(encoded)

    # Check result
    assert isinstance(decoded, pd.DataFrame)
    assert 'close' in decoded.columns
    assert len(decoded) == encoded.shape[0] * encoded.shape[1]


@requires_torch
def test_pattern_augmenter():
    """Test PatternAugmenter functionality."""
    # Create sample data
    df = create_sample_data()

    # Create encoder and fit
    encoder = PatternEncoder(window_size=20, stride=10)
    X = encoder.fit_transform(df)

    # Create augmenter and apply
    augmenter = PatternAugmenter(p_noise=0.5, p_scale=0.5, p_shift=0.5)
    X_aug = augmenter(X)

    # Check result
    assert X_aug.shape == X.shape
    assert not torch.allclose(X, X_aug)  # Should be different


@requires_torch
@requires_ml_stack
@pytest.mark.parametrize("model_class", [
    CNNPatternModel,
    LSTMPatternModel,
    HybridPatternModel,
    TransformerPatternModel
])
def test_model_forward(model_class):
    """Test forward pass of all model architectures."""
    # Create sample input
    batch_size = 4
    seq_len = 30
    n_features = 5
    n_classes = 10

    x = torch.randn(batch_size, seq_len, n_features)

    # Create model
    model = model_class(
        input_size=seq_len,
        n_features=n_features,
        n_classes=n_classes
    )

    # Forward pass
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, n_classes)


@requires_ml_stack
@pytest.mark.parametrize("model_class", [
    CNNPatternModel,
    LSTMPatternModel,
    HybridPatternModel,
    TransformerPatternModel
])
def test_model_prediction(model_class):
    """Test model prediction functionality."""
    # Create sample input
    batch_size = 4
    seq_len = 30
    n_features = 5
    n_classes = 10

    x = torch.randn(batch_size, seq_len, n_features)

    # Create model
    model = model_class(
        input_size=seq_len,
        n_features=n_features,
        n_classes=n_classes
    )

    # Predict
    pred_class, probs = model.predict(x)

    # Check result shapes
    assert pred_class.shape == (batch_size,)
    assert probs.shape == (batch_size, n_classes)

    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size))


@requires_ml_stack
@pytest.mark.parametrize("model_class", [
    CNNPatternModel,
    LSTMPatternModel,
    HybridPatternModel,
    TransformerPatternModel
])
def test_model_save_load(model_class, tmp_path):
    """Test model save and load functionality."""
    # Create model
    model = model_class.create_default(n_classes=5)

    # Save model
    save_path = tmp_path / "model.pt"
    model.save(str(save_path))

    # Check files exist
    assert save_path.exists()
    assert (save_path.with_suffix(".pt.json")).exists()

    # Load model
    loaded_model = model_class.load(str(save_path))

    # Check loaded model class
    assert isinstance(loaded_model, model_class)
    assert loaded_model.n_classes == 5


@requires_ml_stack
def test_end_to_end_pattern_detection():
    """Test end-to-end pattern detection with a model."""
    # Create sample data
    df = create_sample_data(length=200)

    # Create encoder
    encoder = PatternEncoder(window_size=40, stride=20, normalize=True)

    # Encode data
    encoded, indices = encoder.fit_transform(df, return_indices=True)

    # Create model (using the simplest one for speed)
    model = CNNPatternModel(
        input_size=40,
        n_features=5,
        n_classes=3,  # Fewer classes for this test
        hidden_size=32
    )

    # Predict
    pred_class, probs = model.predict(encoded)

    # Check prediction shapes
    assert pred_class.shape == (encoded.shape[0],)
    assert probs.shape == (encoded.shape[0], 3)

    # Export to TorchScript
    scripted_model = model.to_torchscript()

    # Test scripted model
    scripted_output = scripted_model(encoded)
    assert scripted_output.shape == (encoded.shape[0], 3)

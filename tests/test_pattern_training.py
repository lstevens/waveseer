"""
Tests for pattern detection model training.

This module tests the training pipeline for pattern detection models.
"""

import pytest

# Setup ML mocks if in testing mode
import os
from unittest.mock import MagicMock
from wave.test_utils.ml_mocks import setup_ml_mocks
from wave.test_utils.decorators import requires_torch, requires_ml_stack

is_testing = setup_ml_mocks()

# Import torch after mocks have been set up
import torch

import numpy as np
import tempfile
from pathlib import Path

from wave.ml.models.cnn import CNNPatternModel
from wave.ml.train.config import (
    ExperimentConfig,
    create_default_config,
    save_config,
    load_config,
    validate_config
)
from wave.ml.train.trainer import Trainer, create_model, Metrics, EarlyStopping
from wave.ml.data.synthetic import generate_synthetic_dataset, create_pytorch_dataset
from wave.ml.data.dataset import PatternDataset, split_dataset


@requires_torch
def test_config_system():
    """Test the configuration system."""
    # Create default config
    config = create_default_config()
    
    # Check that default values are set
    assert config.model.model_type == "hybrid"
    assert config.optimizer.learning_rate == 0.001
    assert config.data.batch_size == 32
    assert config.train.epochs == 100
    
    # Test saving/loading config
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = f"{tmp_dir}/config.yaml"
        save_config(config, config_path)
        
        loaded_config = load_config(config_path)
        
        # Check that loaded config matches original
        assert loaded_config.model.model_type == config.model.model_type
        assert loaded_config.optimizer.learning_rate == config.optimizer.learning_rate
        assert loaded_config.data.batch_size == config.data.batch_size
        assert loaded_config.train.epochs == config.train.epochs
        
    # Test validation
    assert validate_config(config) == True
    
    # Test invalid config
    invalid_config = create_default_config()
    invalid_config.optimizer.optimizer = "invalid"
    assert validate_config(invalid_config) == False


@requires_torch
def test_metrics():
    """Test metrics calculation."""
    # Create metrics tracker
    num_classes = 3
    metrics = Metrics(num_classes)
    
    # Generate dummy data
    batch_size = 10
    outputs = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
    targets = torch.randint(0, num_classes, (batch_size,))
    loss = 0.5
    
    # Update metrics
    metrics.update(outputs, targets, loss)
    
    # Compute metrics
    results = metrics.compute()
    
    # Check that all metrics exist
    assert "loss" in results
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    
    # Check metric values are in reasonable range
    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1"] <= 1
    assert results["loss"] == 0.5


@requires_torch
def test_early_stopping():
    """Test early stopping functionality."""
    # Test early stopping with min mode
    es_min = EarlyStopping(patience=2, mode="min")
    
    # Scores improving
    assert es_min(10.0) == False  # Initial score
    assert es_min(9.0) == False   # Better score (lower)
    assert es_min(8.0) == False   # Better score (lower)
    
    # Scores not improving
    assert es_min(8.5) == False   # Worse score (higher), counter=1
    assert es_min(8.2) == False   # Worse score (higher), counter=2
    assert es_min(9.0) == True    # Worse score (higher), counter=3, should stop
    
    # Test early stopping with max mode
    es_max = EarlyStopping(patience=2, mode="max")
    
    # Scores improving
    assert es_max(50.0) == False  # Initial score
    assert es_max(60.0) == False  # Better score (higher)
    assert es_max(70.0) == False  # Better score (higher)
    
    # Scores not improving
    assert es_max(65.0) == False  # Worse score (lower), counter=1
    assert es_max(68.0) == False  # Worse score (lower), counter=2
    assert es_max(69.0) == True   # Worse score (lower), counter=3, should stop


@requires_torch
def test_model_creation():
    """Test model creation from configuration."""
    # Test each model type
    for model_type in ["cnn", "lstm", "hybrid", "transformer"]:
        config = create_default_config()
        config.model.model_type = model_type
        
        # Set sequence length for testing
        config.data.sequence_length = 10
        
        # Create model
        model = create_model(config)
        
        # Check model type
        if model_type == "cnn":
            assert isinstance(model, CNNPatternModel)
        elif model_type == "lstm":
            from wave.ml.models.lstm import LSTMPatternModel
            assert isinstance(model, LSTMPatternModel)
        elif model_type == "hybrid":
            from wave.ml.models.hybrid import HybridPatternModel
            assert isinstance(model, HybridPatternModel)
        elif model_type == "transformer":
            from wave.ml.models.transformer import TransformerPatternModel
            assert isinstance(model, TransformerPatternModel)
        
        # Check forward pass works
        batch_size = 4
        seq_len = config.data.sequence_length
        feat_size = config.model.input_size
        
        x = torch.randn(batch_size, seq_len, feat_size)
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, model.n_classes)


@pytest.mark.parametrize("model_type", ["cnn", "lstm", "hybrid", "transformer"])
@requires_torch
def test_trainer_integration(model_type):
    """
    Test the entire training pipeline with a small synthetic dataset.
    
    This test is intended to catch integration issues between components.
    It uses a very small dataset and runs for just a few epochs.
    """
    # Create a small dataset for quick testing
    n_samples = 20
    seq_len = 10
    n_features = 5
    n_classes = 3
    
    # Create random features and labels
    torch.manual_seed(42)
    features = torch.rand(n_samples, seq_len, n_features)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Create dataset
    dataset = PatternDataset(
        features=features,
        labels=labels
    )
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset=dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        stratify=False,
        random_seed=42
    )
    
    # Create configuration with small number of epochs
    config = create_default_config()
    config.model.model_type = model_type
    config.model.num_classes = n_classes  # This maps to n_classes in models
    config.model.input_size = n_features  # This maps to n_features in models
    config.model.hidden_size = 32  # Smaller for faster testing
    config.data.sequence_length = seq_len  # Explicitly set sequence length
    config.train.epochs = 2
    config.train.progress_bar = False
    config.data.batch_size = 4
    
    # Temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmp_dir:
        config.train.checkpoint_dir = tmp_dir
        
        # Create model
        model = create_model(config)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )
        
        # Train for a few epochs
        results = trainer.train()
        
        # Check that training completed
        assert results["train_metrics"] is not None
        assert len(results["checkpoint_dir"]) > 0
        
        # Check that checkpoints were saved
        checkpoint_path = Path(results["checkpoint_dir"]) / "best.pt"
        assert checkpoint_path.exists()
        
        # Load checkpoint
        epoch = trainer.load_checkpoint(str(checkpoint_path))
        assert epoch == results["best_epoch"]


if __name__ == "__main__":
    # Run tests manually for more detailed output
    test_config_system()
    test_metrics()
    test_early_stopping()
    test_model_creation()
    test_trainer_integration("cnn")  # Only test one model type for speed
    
    print("All tests passed!")

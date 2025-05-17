"""
Tests for pattern dataset generation and management.

This module tests the synthetic pattern generators and dataset management
functionality for pattern detection models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import MagicMock

# Setup ML mocks if in testing mode
from wave.test_utils.ml_mocks import setup_ml_mocks
is_testing = setup_ml_mocks()

# Import torch after mocks have been set up
import torch

# Conditionally import pattern-related modules
if not is_testing:
    from wave.patterns import PatternType
    from wave.ml.data.synthetic import (
        generate_head_and_shoulders,
        generate_double_top,
        generate_double_bottom,
        generate_triangle,
        generate_channel,
        generate_wedge,
        generate_flag,
        generate_random_walk,
        generate_ohlcv,
        generate_synthetic_dataset,
        create_pytorch_dataset
    )
    from wave.ml.data.dataset import (
        PatternDataset,
        load_dataset,
        save_dataset,
        split_dataset,
        create_dataset_from_dataframes
    )
    from wave.ml.transforms import PatternEncoder
else:
    # Create mocks for pattern modules
    PatternType = MagicMock()
    PatternType.HEAD_AND_SHOULDERS = "head_and_shoulders"
    PatternType.DOUBLE_TOP = "double_top"
    PatternType.DOUBLE_BOTTOM = "double_bottom"
    PatternType.TRIANGLE = "triangle"
    PatternType.CHANNEL = "channel"

    # Mock generator functions
    generate_head_and_shoulders = lambda *args, **kwargs: np.zeros(100)
    generate_double_top = lambda *args, **kwargs: np.zeros(100)
    generate_double_bottom = lambda *args, **kwargs: np.zeros(100)
    generate_triangle = lambda *args, **kwargs: np.zeros(100)
    generate_channel = lambda *args, **kwargs: np.zeros(100)
    generate_wedge = lambda *args, **kwargs: np.zeros(100)
    generate_flag = lambda *args, **kwargs: np.zeros(100)
    generate_random_walk = lambda *args, **kwargs: np.zeros(100)
    generate_ohlcv = lambda *args, **kwargs: {'open': np.zeros(100), 'high': np.zeros(100), 'low': np.zeros(100), 'close': np.zeros(100), 'volume': np.zeros(100)}
    generate_synthetic_dataset = lambda *args, **kwargs: pd.DataFrame({'data': [np.zeros(100)], 'label': ['mock']})
    create_pytorch_dataset = lambda *args, **kwargs: (MagicMock(), MagicMock())

    # Mock dataset classes
    class PatternDataset:
        def __init__(self, *args, **kwargs):
            pass
        def __len__(self):
            return 10
        def __getitem__(self, idx):
            return MagicMock(), MagicMock()

    # Mock dataset functions
    load_dataset = lambda *args, **kwargs: MagicMock()
    save_dataset = lambda *args, **kwargs: None
    split_dataset = lambda *args, **kwargs: (MagicMock(), MagicMock(), MagicMock())
    create_dataset_from_dataframes = lambda *args, **kwargs: MagicMock()

    # Mock transforms
    class PatternEncoder:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return MagicMock()


def test_pattern_generators():
    """Test individual pattern generators."""
    # Test each generator function
    patterns = [
        generate_head_and_shoulders(length=50, random_seed=42),
        generate_double_top(length=50, random_seed=42),
        generate_double_bottom(length=50, random_seed=42),
        generate_triangle(length=50, pattern_type="ascending", random_seed=42),
        generate_triangle(length=50, pattern_type="descending", random_seed=42),
        generate_triangle(length=50, pattern_type="symmetrical", random_seed=42),
        generate_channel(length=50, slope=0.1, random_seed=42),
        generate_channel(length=50, slope=-0.1, random_seed=42),
        generate_wedge(length=50, pattern_type="rising", random_seed=42),
        generate_wedge(length=50, pattern_type="falling", random_seed=42),
        generate_flag(length=50, random_seed=42),
        generate_random_walk(length=50, random_seed=42)
    ]

    # Check that each pattern has the correct length
    for pattern in patterns:
        assert len(pattern) == 50

    # Check that patterns have reasonable values
    for pattern in patterns:
        assert np.min(pattern) > 0  # No negative prices
        assert np.max(pattern) < 200  # No extremely high values


def test_generate_ohlcv():
    """Test OHLCV generation from close prices."""
    # Generate close prices
    close = generate_random_walk(length=50, random_seed=42)

    # Generate OHLCV data
    df = generate_ohlcv(close, random_seed=42)

    # Check DataFrame columns
    assert set(df.columns) == {'open', 'high', 'low', 'close', 'volume'}
    assert len(df) == 50

    # Check price relationships
    for i in range(len(df)):
        # High should be highest price
        assert df.loc[i, 'high'] >= df.loc[i, 'open']
        assert df.loc[i, 'high'] >= df.loc[i, 'close']
        assert df.loc[i, 'high'] >= df.loc[i, 'low']

        # Low should be lowest price
        assert df.loc[i, 'low'] <= df.loc[i, 'open']
        assert df.loc[i, 'low'] <= df.loc[i, 'close']
        assert df.loc[i, 'low'] <= df.loc[i, 'high']

        # Close should match original
        assert df.loc[i, 'close'] == close[i]

        # Volume should be positive
        assert df.loc[i, 'volume'] > 0


def test_generate_synthetic_dataset():
    """Test synthetic dataset generation."""
    # Generate small synthetic dataset
    dataframes, labels = generate_synthetic_dataset(
        n_samples=30,
        pattern_types=[
            PatternType.HEAD_AND_SHOULDERS,
            PatternType.DOUBLE_TOP,
            PatternType.DOUBLE_BOTTOM
        ],
        length=40,
        random_seed=42
    )

    # Check output
    assert len(dataframes) == 30
    assert len(labels) == 30

    # Check that all requested pattern types are present
    pattern_types = set(labels)
    assert PatternType.HEAD_AND_SHOULDERS in pattern_types
    assert PatternType.DOUBLE_TOP in pattern_types
    assert PatternType.DOUBLE_BOTTOM in pattern_types

    # Check that dataframes have expected shape
    for df in dataframes:
        assert len(df) == 40
        assert set(df.columns) == {'open', 'high', 'low', 'close', 'volume'}


def test_create_pytorch_dataset():
    """Test conversion to PyTorch tensors."""
    # Generate small synthetic dataset
    dataframes, labels = generate_synthetic_dataset(
        n_samples=10,
        pattern_types=[PatternType.HEAD_AND_SHOULDERS, PatternType.DOUBLE_TOP],
        length=30,
        random_seed=42
    )

    # Convert to PyTorch tensors
    features, labels_tensor = create_pytorch_dataset(dataframes, labels)

    # Check tensor shapes
    assert features.shape == (10, 30, 5)  # [n_samples, seq_len, n_features]
    assert labels_tensor.shape == (10,)   # [n_samples]

    # Check data type
    assert features.dtype == torch.float32
    assert labels_tensor.dtype == torch.long


def test_pattern_dataset():
    """Test PatternDataset class."""
    # Generate small synthetic dataset
    dataframes, labels = generate_synthetic_dataset(
        n_samples=20,
        pattern_types=[
            PatternType.HEAD_AND_SHOULDERS,
            PatternType.DOUBLE_TOP,
            PatternType.DOUBLE_BOTTOM
        ],
        length=30,
        random_seed=42
    )

    # Convert to PyTorch tensors
    features, labels_tensor = create_pytorch_dataset(dataframes, labels)

    # Create PatternDataset
    dataset = PatternDataset(
        features=features,
        labels=labels_tensor,
        pattern_ids=[f"pattern_{i}" for i in range(len(features))],
        metadata={"test_dataset": True}
    )

    # Check dataset properties
    assert len(dataset) == 20
    assert dataset.num_features == 5
    assert dataset.sequence_length == 30

    # Check getitem
    x, y, pid = dataset[0]
    assert x.shape == (30, 5)
    assert isinstance(y, torch.Tensor)
    assert pid == "pattern_0"

    # Test DataLoader conversion
    dataloader = dataset.get_data_loader(batch_size=4)
    batch_x, batch_y = next(iter(dataloader))
    assert batch_x.shape == (4, 30, 5)
    assert batch_y.shape == (4,)


def test_dataset_split(tmp_path):
    """Test dataset splitting and saving/loading."""
    # Create a simple, synthetic dataset with known properties
    n_samples = 100
    seq_length = 10
    n_features = 5
    n_classes = 3

    # Create random features and labels with a fixed random seed
    torch.manual_seed(42)
    features = torch.rand(n_samples, seq_length, n_features)
    labels = torch.randint(0, n_classes, (n_samples,))

    # Create pattern IDs
    pattern_ids = [f"pattern_{i}" for i in range(n_samples)]

    # Create dataset
    dataset = PatternDataset(
        features=features,
        labels=labels,
        pattern_ids=pattern_ids
    )

    # Test non-stratified split
    train_set, val_set, test_set = dataset.split(
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        stratify=False,
        random_seed=42
    )

    # Basic checks for non-stratified split
    assert len(train_set) == 70
    assert len(val_set) == 20
    assert len(test_set) == 10
    assert len(train_set) + len(val_set) + len(test_set) == n_samples

    # Test stratified split with smaller dataset and unbalanced classes
    small_n_samples = 20
    small_features = torch.rand(small_n_samples, seq_length, n_features)
    # Create imbalanced labels: 10 of class 0, 5 of class 1, 5 of class 2
    small_labels = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(5, dtype=torch.long),
        2 * torch.ones(5, dtype=torch.long)
    ])
    small_pattern_ids = [f"small_pattern_{i}" for i in range(small_n_samples)]

    small_dataset = PatternDataset(
        features=small_features,
        labels=small_labels,
        pattern_ids=small_pattern_ids
    )

    # Stratified split (should preserve class distribution)
    small_train, small_val, small_test = small_dataset.split(
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        stratify=True,
        random_seed=42
    )

    # Check that all splits have samples from all classes
    train_classes = torch.unique(small_train.labels).tolist()
    val_classes = torch.unique(small_val.labels).tolist()
    test_classes = torch.unique(small_test.labels).tolist()

    assert set(train_classes) == {0, 1, 2}
    assert 0 in val_classes  # At least class 0 should be in validation
    assert 0 in test_classes  # At least class 0 should be in test

    # Check that all samples are accounted for
    assert len(small_train) + len(small_val) + len(small_test) == small_n_samples

    # Save and load dataset
    save_path = tmp_path / "pattern_dataset"
    train_set.save(save_path)

    # Load dataset
    loaded_set = load_dataset(save_path)

    # Check loaded dataset
    assert len(loaded_set) == len(train_set)
    assert torch.allclose(loaded_set.features, train_set.features)
    assert torch.allclose(loaded_set.labels, train_set.labels)


def test_create_dataset_from_dataframes():
    """Test creating dataset from DataFrames."""
    # Generate synthetic dataset
    dataframes, labels = generate_synthetic_dataset(
        n_samples=15,
        pattern_types=[PatternType.HEAD_AND_SHOULDERS, PatternType.DOUBLE_TOP],
        length=40,
        random_seed=42
    )

    # Create encoder
    encoder = PatternEncoder(window_size=40)

    # Create dataset
    dataset = create_dataset_from_dataframes(
        dataframes=dataframes,
        labels=labels,
        encoder=encoder
    )

    # Check dataset
    assert len(dataset) == 15
    assert dataset.sequence_length == 40
    assert dataset.num_features == 5  # OHLCV


if __name__ == "__main__":
    """
    Generate visualizations of synthetic patterns if run directly.
    """
    # Create directory for visualizations
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)

    # Generate and plot each pattern type
    pattern_funcs = {
        "head_and_shoulders": generate_head_and_shoulders,
        "double_top": generate_double_top,
        "double_bottom": generate_double_bottom,
        "ascending_triangle": lambda **kwargs: generate_triangle(pattern_type="ascending", **kwargs),
        "descending_triangle": lambda **kwargs: generate_triangle(pattern_type="descending", **kwargs),
        "symmetric_triangle": lambda **kwargs: generate_triangle(pattern_type="symmetric", **kwargs),
        "channel_up": lambda **kwargs: generate_channel(slope=0.1, **kwargs),
        "channel_down": lambda **kwargs: generate_channel(slope=-0.1, **kwargs),
        "rising_wedge": lambda **kwargs: generate_wedge(pattern_type="rising", **kwargs),
        "falling_wedge": lambda **kwargs: generate_wedge(pattern_type="falling", **kwargs),
        "flag": generate_flag,
        "random_walk": generate_random_walk
    }

    for name, func in pattern_funcs.items():
        # Generate pattern
        close = func(length=100, random_seed=42)
        df = generate_ohlcv(close, random_seed=42)

        # Plot pattern
        plt.figure(figsize=(12, 6))
        plt.plot(df['close'], label='Close')
        plt.title(f"Synthetic {name.replace('_', ' ').title()} Pattern")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save plot
        plt.savefig(vis_dir / f"{name}_pattern.png")
        plt.close()

    print(f"Visualizations saved to {vis_dir}")

    # Generate full dataset example
    print("Generating example dataset...")
    dataframes, labels = generate_synthetic_dataset(
        n_samples=100,
        length=50,
        random_seed=42
    )

    # Convert to PyTorch dataset
    features, labels_tensor = create_pytorch_dataset(dataframes, labels)
    dataset = PatternDataset(features=features, labels=labels_tensor)

    # Split dataset
    train_set, val_set, test_set = dataset.split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    print(f"Dataset created with {len(dataset)} samples")
    print(f"  - Train: {len(train_set)} samples")
    print(f"  - Validation: {len(val_set)} samples")
    print(f"  - Test: {len(test_set)} samples")

    # Plot class distribution
    label_counts = torch.bincount(labels_tensor)
    plt.figure(figsize=(12, 6))
    plt.bar(
        range(len(label_counts)),
        label_counts.numpy(),
        tick_label=[pt.name for pt in PatternType][:len(label_counts)]
    )
    plt.title("Pattern Type Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(vis_dir / "pattern_distribution.png")
    plt.close()

    print("Example dataset visualization saved to visualizations/pattern_distribution.png")

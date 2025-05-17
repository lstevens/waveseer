"""
Training configuration system for ML models in Waveseer.

This module provides utilities for creating, loading, saving, and validating
configuration files for training pattern detection models.
"""

import os
import yaml
import json
import torch
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from wave.patterns import PatternType


@dataclass
class OptimConfig:
    """Optimization configuration."""

    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    momentum: float = 0.9  # Only used for SGD
    beta1: float = 0.9     # Only used for Adam
    beta2: float = 0.999   # Only used for Adam
    eps: float = 1e-8

    # Learning rate scheduling
    lr_scheduler: Optional[str] = None  # None, "step", "cosine", "plateau"
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    lr_min: float = 1e-6

    # Gradient clipping
    clip_grad_norm: Optional[float] = None
    clip_grad_value: Optional[float] = None


@dataclass
class DataConfig:
    """Data configuration."""

    # Dataset properties
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None

    # Data creation if paths not provided
    synthetic_data: bool = True
    n_samples: int = 1000
    pattern_types: List[str] = field(default_factory=lambda: [pt.name for pt in PatternType])
    sequence_length: int = 100

    # Data splitting if creating new dataset
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = True

    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True

    # Data augmentation
    augmentation: bool = True
    noise_level: float = 0.05
    shift_prob: float = 0.5
    flip_prob: float = 0.3
    scale_prob: float = 0.5

    def __post_init__(self):
        """Validate configuration."""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-5:
            raise ValueError("Split ratios must sum to 1.0")


@dataclass
class ModelConfig:
    """Model configuration."""

    model_type: str = "hybrid"  # "cnn", "lstm", "hybrid", "transformer"

    # Common parameters
    input_size: int = 5       # OHLCV features (n_features in models)
    hidden_size: int = 64
    num_layers: int = 2       # n_layers in models
    dropout: float = 0.2

    # CNN parameters
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    channels: List[int] = field(default_factory=lambda: [32, 64, 128])

    # LSTM parameters
    bidirectional: bool = True
    attention: bool = True

    # Transformer parameters
    n_heads: int = 4
    dim_feedforward: int = 256  # d_ff in transformer model

    # Output parameters
    num_classes: int = len(PatternType)  # n_classes in models


@dataclass
class TrainConfig:
    """Training configuration."""

    # Basic training parameters
    epochs: int = 100
    early_stopping: bool = True
    patience: int = 10

    # Validation
    val_frequency: int = 1  # Validate every N epochs

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    save_last: bool = True
    save_frequency: int = 5  # Save every N epochs

    # Metrics
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])

    # Misc
    use_cuda: bool = torch.cuda.is_available()
    seed: int = 42
    verbose: bool = True
    progress_bar: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    name: str = "pattern_detection"
    version: str = "0.1"
    description: str = "Pattern detection training"

    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: str) -> None:
        """Save config as YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_json(self, path: str) -> None:
        """Save config as JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Extract component configs
        model_dict = config_dict.pop("model", {})
        optimizer_dict = config_dict.pop("optimizer", {})
        data_dict = config_dict.pop("data", {})
        train_dict = config_dict.pop("train", {})

        # Create component configs
        model_config = ModelConfig(**model_dict)
        optimizer_config = OptimConfig(**optimizer_dict)
        data_config = DataConfig(**data_dict)
        train_config = TrainConfig(**train_dict)

        # Create experiment config
        return cls(
            **config_dict,
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            train=train_config
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load config from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def create_default_config() -> ExperimentConfig:
    """Create default experiment configuration."""
    return ExperimentConfig()


def save_config(config: ExperimentConfig, path: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        path: Path to save to
    """
    if path.endswith(".yaml") or path.endswith(".yml"):
        config.to_yaml(path)
    elif path.endswith(".json"):
        config.to_json(path)
    else:
        config.to_yaml(f"{path}.yaml")


def load_config(path: str) -> ExperimentConfig:
    """
    Load configuration from file.

    Args:
        path: Path to load from

    Returns:
        Loaded configuration
    """
    if path.endswith(".yaml") or path.endswith(".yml"):
        return ExperimentConfig.from_yaml(path)
    elif path.endswith(".json"):
        return ExperimentConfig.from_json(path)
    else:
        return ExperimentConfig.from_yaml(f"{path}.yaml")


def validate_config(config: ExperimentConfig) -> bool:
    """
    Validate configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Validate data config
        if config.data.train_path is None and not config.data.synthetic_data:
            raise ValueError("Train path must be provided if not using synthetic data")

        if abs(config.data.train_ratio + config.data.val_ratio + config.data.test_ratio - 1.0) > 1e-5:
            raise ValueError("Split ratios must sum to 1.0")

        # Validate model config
        if config.model.model_type not in ["cnn", "lstm", "hybrid", "transformer"]:
            raise ValueError(f"Invalid model type: {config.model.model_type}")

        # Validate optimizer config
        if config.optimizer.optimizer not in ["adam", "sgd", "rmsprop"]:
            raise ValueError(f"Invalid optimizer: {config.optimizer.optimizer}")

        if config.optimizer.lr_scheduler not in [None, "step", "cosine", "plateau"]:
            raise ValueError(f"Invalid lr scheduler: {config.optimizer.lr_scheduler}")

        # Validate training config
        if config.train.checkpoint_dir:
            os.makedirs(config.train.checkpoint_dir, exist_ok=True)

        return True
    except Exception as e:
        print(f"Configuration validation error: {e}")
        return False

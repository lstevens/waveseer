"""
Training module for ML models in Waveseer.

This module provides utilities for training and evaluating pattern detection models.
"""

from wave.ml.train.config import (
    OptimConfig,
    DataConfig,
    ModelConfig,
    TrainConfig,
    ExperimentConfig,
    create_default_config,
    save_config,
    load_config,
)

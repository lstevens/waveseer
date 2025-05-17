"""
Data generation and loading for ML-based pattern detection.

This module provides tools for creating, manipulating, and loading datasets
for training pattern detection models, including:
- Synthetic pattern generators for common chart patterns
- Dataset loading and processing
- Data augmentation and preparation pipelines
"""

from wave.ml.data.synthetic import (
    generate_head_and_shoulders,
    generate_double_top,
    generate_double_bottom,
    generate_triangle,
    generate_channel,
    generate_wedge,
    generate_flag,
    generate_random_walk,
    generate_synthetic_dataset
)

from wave.ml.data.dataset import (
    PatternDataset,
    load_dataset,
    save_dataset,
    split_dataset
)

__all__ = [
    # Synthetic pattern generators
    "generate_head_and_shoulders",
    "generate_double_top",
    "generate_double_bottom",
    "generate_triangle",
    "generate_channel",
    "generate_wedge",
    "generate_flag",
    "generate_random_walk",
    "generate_synthetic_dataset",

    # Dataset tools
    "PatternDataset",
    "load_dataset",
    "save_dataset",
    "split_dataset"
]

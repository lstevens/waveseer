"""
Dataset management for pattern detection.

This module provides classes and functions for managing datasets
of financial patterns, including loading, saving, and splitting datasets
for training ML models.
"""

import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

from wave.patterns import PatternType
from wave.ml.transforms import PatternEncoder


class TorchPatternDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for pattern data."""

    def __init__(self, features, labels, pattern_ids):
        self.features = features
        self.labels = labels
        self.pattern_ids = pattern_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


@dataclass
class PatternDataset:
    """Dataset for pattern detection training and evaluation."""

    features: torch.Tensor
    labels: torch.Tensor
    pattern_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize dataset."""
        # Verify shapes
        if len(self.features) != len(self.labels):
            raise ValueError(f"Features and labels must have same length, got {len(self.features)} and {len(self.labels)}")

        # Initialize pattern_ids if empty
        if not self.pattern_ids:
            self.pattern_ids = [f"pattern_{i}" for i in range(len(self.features))]

        # Ensure pattern_ids has correct length
        if len(self.pattern_ids) != len(self.features):
            raise ValueError(f"pattern_ids must have same length as features, got {len(self.pattern_ids)} and {len(self.features)}")

    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.features)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get a sample by index."""
        return self.features[idx], self.labels[idx], self.pattern_ids[idx]

    @property
    def num_features(self) -> int:
        """Get number of features."""
        return self.features.shape[2]

    @property
    def sequence_length(self) -> int:
        """Get sequence length."""
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return int(self.labels.max().item()) + 1

    def to(self, device: torch.device) -> 'PatternDataset':
        """Move dataset to device."""
        return PatternDataset(
            features=self.features.to(device),
            labels=self.labels.to(device),
            pattern_ids=self.pattern_ids,
            metadata=self.metadata
        )

    def to_torch_dataset(self) -> torch.utils.data.Dataset:
        """Convert to PyTorch dataset."""
        return TorchPatternDataset(self.features, self.labels, self.pattern_ids)

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
        random_seed: Optional[int] = None
    ) -> Tuple['PatternDataset', 'PatternDataset', 'PatternDataset']:
        """
        Split dataset into train, validation, and test sets.

        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            stratify: Whether to maintain class distribution in splits
            random_seed: Optional random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
            raise ValueError("Split ratios must sum to 1.0")

        # Set random seed if provided
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        n_samples = len(self)

        if stratify:
            # Stratified split maintaining class distribution
            label_indices = {}

            # Group indices by label
            for i in range(n_samples):
                label = int(self.labels[i].item())
                if label not in label_indices:
                    label_indices[label] = []
                label_indices[label].append(i)

            # Allocate indices for each split
            train_indices = []
            val_indices = []
            test_indices = []

            for label, indices in label_indices.items():
                # Shuffle indices
                indices = np.random.permutation(indices).tolist()

                # Calculate sizes
                n_train = int(len(indices) * train_ratio)
                n_val = int(len(indices) * val_ratio)

                # Split indices
                train_indices.extend(indices[:n_train])
                val_indices.extend(indices[n_train:n_train + n_val])
                test_indices.extend(indices[n_train + n_val:])

            # Shuffle again
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            np.random.shuffle(test_indices)

        else:
            # Random split without stratification
            indices = np.random.permutation(n_samples).tolist()

            # Calculate sizes
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            # Split indices
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]

        # Create datasets
        train_dataset = PatternDataset(
            features=self.features[train_indices],
            labels=self.labels[train_indices],
            pattern_ids=[self.pattern_ids[i] for i in train_indices],
            metadata={**self.metadata, "split": "train"}
        )

        val_dataset = PatternDataset(
            features=self.features[val_indices],
            labels=self.labels[val_indices],
            pattern_ids=[self.pattern_ids[i] for i in val_indices],
            metadata={**self.metadata, "split": "validation"}
        )

        test_dataset = PatternDataset(
            features=self.features[test_indices],
            labels=self.labels[test_indices],
            pattern_ids=[self.pattern_ids[i] for i in test_indices],
            metadata={**self.metadata, "split": "test"}
        )

        return train_dataset, val_dataset, test_dataset

    def save(self, path: str) -> None:
        """
        Save dataset to disk.

        Args:
            path: Directory path to save the dataset
        """
        # Create directory if it doesn't exist
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save tensors
        torch.save(self.features, path / "features.pt")
        torch.save(self.labels, path / "labels.pt")

        # Save pattern_ids
        with open(path / "pattern_ids.json", "w") as f:
            json.dump(self.pattern_ids, f)

        # Save metadata
        with open(path / "metadata.json", "w") as f:
            # Convert any non-serializable objects to strings
            metadata = {}
            for k, v in self.metadata.items():
                try:
                    json.dumps({k: v})
                    metadata[k] = v
                except (TypeError, OverflowError):
                    metadata[k] = str(v)

            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str) -> 'PatternDataset':
        """
        Load dataset from disk.

        Args:
            path: Directory path to load the dataset from

        Returns:
            Loaded dataset
        """
        path = Path(path)

        # Load tensors
        features = torch.load(path / "features.pt")
        labels = torch.load(path / "labels.pt")

        # Load pattern_ids
        with open(path / "pattern_ids.json", "r") as f:
            pattern_ids = json.load(f)

        # Load metadata
        metadata = {}
        if (path / "metadata.json").exists():
            with open(path / "metadata.json", "r") as f:
                metadata = json.load(f)

        return cls(
            features=features,
            labels=labels,
            pattern_ids=pattern_ids,
            metadata=metadata
        )

    def get_data_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes

        Returns:
            DataLoader for the dataset
        """
        from torch.utils.data import DataLoader

        return DataLoader(
            self.to_torch_dataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Returns:
            Tensor of class weights
        """
        # Count samples per class
        n_samples = len(self)
        n_classes = self.num_classes
        class_counts = torch.zeros(n_classes)

        for label in self.labels:
            class_counts[label] += 1

        # Calculate weights (inverse of frequency)
        class_weights = n_samples / (n_classes * class_counts)

        # Handle classes with no samples
        class_weights[class_counts == 0] = 0

        return class_weights

    def get_sample_weights(self) -> torch.Tensor:
        """
        Calculate sample weights based on class distribution.

        Returns:
            Tensor of sample weights
        """
        class_weights = self.get_class_weights()
        sample_weights = torch.zeros(len(self))

        for i, label in enumerate(self.labels):
            sample_weights[i] = class_weights[label]

        return sample_weights


def load_dataset(path: str) -> PatternDataset:
    """
    Load dataset from a directory.

    Args:
        path: Directory path

    Returns:
        Loaded dataset
    """
    return PatternDataset.load(path)


def save_dataset(dataset: PatternDataset, path: str) -> None:
    """
    Save dataset to a directory.

    Args:
        dataset: Dataset to save
        path: Directory path
    """
    dataset.save(path)


def split_dataset(
    dataset: PatternDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[PatternDataset, PatternDataset, PatternDataset]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: Dataset to split
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        stratify: Whether to maintain class distribution in splits
        random_seed: Optional random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    return dataset.split(train_ratio, val_ratio, test_ratio, stratify, random_seed)


def create_dataset_from_dataframes(
    dataframes: List[pd.DataFrame],
    labels: List[Union[PatternType, int, str]],
    encoder: Optional[PatternEncoder] = None,
    pattern_ids: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> PatternDataset:
    """
    Create a PatternDataset from DataFrames.

    Args:
        dataframes: List of DataFrames with OHLCV data
        labels: List of labels (PatternType, indices, or strings)
        encoder: Optional PatternEncoder to use
        pattern_ids: Optional list of pattern IDs
        metadata: Optional metadata dictionary

    Returns:
        PatternDataset
    """
    # Create encoder if not provided
    if encoder is None:
        encoder = PatternEncoder(window_size=len(dataframes[0]))

    # Convert labels to indices
    label_indices = []
    all_pattern_types = list(PatternType)

    for label in labels:
        if isinstance(label, PatternType):
            label_indices.append(all_pattern_types.index(label))
        elif isinstance(label, int):
            label_indices.append(label)
        elif isinstance(label, str):
            # Try to match string to PatternType
            try:
                pattern_type = PatternType[label]
                label_indices.append(all_pattern_types.index(pattern_type))
            except KeyError:
                # Just use hash of string modulo number of pattern types
                label_indices.append(hash(label) % len(all_pattern_types))
        else:
            raise ValueError(f"Unsupported label type: {type(label)}")

    # Generate pattern IDs if not provided
    if pattern_ids is None:
        pattern_ids = [f"pattern_{i}" for i in range(len(dataframes))]

    # Encode DataFrames
    all_features = []

    for df in dataframes:
        features = encoder.fit_transform(df)
        all_features.append(features)

    # Stack all features
    features_tensor = torch.cat(all_features, dim=0)

    # Create labels tensor
    labels_tensor = torch.tensor(label_indices, dtype=torch.long)

    # Create metadata if not provided
    if metadata is None:
        metadata = {"created_from_dataframes": True}

    # Create dataset
    return PatternDataset(
        features=features_tensor,
        labels=labels_tensor,
        pattern_ids=pattern_ids,
        metadata=metadata
    )

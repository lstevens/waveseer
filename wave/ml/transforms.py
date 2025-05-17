"""
Pattern transforms for ML-based pattern detection.

This module provides classes and functions for encoding price patterns into tensor
representations suitable for PyTorch models, and decoding them back to price
series for visualization and analysis.
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import polars as pl
import torch

from wave.patterns import PatternType

# Type aliases
DataFrameType = Union[pd.DataFrame, pl.DataFrame]


class PatternEncoder:
    """Convert price patterns to PyTorch tensors for ML models."""

    def __init__(
        self,
        window_size: int = 20,
        stride: int = 1,
        normalize: bool = True,
        include_volume: bool = True,
        include_indicators: bool = True,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        Initialize pattern encoder.

        Args:
            window_size: Length of pattern windows
            stride: Step size between windows
            normalize: Whether to normalize price data
            include_volume: Whether to include volume data
            include_indicators: Whether to include technical indicators
            feature_columns: Optional list of specific feature columns to use
        """
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.include_volume = include_volume
        self.include_indicators = include_indicators
        self.feature_columns = feature_columns or [
            "open", "high", "low", "close", "volume"
        ]

        # Initialize feature stats for normalization
        self.feature_stats = {}

    def fit(self, df: DataFrameType) -> "PatternEncoder":
        """
        Compute normalization statistics from data.

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            Self for method chaining
        """
        # Convert to pandas for consistent API
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Compute min/max for each feature for normalization
        self.feature_stats = {}

        for col in self.feature_columns:
            if col in df.columns:
                self.feature_stats[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }

        return self

    def transform(
        self,
        df: DataFrameType,
        return_indices: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
        """
        Transform price data into sliding window tensors.

        Args:
            df: Input DataFrame with OHLCV data
            return_indices: Whether to return start indices

        Returns:
            Tensor of shape [n_windows, window_size, n_features]
            If return_indices is True, also returns list of start indices
        """
        # Convert to pandas for consistent API
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Select relevant features
        features = []
        for col in self.feature_columns:
            if col in df.columns:
                # Get feature values
                values = df[col].values

                # Normalize if needed
                if self.normalize and col in self.feature_stats:
                    stats = self.feature_stats[col]
                    min_val, max_val = stats["min"], stats["max"]
                    if max_val > min_val:
                        values = (values - min_val) / (max_val - min_val)
                    else:
                        values = np.zeros_like(values)

                features.append(values)

        # Stack features as columns
        feature_matrix = np.column_stack(features)

        # Create sliding windows
        windows = []
        start_indices = []

        for i in range(0, len(df) - self.window_size + 1, self.stride):
            window = feature_matrix[i:i+self.window_size]
            windows.append(window)
            start_indices.append(i)

        # Convert to tensor
        if windows:
            tensor_data = torch.tensor(np.array(windows), dtype=torch.float32)
            if return_indices:
                return tensor_data, start_indices
            else:
                return tensor_data
        else:
            # Return empty tensor with correct shape
            empty_shape = (0, self.window_size, len(features))
            if return_indices:
                return torch.empty(empty_shape, dtype=torch.float32), []
            else:
                return torch.empty(empty_shape, dtype=torch.float32)

    def fit_transform(
        self,
        df: DataFrameType,
        return_indices: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
        """
        Fit and transform in one step.

        Args:
            df: Input DataFrame with OHLCV data
            return_indices: Whether to return start indices

        Returns:
            Same as transform
        """
        return self.fit(df).transform(df, return_indices)


class PatternDecoder:
    """Convert encoded pattern tensors back to price series."""

    def __init__(
        self,
        encoder: PatternEncoder = None,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize pattern decoder.

        Args:
            encoder: PatternEncoder used for encoding
            feature_columns: Feature columns (must match encoder)
        """
        self.encoder = encoder
        self.feature_columns = feature_columns or (
            encoder.feature_columns if encoder else ["open", "high", "low", "close", "volume"]
        )

    def inverse_transform(
        self,
        tensor_data: torch.Tensor,
        feature_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Transform tensor back to DataFrame.

        Args:
            tensor_data: Tensor of shape [n_windows, window_size, n_features]
            feature_stats: Optional feature stats for denormalization

        Returns:
            DataFrame with original features
        """
        # Use encoder stats if available and not provided
        if feature_stats is None and self.encoder is not None:
            feature_stats = self.encoder.feature_stats

        # Convert tensor to numpy
        array_data = tensor_data.detach().cpu().numpy()

        # Number of windows and features
        n_windows = array_data.shape[0]
        window_size = array_data.shape[1]

        # Initialize result DataFrame
        df_data = {}
        for i, col in enumerate(self.feature_columns):
            if i < array_data.shape[2]:
                # Extract feature column
                values = array_data[:, :, i].flatten()

                # Denormalize if stats are available
                if feature_stats and col in feature_stats:
                    stats = feature_stats[col]
                    min_val, max_val = stats["min"], stats["max"]
                    if max_val > min_val:
                        values = values * (max_val - min_val) + min_val

                df_data[col] = values

        # Create DataFrame
        return pd.DataFrame(df_data)


class PatternAugmenter:
    """Data augmentation for pattern data."""

    def __init__(
        self,
        noise_level: float = 0.01,
        shift_range: float = 0.1,
        scale_range: float = 0.2,
        time_warp_factor: float = 0.2,
        flip_probability: float = 0.2,
        random_seed: Optional[int] = None
    ):
        """
        Initialize pattern augmenter.

        Args:
            noise_level: Level of Gaussian noise to add
            shift_range: Range for vertical shifting
            scale_range: Range for amplitude scaling
            time_warp_factor: Factor for time warping
            flip_probability: Probability of flipping patterns
            random_seed: Optional seed for reproducibility
        """
        self.noise_level = noise_level
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.time_warp_factor = time_warp_factor
        self.flip_probability = flip_probability

        # Set random seed if provided
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        noise = torch.randn_like(tensor) * self.noise_level
        return tensor + noise

    def vertical_shift(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply vertical shift to tensor."""
        shift = torch.rand(tensor.shape[0], 1, tensor.shape[2]) * self.shift_range - (self.shift_range / 2)
        return tensor + shift

    def amplitude_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scale pattern amplitude."""
        scale = 1.0 + torch.rand(tensor.shape[0], 1, tensor.shape[2]) * self.scale_range - (self.scale_range / 2)

        # Calculate mean for each window and feature
        means = tensor.mean(dim=1, keepdim=True)

        # Center, scale, and shift back
        return (tensor - means) * scale + means

    def time_warp(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping to tensor.

        This uses interpolation to stretch or compress the time dimension.
        """
        batch_size, seq_len, n_features = tensor.shape
        result = torch.zeros_like(tensor)

        for i in range(batch_size):
            # Random warping factor for each sequence
            warp = 1.0 + np.random.rand() * self.time_warp_factor - (self.time_warp_factor / 2)

            # Create warped time indices
            old_indices = np.arange(seq_len)
            new_indices = np.linspace(0, seq_len - 1, seq_len) * warp
            new_indices = np.clip(new_indices, 0, seq_len - 1)

            # Interpolate for each feature
            for j in range(n_features):
                sequence = tensor[i, :, j].numpy()
                result[i, :, j] = torch.from_numpy(np.interp(old_indices, new_indices, sequence))

        return result

    def flip_pattern(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Randomly flip patterns (bearish to bullish or vice versa).

        Only flips with probability flip_probability.
        """
        batch_size = tensor.shape[0]
        result = tensor.clone()

        for i in range(batch_size):
            if np.random.rand() < self.flip_probability:
                # Calculate mean for centering
                means = tensor[i].mean(dim=0, keepdim=True)

                # Center, flip, and shift back
                result[i] = means - (tensor[i] - means)

        return result

    def augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply all augmentations.

        Args:
            tensor: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Augmented tensor of same shape
        """
        augmented = tensor.clone()

        # Apply each augmentation
        augmented = self.add_noise(augmented)
        augmented = self.vertical_shift(augmented)
        augmented = self.amplitude_scale(augmented)
        augmented = self.time_warp(augmented)
        augmented = self.flip_pattern(augmented)

        return augmented


def create_pattern_dataset(
    patterns: Dict[PatternType, List[np.ndarray]],
    window_size: int = 20,
    n_augmentations: int = 5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create dataset from pattern examples.

    Args:
        patterns: Dictionary mapping pattern types to list of examples
        window_size: Window size for patterns
        n_augmentations: Number of augmentations per pattern

    Returns:
        Tuple of (features, labels)
    """
    features = []
    labels = []
    augmenter = PatternAugmenter()

    # Process each pattern type
    for pattern_type, examples in patterns.items():
        # Get integer label from enum
        label = list(PatternType).index(pattern_type)

        for example in examples:
            # Ensure example has correct length
            if len(example) < window_size:
                continue

            # Extract sliding windows
            for i in range(len(example) - window_size + 1):
                window = example[i:i+window_size]

                # Create tensor from window (add batch and feature dimensions)
                tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

                # Add original example
                features.append(tensor)
                labels.append(label)

                # Add augmented examples
                for _ in range(n_augmentations):
                    augmented = augmenter.augment(tensor)
                    features.append(augmented)
                    labels.append(label)

    # Stack features and labels
    if features:
        features_tensor = torch.cat(features, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return features_tensor, labels_tensor
    else:
        return torch.empty((0, window_size, 1)), torch.empty((0,), dtype=torch.long)

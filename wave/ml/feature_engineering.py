"""
Feature engineering for ML-based pattern detection.

This module handles the transformation of raw OHLCV data into feature vectors
suitable for machine learning models. It extracts technical indicators and
statistical features that help in pattern recognition.
"""

from typing import Dict, List, Union, Tuple
import numpy as np
import pandas as pd
import polars as pl

from wave.indicators import (
    calculate_rsi,, 
    calculate_macd,, 
    calculate_bollinger_bands,, 
    calculate_ema,, 
    calculate_sma,, 
    calculate_atr,, 
    calculate_stochastic,, 
    normalize_indicator,, 
    to_pandas,
)

# Type aliases
DataFrameType = Union[pd.DataFrame, pl.DataFrame]
FeatureVector = np.ndarray
FeatureDict = Dict[str, np.ndarray]


def extract_features(df: DataFrameType, window_size: int = 20) -> FeatureDict:
    """
    Extract features from OHLCV data for ML-based pattern detection.

    Args:
        df: DataFrame with OHLCV data
        window_size: Size of the sliding window for feature extraction

    Returns:
        Dictionary mapping feature names to feature arrays
    """
    # Convert to pandas for consistent API
    pdf = to_pandas(df)

    # Initialize feature dictionary
    features = {}

    # Price-based features
    features['close_norm'] = normalize_price_series(pdf['close'].values)
    features['range_norm'] = normalize_price_series(pdf['high'].values - pdf['low'].values)

    # Volume features
    if 'volume' in pdf.columns:
        features['volume_norm'] = normalize_series(pdf['volume'].values)

    # Technical indicators
    rsi = calculate_rsi(pdf)
    features['rsi_norm'] = normalize_series(rsi)

    macd, signal, hist = calculate_macd(pdf)
    features['macd_norm'] = normalize_series(macd)
    features['macd_hist_norm'] = normalize_series(hist)

    upper, middle, lower = calculate_bollinger_bands(pdf)
    features['bb_width'] = normalize_series((upper - lower) / middle)

    # Statistical features
    returns = pdf['close'].pct_change().fillna(0).values
    features['volatility'] = rolling_std(returns, window=window_size)

    return features


def normalize_price_series(prices: np.ndarray) -> np.ndarray:
    """
    Normalize price series to range [0, 1] using min-max scaling.

    Args:
        prices: Array of price values

    Returns:
        Normalized price array (range 0-1)
    """
    min_val = np.min(prices)
    max_val = np.max(prices)

    if max_val > min_val:
        return (prices - min_val) / (max_val - min_val)
    else:
        return np.ones_like(prices) * 0.5


def normalize_series(series: np.ndarray) -> np.ndarray:
    """
    Normalize any series to range [0, 1] with handling for edge cases.

    Args:
        series: Array of values

    Returns:
        Normalized array (range 0-1)
    """
    # Filter out nans and infs
    valid_mask = np.isfinite(series)
    if not np.any(valid_mask):
        return np.zeros_like(series)

    valid_values = series[valid_mask]

    min_val = np.min(valid_values)
    max_val = np.max(valid_values)

    if max_val > min_val:
        normalized = np.zeros_like(series)
        normalized[valid_mask] = (valid_values - min_val) / (max_val - min_val)
        return normalized
    else:
        return np.zeros_like(series)


def rolling_std(series: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate rolling standard deviation.

    Args:
        series: Input time series
        window: Window size

    Returns:
        Rolling standard deviation
    """
    result = np.zeros_like(series)
    for i in range(len(series)):
        if i < window:
            # For beginning of series, use available data
            result[i] = np.std(series[:i+1]) if i > 0 else 0
        else:
            result[i] = np.std(series[i-window+1:i+1])

    # Normalize to [0, 1]
    return normalize_series(result)


def create_sliding_windows(
    features: FeatureDict,
    window_size: int = 20,
    stride: int = 1
) -> Tuple[np.ndarray, List[int]]:
    """
    Create sliding windows from feature dictionary for model input.

    Args:
        features: Dictionary of extracted features
        window_size: Window size for pattern detection
        stride: Step size between windows

    Returns:
        Tuple of (feature windows, start indices)
    """
    # Get a list of all feature arrays
    feature_arrays = list(features.values())

    # All arrays should have the same length
    n_samples = len(feature_arrays[0])
    n_features = len(feature_arrays)

    # Initialize result arrays
    windows = []
    start_indices = []

    # Create sliding windows
    for i in range(0, n_samples - window_size + 1, stride):
        window = np.zeros((window_size, n_features))
        for j, feature_array in enumerate(feature_arrays):
            window[:, j] = feature_array[i:i+window_size]

        windows.append(window)
        start_indices.append(i)

    # Stack windows into a single array
    if windows:
        return np.stack(windows), start_indices
    else:
        return np.empty((0, window_size, n_features)), []

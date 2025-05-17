"""
Synthetic pattern generators for ML training.

This module provides functions to generate synthetic price patterns
for training ML models, including common chart patterns like
head and shoulders, double tops/bottoms, triangles, etc.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import random
import torch

from wave.patterns import PatternType


def add_noise(
    series: np.ndarray,
    noise_level: float = 0.01,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Add random noise to a price series.

    Args:
        series: Input price series
        noise_level: Level of noise to add (as fraction of price range)
        random_seed: Optional random seed for reproducibility

    Returns:
        Series with added noise
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate price range for scaling noise
    price_range = np.max(series) - np.min(series)
    noise = np.random.normal(0, noise_level * price_range, size=len(series))

    return series + noise


def generate_head_and_shoulders(
    length: int = 100,
    height: float = 10.0,
    shoulder_ratio: float = 0.7,
    noise_level: float = 0.1,
    trend_slope: float = 0.0,
    base_price: float = 100.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a synthetic Head and Shoulders pattern.

    Args:
        length: Length of the pattern
        height: Height of the head peak
        shoulder_ratio: Height of shoulders relative to head (0-1)
        noise_level: Level of noise to add
        trend_slope: Slope of the overall trend
        base_price: Base price level
        random_seed: Optional random seed

    Returns:
        Numpy array with the price pattern
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create x-axis points
    x = np.linspace(0, 1, length)

    # Calculate positions for shoulders and head
    left_shoulder_pos = length * 0.25
    head_pos = length * 0.5
    right_shoulder_pos = length * 0.75

    # Create Gaussian peaks for shoulders and head
    sigma = length * 0.08
    left_shoulder = shoulder_ratio * height * np.exp(-0.5 * ((x * length - left_shoulder_pos) / sigma) ** 2)
    head = height * np.exp(-0.5 * ((x * length - head_pos) / sigma) ** 2)
    right_shoulder = shoulder_ratio * height * np.exp(-0.5 * ((x * length - right_shoulder_pos) / sigma) ** 2)

    # Combine the components
    pattern = np.maximum.reduce([left_shoulder, head, right_shoulder])

    # Add trend
    trend = trend_slope * (x - 0.5) * length

    # Combine pattern with trend and base price
    price = base_price + pattern + trend

    # Add noise
    price = add_noise(price, noise_level)

    return price


def generate_double_top(
    length: int = 100,
    height: float = 10.0,
    peak_distance_ratio: float = 0.3,
    trough_depth_ratio: float = 0.5,
    noise_level: float = 0.1,
    base_price: float = 100.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a synthetic Double Top pattern.

    Args:
        length: Length of the pattern
        height: Height of the peaks
        peak_distance_ratio: Distance between peaks as ratio of length
        trough_depth_ratio: Depth of the trough between peaks (0-1)
        noise_level: Level of noise to add
        base_price: Base price level
        random_seed: Optional random seed

    Returns:
        Numpy array with the price pattern
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create x-axis points
    x = np.linspace(0, 1, length)

    # Calculate positions for peaks and trough
    peak1_pos = length * (0.5 - peak_distance_ratio / 2)
    peak2_pos = length * (0.5 + peak_distance_ratio / 2)
    trough_pos = length * 0.5

    # Create Gaussian peaks
    sigma = length * 0.08
    peak1 = height * np.exp(-0.5 * ((x * length - peak1_pos) / sigma) ** 2)
    peak2 = height * np.exp(-0.5 * ((x * length - peak2_pos) / sigma) ** 2)
    trough = -trough_depth_ratio * height * np.exp(-0.5 * ((x * length - trough_pos) / (sigma / 2)) ** 2)

    # Combine components
    peaks = np.maximum(peak1, peak2)
    pattern = peaks + trough

    # Add downtrend after second peak
    downtrend = np.zeros_like(x)
    downtrend_start = int(peak2_pos) + int(sigma)
    if downtrend_start < length:
        slope = -height / (length - downtrend_start)
        downtrend[downtrend_start:] = slope * np.arange(length - downtrend_start)

    # Combine with base price
    price = base_price + pattern + downtrend

    # Add noise
    price = add_noise(price, noise_level)

    return price


def generate_double_bottom(
    length: int = 100,
    depth: float = 10.0,
    trough_distance_ratio: float = 0.3,
    peak_height_ratio: float = 0.5,
    noise_level: float = 0.1,
    base_price: float = 100.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a synthetic Double Bottom pattern.

    Args:
        length: Length of the pattern
        depth: Depth of the troughs
        trough_distance_ratio: Distance between troughs as ratio of length
        peak_height_ratio: Height of the peak between troughs (0-1)
        noise_level: Level of noise to add
        base_price: Base price level
        random_seed: Optional random seed

    Returns:
        Numpy array with the price pattern
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate a double top and invert it
    double_top = generate_double_top(
        length=length,
        height=depth,
        peak_distance_ratio=trough_distance_ratio,
        trough_depth_ratio=peak_height_ratio,
        noise_level=0,  # Add noise later
        base_price=0,
        random_seed=random_seed
    )

    # Invert the pattern
    pattern = -double_top

    # Add base price
    price = base_price + pattern

    # Add noise
    price = add_noise(price, noise_level)

    return price


def generate_triangle(
    length: int = 100,
    height: float = 10.0,
    pattern_type: str = "ascending",
    noise_level: float = 0.1,
    base_price: float = 100.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a synthetic triangle pattern (ascending, descending, or symmetric).

    Args:
        length: Length of the pattern
        height: Height/depth of the triangle
        pattern_type: One of "ascending", "descending", or "symmetric"
        noise_level: Level of noise to add
        base_price: Base price level
        random_seed: Optional random seed

    Returns:
        Numpy array with the price pattern
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create x-axis points
    x = np.linspace(0, 1, length)

    # Calculate baseline with appropriate slope
    baseline = np.zeros_like(x)

    # Create upper and lower bounds based on pattern type
    if pattern_type == "ascending":
        lower_bound = height * x
        upper_bound = height * (1 - 0.7 * x)
    elif pattern_type == "descending":
        lower_bound = height * (0.3 + 0.7 * x)
        upper_bound = height * (1 - x)
    else:  # symmetric
        lower_bound = height * (0.5 + 0.5 * x)
        upper_bound = height * (1 - 0.5 * x)

    # Create oscillating pattern between bounds
    oscillation = 0.5 * (upper_bound - lower_bound) * np.sin(x * np.pi * 6)
    pattern = lower_bound + 0.5 * (upper_bound - lower_bound) + oscillation

    # Ensure bounds are respected
    pattern = np.minimum(pattern, upper_bound)
    pattern = np.maximum(pattern, lower_bound)

    # Add base price
    price = base_price + pattern

    # Add noise
    price = add_noise(price, noise_level)

    return price


def generate_channel(
    length: int = 100,
    height: float = 10.0,
    slope: float = 0.1,
    noise_level: float = 0.1,
    base_price: float = 100.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a price channel pattern.

    Args:
        length: Length of the pattern
        height: Height of the channel
        slope: Slope of the channel (+ for ascending, - for descending)
        noise_level: Level of noise to add
        base_price: Base price level
        random_seed: Optional random seed

    Returns:
        Numpy array with the price pattern
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create x-axis points
    x = np.linspace(0, 1, length)

    # Create trend line
    trend = slope * height * length * x

    # Create oscillations within channel
    oscillations = height * 0.8 * np.sin(x * np.pi * 5)

    # Combine components
    pattern = trend + oscillations

    # Add base price
    price = base_price + pattern

    # Add noise
    price = add_noise(price, noise_level)

    return price


def generate_wedge(
    length: int = 100,
    height: float = 10.0,
    pattern_type: str = "rising",
    noise_level: float = 0.1,
    base_price: float = 100.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a wedge pattern (rising or falling).

    Args:
        length: Length of the pattern
        height: Height/depth of the wedge
        pattern_type: Either "rising" or "falling"
        noise_level: Level of noise to add
        base_price: Base price level
        random_seed: Optional random seed

    Returns:
        Numpy array with the price pattern
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create x-axis points
    x = np.linspace(0, 1, length)

    if pattern_type == "rising":
        # Rising wedge - resistance and support lines converge upward
        resistance = height * (0.5 + 0.8 * x)
        support = height * (0.2 + 0.5 * x)
        slope = 0.5
    else:  # falling
        # Falling wedge - resistance and support lines converge downward
        resistance = height * (1.0 - 0.5 * x)
        support = height * (0.7 - 0.5 * x)
        slope = -0.5

    # Create oscillating pattern between bounds
    oscillation = 0.4 * (resistance - support) * np.sin(x * np.pi * 4.5)
    midline = support + 0.5 * (resistance - support)
    pattern = midline + oscillation

    # Ensure bounds are respected
    pattern = np.minimum(pattern, resistance)
    pattern = np.maximum(pattern, support)

    # Add trend
    trend = slope * height * x

    # Add base price
    price = base_price + pattern + trend

    # Add noise
    price = add_noise(price, noise_level)

    return price


def generate_flag(
    length: int = 100,
    pole_height: float = 15.0,
    flag_height: float = 5.0,
    flag_slope: float = -0.1,
    noise_level: float = 0.1,
    base_price: float = 100.0,
    trend: str = "up",
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a flag pattern with pole.

    Args:
        length: Length of the pattern
        pole_height: Height of the flag pole
        trend: Direction of the trend ('up' for bull flag, 'down' for bear flag)
        flag_height: Height of the flag part
        flag_slope: Slope of the flag (+ for upward, - for downward)
        noise_level: Level of noise to add
        base_price: Base price level
        random_seed: Optional random seed

    Returns:
        Numpy array with the price pattern
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create x-axis points
    x = np.linspace(0, 1, length)

    # Adjust for bear flag (invert the pattern)
    multiplier = 1.0 if trend == "up" else -1.0

    # Flag pole (sharp rise or fall)
    pole_end = int(length * 0.3)
    pole = np.zeros(length)
    pole[:pole_end] = multiplier * np.linspace(0, pole_height, pole_end)

    # Flag pattern
    flag = np.zeros(length)
    flag_start = pole_end
    flag_slope_amount = flag_slope * flag_height * length
    flag_trend = np.linspace(0, flag_slope_amount, length - flag_start)

    # Oscillations in the flag
    oscillations = flag_height * 0.4 * np.sin(np.linspace(0, 4 * np.pi, length - flag_start))

    flag[flag_start:] = multiplier * (flag_trend + oscillations)

    # Combine components
    pattern = pole + flag

    # Add base price
    price = base_price + pattern

    # Add noise
    price = add_noise(price, noise_level)

    return price


def generate_random_walk(
    length: int = 100,
    volatility: float = 1.0,
    drift: float = 0.0,
    base_price: float = 100.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a random walk price series.

    Args:
        length: Length of the series
        volatility: Volatility of the random walk
        drift: Drift component (price change bias)
        base_price: Starting price
        random_seed: Optional random seed

    Returns:
        Numpy array with the price series
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random steps
    steps = np.random.normal(drift, volatility, size=length-1)

    # Cumulative sum to create random walk
    walks = np.cumsum(steps)

    # Add base price
    price = np.append([base_price], base_price + walks)

    return price


def generate_ohlcv(
    close_prices: np.ndarray,
    volatility_ratio: float = 0.2,
    volume_correlation: float = 0.3,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate OHLCV data from close prices.

    Args:
        close_prices: Array of close prices
        volatility_ratio: Ratio of high-low range to price
        volume_correlation: Correlation of volume with price change
        random_seed: Optional random seed

    Returns:
        DataFrame with OHLCV data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    length = len(close_prices)

    # Calculate price changes
    price_changes = np.zeros(length)
    price_changes[1:] = np.abs(close_prices[1:] - close_prices[:-1])

    # Generate high and low prices
    price_range = volatility_ratio * close_prices
    high_offset = np.random.uniform(0, price_range)
    low_offset = np.random.uniform(0, price_range)

    high_prices = close_prices + high_offset
    low_prices = close_prices - low_offset

    # Make sure lows don't exceed highs
    low_prices = np.minimum(low_prices, high_prices * 0.999)

    # Generate open prices between previous close and current close
    open_prices = np.zeros_like(close_prices)
    open_prices[0] = close_prices[0] * (1 - volatility_ratio / 2)

    for i in range(1, length):
        prev_close = close_prices[i-1]
        curr_close = close_prices[i]
        weight = np.random.uniform(0, 1)
        open_prices[i] = prev_close * weight + curr_close * (1 - weight)

    # Ensure OHLC relationship
    for i in range(length):
        low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])

    # Generate volumes correlated with price changes
    base_volume = 1000
    volume_noise = np.random.lognormal(0, 0.5, size=length)
    volumes = base_volume + volume_correlation * price_changes * 1000 + volume_noise * 500

    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })

    return df


def generate_synthetic_dataset(
    n_samples: int = 1000,
    pattern_types: Optional[List[PatternType]] = None,
    length: int = 100,
    base_price: float = 100.0,
    random_walks: float = 0.3,
    augmentation: bool = True,
    random_seed: Optional[int] = None
) -> Tuple[List[pd.DataFrame], List[PatternType]]:
    """
    Generate a synthetic dataset with various patterns.

    Args:
        n_samples: Number of samples to generate
        pattern_types: List of pattern types to include, or None for all
        length: Length of each pattern
        base_price: Base price level
        random_walks: Fraction of samples that should be random walks
        augmentation: Whether to apply augmentation techniques
        random_seed: Optional random seed

    Returns:
        Tuple of (list of DataFrames, list of pattern types)
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # If pattern_types not specified, use all supported types
    if pattern_types is None:
        pattern_types = [
            PatternType.HEAD_AND_SHOULDERS,
            PatternType.DOUBLE_TOP,
            PatternType.DOUBLE_BOTTOM,
            PatternType.ASCENDING_TRIANGLE,
            PatternType.DESCENDING_TRIANGLE,
            PatternType.SYMMETRICAL_TRIANGLE,
            PatternType.RISING_WEDGE,
            PatternType.FALLING_WEDGE,
            PatternType.RECTANGLE,
            PatternType.BULL_FLAG,
            PatternType.BEAR_FLAG
        ]

    # Function to generate each pattern type
    pattern_generators = {
        PatternType.HEAD_AND_SHOULDERS: generate_head_and_shoulders,
        PatternType.DOUBLE_TOP: generate_double_top,
        PatternType.DOUBLE_BOTTOM: generate_double_bottom,
        PatternType.ASCENDING_TRIANGLE: lambda **kwargs: generate_triangle(pattern_type="ascending", **kwargs),
        PatternType.DESCENDING_TRIANGLE: lambda **kwargs: generate_triangle(pattern_type="descending", **kwargs),
        PatternType.SYMMETRICAL_TRIANGLE: lambda **kwargs: generate_triangle(pattern_type="symmetrical", **kwargs),
        PatternType.RECTANGLE: lambda **kwargs: generate_channel(slope=random.choice([0.1, -0.1]), **kwargs),
        PatternType.RISING_WEDGE: lambda **kwargs: generate_wedge(pattern_type="rising", **kwargs),
        PatternType.FALLING_WEDGE: lambda **kwargs: generate_wedge(pattern_type="falling", **kwargs),
        PatternType.BULL_FLAG: generate_flag,
        PatternType.BEAR_FLAG: lambda **kwargs: generate_flag(trend="down", **kwargs),
    }

    # Calculate number of each pattern
    n_random_walks = int(n_samples * random_walks)
    n_patterns = n_samples - n_random_walks

    # Initialize result lists
    dataframes = []
    labels = []

    # Generate patterns
    patterns_per_type = n_patterns // len(pattern_types)
    remainder = n_patterns % len(pattern_types)

    for pattern_type in pattern_types:
        # Calculate how many of this pattern to generate
        count = patterns_per_type + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1

        # Skip if this is the RANDOM type (handled separately)
        if pattern_type == PatternType.RECTANGLE:
            continue

        for i in range(count):
            # Generate pattern with random parameters
            noise_level = random.uniform(0.05, 0.15)
            pattern_generator = pattern_generators.get(pattern_type)

            if pattern_generator:
                close_prices = pattern_generator(
                    length=length,
                    noise_level=noise_level,
                    base_price=base_price + random.uniform(-10, 10),
                    random_seed=None  # Use different seed each time
                )

                # Convert to OHLCV DataFrame
                df = generate_ohlcv(close_prices)

                dataframes.append(df)
                labels.append(pattern_type)

    # Generate random walks
    for i in range(n_random_walks):
        close_prices = generate_random_walk(
            length=length,
            volatility=random.uniform(0.5, 1.5),
            drift=random.uniform(-0.05, 0.05),
            base_price=base_price + random.uniform(-10, 10),
            random_seed=None
        )

        df = generate_ohlcv(close_prices)

        dataframes.append(df)
        labels.append(PatternType.RECTANGLE)

    # Shuffle the data
    combined = list(zip(dataframes, labels))
    random.shuffle(combined)
    dataframes, labels = zip(*combined)

    return list(dataframes), list(labels)


def create_pytorch_dataset(
    dataframes: List[pd.DataFrame],
    labels: List[PatternType],
    window_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert list of DataFrames and labels to PyTorch tensors.

    Args:
        dataframes: List of DataFrames with OHLCV data
        labels: List of pattern types
        window_size: Optional window size to truncate or pad sequences

    Returns:
        Tuple of (features tensor, labels tensor)
    """
    # Convert pattern types to integers
    all_pattern_types = list(PatternType)
    label_indices = [all_pattern_types.index(label) for label in labels]

    # Determine sequence length
    if window_size is None:
        # Use length of first DataFrame
        window_size = len(dataframes[0])

    # Initialize tensors
    n_samples = len(dataframes)
    n_features = 5  # OHLCV
    features = torch.zeros((n_samples, window_size, n_features), dtype=torch.float32)

    for i, df in enumerate(dataframes):
        # Extract OHLCV values
        ohlcv = df[['open', 'high', 'low', 'close', 'volume']].values

        # Handle sequences of different lengths
        seq_len = min(len(ohlcv), window_size)

        # Normalize each feature
        for j in range(4):  # OHLC
            # Min-max normalization for OHLC
            min_val = ohlcv[:seq_len, j].min()
            max_val = ohlcv[:seq_len, j].max()
            if max_val > min_val:
                ohlcv[:seq_len, j] = (ohlcv[:seq_len, j] - min_val) / (max_val - min_val)
            else:
                ohlcv[:seq_len, j] = 0.5

        # Log-normalize volume
        if ohlcv[:seq_len, 4].min() > 0:
            log_vol = np.log(ohlcv[:seq_len, 4])
            min_vol = log_vol.min()
            max_vol = log_vol.max()
            if max_vol > min_vol:
                ohlcv[:seq_len, 4] = (log_vol - min_vol) / (max_vol - min_vol)
            else:
                ohlcv[:seq_len, 4] = 0.5

        # Copy to tensor (handles padding or truncation)
        features[i, :seq_len, :] = torch.tensor(ohlcv[:seq_len], dtype=torch.float32)

    # Create labels tensor
    labels_tensor = torch.tensor(label_indices, dtype=torch.long)

    return features, labels_tensor

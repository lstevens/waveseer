"""
Technical indicators module for Waveseer pattern detection.

This module contains implementations of various technical indicators used
for financial market analysis and pattern detection. The implementations
are optimized for Polars DataFrames but maintain compatibility with Pandas.

Ported and enhanced from crypto_heatmap indicator implementations.
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Union, Dict, Callable
from dataclasses import dataclass


def to_polars(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """Convert pandas DataFrame to polars if needed."""
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def to_pandas(df: Union[pd.DataFrame, pl.DataFrame]) -> pd.DataFrame:
    """Convert polars DataFrame to pandas if needed."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def calculate_rsi(df: Union[pd.DataFrame, pl.DataFrame], period: int = 14,
                 column: str = 'close') -> pl.Series:
    """Calculate the Relative Strength Index (RSI).

    Args:
        df: DataFrame with OHLCV data (polars or pandas)
        period: RSI period (default: 14)
        column: Column name to use for calculation (default: 'close')

    Returns:
        Series with RSI values
    """
    # Convert to pandas for calculation (more efficient RSI implementation)
    pdf = to_pandas(df)

    # Calculate RSI using pandas
    delta = pdf[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Convert back to polars
    return pl.from_pandas(rsi)


def calculate_macd(df: Union[pd.DataFrame, pl.DataFrame], fast_period: int = 12,
                  slow_period: int = 26, signal_period: int = 9,
                  column: str = 'close') -> Dict[str, pl.Series]:
    """Calculate the Moving Average Convergence Divergence (MACD).

    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal EMA period (default: 9)
        column: Column name to use for calculation (default: 'close')

    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' Series
    """
    # Convert to pandas for calculation
    pdf = to_pandas(df)

    # Calculate MACD components
    exp1 = pdf[column].ewm(span=fast_period, adjust=False).mean()
    exp2 = pdf[column].ewm(span=slow_period, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    # Convert back to polars
    return {
        'macd': pl.from_pandas(macd_line),
        'signal': pl.from_pandas(signal_line),
        'histogram': pl.from_pandas(histogram)
    }


def calculate_bollinger_bands(df: Union[pd.DataFrame, pl.DataFrame], period: int = 20,
                             std_dev: float = 2.0, column: str = 'close') -> Dict[str, pl.Series]:
    """Calculate Bollinger Bands.

    Args:
        df: DataFrame with OHLCV data
        period: Moving average period (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
        column: Column name to use for calculation (default: 'close')

    Returns:
        Dictionary with 'middle', 'upper', and 'lower' band Series
    """
    # Convert to pandas for calculation
    pdf = to_pandas(df)

    # Calculate Bollinger Bands
    middle = pdf[column].rolling(window=period).mean()
    stdev = pdf[column].rolling(window=period).std()
    upper = middle + (stdev * std_dev)
    lower = middle - (stdev * std_dev)

    # Convert back to polars
    return {
        'middle': pl.from_pandas(middle),
        'upper': pl.from_pandas(upper),
        'lower': pl.from_pandas(lower)
    }


def calculate_ema(df: Union[pd.DataFrame, pl.DataFrame], period: int,
                 column: str = 'close') -> pl.Series:
    """Calculate the Exponential Moving Average (EMA).

    Args:
        df: DataFrame with OHLCV data
        period: EMA period
        column: Column name to use for calculation (default: 'close')

    Returns:
        Series with EMA values
    """
    # Convert to pandas for calculation
    pdf = to_pandas(df)

    # Calculate EMA
    ema = pdf[column].ewm(span=period, adjust=False).mean()

    # Convert back to polars
    return pl.from_pandas(ema)


def calculate_sma(df: Union[pd.DataFrame, pl.DataFrame], period: int,
                 column: str = 'close') -> pl.Series:
    """Calculate the Simple Moving Average (SMA).

    Args:
        df: DataFrame with OHLCV data
        period: SMA period
        column: Column name to use for calculation (default: 'close')

    Returns:
        Series with SMA values
    """
    # Convert to polars for calculation
    pldf = to_polars(df)

    # Calculate SMA directly in polars
    return pldf.select(
        pl.col(column).rolling_mean(period)
    ).to_series()


def calculate_atr(df: Union[pd.DataFrame, pl.DataFrame], period: int = 14) -> pl.Series:
    """Calculate the Average True Range (ATR).

    Args:
        df: DataFrame with OHLCV data
        period: ATR period (default: 14)

    Returns:
        Series with ATR values
    """
    # Convert to pandas for calculation
    pdf = to_pandas(df)

    # Calculate True Range
    high = pdf['high']
    low = pdf['low']
    close_prev = pdf['close'].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate ATR
    atr = true_range.rolling(window=period).mean()

    # Convert back to polars
    return pl.from_pandas(atr)


def calculate_stochastic(df: Union[pd.DataFrame, pl.DataFrame], k_period: int = 14,
                        d_period: int = 3) -> Dict[str, pl.Series]:
    """Calculate Stochastic Oscillator.

    Args:
        df: DataFrame with OHLCV data
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)

    Returns:
        Dictionary with 'k' and 'd' Series
    """
    # Convert to pandas for calculation
    pdf = to_pandas(df)

    # Calculate %K
    low_min = pdf['low'].rolling(window=k_period).min()
    high_max = pdf['high'].rolling(window=k_period).max()

    k = 100 * ((pdf['close'] - low_min) / (high_max - low_min))

    # Calculate %D
    d = k.rolling(window=d_period).mean()

    # Convert back to polars
    return {
        'k': pl.from_pandas(k),
        'd': pl.from_pandas(d)
    }


@dataclass
class IndicatorFunction:
    """Configuration for indicator mathematical transformations."""
    name: str
    function: Callable
    description: str
    parameters: Dict[str, float]


class MathematicalFunctions:
    """Mathematical functions for indicator normalization.

    Ported from crypto_heatmap indicator_functions.py
    """

    @staticmethod
    def gaussian(x: np.ndarray, mu: float = 0, sigma: float = 1) -> np.ndarray:
        """Gaussian (Bell Curve) Normalization"""
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def sigmoid(x: np.ndarray, k: float = 1, x0: float = 0) -> np.ndarray:
        """Sigmoid (Logistic) Normalization"""
        return 1 / (1 + np.exp(-k * (x - x0)))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh (Hyperbolic Tangent) Normalization"""
        return np.tanh(x)

    @staticmethod
    def zscore(x: np.ndarray, window: int = 20) -> np.ndarray:
        """Z-score Normalization"""
        series = pd.Series(x)
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        return (series - rolling_mean) / rolling_std

    @staticmethod
    def raw(x: np.ndarray) -> np.ndarray:
        """Raw Value (No Transformation)"""
        return x

    @staticmethod
    def clip(x: np.ndarray, min_val: float = -1, max_val: float = 1) -> np.ndarray:
        """Clip Values to Range"""
        return np.clip(x, min_val, max_val)


# Indicator configuration with optimal mathematical functions
INDICATOR_CONFIGS = {
    'rsi': {
        'title': 'RSI',
        'range': (0, 100),
        'neutral': 50,
        'description': 'Relative Strength Index (0-100)',
        'functions': [
            IndicatorFunction(
                name='gaussian',
                function=MathematicalFunctions.gaussian,
                description='Gaussian normalization for RSI extremes',
                parameters={'mu': 50, 'sigma': 20}
            ),
            IndicatorFunction(
                name='sigmoid',
                function=MathematicalFunctions.sigmoid,
                description='Sigmoid normalization for RSI transitions',
                parameters={'k': 0.1, 'x0': 50}
            )
        ]
    },
    'macd': {
        'title': 'MACD',
        'range': (-10, 10),  # Varies by asset
        'neutral': 0,
        'description': 'Moving Average Convergence Divergence',
        'functions': [
            IndicatorFunction(
                name='tanh',
                function=MathematicalFunctions.tanh,
                description='Tanh normalization for MACD',
                parameters={}
            ),
            IndicatorFunction(
                name='zscore',
                function=MathematicalFunctions.zscore,
                description='Z-score normalization for MACD',
                parameters={'window': 20}
            )
        ]
    },
    'bollinger': {
        'title': 'Bollinger Bands %B',
        'range': (0, 1),
        'neutral': 0.5,
        'description': 'Position within Bollinger Bands (0-1)',
        'functions': [
            IndicatorFunction(
                name='sigmoid',
                function=MathematicalFunctions.sigmoid,
                description='Sigmoid for Bollinger Band position',
                parameters={'k': 5, 'x0': 0.5}
            )
        ]
    }
}


def normalize_indicator(values: np.ndarray, indicator: str,
                       function: str = 'gaussian', sensitivity: float = 1.0) -> np.ndarray:
    """Normalize indicator values using the specified function.

    Args:
        values: Array of indicator values
        indicator: Name of the indicator (e.g., 'rsi', 'macd')
        function: Function to use for normalization ('gaussian', 'sigmoid', 'tanh', etc.)
        sensitivity: Sensitivity factor to adjust the normalization (default: 1.0)
            Higher sensitivity makes the response curve steeper, emphasizing changes
            around the neutral point. For sigmoid, this affects k; for gaussian, this
            affects sigma inversely.

    Returns:
        Normalized values in range [-1, 1]
    """
    if indicator not in INDICATOR_CONFIGS:
        raise ValueError(f"Unknown indicator: {indicator}")

    # Get indicator config
    config = INDICATOR_CONFIGS[indicator]

    # Find the specified function
    func_config = None
    for func in config['functions']:
        if func.name == function:
            func_config = func
            break

    if func_config is None:
        # Default to first function if specified one not found
        func_config = config['functions'][0]

    # Scale the values to appropriate range if needed
    indicator_min, indicator_max = config['range']
    neutral = config['neutral']

    # Apply function with specified parameters
    params = func_config.parameters.copy()

    # Apply sensitivity factor to function parameters
    # Note: sensitivity affects parameters differently depending on function
    for param in params:
        if param == 'sigma':
            # For Gaussian, higher sensitivity means narrower bell curve (smaller sigma)
            params[param] /= sensitivity
        elif param == 'k':
            # For Sigmoid, higher sensitivity means steeper slope (larger k)
            params[param] *= sensitivity

    # Apply the normalization function
    normalized = func_config.function(values, **params)

    # Ensure output is in range [-1, 1]
    return MathematicalFunctions.clip(normalized, -1, 1)


def calculate_all_indicators(df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """Calculate all technical indicators for the given DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicator columns
    """
    # Convert to polars
    pldf = to_polars(df)

    # RSI
    rsi = calculate_rsi(pldf)

    # MACD
    macd_components = calculate_macd(pldf)

    # Bollinger Bands
    bb = calculate_bollinger_bands(pldf)

    # Stochastic
    stoch = calculate_stochastic(pldf)

    # ATR
    atr = calculate_atr(pldf)

    # Moving Averages
    ema20 = calculate_ema(pldf, 20)
    sma50 = calculate_sma(pldf, 50)
    sma200 = calculate_sma(pldf, 200)

    # Create result DataFrame with indicators
    result = pldf.with_columns([
        pl.lit(rsi).alias('rsi'),
        pl.lit(macd_components['histogram']).alias('macd_hist'),
        pl.lit(macd_components['macd']).alias('macd'),
        pl.lit(macd_components['signal']).alias('macd_signal'),
        pl.lit(bb['middle']).alias('bb_middle'),
        pl.lit(bb['upper']).alias('bb_upper'),
        pl.lit(bb['lower']).alias('bb_lower'),
        pl.lit(stoch['k']).alias('stoch_k'),
        pl.lit(stoch['d']).alias('stoch_d'),
        pl.lit(atr).alias('atr'),
        pl.lit(ema20).alias('ema20'),
        pl.lit(sma50).alias('sma50'),
        pl.lit(sma200).alias('sma200')
    ])

    return result

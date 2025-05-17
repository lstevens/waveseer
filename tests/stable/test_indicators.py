"""
Tests for the technical indicators module.

These tests validate the accuracy of various technical indicators used
for financial market analysis and pattern detection.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from wave.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_sma,
    calculate_atr,
    calculate_stochastic,
    normalize_indicator,
    MathematicalFunctions,
    calculate_all_indicators
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing indicators."""
    # Create a datetime index
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a price series with a trend component and random noise
    # This gives more realistic price action for testing indicators
    trend = np.linspace(100, 150, 100)
    cycle = 10 * np.sin(np.linspace(0, 4*np.pi, 100))
    noise = np.random.normal(0, 3, 100)
    
    close = trend + cycle + noise
    
    # Create realistic OHLCV data
    high = close + np.random.uniform(0, 5, 100)
    low = close - np.random.uniform(0, 5, 100)
    open_price = close.copy()
    np.random.shuffle(open_price)  # Randomize open prices for realism
    volume = np.random.uniform(1000, 5000, 100) * (1 + 0.1 * np.random.randn(100))
    
    # Create pandas DataFrame
    df_pd = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # Create polars DataFrame with same data
    df_pl = pl.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return {
        'pandas': df_pd,
        'polars': df_pl,
        'original_close': close  # Keep original for validation
    }


def test_calculate_rsi(sample_data):
    """Test RSI calculation with both pandas and polars."""
    # Test with pandas input
    rsi_pd = calculate_rsi(sample_data['pandas'])
    
    # Test with polars input
    rsi_pl = calculate_rsi(sample_data['polars'])
    
    # Verify results match
    np.testing.assert_allclose(
        rsi_pd.to_numpy(), 
        rsi_pl.to_numpy(), 
        rtol=1e-10, atol=1e-10
    )
    
    # Verify RSI properties
    assert rsi_pd.min() >= 0
    assert rsi_pd.max() <= 100
    
    # Verify length
    assert len(rsi_pd) == len(sample_data['pandas'])


def test_calculate_macd(sample_data):
    """Test MACD calculation with both pandas and polars."""
    # Test with pandas input
    macd_pd = calculate_macd(sample_data['pandas'])
    
    # Test with polars input
    macd_pl = calculate_macd(sample_data['polars'])
    
    # Verify components match between pandas and polars
    for component in ['macd', 'signal', 'histogram']:
        np.testing.assert_allclose(
            macd_pd[component].to_numpy(),
            macd_pl[component].to_numpy(),
            rtol=1e-10, atol=1e-10
        )
    
    # Verify MACD properties
    # Histogram should equal macd - signal
    expected_hist = macd_pd['macd'] - macd_pd['signal']
    np.testing.assert_allclose(
        macd_pd['histogram'].to_numpy(),
        expected_hist.to_numpy(),
        rtol=1e-10, atol=1e-10
    )


def test_calculate_bollinger_bands(sample_data):
    """Test Bollinger Bands calculation."""
    # Calculate with default parameters
    bands = calculate_bollinger_bands(sample_data['polars'])
    
    # Verify components
    assert 'middle' in bands
    assert 'upper' in bands
    assert 'lower' in bands
    
    # Verify properties
    upper = bands['upper'].to_numpy()
    middle = bands['middle'].to_numpy()
    lower = bands['lower'].to_numpy()
    
    # Upper should be above middle, lower should be below
    # (ignoring NaN values at start from rolling calculation)
    valid_idx = ~np.isnan(middle)
    assert np.all(upper[valid_idx] >= middle[valid_idx])
    assert np.all(lower[valid_idx] <= middle[valid_idx])
    
    # Test with different parameters
    # For proper comparison, we need to use the same period but different std_dev
    bands_3std = calculate_bollinger_bands(sample_data['polars'], period=20, std_dev=3.0)
    
    # 3std bands should be wider than 2std bands
    upper_3std = bands_3std['upper'].to_numpy()
    lower_3std = bands_3std['lower'].to_numpy()
    middle_3std = bands_3std['middle'].to_numpy()
    
    # Ensure we're comparing the same time periods
    valid_idx = ~np.isnan(middle) & ~np.isnan(middle_3std)
    
    # The upper band of 3std should be higher than 2std
    assert np.all(upper_3std[valid_idx] >= upper[valid_idx])
    # The lower band of 3std should be lower than 2std
    assert np.all(lower_3std[valid_idx] <= lower[valid_idx])


def test_calculate_ema(sample_data):
    """Test EMA calculation."""
    # Calculate EMAs with different periods
    ema20 = calculate_ema(sample_data['polars'], 20)
    ema50 = calculate_ema(sample_data['polars'], 50)
    
    # Verify that EMA responds faster for shorter periods
    # Get the original close values
    close = sample_data['original_close']
    
    # Create a 20% price jump at index 50
    modified_close = close.copy()
    modified_close[50] = close[49] * 1.2
    
    # Create a DataFrame with the modified close
    df = pd.DataFrame({'close': modified_close})
    
    # Calculate EMAs
    ema20_mod = calculate_ema(df, 20)
    ema50_mod = calculate_ema(df, 50)
    
    # Get values after the price jump
    ema20_after = ema20_mod.to_numpy()[51]
    ema50_after = ema50_mod.to_numpy()[51]
    
    # Calculate percentage change in EMAs
    ema20_pct_change = (ema20_after - ema20_mod.to_numpy()[49]) / ema20_mod.to_numpy()[49]
    ema50_pct_change = (ema50_after - ema50_mod.to_numpy()[49]) / ema50_mod.to_numpy()[49]
    
    # 20-period EMA should respond more (have a bigger % change)
    assert ema20_pct_change > ema50_pct_change


def test_calculate_stochastic(sample_data):
    """Test Stochastic Oscillator calculation."""
    # Calculate Stochastic
    stoch = calculate_stochastic(sample_data['polars'])
    
    # Verify components
    assert 'k' in stoch
    assert 'd' in stoch
    
    # Verify properties
    k = stoch['k'].to_numpy()
    d = stoch['d'].to_numpy()
    
    # Values should be in range [0, 100]
    valid_k = k[~np.isnan(k)]
    valid_d = d[~np.isnan(d)]
    
    assert np.all(valid_k >= 0) and np.all(valid_k <= 100)
    assert np.all(valid_d >= 0) and np.all(valid_d <= 100)
    
    # %D should be smoother than %K
    # Compare standard deviations of valid values
    assert np.nanstd(valid_d) <= np.nanstd(valid_k)


def test_normalize_indicator():
    """Test indicator normalization functions."""
    # Create sample RSI values across full range
    rsi_values = np.linspace(0, 100, 101)
    
    # Test Gaussian normalization
    norm_gaussian = normalize_indicator(rsi_values, 'rsi', 'gaussian')
    
    # Verify properties
    assert norm_gaussian.min() >= -1
    assert norm_gaussian.max() <= 1
    
    # Test sigmoid normalization which has a clearer sensitivity effect
    norm_sensitive = normalize_indicator(rsi_values, 'rsi', 'sigmoid', sensitivity=2.0)
    norm_less_sensitive = normalize_indicator(rsi_values, 'rsi', 'sigmoid', sensitivity=0.5)
    
    # For sigmoid, higher sensitivity (k parameter) makes the transition from -1 to 1 steeper
    # Calculate where the values cross 0
    sensitive_zero_crossing = np.where(np.diff(np.signbit(norm_sensitive)))[0]
    less_sensitive_zero_crossing = np.where(np.diff(np.signbit(norm_less_sensitive)))[0]
    
    # If no zero crossing (possible with extreme sensitivities), use the midpoint
    if len(sensitive_zero_crossing) == 0:
        sensitive_zero_crossing = np.array([50])
    if len(less_sensitive_zero_crossing) == 0:
        less_sensitive_zero_crossing = np.array([50])
    
    # Check values around midpoint
    midpoint = int(len(rsi_values) / 2)
    range_size = 5  # Check 5 values on each side
    
    # For sigmoid with higher sensitivity (k), the values will change more rapidly around the midpoint
    sensitive_change = np.abs(norm_sensitive[midpoint + range_size] - norm_sensitive[midpoint - range_size])
    less_sensitive_change = np.abs(norm_less_sensitive[midpoint + range_size] - norm_less_sensitive[midpoint - range_size])
    
    # Higher sensitivity should show more change over the same range
    assert sensitive_change > less_sensitive_change


def test_calculate_all_indicators(sample_data):
    """Test the omnibus indicator calculation function."""
    # Call the function
    result = calculate_all_indicators(sample_data['polars'])
    
    # Check that result is a polars DataFrame
    assert isinstance(result, pl.DataFrame)
    
    # Check that all expected columns are present
    expected_columns = [
        'rsi', 'macd_hist', 'macd', 'macd_signal',
        'bb_middle', 'bb_upper', 'bb_lower',
        'stoch_k', 'stoch_d', 'atr',
        'ema20', 'sma50', 'sma200'
    ]
    
    for col in expected_columns:
        assert col in result.columns
    
    # Check that original data is preserved
    assert 'open' in result.columns
    assert 'high' in result.columns
    assert 'low' in result.columns
    assert 'close' in result.columns
    assert 'volume' in result.columns
    
    # Check row count is preserved
    assert len(result) == len(sample_data['polars'])


def test_mathematical_functions():
    """Test the mathematical normalization functions."""
    # Create sample data
    x = np.linspace(-3, 3, 100)
    
    # Test Gaussian
    gaussian = MathematicalFunctions.gaussian(x)
    assert gaussian.max() <= 1.0
    assert gaussian.min() >= 0.0
    assert np.argmax(gaussian) == np.argmin(np.abs(x))  # Peak at x=0
    
    # Test Sigmoid
    sigmoid = MathematicalFunctions.sigmoid(x)
    assert sigmoid.max() <= 1.0
    assert sigmoid.min() >= 0.0
    assert sigmoid[x > 0].min() >= 0.5  # Values for x>0 should be >0.5
    assert sigmoid[x < 0].max() <= 0.5  # Values for x<0 should be <0.5
    
    # Test Tanh
    tanh = MathematicalFunctions.tanh(x)
    assert tanh.max() <= 1.0
    assert tanh.min() >= -1.0
    assert np.allclose(tanh, np.tanh(x))  # Should match numpy's tanh
    
    # Test Clipping
    values = np.array([-2, -1, 0, 1, 2])
    clipped = MathematicalFunctions.clip(values, -1, 1)
    assert clipped.min() == -1
    assert clipped.max() == 1
    assert np.array_equal(clipped, np.array([-1, -1, 0, 1, 1]))


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

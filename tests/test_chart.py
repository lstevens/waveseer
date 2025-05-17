"""
Unit tests for the chart drawing module.
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from datetime import datetime, timedelta

from wave.chart import draw_candlestick_chart


@pytest.fixture
def sample_data():
    """Create sample OHLCV dataframe for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": np.random.rand(100) * 100 + 20000,
        "high": np.random.rand(100) * 100 + 20100,
        "low": np.random.rand(100) * 100 + 19900,
        "close": np.random.rand(100) * 100 + 20050,
        "volume": np.random.rand(100) * 10
    })
    return df


def test_draw_candlestick_chart_basic(sample_data):
    """Test basic chart generation with sample data."""
    img_str = draw_candlestick_chart(sample_data, title="Test Chart")
    
    # Verify we get a non-empty base64 string
    assert isinstance(img_str, str)
    assert len(img_str) > 1000
    # Base64 might not have padding if length is divisible by 3
    assert all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/' or c == '=' for c in img_str)


def test_draw_candlestick_chart_empty():
    """Test chart generation with empty dataframe."""
    df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Should handle empty dataframe gracefully
    with pytest.raises(ValueError):
        draw_candlestick_chart(df)


def test_draw_candlestick_chart_single_row():
    """Test chart generation with single data point."""
    df = pd.DataFrame({
        'open': [100.0],
        'high': [105.0],
        'low': [95.0],
        'close': [102.0],
        'volume': [1000.0]
    })
    
    img_str = draw_candlestick_chart(df)
    assert isinstance(img_str, str)
    assert len(img_str) > 1000


def test_draw_candlestick_chart_custom_size(sample_data):
    """Test chart generation with custom figure size."""
    img_str = draw_candlestick_chart(sample_data, figsize=(15, 8))
    
    # Larger figure should produce longer base64 string
    assert isinstance(img_str, str)
    assert len(img_str) > 1000


def test_draw_candlestick_chart_datetime_index(sample_data):
    """Test chart with datetime index."""
    df_indexed = sample_data.set_index('datetime')
    
    img_str = draw_candlestick_chart(df_indexed)
    assert isinstance(img_str, str)
    assert len(img_str) > 1000


def test_draw_candlestick_chart_no_volume(sample_data):
    """Test chart generation with missing volume data."""
    df_no_volume = sample_data.copy()
    df_no_volume['volume'] = 0  # Replace volume with zeros
    
    img_str = draw_candlestick_chart(df_no_volume)
    assert isinstance(img_str, str)
    assert len(img_str) > 1000

"""
Tests for pattern visualization module.
"""
import os
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64
from PIL import Image

from wave.patterns import PatternType, PatternMatch
from wave.ml.viz.pattern_viz import (
    PatternVisualizer,
    overlay_pattern,
    draw_patterns_on_chart,
    create_confidence_indicator,
    visualize_attention_map
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100)
    
    # Create some realistic price movements
    base = 100
    moves = np.cumsum(np.random.normal(0, 1, 100))
    close = base + moves
    
    # Create open, high, low based on close
    daily_volatility = 1.0
    high = close + np.random.uniform(0, daily_volatility, 100)
    low = close - np.random.uniform(0, daily_volatility, 100)
    open_prices = close - np.random.uniform(-daily_volatility/2, daily_volatility/2, 100)
    
    # Create volume
    volume = np.random.uniform(1000, 5000, 100)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


@pytest.fixture
def sample_patterns():
    """Create sample pattern matches for testing."""
    patterns = {
        PatternType.HEAD_AND_SHOULDERS: [
            PatternMatch(
                pattern_id="hs_1",
                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                score=0.85,
                start_idx=10,
                end_idx=30,
                bars_matched=21,
                indicator_scores={
                    "price": 0.9,
                    "volume": 0.7
                }
            )
        ],
        PatternType.DOUBLE_BOTTOM: [
            PatternMatch(
                pattern_id="db_1",
                pattern_type=PatternType.DOUBLE_BOTTOM,
                score=0.75,
                start_idx=50,
                end_idx=70,
                bars_matched=21,
                indicator_scores={
                    "price": 0.8,
                    "volume": 0.6
                }
            )
        ]
    }
    return patterns


def test_pattern_visualizer_init():
    """Test PatternVisualizer initialization."""
    visualizer = PatternVisualizer()
    assert visualizer is not None
    
    # With custom colors
    custom_colors = {
        PatternType.HEAD_AND_SHOULDERS: "blue",
        PatternType.DOUBLE_BOTTOM: "green"
    }
    visualizer = PatternVisualizer(pattern_colors=custom_colors)
    assert visualizer.pattern_colors[PatternType.HEAD_AND_SHOULDERS] == "blue"


def test_overlay_pattern(sample_ohlcv_data, sample_patterns):
    """Test overlay_pattern function."""
    # Create a figure with sample data
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Create a sample candlestick chart
    x = np.arange(len(sample_ohlcv_data))
    width = 0.6
    up = sample_ohlcv_data[sample_ohlcv_data['close'] >= sample_ohlcv_data['open']]
    down = sample_ohlcv_data[sample_ohlcv_data['close'] < sample_ohlcv_data['open']]
    
    # For testing, just draw some lines instead of full candlesticks
    ax.plot(x, sample_ohlcv_data['close'])
    
    # Get a pattern to overlay
    pattern = sample_patterns[PatternType.HEAD_AND_SHOULDERS][0]
    
    # Call the overlay function
    result_ax = overlay_pattern(ax, pattern, sample_ohlcv_data)
    
    # Verify the result is the same axis
    assert result_ax is ax
    
    # Close figure to avoid memory leak
    plt.close(fig)


def test_draw_patterns_on_chart(sample_ohlcv_data, sample_patterns):
    """Test draw_patterns_on_chart function."""
    # Call the function
    img_str = draw_patterns_on_chart(sample_ohlcv_data, sample_patterns)
    
    # Verify the result is a base64 string
    assert isinstance(img_str, str)
    assert img_str.startswith("data:image/png;base64,") or len(img_str) > 1000
    
    # Try to decode and load the image to ensure it's valid
    img_data = img_str
    if "base64," in img_str:
        # Handle if the function returns with data URL format
        img_data = img_str.split("base64,")[1]
    
    try:
        # This will fail if the image data is invalid
        Image.open(io.BytesIO(base64.b64decode(img_data)))
    except Exception as e:
        pytest.fail(f"Failed to decode image: {e}")


def test_create_confidence_indicator():
    """Test create_confidence_indicator function."""
    # Test with different confidence levels
    high_confidence = create_confidence_indicator(0.9)
    medium_confidence = create_confidence_indicator(0.6)
    low_confidence = create_confidence_indicator(0.3)
    
    # Verify the results produce different visual properties
    assert high_confidence != medium_confidence
    assert medium_confidence != low_confidence
    
    # Check correct keys are present
    for conf in [high_confidence, medium_confidence, low_confidence]:
        assert "alpha" in conf or "color" in conf or "linewidth" in conf


def test_visualize_attention_map(sample_ohlcv_data):
    """Test visualize_attention_map function."""
    # Create fake attention weights
    attention_weights = np.random.random((len(sample_ohlcv_data), len(sample_ohlcv_data)))
    
    # Call the function
    img_str = visualize_attention_map(sample_ohlcv_data, attention_weights, head_idx=0)
    
    # Verify the result is a base64 string
    assert isinstance(img_str, str)
    assert img_str.startswith("data:image/png;base64,") or len(img_str) > 1000

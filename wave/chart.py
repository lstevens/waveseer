"""
Candlestick chart generation module for Waveseer.
Adapts visualization code from crypto_heatmap.
"""
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple


def draw_candlestick_chart(
    df,
    title: str = None,
    figsize: Tuple[int, int] = (10, 6)
) -> str:
    """Draw price chart with candlesticks and volume.

    Args:
        df: DataFrame with OHLCV data (must have open, high, low, close, volume)
        title: Optional chart title
        figsize: Figure size as (width, height) tuple

    Returns:
        Base64 encoded PNG image

    Raises:
        ValueError: If dataframe is empty or missing required columns
    """
    # Validate dataframe has required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame missing required column: {col}")

    # Check if dataframe is empty
    if len(df) == 0:
        raise ValueError("Cannot generate chart: DataFrame is empty")
    # Create figure
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Plot candlesticks
    width = 0.6
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]

    # Convert index to numeric for plotting
    x = np.arange(len(df))

    # Handle different index types (numeric, datetime, etc.)
    if hasattr(up, 'index') and not isinstance(up.index, pd.RangeIndex):
        # For datetime index or other non-numeric indices, map to positions
        up_idx = df.index.get_indexer(up.index)
        down_idx = df.index.get_indexer(down.index)
    else:
        # For default RangeIndex
        up_idx = up.index
        down_idx = down.index

    # Up candlesticks (green)
    if len(up) > 0:
        ax.bar(x[up_idx], up['close'] - up['open'], width, bottom=up['open'], color='green', alpha=0.6)
        ax.bar(x[up_idx], up['high'] - up['close'], width*0.2, bottom=up['close'], color='green', alpha=0.6)
        ax.bar(x[up_idx], up['low'] - up['open'], width*0.2, bottom=up['open'], color='green', alpha=0.6)

    # Down candlesticks (red)
    if len(down) > 0:
        ax.bar(x[down_idx], down['close'] - down['open'], width, bottom=down['open'], color='red', alpha=0.6)
        ax.bar(x[down_idx], down['high'] - down['open'], width*0.2, bottom=down['open'], color='red', alpha=0.6)
        ax.bar(x[down_idx], down['low'] - down['close'], width*0.2, bottom=down['close'], color='red', alpha=0.6)

    # Add volume bars
    volume_ax = ax.twinx()
    volume_ax.bar(x, df['volume'], width=0.6, color='gray', alpha=0.3)

    # Set y-limits for volume with minimum range to avoid warning
    max_volume = df['volume'].max()
    if max_volume > 0:
        volume_ax.set_ylim(0, max_volume * 4)
    else:
        # Use a small non-zero range when all volumes are zero
        volume_ax.set_ylim(0, 1.0)
    volume_ax.grid(False)

    # Format axes
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Price')
    volume_ax.set_ylabel('Volume')

    # Set title if provided
    if title:
        ax.set_title(title)

    # Set x-axis ticks - ensure we have a reasonable number of ticks
    if len(x) > 1:
        tick_step = max(1, len(x)//10)  # Show ~10 ticks
        tick_indices = x[::tick_step]
        ax.set_xticks(tick_indices)

        # Handle different index types safely
        if hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_dtype(df.index):
            # If index is datetime
            tick_labels = [df.index[i].strftime('%Y-%m-%d %H:%M') for i in range(0, len(df), tick_step) if i < len(df)]
        elif any(hasattr(idx, 'strftime') for idx in df.index):
            # If index contains datetime-like objects
            tick_labels = [df.index[i].strftime('%Y-%m-%d %H:%M') if hasattr(df.index[i], 'strftime') else str(df.index[i])
                          for i in range(0, len(df), tick_step) if i < len(df)]
        else:
            # Otherwise use string representation
            tick_labels = [str(df.index[i]) for i in range(0, len(df), tick_step) if i < len(df)]

        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    # Adjust layout
    fig.tight_layout()

    # Convert to base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    # Close the figure to free memory
    plt.close(fig)

    return img_str

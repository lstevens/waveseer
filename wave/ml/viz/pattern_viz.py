"""
Pattern visualization module for ML-detected patterns.

This module provides functions to visualize ML-detected patterns on candlestick charts,
including confidence indicators, overlays, and attention maps for transformer models.
"""

import io
import base64
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches

from wave.patterns import PatternType, PatternMatch


class PatternVisualizer:
    """Class for visualizing pattern detections on candlestick charts."""

    def __init__(self,
                 pattern_colors: Optional[Dict[PatternType, str]] = None,
                 default_alpha: float = 0.4,
                 confidence_thresholds: Optional[Dict[str, float]] = None):
        """Initialize the pattern visualizer.

        Args:
            pattern_colors: Optional dictionary mapping pattern types to colors
            default_alpha: Default transparency for pattern overlays
            confidence_thresholds: Thresholds for confidence levels (low, medium, high)
        """
        # Default pattern colors based on pattern types
        self.pattern_colors = {
            PatternType.HEAD_AND_SHOULDERS: "#FF5733",  # Red-orange
            PatternType.INVERSE_HEAD_AND_SHOULDERS: "#33FF57",  # Green
            PatternType.DOUBLE_TOP: "#FF3333",  # Red
            PatternType.DOUBLE_BOTTOM: "#33FF33",  # Green
            PatternType.TRIPLE_TOP: "#FF3380",  # Pink
            PatternType.TRIPLE_BOTTOM: "#33FF80",  # Light green
            PatternType.ASCENDING_TRIANGLE: "#FF8033",  # Orange
            PatternType.DESCENDING_TRIANGLE: "#3380FF",  # Blue
            PatternType.SYMMETRICAL_TRIANGLE: "#8033FF",  # Purple
            PatternType.RISING_WEDGE: "#FF33D4",  # Magenta
            PatternType.FALLING_WEDGE: "#33D4FF",  # Cyan
            PatternType.RECTANGLE: "#D4FF33",  # Yellow-green
            PatternType.CUP_AND_HANDLE: "#33D4D4",  # Teal
            PatternType.INVERSE_CUP_AND_HANDLE: "#D433D4",  # Purple-pink
            PatternType.BULL_FLAG: "#33FF33",  # Green
            PatternType.BEAR_FLAG: "#FF3333",  # Red
        }

        # Override with custom colors if provided
        if pattern_colors:
            for pattern_type, color in pattern_colors.items():
                self.pattern_colors[pattern_type] = color

        self.default_alpha = default_alpha

        # Confidence thresholds for visualization
        self.confidence_thresholds = {
            "low": 0.4,    # Below this is low confidence
            "medium": 0.7  # Below this is medium, above is high
        }
        if confidence_thresholds:
            self.confidence_thresholds.update(confidence_thresholds)

    def _get_pattern_color(self, pattern_type: PatternType) -> str:
        """Get the color for a given pattern type."""
        return self.pattern_colors.get(pattern_type, "#888888")  # Default to gray if not found

    def overlay_pattern(self,
                       ax: Axes,
                       pattern: PatternMatch,
                       df: pd.DataFrame,
                       show_confidence: bool = True) -> Axes:
        """Overlay a pattern on an existing chart.

        Args:
            ax: Matplotlib axis to draw on
            pattern: PatternMatch object with detection information
            df: DataFrame with OHLCV data
            show_confidence: Whether to show confidence indicators

        Returns:
            The updated matplotlib axis
        """
        # Extract pattern information
        start_idx = pattern.start_idx
        end_idx = pattern.end_idx

        # Ensure indices are within dataframe bounds
        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(df):
            end_idx = len(df) - 1

        # Get pattern range in data coordinates
        x_start, x_end = start_idx, end_idx

        # Get y range with some padding
        pattern_prices = df.iloc[start_idx:end_idx+1]
        y_min = pattern_prices['low'].min() * 0.99
        y_max = pattern_prices['high'].max() * 1.01

        # Get color with alpha based on confidence
        base_color = self._get_pattern_color(pattern.pattern_type)

        # Create pattern highlight rectangle
        conf_style = create_confidence_indicator(pattern.score)
        rect_alpha = conf_style.get("alpha", self.default_alpha)

        # Create rectangle patch for the pattern region
        rect = patches.Rectangle(
            (x_start, y_min),
            width=(x_end - x_start),
            height=(y_max - y_min),
            linewidth=conf_style.get("linewidth", 1.5),
            edgecolor=base_color,
            facecolor=base_color,
            alpha=rect_alpha,
            zorder=1  # Ensure it's below the candlesticks
        )
        ax.add_patch(rect)

        # Add pattern label
        pattern_name = pattern.pattern_type.value.replace("_", " ").title()
        confidence = f"{pattern.score:.0%}"

        # Position label at the top of the pattern
        text_x = (x_start + x_end) / 2
        text_y = y_max * 1.01

        label = f"{pattern_name}\n{confidence}"
        ax.text(text_x, text_y, label,
                fontsize=9,
                ha='center',
                va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=base_color, boxstyle='round,pad=0.3'))

        return ax

    def draw_patterns_on_chart(self,
                              df: pd.DataFrame,
                              patterns: Dict[PatternType, List[PatternMatch]],
                              title: str = "Price Chart with Detected Patterns",
                              figsize: Tuple[int, int] = (12, 8)) -> str:
        """Draw candlestick chart with pattern overlays.

        Args:
            df: DataFrame with OHLCV data
            patterns: Dictionary of pattern matches from detect_patterns()
            title: Chart title
            figsize: Figure size as (width, height) tuple

        Returns:
            Base64 encoded PNG image
        """
        # Create figure
        fig = Figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # Create candlestick chart
        x = np.arange(len(df))
        width = 0.6

        # Plot candlesticks
        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]

        # Handle different index types (numeric, datetime, etc.)
        if hasattr(up, 'index') and not isinstance(up.index, pd.RangeIndex):
            up_idx = df.index.get_indexer(up.index)
            down_idx = df.index.get_indexer(down.index)
        else:
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

        # Set y-limits for volume
        max_volume = df['volume'].max()
        if max_volume > 0:
            volume_ax.set_ylim(0, max_volume * 4)
        else:
            volume_ax.set_ylim(0, 1.0)
        volume_ax.grid(False)

        # Format axes
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Price')
        volume_ax.set_ylabel('Volume')

        # Set title
        ax.set_title(title)

        # Set x-axis ticks
        if len(x) > 1:
            tick_step = max(1, len(x) // 10)  # Show ~10 ticks
            tick_indices = x[::tick_step]
            ax.set_xticks(tick_indices)

            # Handle different index types
            if hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_dtype(df.index):
                tick_labels = [df.index[i].strftime('%Y-%m-%d %H:%M')
                              for i in range(0, len(df), tick_step) if i < len(df)]
            else:
                tick_labels = [str(df.index[i])
                              for i in range(0, len(df), tick_step) if i < len(df)]

            ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        # Overlay patterns
        count = 0
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                self.overlay_pattern(ax, pattern, df)
                count += 1

        # Add pattern count as subtitle if any patterns found
        if count > 0:
            ax.set_title(f"{title}\n{count} patterns detected", fontsize=12)

        # Adjust layout for better spacing
        fig.tight_layout()

        # Convert to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        # Close the figure to free memory
        plt.close(fig)

        return img_str

    def visualize_attention_map(self,
                               df: pd.DataFrame,
                               attention_weights: np.ndarray,
                               head_idx: int = 0,
                               layer_idx: int = 0,
                               title: str = "Transformer Attention Map",
                               figsize: Tuple[int, int] = (12, 10)) -> str:
        """Visualize attention weights from transformer model.

        Args:
            df: DataFrame with OHLCV data
            attention_weights: Attention weights array [batch, heads, seq_len, seq_len]
                or [heads, seq_len, seq_len] or [seq_len, seq_len]
            head_idx: Attention head to visualize
            layer_idx: Transformer layer to visualize
            title: Figure title
            figsize: Figure size as (width, height) tuple

        Returns:
            Base64 encoded PNG image
        """
        # Handle different dimensions for attention weights
        if attention_weights.ndim == 4:  # [batch, heads, seq_len, seq_len]
            weights = attention_weights[0, head_idx]
        elif attention_weights.ndim == 3:  # [heads, seq_len, seq_len]
            weights = attention_weights[head_idx]
        else:  # [seq_len, seq_len]
            weights = attention_weights

        # Create figure with 2 subplots: price chart and attention heatmap
        fig = Figure(figsize=figsize)

        # Price chart with price line
        ax1 = fig.add_subplot(211)
        ax1.plot(df['close'], color='black', alpha=0.7)
        ax1.set_title(f"Price Chart")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)

        # Set x-axis ticks
        if len(df) > 1:
            tick_step = max(1, len(df) // 10)  # Show ~10 ticks
            tick_indices = np.arange(len(df))[::tick_step]
            ax1.set_xticks(tick_indices)

            # Handle different index types
            if hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_dtype(df.index):
                tick_labels = [df.index[i].strftime('%Y-%m-%d %H:%M')
                              for i in range(0, len(df), tick_step) if i < len(df)]
            else:
                tick_labels = [str(df.index[i])
                              for i in range(0, len(df), tick_step) if i < len(df)]

            ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

        # Attention heatmap
        ax2 = fig.add_subplot(212)

        # If attention map is too large, downsample for visualization
        max_display_size = 100
        if weights.shape[0] > max_display_size:
            downsample_factor = weights.shape[0] // max_display_size + 1
            weights_display = weights[::downsample_factor, ::downsample_factor]
        else:
            weights_display = weights

        im = ax2.imshow(weights_display, cmap='viridis')
        ax2.set_title(f"Attention Map (Layer {layer_idx}, Head {head_idx})")
        ax2.set_xlabel("Token Position (Time)")
        ax2.set_ylabel("Token Position (Time)")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax2)
        cbar.set_label('Attention Weight')

        # Overall title
        fig.suptitle(title, fontsize=14)

        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

        # Convert to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')

        # Close the figure to free memory
        plt.close(fig)

        return img_str


# Standalone functions that delegate to PatternVisualizer for easier use
def overlay_pattern(ax: Axes,
                   pattern: PatternMatch,
                   df: pd.DataFrame) -> Axes:
    """Overlay a pattern on an existing chart (standalone function)."""
    visualizer = PatternVisualizer()
    return visualizer.overlay_pattern(ax, pattern, df)


def draw_patterns_on_chart(df: pd.DataFrame,
                          patterns: Dict[PatternType, List[PatternMatch]],
                          title: str = "Price Chart with Detected Patterns",
                          figsize: Tuple[int, int] = (12, 8)) -> str:
    """Draw candlestick chart with pattern overlays (standalone function)."""
    visualizer = PatternVisualizer()
    return visualizer.draw_patterns_on_chart(df, patterns, title, figsize)


def create_confidence_indicator(confidence: float) -> Dict[str, Union[float, str]]:
    """Create visual style settings based on confidence score.

    Args:
        confidence: Confidence score (0-1)

    Returns:
        Dictionary with visual properties (alpha, linewidth, etc.)
    """
    # Set alpha based on confidence
    alpha = min(0.2 + confidence * 0.6, 0.8)

    # Set linewidth based on confidence
    linewidth = 1.0 + confidence * 2.0  # 1.0 to 3.0

    return {
        "alpha": alpha,
        "linewidth": linewidth,
    }


def visualize_attention_map(df: pd.DataFrame,
                           attention_weights: np.ndarray,
                           head_idx: int = 0,
                           layer_idx: int = 0,
                           title: str = "Transformer Attention Map",
                           figsize: Tuple[int, int] = (12, 10)) -> str:
    """Visualize attention weights from transformer model (standalone function)."""
    visualizer = PatternVisualizer()
    return visualizer.visualize_attention_map(
        df, attention_weights, head_idx, layer_idx, title, figsize)

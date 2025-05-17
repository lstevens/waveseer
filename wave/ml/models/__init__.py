"""
PyTorch model architectures for pattern detection.

This module contains model architectures for pattern detection in financial time series:
- CNNPatternModel: Convolutional Neural Network for patterns
- LSTMPatternModel: Long Short-Term Memory network for sequence patterns
- HybridPatternModel: Combined CNN+LSTM architecture
- TransformerPatternModel: Self-attention based architecture for complex patterns
"""

from wave.ml.models.base import PatternModel
from wave.ml.models.cnn import CNNPatternModel
from wave.ml.models.lstm import LSTMPatternModel
from wave.ml.models.hybrid import HybridPatternModel
from wave.ml.models.transformer import TransformerPatternModel

__all__ = [
    "PatternModel",
    "CNNPatternModel",
    "LSTMPatternModel",
    "HybridPatternModel",
    "TransformerPatternModel"
]

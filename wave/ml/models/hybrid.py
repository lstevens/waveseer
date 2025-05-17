"""
Hybrid CNN-LSTM model for pattern detection.

This module implements a hybrid architecture combining CNN for feature extraction
with LSTM for temporal pattern analysis, ideal for chart pattern detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave.ml.models.base import PatternModel


class HybridPatternModel(PatternModel):
    """Hybrid CNN-LSTM model for pattern detection."""

    model_type = "hybrid"

    def __init__(
        self,
        input_size: int = 20,
        n_features: int = 5,
        n_classes: int = 10,
        hidden_size: int = 64,
        dropout: float = 0.2,
        cnn_filters: list = None,
        cnn_kernel_sizes: list = None,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        **kwargs
    ):
        """
        Initialize hybrid CNN-LSTM model.

        Args:
            input_size: Length of input sequence
            n_features: Number of input features
            n_classes: Number of pattern classes
            hidden_size: Size of hidden layers
            dropout: Dropout probability
            cnn_filters: List of filter counts for CNN layers
            cnn_kernel_sizes: List of kernel sizes for CNN layers
            lstm_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        # Handle aliases and backward compatibility
        if 'kernel_sizes' in kwargs:
            cnn_kernel_sizes = kwargs.pop('kernel_sizes')  # type: ignore
        if 'channels' in kwargs:
            cnn_filters = kwargs.pop('channels')  # type: ignore
        if 'num_layers' in kwargs:
            lstm_layers = kwargs.pop('num_layers')  # type: ignore
        # Support lstm_hidden_size alias for hidden_size
        if 'lstm_hidden_size' in kwargs:
            hidden_size = kwargs.pop('lstm_hidden_size')  # type: ignore
        # Pop unused aliases
        kwargs.pop('cnn_hidden_size', None)
        super().__init__(
            input_size=input_size,
            n_features=n_features,
            n_classes=n_classes,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # Default CNN parameters if not provided
        self.cnn_filters = cnn_filters or [32, 64, 64]
        self.cnn_kernel_sizes = cnn_kernel_sizes or [3, 5, 3]
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        # Update config
        self.config.update({
            "cnn_filters": self.cnn_filters,
            "cnn_kernel_sizes": self.cnn_kernel_sizes,
            "lstm_layers": lstm_layers,
            "bidirectional": bidirectional
        })

        # CNN feature extraction layers
        self.conv_layers = nn.ModuleList()

        # Input is in shape [batch, seq, features], will be transposed for CNN
        in_channels = n_features

        for i, (n_filt, kernel_size) in enumerate(zip(self.cnn_filters, self.cnn_kernel_sizes)):
            # Padding to keep sequence length
            padding = kernel_size // 2

            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else self.cnn_filters[i-1],
                    out_channels=n_filt,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )

        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_filters[-1],  # Input from last CNN layer
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Attention mechanism
        self.attention = nn.Linear(lstm_output_size, 1)

        # Final classification layers
        self.fc = nn.Linear(lstm_output_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Output tensor of shape [batch_size, n_classes]
        """
        batch_size, seq_len, n_features = x.shape

        # Transpose to [batch, features, seq] for 1D CNN
        x = x.transpose(1, 2)

        # CNN feature extraction
        for conv in self.conv_layers:
            x = F.relu(conv(x))

        # Transpose back to [batch, seq, features] for LSTM
        x = x.transpose(1, 2)

        # LSTM sequence modeling
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        attention_weights = attention_weights.unsqueeze(2)

        # Context vector as weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attention_weights, dim=1)

        # Final classification
        x = self.dropout(context)
        x = self.fc(x)

        return x

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get CNN features for visualization.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            CNN features of shape [batch_size, seq_len, n_filters[-1]]
        """
        # Transpose to [batch, features, seq] for 1D CNN
        x = x.transpose(1, 2)

        # CNN feature extraction
        for conv in self.conv_layers:
            x = F.relu(conv(x))

        # Transpose back to [batch, seq, features]
        return x.transpose(1, 2)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights over the sequence.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Attention weights of shape [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            # CNN feature extraction
            cnn_features = self.get_cnn_features(x)

            # LSTM processing
            lstm_out, _ = self.lstm(cnn_features)

            # Attention weights
            return F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)

    @classmethod
    def create_default(cls, n_classes=None) -> "HybridPatternModel":
        """Create a default hybrid model with recommended parameters."""
        return cls(
            input_size=40,
            n_features=5,
            n_classes=n_classes or 10,
            hidden_size=128,
            dropout=0.3,
            cnn_filters=[32, 64, 128],
            cnn_kernel_sizes=[3, 5, 3],
            lstm_layers=2,
            bidirectional=True
        )

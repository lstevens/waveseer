"""
LSTM model for pattern detection.

This module implements a Long Short-Term Memory (LSTM) network
for detecting patterns in financial time series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave.ml.models.base import PatternModel


class LSTMPatternModel(PatternModel):
    """LSTM Network for sequential pattern detection."""

    model_type = "lstm"

    def __init__(
        self,
        input_size: int = 20,
        n_features: int = 5,
        n_classes: int = 10,
        hidden_size: int = 64,
        dropout: float = 0.2,
        n_layers: int = 2,
        bidirectional: bool = True,
        **kwargs
    ):
        """
        Initialize LSTM model for pattern detection.

        Args:
            input_size: Length of input sequence
            n_features: Number of input features
            n_classes: Number of pattern classes
            hidden_size: Size of LSTM hidden state
            dropout: Dropout probability
            n_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        # Alias for num_layers
        if 'num_layers' in kwargs:
            n_layers = kwargs.pop('num_layers')  # type: ignore
        super().__init__(
            input_size=input_size,
            n_features=n_features,
            n_classes=n_classes,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # LSTM-specific parameters
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # Update config with LSTM-specific parameters
        self.config.update({
            "n_layers": n_layers,
            "bidirectional": bidirectional
        })

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate output size of LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Attention layer
        self.attention = nn.Linear(lstm_output_size, 1)

        # Output layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Output tensor of shape [batch_size, n_classes]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention to focus on important parts of the sequence
        # Shape of lstm_out: [batch_size, seq_len, hidden_size * (2 if bidirectional)]
        attention_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        attention_weights = attention_weights.unsqueeze(2)

        # Weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attention_weights, dim=1)

        # Final classification
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Attention weights of shape [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            # LSTM forward
            lstm_out, _ = self.lstm(x)

            # Get attention weights
            weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)

            return weights

    @classmethod
    def create_default(cls, n_classes=None) -> "LSTMPatternModel":
        """Create a default LSTM model with recommended parameters."""
        return cls(
            input_size=40,
            n_features=5,
            n_classes=n_classes or 10,
            hidden_size=128,
            dropout=0.3,
            n_layers=2,
            bidirectional=True
        )

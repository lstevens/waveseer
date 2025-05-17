"""
Transformer model for pattern detection.

This module implements a Transformer-based architecture using self-attention
mechanisms for detecting complex patterns in financial time series.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from wave.ml.models.base import PatternModel


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register as buffer (not model parameter but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPatternModel(PatternModel):
    """Transformer model for pattern detection."""

    model_type = "transformer"

    def __init__(
        self,
        input_size: int = 20,
        n_features: int = 5,
        n_classes: int = 10,
        hidden_size: int = 64,
        dropout: float = 0.2,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        **kwargs
    ):
        """
        Initialize Transformer model.

        Args:
            input_size: Length of input sequence
            n_features: Number of input features
            n_classes: Number of pattern classes
            hidden_size: Size of hidden layers (d_model)
            dropout: Dropout probability
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Dimension of feedforward network
        """
        # Alias and backward compatibility
        if 'num_layers' in kwargs:
            n_layers = kwargs.pop('num_layers')  # type: ignore
        if 'dim_feedforward' in kwargs:
            d_ff = kwargs.pop('dim_feedforward')  # type: ignore
        super().__init__(
            input_size=input_size,
            n_features=n_features,
            n_classes=n_classes,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # Transformer-specific parameters
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        # Expose alias for exporter
        self.dim_feedforward = d_ff

        # Update config
        self.config.update({
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            # include alias
            "dim_feedforward": d_ff
        })

        # Feature dimension linear projection
        self.feature_proj = nn.Linear(n_features, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=input_size, dropout=dropout)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output layers
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Output tensor of shape [batch_size, n_classes]
        """
        # Project features to hidden dimension
        x = self.feature_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Global pooling (mean of sequence)
        x = torch.mean(x, dim=1)

        # Classification
        x = self.fc(x)

        return x

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps from all layers.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            List of attention maps, each with shape [batch_size, n_heads, seq_len, seq_len]
        """
        self.eval()
        attention_maps = []

        # Forward through each component
        x = self.feature_proj(x)
        x = self.pos_encoder(x)

        # Hook to capture attention weights
        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                # Extract attention weights from multihead attention
                # This needs to be adapted to the specific PyTorch version and implementation
                # as the internal structure might vary
                attention_maps.append(output[1])  # Assuming attention weights are returned
            return hook

        # Register hooks for each layer
        hooks = []
        for i, layer in enumerate(self.transformer_encoder.layers):
            hook = layer.self_attn.register_forward_hook(get_attention_hook(i))
            hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            self.transformer_encoder(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return attention_maps

    @classmethod
    def create_default(cls, n_classes=None) -> "TransformerPatternModel":
        """Create a default Transformer model with recommended parameters."""
        return cls(
            input_size=40,
            n_features=5,
            n_classes=n_classes or 10,
            hidden_size=128,
            dropout=0.1,
            n_heads=8,
            n_layers=4,
            d_ff=512
        )

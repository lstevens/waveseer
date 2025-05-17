"""
CNN model for pattern detection.

This module implements a Convolutional Neural Network (CNN) architecture
for detecting patterns in financial time series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave.ml.models.base import PatternModel


class CNNPatternModel(PatternModel):
    """Convolutional Neural Network for pattern detection."""

    model_type = "cnn"

    def __init__(
        self,
        input_size: int = 20,
        n_features: int = 5,
        n_classes: int = 10,
        hidden_size: int = 64,
        kernel_sizes: list = None,
        channels: list = None,
        dropout: float = 0.2,
        n_filters: list = None
    ):
        """
        Initialize CNN model for pattern detection.

        Args:
            input_size: Length of input sequence
            n_features: Number of input features
            n_classes: Number of pattern classes
            hidden_size: Size of hidden layers
            kernel_sizes: List of kernel sizes for each conv layer
            channels: List of filter counts for each conv layer (alias for n_filters)
            dropout: Dropout probability
            n_filters: List of filter counts for each conv layer
        """
        super().__init__(
            input_size=input_size,
            n_features=n_features,
            n_classes=n_classes,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # Default kernel sizes and filter counts if not provided
        self.kernel_sizes = kernel_sizes or [3, 5, 7]
        # Support 'channels' alias for n_filters
        self.n_filters = channels or n_filters or [32, 64, 128]

        # Update config with CNN-specific parameters
        self.config.update({
            "kernel_sizes": self.kernel_sizes,
            "n_filters": self.n_filters
        })

        # Build CNN layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_channels = n_features
        seq_length = input_size

        # Create multiple convolutional blocks
        for i, (kernel_size, n_filter) in enumerate(zip(self.kernel_sizes, self.n_filters)):
            # Add padding to maintain sequence length
            padding = kernel_size // 2

            # Add convolutional layer
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else self.n_filters[i-1],
                    out_channels=n_filter,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )

            # Add pooling layer (reducing sequence length by half)
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            seq_length = seq_length // 2

        # Calculate flattened size after conv layers
        self.flattened_size = seq_length * self.n_filters[-1]

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Output tensor of shape [batch_size, n_classes]
        """
        # Transpose to [batch_size, n_features, seq_len] for 1D CNN
        x = x.transpose(1, 2)

        # Apply convolutional blocks
        for i, (conv, pool) in enumerate(zip(self.conv_layers, self.pool_layers)):
            x = F.relu(conv(x))
            x = pool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_feature_maps(self, x: torch.Tensor) -> list:
        """
        Extract feature maps from intermediate CNN layers.
        Useful for visualization and understanding what the model learns.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            List of intermediate feature maps
        """
        # Transpose to [batch_size, n_features, seq_len] for 1D CNN
        x = x.transpose(1, 2)

        feature_maps = []

        # Apply convolutional blocks and save intermediate outputs
        for i, (conv, pool) in enumerate(zip(self.conv_layers, self.pool_layers)):
            x = F.relu(conv(x))
            feature_maps.append(x.clone())
            x = pool(x)

        return feature_maps

    @classmethod
    def create_default(cls, n_classes=None) -> "CNNPatternModel":
        """Create a default CNN model with recommended parameters."""
        return cls(
            input_size=40,
            n_features=5,
            n_classes=n_classes or 10,
            hidden_size=128,
            kernel_sizes=[3, 5, 7, 9],
            channels=[32, 64, 128, 256],
            dropout=0.3
        )

    def to_torchscript(self, path=None, method: str = "trace", optimize: bool = True):
        """
        Export or return a TorchScript model.
        If `path` is provided, export to file and return the file path.
        Otherwise, return the scripted/traced model instance.
        """
        import torch
        from wave.ml.export.model_export import export_to_torchscript
        # Prepare example input matching expected shape [batch, seq_len, features]
        example_input = torch.randn(1, self.input_size, self.n_features)
        # Export to path
        if path:
            return export_to_torchscript(self, example_input, path, method=method, optimize=optimize)
        # Return in-memory TorchScript model
        self.eval()
        with torch.no_grad():
            if method == "trace":
                ts_model = torch.jit.trace(self, example_input)
            else:
                ts_model = torch.jit.script(self)
            if optimize:
                ts_model = torch.jit.optimize_for_inference(ts_model)
        return ts_model

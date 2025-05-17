"""
Base pattern model class for PyTorch pattern detection.

This module defines the base PatternModel class that all pattern models inherit from,
ensuring consistent interfaces and functionality across model architectures.
"""

from typing import List, Tuple, ClassVar
import os
import json
import torch
import torch.nn as nn

from wave.patterns import PatternType


class PatternModel(nn.Module):
    """Base class for all pattern detection models."""

    model_type: ClassVar[str] = "base"

    def __init__(
        self,
        input_size: int = 20,
        n_features: int = 5,
        n_classes: int = len(PatternType),
        hidden_size: int = 64,
        dropout: float = 0.2
    ):
        """
        Initialize base pattern model.

        Args:
            input_size: Sequence length of input pattern
            n_features: Number of input features
            n_classes: Number of pattern classes
            hidden_size: Size of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        self.input_size = input_size
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.dropout = dropout

        # Config dictionary for serialization
        self.config = {
            "model_type": self.model_type,
            "input_size": input_size,
            "n_features": n_features,
            "n_classes": n_classes,
            "hidden_size": hidden_size,
            "dropout": dropout
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Output tensor of shape [batch_size, n_classes]
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with class and confidence.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Tuple of (predicted class indices, class probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            return pred_class, probs

    def predict_pattern_type(self, x: torch.Tensor) -> Tuple[List[PatternType], torch.Tensor]:
        """
        Predict pattern types from input sequence.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]

        Returns:
            Tuple of (list of pattern types, class probabilities)
        """
        pred_class, probs = self.predict(x)

        # Convert class indices to PatternType
        pattern_types = []
        pattern_type_list = list(PatternType)

        for cls_idx in pred_class:
            if 0 <= cls_idx < len(pattern_type_list):
                pattern_types.append(pattern_type_list[cls_idx])
            else:
                pattern_types.append(PatternType.UNKNOWN)

        return pattern_types, probs

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model and config
        state_dict = self.state_dict()
        save_dict = {
            "config": self.config,
            "state_dict": state_dict
        }
        torch.save(save_dict, path)

        # Save config as JSON for easy inspection
        config_path = path + ".json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def to_torchscript(self, path: str = None) -> torch.jit.ScriptModule:
        """
        Convert model to TorchScript for deployment.

        Args:
            path: Optional path to save the script module

        Returns:
            Scripted model
        """
        self.eval()

        # Example input for tracing
        example_input = torch.randn(1, self.input_size, self.n_features)

        # Script the model
        scripted_model = torch.jit.trace(self, example_input)

        # Save if path is provided
        if path:
            scripted_model.save(path)

        return scripted_model

    @classmethod
    def load(cls, path: str) -> "PatternModel":
        """
        Load model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model
        """
        # Load saved dictionary
        save_dict = torch.load(path, map_location=torch.device("cpu"))

        # Get config
        config = save_dict["config"]

        # Choose correct model class based on model_type
        model_cls = {
            "cnn": globals().get("CNNPatternModel", cls),
            "lstm": globals().get("LSTMPatternModel", cls),
            "hybrid": globals().get("HybridPatternModel", cls),
            "transformer": globals().get("TransformerPatternModel", cls),
            "base": cls
        }.get(config.get("model_type", "base"), cls)

        # Create instance with saved config
        model_args = {k: v for k, v in config.items() if k not in ["model_type"]}
        model = model_cls(**model_args)

        # Load state dictionary
        model.load_state_dict(save_dict["state_dict"])

        return model

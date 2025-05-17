"""
TensorBoard visualization utilities for pattern detection models.

This module provides helper functions for integrating TensorBoard visualizations
with the pattern detection models, including confusion matrices, model graphs,
feature importance, and prediction visualizations.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from sklearn.metrics import confusion_matrix


def setup_tensorboard(model_name: str, base_dir: str = "runs") -> SummaryWriter:
    """Set up a TensorBoard writer with a unique run ID.

    Args:
        model_name: Name of the model or experiment
        base_dir: Base directory for TensorBoard logs

    Returns:
        TensorBoard SummaryWriter instance
    """
    # Create unique run ID with timestamp
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir) / f"{model_name}_{run_id}"

    # Create SummaryWriter
    writer = SummaryWriter(log_dir=str(log_dir))

    return writer


def create_confusion_matrix_figure(
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues"
) -> plt.Figure:
    """Create a matplotlib figure with a confusion matrix visualization.

    Args:
        conf_matrix: Confusion matrix as a numpy array
        class_names: List of class names for axis labels
        title: Figure title
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size as (width, height)
        cmap: Colormap for the visualization

    Returns:
        Matplotlib Figure object
    """
    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(conf_matrix.shape[0])]

    # Normalize the confusion matrix if requested
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar=True, ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    fig.tight_layout()

    return fig


def log_confusion_matrix(
    writer: SummaryWriter,
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    step: int = 0,
    tag: str = "confusion_matrix",
    normalize: bool = False
) -> None:
    """Log a confusion matrix to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        conf_matrix: Confusion matrix as a numpy array
        class_names: List of class names for axis labels
        step: Global step value to log
        tag: Tag name for the visualization
        normalize: Whether to normalize the confusion matrix
    """
    fig = create_confusion_matrix_figure(
        conf_matrix=conf_matrix,
        class_names=class_names,
        normalize=normalize
    )

    # Log the figure to TensorBoard
    writer.add_figure(tag, fig, global_step=step)


def log_model_graph(
    writer: SummaryWriter,
    model: nn.Module,
    input_tensor: torch.Tensor
) -> None:
    """Log a model graph to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        model: PyTorch model
        input_tensor: Example input tensor for the model
    """
    # Make sure the model is in eval mode
    model.eval()

    # Add graph to TensorBoard
    writer.add_graph(model, input_tensor)


def log_feature_importance(
    writer: SummaryWriter,
    importance_scores: np.ndarray,
    feature_names: List[str],
    step: int = 0,
    tag: str = "feature_importance"
) -> None:
    """Log feature importance scores to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        importance_scores: Array of importance scores
        feature_names: List of feature names
        step: Global step value to log
        tag: Tag name for the visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by importance
    indices = np.argsort(importance_scores)
    sorted_scores = importance_scores[indices]
    sorted_names = [feature_names[i] for i in indices]

    # Create horizontal bar chart
    ax.barh(sorted_names, sorted_scores)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    fig.tight_layout()

    # Log to TensorBoard
    writer.add_figure(tag, fig, global_step=step)


def log_predictions_vs_actual(
    writer: SummaryWriter,
    predictions: np.ndarray,
    actual: np.ndarray,
    class_names: List[str],
    step: int = 0,
    tag: str = "predictions_vs_actual",
    sample_limit: int = 100
) -> None:
    """Log a visualization of predictions vs actual values.

    Args:
        writer: TensorBoard SummaryWriter instance
        predictions: Array of predicted class indices
        actual: Array of actual class indices
        class_names: List of class names
        step: Global step value to log
        tag: Tag name for the visualization
        sample_limit: Maximum number of samples to visualize
    """
    # Limit the number of samples if needed
    if len(predictions) > sample_limit:
        indices = np.random.choice(len(predictions), sample_limit, replace=False)
        predictions = predictions[indices]
        actual = actual[indices]

    # Create confusion matrix
    conf_mat = confusion_matrix(actual, predictions,
                                labels=range(len(class_names)))

    # Log confusion matrix
    log_confusion_matrix(
        writer=writer,
        conf_matrix=conf_mat,
        class_names=class_names,
        step=step,
        tag=f"{tag}/confusion_matrix"
    )

    # Create comparison figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sample indices for x-axis
    x = np.arange(len(predictions))

    # Plot actual and predicted
    ax.plot(x, actual, 'o-', label='Actual', alpha=0.7)
    ax.plot(x, predictions, 'x-', label='Predicted', alpha=0.7)

    # Customize plot
    ax.set_ylabel('Class')
    ax.set_xlabel('Sample Index')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.legend()
    ax.set_title('Predictions vs Actual Values')
    ax.grid(alpha=0.3)

    # Log figure to TensorBoard
    writer.add_figure(f"{tag}/comparison", fig, global_step=step)

    # Calculate and log accuracy
    accuracy = np.mean(predictions == actual)
    writer.add_scalar(f"{tag}/accuracy", accuracy, global_step=step)


def generate_tensor_histogram(
    writer: SummaryWriter,
    tensor: torch.Tensor,
    tag: str,
    step: int = 0,
    bins: str = 'auto'
) -> None:
    """Generate and log histogram of tensor values.

    Args:
        writer: TensorBoard SummaryWriter instance
        tensor: Tensor to visualize
        tag: Tag name for the histogram
        step: Global step value to log
        bins: Number of bins or bin strategy
    """
    # Log the histogram to TensorBoard
    writer.add_histogram(tag, tensor, global_step=step, bins=bins)

    # Also add as a matplotlib figure for more control
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert tensor to numpy for matplotlib
    values = tensor.detach().cpu().numpy().flatten()

    # Plot histogram
    ax.hist(values, bins=bins, alpha=0.7)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {tag}')
    ax.grid(alpha=0.3)

    # Log figure to TensorBoard
    writer.add_figure(f"{tag}_hist", fig, global_step=step)


def log_model_weights_and_gradients(
    writer: SummaryWriter,
    model: nn.Module,
    step: int = 0
) -> None:
    """Log model weights and gradients to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        model: PyTorch model
        step: Global step value to log
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Log weights
            writer.add_histogram(f"weights/{name}", param.data, global_step=step)

            # Log gradients if they exist
            if param.grad is not None:
                writer.add_histogram(f"gradients/{name}", param.grad.data, global_step=step)


def log_learning_curve(
    writer: SummaryWriter,
    train_losses: List[float],
    val_losses: List[float],
    metric_name: str = "loss",
    step: int = 0
) -> None:
    """Log learning curves to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        train_losses: List of training losses
        val_losses: List of validation losses
        metric_name: Name of the metric being logged
        step: Global step value to log
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create x-axis (epochs)
    epochs = range(1, len(train_losses) + 1)

    # Plot training and validation losses
    ax.plot(epochs, train_losses, 'b-', label=f'Training {metric_name}')
    ax.plot(epochs, val_losses, 'r-', label=f'Validation {metric_name}')

    # Customize plot
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f'Training and Validation {metric_name.capitalize()}')
    ax.legend()
    ax.grid(alpha=0.3)

    # Log figure to TensorBoard
    writer.add_figure(f"learning_curve/{metric_name}", fig, global_step=step)


def log_attention_maps(
    writer: SummaryWriter,
    attention_weights: torch.Tensor,
    step: int = 0,
    max_heads: int = 4,
    tag: str = "attention_maps"
) -> None:
    """Log attention maps from transformer models.

    Args:
        writer: TensorBoard SummaryWriter instance
        attention_weights: Tensor of attention weights [heads, seq_len, seq_len]
        step: Global step value to log
        max_heads: Maximum number of attention heads to visualize
        tag: Tag name for the visualization
    """
    # Handle different dimensions
    if attention_weights.ndim == 4:  # [batch, heads, seq_len, seq_len]
        attention_weights = attention_weights[0]  # Take first batch

    # Ensure tensor is on CPU and convert to numpy
    attention_weights = attention_weights.detach().cpu().numpy()

    # Limit number of heads to visualize
    num_heads = min(attention_weights.shape[0], max_heads)

    # Create a figure for each attention head
    for h in range(num_heads):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot attention weights as heatmap
        im = ax.imshow(attention_weights[h], cmap='viridis')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')

        # Set labels
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Attention Map - Head {h+1}')

        # Log figure to TensorBoard
        writer.add_figure(f"{tag}/head_{h+1}", fig, global_step=step)


def log_pattern_distribution(
    writer: SummaryWriter,
    pattern_counts: Dict[str, int],
    step: int = 0,
    tag: str = "pattern_distribution"
) -> None:
    """Log distribution of detected patterns.

    Args:
        writer: TensorBoard SummaryWriter instance
        pattern_counts: Dictionary mapping pattern types to counts
        step: Global step value to log
        tag: Tag name for the visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort patterns by count
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    patterns = [p[0] for p in sorted_patterns]
    counts = [p[1] for p in sorted_patterns]

    # Create bar chart
    ax.bar(patterns, counts)

    # Customize plot
    ax.set_xlabel('Pattern Type')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Detected Patterns')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()

    # Log figure to TensorBoard
    writer.add_figure(tag, fig, global_step=step)

    # Also log as scalars for time series tracking
    for pattern, count in pattern_counts.items():
        writer.add_scalar(f"{tag}/{pattern}", count, global_step=step)

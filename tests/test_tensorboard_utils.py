"""
Tests for TensorBoard utilities.
"""
import os
import shutil
import tempfile
import pytest

# Setup ML mocks if in testing mode
import os
from wave.test_utils.ml_mocks import setup_ml_mocks
from wave.test_utils.decorators import requires_torch

is_testing = setup_ml_mocks()

# Import torch after mocks have been set up
import torch

import numpy as np
from pathlib import Path

from wave.ml.viz.tensorboard_utils import (
    create_confusion_matrix_figure,, 
    log_confusion_matrix,, 
    log_model_graph,, 
    log_feature_importance,, 
    log_predictions_vs_actual,, 
    generate_tensor_histogram,, 
    setup_tensorboard
)

# Create temporary test directory
@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for TensorBoard logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture
def sample_confusion_matrix():
    """Create a sample confusion matrix."""
    return np.array([
        [10, 2, 0],
        [1, 12, 3],
        [0, 2, 8]
    ])

@pytest.fixture
def sample_model():
    """Create a simple sample model."""
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(1, 16, 3)
            self.relu = torch.nn.ReLU()
            self.flatten = torch.nn.Flatten()
            self.fc1 = torch.nn.Linear(16 * 8, 32)
            self.fc2 = torch.nn.Linear(32, 3)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    return SimpleModel()

@requires_torch
def test_create_confusion_matrix_figure(sample_confusion_matrix):
    """Test creating confusion matrix figure."""
    class_names = ["Class A", "Class B", "Class C"]
    fig = create_confusion_matrix_figure(
        sample_confusion_matrix,
        class_names=class_names
    )
    assert fig is not None

@requires_torch
def test_log_confusion_matrix(sample_confusion_matrix, temp_log_dir):
    """Test logging confusion matrix to TensorBoard."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=temp_log_dir)
    class_names = ["Class A", "Class B", "Class C"]

    # Log the confusion matrix
    log_confusion_matrix(
        writer,
        sample_confusion_matrix,
        class_names=class_names,
        step=1
    )

    # Check if the event file was created
    event_files = list(Path(temp_log_dir).glob("events.out.tfevents.*"))
    assert len(event_files) > 0

    # Close writer
    writer.close()

@requires_torch
def test_log_model_graph(sample_model, temp_log_dir):
    """Test logging model graph to TensorBoard."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=temp_log_dir)

    # Sample input for the model
    sample_input = torch.randn(1, 1, 10)

    # Log the model graph
    log_model_graph(writer, sample_model, sample_input)

    # Check if the event file was created
    event_files = list(Path(temp_log_dir).glob("events.out.tfevents.*"))
    assert len(event_files) > 0

    # Close writer
    writer.close()

@requires_torch
def test_log_feature_importance(temp_log_dir):
    """Test logging feature importance to TensorBoard."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=temp_log_dir)

    # Sample feature names and importance scores
    feature_names = ["Feature A", "Feature B", "Feature C", "Feature D"]
    importance_scores = np.array([0.3, 0.1, 0.5, 0.1])

    # Log feature importance
    log_feature_importance(writer, importance_scores, feature_names)

    # Check if the event file was created
    event_files = list(Path(temp_log_dir).glob("events.out.tfevents.*"))
    assert len(event_files) > 0

    # Close writer
    writer.close()

@requires_torch
def test_log_predictions_vs_actual(temp_log_dir):
    """Test logging predictions vs actual values to TensorBoard."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=temp_log_dir)

    # Sample prediction and actual arrays
    predictions = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    actual = np.array([0, 1, 2, 2, 0, 2, 0, 0])

    # Class names
    class_names = ["Class A", "Class B", "Class C"]

    # Log predictions vs actual
    log_predictions_vs_actual(writer, predictions, actual, class_names)

    # Check if the event file was created
    event_files = list(Path(temp_log_dir).glob("events.out.tfevents.*"))
    assert len(event_files) > 0

    # Close writer
    writer.close()

@requires_torch
def test_generate_tensor_histogram(temp_log_dir):
    """Test generating and logging tensor histograms to TensorBoard."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=temp_log_dir)

    # Sample tensor
    tensor = torch.randn(100, 10)

    # Generate and log histogram
    generate_tensor_histogram(writer, tensor, "sample_tensor", step=0)

    # Check if the event file was created
    event_files = list(Path(temp_log_dir).glob("events.out.tfevents.*"))
    assert len(event_files) > 0

    # Close writer
    writer.close()

@requires_torch
def test_setup_tensorboard():
    """Test setting up a TensorBoard instance with run ID."""
    # Create a temporary directory for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = setup_tensorboard("test_model", base_dir=temp_dir)

        # Check if the writer was created with a proper log directory
        assert writer is not None
        assert hasattr(writer, "log_dir")
        assert "test_model_" in writer.log_dir

        # Close writer
        writer.close()

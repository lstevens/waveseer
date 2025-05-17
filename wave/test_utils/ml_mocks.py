"""
Mock implementations of ML dependencies for testing.

This module provides mock implementations of torch, numpy, and related ML
dependencies to allow tests to run without requiring the full ML stack.
"""

import os
import logging
from unittest.mock import MagicMock
from types import ModuleType

logger = logging.getLogger(__name__)


class MockTorch(ModuleType):
    """Mock implementation of torch module."""

    def __init__(self):
        super().__init__('torch')
        # Create nn module
        self.nn = ModuleType('nn')

        # Add Module class
        self.nn.Module = type('Module', (), {
            '__init__': lambda self: None,
            'forward': lambda self, x: x,
            'eval': lambda self: self,
            'parameters': lambda self: [MagicMock()],
            'to': lambda self, device: self
        })

        # Add ModuleList class
        class MockModuleList(list):
            def __init__(self, modules=None):
                super().__init__(modules or [])

            def append(self, module):
                super().append(module)
                return self

            def extend(self, modules):
                super().extend(modules)
                return self

            def parameters(self):
                return [MagicMock()]

            def to(self, device):
                return self

            def eval(self):
                return self

        self.nn.ModuleList = MockModuleList

        # Create mock layer factory function
        def create_layer_class(layer_name):
            return type(layer_name, (self.nn.Module,), {
                '__init__': lambda self, *args, **kwargs: None,
                'forward': lambda self, x: x,
                'weight': MagicMock(),
                'bias': MagicMock()
            })

        # Add common layer types
        for layer in ['Conv1d', 'Conv2d', 'Conv3d',
                      'Linear', 'LSTM', 'GRU', 'RNN',
                      'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
                      'Dropout', 'MaxPool1d', 'MaxPool2d', 'AvgPool1d', 'AvgPool2d',
                      'ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU',
                      'Embedding', 'Flatten', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d']:
            setattr(self.nn, layer, create_layer_class(layer))

        # Create nn.functional module
        nn_functional = ModuleType('functional')
        # Add common functions to nn.functional
        for func_name in ['relu', 'softmax', 'cross_entropy', 'mse_loss',
                          'dropout', 'conv1d', 'conv2d', 'max_pool1d', 'max_pool2d',
                          'avg_pool1d', 'avg_pool2d', 'linear', 'batch_norm',
                          'embedding', 'pad', 'binary_cross_entropy']:
            setattr(nn_functional, func_name, lambda *args, **kwargs: MagicMock())

        self.nn.functional = nn_functional

        self.optim = MagicMock()
        self.optim.Adam = MagicMock(return_value=MagicMock())

        self.cuda = MagicMock()
        self.cuda.is_available = MagicMock(return_value=False)

        self.Tensor = MagicMock()
        self.tensor = lambda x: MagicMock()
        self.load = lambda *args, **kwargs: MagicMock()
        self.save = lambda *args, **kwargs: None
        self.no_grad = MagicMock()
        self.no_grad.__enter__ = lambda x: None
        self.no_grad.__exit__ = lambda x, *args: None

        # Create mock device
        self.device = MagicMock()
        self.device.type = 'cpu'

        # Mock common tensor functions
        self.zeros = lambda *args, **kwargs: MagicMock()
        self.ones = lambda *args, **kwargs: MagicMock()
        self.randn = lambda *args, **kwargs: MagicMock()
        self.cat = lambda *args, **kwargs: MagicMock()
        self.stack = lambda *args, **kwargs: MagicMock()
        self.transpose = lambda *args, **kwargs: MagicMock()
        self.mean = lambda *args, **kwargs: MagicMock()
        self.sum = lambda *args, **kwargs: MagicMock()

        # Mock jit module (needed for to_torchscript)
        self.jit = MagicMock()
        self.jit.ScriptModule = MagicMock
        self.jit.script = lambda x: x
        self.jit.trace = lambda model, example_inputs: model

        # Create a dtype class for torch.dtype
        class MockDType:
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return f"torch.{self.name}"

        # Create dtype object
        self.dtype = ModuleType('dtype')

        # Define quantization types
        for dtype_name in ['int', 'int8', 'int16', 'int32', 'int64',
                      'float', 'float16', 'float32', 'float64',
                      'uint8', 'long', 'double',
                      'qint8', 'qint32', 'quint8', 'quint4x2', 'quint2x4',
                      'bfloat16', 'bool']:
            dtype_obj = MockDType(dtype_name)
            setattr(self, dtype_name, dtype_obj)

        # Return a value for torch.device()
        self.device = lambda *args, **kwargs: 'cpu'

    def __call__(self, *args, **kwargs):
        return MagicMock()


class MockDataset:
    """Mock implementation of torch.utils.data.Dataset."""

    def __init__(self, *args, **kwargs):
        self.data = [MagicMock() for _ in range(10)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MockDataLoader:
    """Mock implementation of torch.utils.data.DataLoader."""

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = [self.dataset[j] for j in range(i, min(i + self.batch_size, len(self.dataset)))]
            yield batch


def setup_ml_mocks():
    """
    Setup mocks for ML-related modules when running in test mode.

    This function should be imported and called at the beginning of test files
    that require ML dependencies but are running in test mode.
    """
    if os.getenv("TESTING") == "true":
        import sys

        # Create mock torch module
        mock_torch = MockTorch()
        sys.modules['torch'] = mock_torch

        # Create mock torch.utils.data
        mock_data = ModuleType('data')
        mock_data.Dataset = MockDataset
        mock_data.DataLoader = MockDataLoader

        # Create mock torch.utils.tensorboard
        class MockSummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def add_scalars(self, *args, **kwargs):
                pass

            def add_image(self, *args, **kwargs):
                pass

            def add_figure(self, *args, **kwargs):
                pass

            def add_graph(self, *args, **kwargs):
                pass

            def close(self):
                pass

        mock_tensorboard = ModuleType('tensorboard')
        mock_tensorboard.SummaryWriter = MockSummaryWriter

        # Create mock torch.utils.mobile_optimizer
        mock_mobile_optimizer = ModuleType('mobile_optimizer')
        mock_mobile_optimizer.optimize_for_mobile = lambda model, *args, **kwargs: model

        # Assemble torch.utils
        mock_utils = ModuleType('utils')
        mock_utils.data = mock_data
        mock_utils.tensorboard = mock_tensorboard
        mock_utils.mobile_optimizer = mock_mobile_optimizer
        sys.modules['torch.utils'] = mock_utils
        mock_torch.utils = mock_utils

        # Register submmodules
        sys.modules['torch.utils.data'] = mock_data
        sys.modules['torch.utils.tensorboard'] = mock_tensorboard
        sys.modules['torch.utils.mobile_optimizer'] = mock_mobile_optimizer

        # Create mock modules for other ML libraries
        sys.modules['torch.nn'] = mock_torch.nn
        sys.modules['torch.nn.functional'] = mock_torch.nn.functional
        sys.modules['torch.optim'] = mock_torch.optim

        # Create proper torch.jit module
        mock_jit = ModuleType('jit')
        mock_jit.ScriptModule = MagicMock
        mock_jit.script = lambda x: x
        mock_jit.trace = lambda model, example_inputs: model
        sys.modules['torch.jit'] = mock_jit

        sys.modules['torchvision'] = MagicMock()
        sys.modules['tensorboard'] = MagicMock()
        sys.modules['pytorch_lightning'] = MagicMock()

        logger.info("ML dependencies mocked for testing")
        return True

    return False

#!/usr/bin/env python3
"""
PyTorch Installation Verification Script

This script verifies that PyTorch is properly installed
and outputs system information for debugging.
"""

import sys
import platform
import torch
import numpy as np


def verify_pytorch():
    """Verify PyTorch installation and print system info."""
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    print("=" * 50)

    # Basic tensor operations
    print("\nBASIC TENSOR OPERATIONS")
    print("=" * 50)

    # Create a tensor
    x = torch.randn(5, 3)
    print(f"Random tensor x:\n{x}")

    # Basic operations
    y = torch.ones(5, 3)
    print(f"\nTensor y (ones):\n{y}")

    z = x + y
    print(f"\nAddition (x + y):\n{z}")

    # Matrix multiplication
    a = torch.randn(3, 4)
    b = torch.randn(4, 2)
    c = torch.matmul(a, b)
    print(f"\nMatrix multiplication (a @ b):\n{c}")

    # Test GPU if available
    if torch.cuda.is_available():
        print("\nTesting GPU operations...")
        x_cuda = x.cuda()
        y_cuda = y.cuda()
        z_cuda = x_cuda + y_cuda
        print(f"GPU addition result shape: {z_cuda.shape}, device: {z_cuda.device}")
        print("GPU operations successful!")

    print("\nPyTorch verification complete! All operations successful.")
    return True

if __name__ == "__main__":
    try:
        success = verify_pytorch()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error during PyTorch verification: {e}")
        sys.exit(1)

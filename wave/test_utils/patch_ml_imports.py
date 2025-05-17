#!/usr/bin/env python3
"""
Patch test files to use ML mocks when in testing mode.

This script adds import statements for ML mocks and decorators to test files
that use PyTorch and other ML dependencies. This allows tests to run in
TESTING mode without requiring the actual ML dependencies.
"""

import re
import argparse
from pathlib import Path

MOCK_IMPORTS_TEMPLATE = """
# Setup ML mocks if in testing mode
from unittest.mock import MagicMock
from wave.test_utils.ml_mocks import setup_ml_mocks
from wave.test_utils.decorators import requires_torch, requires_ml_stack

is_testing = setup_ml_mocks()

# Import torch after mocks have been set up
import torch
"""

FUNCTION_DECORATOR_PATTERN = r"def (test_[a-zA-Z0-9_]+\()"
REPLACEMENT = r"@requires_torch\ndef \1"


def patch_test_file(filepath):
    """
    Patch a test file to use the ML mocks and decorators.

    Args:
        filepath: Path to the test file to patch.

    Returns:
        True if the file was modified, False otherwise.
    """
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if the file already has the mock imports
    if "from wave.test_utils.ml_mocks import setup_ml_mocks" in content:
        print(f"File {filepath} already patched. Skipping.")
        return False

    # Find the import statements
    import_pattern = r"import torch|from torch"
    if not re.search(import_pattern, content):
        print(f"File {filepath} does not import torch. Skipping.")
        return False

    # Replace the first occurrence of torch import with our mock imports
    patched_content = re.sub(import_pattern, MOCK_IMPORTS_TEMPLATE, content, count=1)

    # Add decorator to test functions
    patched_content = re.sub(FUNCTION_DECORATOR_PATTERN, REPLACEMENT, patched_content)

    # Write the modified content back to the file
    with open(filepath, 'w') as f:
        f.write(patched_content)

    print(f"Patched file: {filepath}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch test files to use ML mocks")
    parser.add_argument("--dir", type=str, default="tests", help="Directory containing test files")
    parser.add_argument("--pattern", type=str, default="test_*.py", help="Pattern to match test files")
    args = parser.parse_args()

    root = Path.cwd()
    test_dir = root / args.dir

    if not test_dir.exists():
        print(f"Directory {test_dir} does not exist.")
        return

    # Find all test files matching the pattern
    test_files = list(test_dir.glob(args.pattern))

    # Add specific files from other directories
    wave_test_files = list((root / "wave").glob(args.pattern))
    test_files.extend(wave_test_files)

    if not test_files:
        print(f"No test files found matching pattern {args.pattern} in {test_dir}.")
        return

    count = 0
    for tf in test_files:
        if patch_test_file(tf):
            count += 1

    print(f"Patched {count} test files.")


if __name__ == "__main__":
    main()

"""
Test decorators for WaveSeer.

This module provides decorators for test functions to handle
specific testing scenarios like skipping ML tests in TESTING mode.
"""

import os
import functools
import pytest
import logging

logger = logging.getLogger(__name__)


def skip_if_testing(reason="Test requires ML dependencies unavailable in TESTING mode"):
    """
    Skip a test when the TESTING environment variable is set to "true".

    This decorator is useful for tests that require actual ML dependencies
    that can't be reasonably mocked.

    Args:
        reason: The reason for skipping the test.

    Returns:
        Decorated test function that will be skipped when TESTING=true.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.getenv("TESTING") == "true":
                pytest.skip(reason)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def requires_torch(func):
    """
    Mark a test as requiring PyTorch.

    Tests with this decorator will be skipped when TESTING=true.

    Args:
        func: The test function to decorate.

    Returns:
        Decorated test function.
    """
    return skip_if_testing("Test requires PyTorch")(func)


def requires_ml_stack(func):
    """
    Mark a test as requiring the full ML stack.

    Tests with this decorator will be skipped when TESTING=true.

    Args:
        func: The test function to decorate.

    Returns:
        Decorated test function.
    """
    return skip_if_testing("Test requires full ML stack")(func)

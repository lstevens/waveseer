"""
Basic smoke tests to verify imports and application boot.
These tests should complete in under 30 seconds.
"""
import pytest


def test_core_imports():
    """Verify that core modules can be imported."""
    import wave
    import wave.ingest
    import wave.ml
    assert True, "Core modules imported successfully"


def test_api_imports():
    """Verify that API modules can be imported."""
    import wave.api
    assert True, "API modules imported successfully"


def test_app_creation():
    """Verify that the FastAPI application can be created."""
    from wave.api import create_app
    app = create_app()
    assert app is not None, "Application created successfully"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

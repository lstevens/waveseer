"""
Smoke test to verify that our test structure follows the vibe-coding playbook.
This test ensures we have the proper test zones and organization.
"""
import pytest
import os
from pathlib import Path


def test_test_structure():
    """Verify that the proper test directories exist and are organized correctly."""
    # Get the root test directory
    test_dir = Path(__file__).parent.parent
    
    # Check that all required test zone directories exist
    smoke_dir = test_dir / "smoke"
    stable_dir = test_dir / "stable"
    playground_dir = test_dir / "playground"
    full_dir = test_dir / "full"
    
    assert smoke_dir.exists(), "Smoke test directory should exist"
    assert stable_dir.exists(), "Stable test directory should exist"
    assert playground_dir.exists(), "Playground test directory should exist"
    assert full_dir.exists(), "Full test directory should exist"
    
    # Verify smoke tests contain basic sanity checks
    smoke_files = list(smoke_dir.glob("test_*.py"))
    assert len(smoke_files) > 0, "Smoke tests directory should contain test files"
    
    # Check that README.md exists and contains test zone documentation
    readme_path = test_dir / "README.md"
    assert readme_path.exists(), "Test README.md should exist"
    
    with open(readme_path, "r") as f:
        readme_content = f.read()
        assert "Smoke" in readme_content, "README should document Smoke tests"
        assert "Stable" in readme_content, "README should document Stable tests"
        assert "Playground" in readme_content, "README should document Playground tests"
        assert "Full" in readme_content, "README should document Full tests"


def test_test_organization():
    """Verify that we have at least some tests in each test zone."""
    # Get the root test directory
    test_dir = Path(__file__).parent.parent
    
    # Count test files in each zone
    smoke_count = len(list(Path(test_dir / "smoke").glob("test_*.py")))
    stable_count = len(list(Path(test_dir / "stable").glob("test_*.py")))
    playground_count = len(list(Path(test_dir / "playground").glob("test_*.py")))
    
    # We should have at least some tests in each directory
    assert smoke_count > 0, "Should have some smoke tests"
    assert stable_count > 0, "Should have some stable tests"
    
    # Our recent work should have added at least one test to stable
    stable_tests = list(Path(test_dir / "stable").glob("test_*.py"))
    stable_test_names = [t.name for t in stable_tests]
    assert "test_indicators.py" in stable_test_names, "Should have promoted indicators test to stable"
    assert "test_ws_integration.py" in stable_test_names, "Should have promoted WebSocket integration test to stable"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

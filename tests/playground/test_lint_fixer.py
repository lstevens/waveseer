"""
Tests for the lint_fixer utility.

These tests verify that the lint_fixer properly handles common flake8 issues.
"""

import os
import sys
import tempfile
from pathlib import Path


# Ensure wave package is in path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from wave.utils.lint_fixer import (
    fix_blank_line_whitespace,, 
    fix_trailing_whitespace,, 
    fix_blank_line_count,, 
    fix_unused_imports,, 
    process_file
)


def test_fix_blank_line_whitespace():
    """Test fixing blank lines with whitespace."""
    # Arrange
    content = "def test():\n    return True\n    \n\ndef another():\n    pass"
    expected = "def test():\n    return True\n\n\ndef another():\n    pass"


    # Act
    result = fix_blank_line_whitespace(content)

    # Assert
    assert result == expected


def test_fix_trailing_whitespace():
    """Test fixing trailing whitespace."""
    # Arrange
    content = "def test(): \n    return True \n"
    expected = "def test():\n    return True\n"


    # Act
    result = fix_trailing_whitespace(content)

    # Assert
    assert result == expected


def test_fix_blank_line_count():
    """Test fixing blank line count."""
    # Arrange
    content = "import os\n\ndef test():\n    return True\n\nclass Test:\n    pass"
    expected = "import os\n\n\ndef test():\n    return True\n\n\nclass Test:\n    pass"


    # Act
    result = fix_blank_line_count(content)

    # Assert
    assert result == expected


def test_process_file():
    """Test processing a file with linting issues."""
    # Arrange
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as f:
        f.write("def test(): \n    return True\n    \n")
        temp_path = Path(f.name)

    try:
        # Act
        changes_made, messages = process_file(temp_path)

        # Assert
        assert changes_made is True
        assert len(messages) == 1
        assert "Fixed linting issues" in messages[0]

        # Verify the content was fixed
        with open(temp_path, 'r') as f:
            content = f.read()
            assert content == "def test():\n    return True\n\n"
    finally:
        # Clean up
        os.unlink(temp_path)


def test_process_nonexistent_file():
    """Test processing a file that doesn't exist."""
    # Arrange
    nonexistent_path = Path("/path/to/nonexistent/file.py")

    # Act
    changes_made, messages = process_file(nonexistent_path)

    # Assert
    assert changes_made is False
    assert len(messages) == 1
    assert "File not found" in messages[0]


def test_fix_unused_imports():
    """Test fixing unused imports (F401)."""
    # Arrange
    content = "import os\nimport sys\nimport re  # unused\nfrom typing import List, Dict  # Dict unused\n\ndef test():\n    return os.path.join(sys.path[0], 'test')"
    expected = "import os\nimport sys\nfrom typing import List\n\ndef test():\n    return os.path.join(sys.path[0], 'test')"


    # Act
    result = fix_unused_imports(content, {'re': None, 'Dict': 'typing'})

    # Assert
    assert result == expected


def test_fix_multiline_imports():
    """Test fixing multi-line parenthesized imports (F401)."""
    # Arrange
    content = "from module import (\n    used_func,\n    unused_func,\n    another_used_func,\n    yet_another_unused_func\n)\n\ndef test():\n    used_func()\n    another_used_func()"
    expected = "from module import (\n    used_func,\n    another_used_func\n)\n\ndef test():\n    used_func()\n    another_used_func()"


    # Act
    result = fix_unused_imports(content, {'unused_func': 'module', 'yet_another_unused_func': 'module'})

    # Assert
    assert result == expected

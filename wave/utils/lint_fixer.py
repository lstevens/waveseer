"""
Utility script to automatically fix common flake8 issues.

This script focuses on fixing:
- Blank lines with whitespace (W293)
- Trailing whitespace (W291)
- Import ordering (E402)
- Blank line formatting (E302, E305)
- Unused imports (F401)

It doesn't attempt to fix line length issues (E501) which often require
manual intervention.
"""

import re
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def fix_blank_line_whitespace(content: str) -> str:
    """Fix blank lines containing whitespace (W293)."""
    # Replace any line that contains only whitespace with an empty line
    # Careful to preserve the same number of newlines
    lines = content.splitlines(True)  # Keep newlines
    result = ""
    for line in lines:
        if line.strip() == "":
            result += "\n"  # Replace whitespace-only line with a clean newline
        else:
            result += line
    return result


def fix_trailing_whitespace(content: str) -> str:
    """Fix trailing whitespace (W291)."""
    # Replace trailing whitespace on any line, but preserve newlines
    lines = content.splitlines(True)  # Keep newlines
    result = ""
    for line in lines:
        # Remove trailing whitespace but keep the newline if present
        if line.endswith('\n'):
            result += line.rstrip() + '\n'
        else:
            result += line.rstrip()
    return result


def fix_blank_line_count(content: str) -> str:
    """Fix blank line count issues (E302, E305)."""
    # This is a simplified version - complex cases require manual fixing
    # Fix single blank line before class/def where two are needed
    content = re.sub(r'([^\n])\n\n(class |def )', r'\1\n\n\n\2', content)
    # Fix single blank line after class/def where two are needed
    content = re.sub(r'(def .*?:.*?\n)(\n)([^\n])', r'\1\n\n\3', content)
    return content


def fix_comma_whitespace(content: str) -> str:
    """Fix missing whitespace after commas (E231).

    Adds a space after commas where one is missing, except in special contexts like
    string literals or comments.
    """
    lines = content.splitlines(True)
    result = ""

    for line in lines:
        # Skip processing inside comments
        if '#' in line:
            code_part, comment_part = line.split('#', 1)
            processed_line = fix_comma_in_code(code_part) + '#' + comment_part
        else:
            processed_line = fix_comma_in_code(line)

        result += processed_line

    return result


def fix_comma_in_code(code: str) -> str:
    """Helper function for fix_comma_whitespace that processes non-comment code."""
    # This is a simplified approach - a full parser would be more accurate
    # Replace a comma followed by non-whitespace with comma+space
    # Exclude patterns like closing brackets that shouldn't have a space after comma
    result = re.sub(r', ([^\s\)\]\}])', r', \1', code)

    return result


def get_unused_imports(file_path: Path) -> Dict[str, Optional[str]]:
    """
    Find unused imports in a file using flake8.

    Args:
        file_path: Path to the file to check

    Returns:
        Dictionary mapping import names to their optional module names
    """
    unused_imports = {}

    try:
        # Run flake8 to identify unused imports (F401)
        cmd = ["python3", "-m", "flake8", str(file_path), "--select=F401"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Parse the output to extract unused import names
        for line in result.stdout.splitlines():
            if "F401" not in line:
                continue

            # Extract the import name from the error message
            # Format: path:line:col: F401 'module.name' imported but unused
            match = re.search(r"F401 '([^']+)' imported but unused", line)
            if not match:
                continue

            import_path = match.group(1)

            # Handle different import formats
            if '.' in import_path:
                # For 'from module import name'
                module, name = import_path.rsplit('.', 1)
                unused_imports[name] = module
            else:
                # For 'import module'
                unused_imports[import_path] = None

    except Exception:
        # If flake8 fails, return empty dict
        pass

    return unused_imports


def fix_unused_imports(content: str, unused_imports: Dict[str, Optional[str]]) -> str:
    """
    Fix unused imports (F401) by removing them from the file.

    Args:
        content: File content to fix
        unused_imports: Dictionary mapping import names to their optional module names

    Returns:
        Fixed content with unused imports removed
    """
    if not unused_imports:
        return content

    lines = content.splitlines()
    result_lines = []
    skip_next = False

    # State tracking for multi-line imports
    in_multiline_import = False
    multiline_module = ""
    multiline_imports = []
    multiline_indent = ""

    i = 0
    while i < len(lines):
        line = lines[i]

        if skip_next:
            skip_next = False
            i += 1
            continue

        # Continue processing a multi-line import if we're in one
        if in_multiline_import:
            # Check if this line contains the closing parenthesis
            stripped = line.strip()

            if ')' in stripped:
                # End of multi-line import
                # Extract any imports on this line before the closing paren
                before_close = stripped.split(')', 1)[0].strip()
                if before_close and before_close != "":
                    multiline_imports.append(before_close)

                # Process all collected imports
                cleaned_imports = []

                # First, clean each import (strip comments, trailing commas)
                processed_imports = []
                for imp in multiline_imports:
                    # Handle trailing commas and comments
                    if ', ' in imp:
                        imp = imp.rstrip(', ')

                    if '#' in imp:
                        imp = imp.split('#', 1)[0].strip()

                    # Skip empty entries
                    if imp and imp != "":
                        processed_imports.append(imp.strip())

                # Then filter out unused imports
                for imp in processed_imports:
                    # Check if this is an unused import from this module
                    if imp in unused_imports and unused_imports[imp] == multiline_module:
                        continue  # Skip this import
                    cleaned_imports.append(imp)

                # Only add the import if we have items remaining
                if cleaned_imports:
                    # Reconstruct the multi-line import
                    result_lines.append(f"from {multiline_module} import (")
                    # Add all imports except the last one with commas
                    for idx, imp in enumerate(cleaned_imports):
                        if idx < len(cleaned_imports) - 1:
                            result_lines.append(f"{multiline_indent}{imp}, ")
                        else:
                            result_lines.append(f"{multiline_indent}{imp}")
                    result_lines.append(")")

                # Reset multi-line state
                in_multiline_import = False
                multiline_imports = []
                multiline_module = ""
            else:
                # Still in multi-line import, collect this import
                multiline_imports.append(stripped)

            i += 1
            continue

        # Handle direct imports: "import module"
        if line.startswith('import '):
            imports = line[7:].split(', ')
            cleaned_imports = []

            for imp in imports:
                imp = imp.strip()
                # Remove comments if present
                if '#' in imp:
                    imp = imp.split('#')[0].strip()

                # Keep only imports that aren't unused
                if imp not in unused_imports:
                    cleaned_imports.append(imp)

            if cleaned_imports:
                result_lines.append('import ' + ', '.join(cleaned_imports))
            # Skip empty import lines

        # Handle "from module import name" style
        elif line.startswith('from '):
            # Find the "import" keyword
            parts = line.split(' import ', 1)
            if len(parts) != 2:
                result_lines.append(line)
                i += 1
                continue

            module = parts[0][5:].strip()
            imports_part = parts[1].strip()

            # Check if this is the start of a multi-line import
            if imports_part.startswith('(') and not imports_part.endswith(')'):
                # Start multi-line import tracking
                in_multiline_import = True
                multiline_module = module
                multiline_indent = ' ' * 4  # Standard indent

                # Extract any imports on the same line after the opening paren
                if len(imports_part) > 1:
                    first_import = imports_part[1:].strip()
                    if first_import:
                        multiline_imports.append(first_import)
            else:
                # Single-line import (potentially with parentheses)
                if imports_part.startswith('(') and imports_part.endswith(')'):
                    # Handle single-line with parentheses: from x import (a, b, c)
                    imports_part = imports_part[1:-1]  # Remove parentheses

                imports = imports_part.split(', ')
                cleaned_imports = []

                for imp in imports:
                    imp = imp.strip()
                    # Remove comments if present
                    if '#' in imp:
                        imp = imp.split('#')[0].strip()

                    # Keep only imports that aren't unused or from different modules
                    if imp not in unused_imports or unused_imports[imp] != module:
                        cleaned_imports.append(imp)

                if cleaned_imports:
                    result_lines.append(f'from {module} import {", ".join(cleaned_imports)}')
                # Skip empty import lines

        else:
            result_lines.append(line)

        i += 1

    return '\n'.join(result_lines)


def process_file(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Process a single file to fix linting issues.

    Args:
        file_path: Path to the file to process

    Returns:
        Tuple of (changes_made, messages)
    """
    if not file_path.exists():
        return False, [f"File not found: {file_path}"]

    with open(file_path, 'r') as f:
        original_content = f.read()

    # Apply the fixes
    content = original_content
    content = fix_blank_line_whitespace(content)
    content = fix_trailing_whitespace(content)
    content = fix_blank_line_count(content)
    content = fix_comma_whitespace(content)  # Add fix for E231 errors

    # Fix unused imports if the file is a Python file
    if file_path.suffix == '.py':
        unused_imports = get_unused_imports(file_path)
        if unused_imports:
            content = fix_unused_imports(content, unused_imports)

    # Ensure file ends with a newline
    if not content.endswith('\n'):
        content += '\n'

    changes_made = content != original_content

    if changes_made:
        with open(file_path, 'w') as f:
            f.write(content)
        return True, [f"Fixed linting issues in {file_path}"]
    else:
        return False, [f"No automatic fixes applied to {file_path}"]


def main(file_paths: List[str]) -> None:
    """
    Main entry point for the linting fixer.

    Args:
        file_paths: List of file paths to process
    """
    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        changes_made, messages = process_file(file_path)
        for message in messages:
            print(message)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lint_fixer.py <file_path> [<file_path> ...]")
        sys.exit(1)

    main(sys.argv[1:])

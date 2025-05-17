#!/usr/bin/env python
"""
Script to run the lint_fixer utility on all Python files in the project.
"""
import os
import subprocess
from pathlib import Path


def find_python_files(base_dir="."):
    """Find all Python files in the project."""
    python_files = []
    for root, _, files in os.walk(base_dir):
        if ".git" in root or "__pycache__" in root or "venv" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def fix_linting_issues():
    """Run lint_fixer on all Python files."""
    base_dir = Path(__file__).parent
    python_files = find_python_files(base_dir)

    # Process files in batches to avoid command line length limits
    batch_size = 10
    total_files = len(python_files)
    total_fixed = 0

    print(f"Found {total_files} Python files to process")

    for i in range(0, total_files, batch_size):
        batch = python_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}...")

        cmd = ["python", "-m", "wave.utils.lint_fixer"] + batch
        result = subprocess.run(cmd, capture_output=True, text=True)

        output = result.stdout.strip()
        if output:
            print(output)
            # Count files that had fixes applied
            fixed_count = output.count("Applied automatic fixes to")
            total_fixed += fixed_count

    print(f"Completed linting fixes on {total_fixed}/{total_files} files")

if __name__ == "__main__":
    fix_linting_issues()

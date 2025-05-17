#!/usr/bin/env python3
"""
Setup script for git pre-commit hooks.

This script installs pre-commit and sets up the hooks defined in
.pre-commit-config.yaml to run automatically on git commit.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_prerequisites():
    """Check if pre-commit is installed."""
    try:
        subprocess.run(
            ["pre-commit", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def install_pre_commit():
    """Install pre-commit if not already installed."""
    print("⚙️ Installing pre-commit...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pre-commit"],
            check=True
        )
        return True
    except subprocess.SubprocessError as e:
        print(f"❌ Failed to install pre-commit: {e}")
        return False


def setup_hooks():
    """Set up the pre-commit hooks."""
    print("⚙️ Setting up pre-commit hooks...")
    try:
        subprocess.run(
            ["pre-commit", "install"],
            check=True
        )
        print("✅ Pre-commit hooks installed successfully!")
        return True
    except subprocess.SubprocessError as e:
        print(f"❌ Failed to set up pre-commit hooks: {e}")
        return False


def main():
    """Main entry point."""
    print("🌊 WaveSeer Pre-commit Hook Setup 🌊")

    # Change to project root directory
    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)

    # Check if .pre-commit-config.yaml exists
    if not Path(".pre-commit-config.yaml").exists():
        print("❌ .pre-commit-config.yaml not found in project root!")
        return 1

    # Check for pre-commit
    if not check_prerequisites():
        print("➡️ pre-commit not found. Installing...")
        if not install_pre_commit():
            return 1

    # Set up hooks
    if not setup_hooks():
        return 1

    print("\n🎉 Setup complete! Pre-commit hooks will now run on each commit.")
    print("➡️ To run manually on all files: pre-commit run --all-files")

    return 0


if __name__ == "__main__":
    sys.exit(main())

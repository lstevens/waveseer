# Migrating from Anaconda to Standard Python

This guide provides step-by-step instructions for migrating the WaveSeer development environment from Anaconda to standard Python. Following these instructions will help ensure consistent behavior across all development environments.

## Why Migrate?

Anaconda provides a convenient scientific Python distribution but can sometimes lead to dependency conflicts and inconsistent behavior across environments. Moving to standard Python:

- Simplifies dependency management
- Ensures consistent package versions across all environments
- Reduces potential conflicts with system packages
- Makes CI/CD pipelines more reliable
- Enables cleaner virtual environments

## Prerequisites

- Python 3.12+ (recommended) or Python 3.10+ (minimum)
- Git
- A terminal/command prompt
- Basic knowledge of command-line operations

## Migration Steps

### 1. Install Standard Python

#### macOS
```bash
# Using Homebrew
brew install python@3.12

# Verify installation
python3 --version
```

#### Windows
- Download the official Python installer from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"
- Verify with `python --version` in Command Prompt

#### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# Verify installation
python3.12 --version
```

### 2. Create a Fresh Virtual Environment

```bash
# Navigate to project directory
cd /path/to/waveseer

# Create a new virtual environment
python3.12 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Environment Variables Configuration

WaveSeer requires several environment variables for proper functionality, especially during testing:

```bash
# Core application variables
export WAVESEER_ENV="development"  # Options: development, testing, production
export WAVESEER_LOG_LEVEL="DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR

# Testing variables
export TESTING="true"  # When running tests
export PIPELINE_BYPASS_ENABLED="false"  # Controls pattern pipeline behavior in tests
export PIPELINE_ECHO_RAW_EVENTS="false"  # Controls event broadcasting in tests

# Database configuration
export WAVESEER_DB_HOST="localhost"
export WAVESEER_DB_PORT="5432" 
export WAVESEER_DB_NAME="waveseer"
export WAVESEER_DB_USER="postgres"
export WAVESEER_DB_PASSWORD="postgres"

# For Windows CMD:
# set VARIABLE_NAME=value
# For Windows PowerShell:
# $env:VARIABLE_NAME="value"
```

### 5. Setting Up PostgreSQL for Testing

WaveSeer uses PostgreSQL for data storage. For testing:

1. Install PostgreSQL (if not already installed)
   ```bash
   # macOS
   brew install postgresql
   
   # Ubuntu/Debian
   sudo apt install postgresql postgresql-contrib
   ```

2. Create test database
   ```bash
   # Start PostgreSQL service
   # macOS:
   brew services start postgresql
   # Ubuntu/Debian:
   sudo service postgresql start
   
   # Create database
   createdb -U postgres waveseer
   ```

### 6. Common Migration Issues & Solutions

#### Issue: NumPy Compatibility
Anaconda often uses custom-built numpy binaries. If you encounter errors with numpy:
```bash
pip uninstall numpy
pip install numpy==1.25.0
```

#### Issue: Missing Compiled Extensions
```bash
# macOS
brew install cmake libomp

# Ubuntu/Debian
sudo apt install build-essential cmake libomp-dev
```

#### Issue: PyTorch Installation
Install PyTorch according to your system configuration:
```bash
# CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# With CUDA support (replace X.Y with your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuX.Y
```

#### Issue: Dash Compatibility
WaveSeer requires specific versions of Dash and Werkzeug:
```bash
pip install dash==2.1.0 dash-bootstrap-components==1.0.0 Werkzeug==2.0.3
```

### 7. Verifying Your Environment

Run the following checks to ensure everything is set up correctly:

```bash
# Check imports
python -c "import torch; import pandas; import fastapi; print('Success!')"

# Run smoke tests
python -m pytest tests/smoke -v

# Run WebSocket tests
python -m pytest tests/test_ws_ingest_integration.py::test_ws_integration_enriched_events -v
```

### 8. IDE Integration

#### VS Code
1. Select your new virtual environment:
   - View > Command Palette > Python: Select Interpreter
   - Choose the interpreter in the `venv` directory
   
2. Install extensions:
   - Python
   - Pylance
   - Jupyter

#### PyCharm
1. Go to Settings/Preferences > Project > Python Interpreter
2. Click the gear icon > Add...
3. Select "Existing Environment" and point to the Python in your venv

## Additional Resources

- [Python Virtual Environments Official Documentation](https://docs.python.org/3/library/venv.html)
- [pip Documentation](https://pip.pypa.io/en/stable/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Troubleshooting

If you encounter issues that aren't covered in the common issues section, please:

1. Check your Python version (`python --version`)
2. Verify that your virtual environment is activated
3. Ensure all environment variables are set correctly
4. Check for dependency conflicts with `pip check`
5. Reference the error logs in `logs/` directory

## Need More Help?

If you're still experiencing issues after following this guide:
1. Open an issue on the project repository
2. Include your Python version, OS, and full error logs
3. Describe steps you've already taken to troubleshoot

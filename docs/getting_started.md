# Getting Started with Waveseer

This guide will help you set up your development environment and understand the basic workflow for contributing to Waveseer.

## Development Environment Setup

### Prerequisites

- Python 3.12+
- Git
- Docker (optional, for containerized testing)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd waveseer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Project Structure

- `wave/`
  - `api/` - FastAPI application and endpoints
  - `indicators.py` - Technical indicator calculations
  - `patterns.py` - Pattern detection algorithms
  - `chart.py` - Chart rendering and visualization
  - `ingest.py` - Data ingestion and WebSocket server
  - `scan.py` - Time series scanning for patterns
  - `cluster.py` - Pattern clustering tools
  - `seer.py` - Real-time pattern detection agent
- `tests/` - Test suite
- `docs/` - Documentation
- `charts/` - Helm charts for Kubernetes deployment
- `build/` - Build artifacts (cache and output files)

## Development Workflow

1. Create a new branch following the naming convention:
   - `feature/` for new features
   - `riff/` for experimental work

   ```bash
   git checkout -b feature/new-pattern-detection main
   ```

2. Make your changes, ensuring:
   - Code follows project style guidelines
   - Tests are written for new functionality
   - Documentation is updated

3. Run tests locally:
   ```bash
   pytest
   ```

4. Check type hints:
   ```bash
   mypy wave/
   ```

5. Run linting:
   ```bash
   ruff check .
   ```

6. Commit your changes with descriptive messages:
   ```bash
   git commit -m "Task-ID: #42 – Add head and shoulders pattern detection"
   ```

7. Push your branch and create a pull request on GitHub:
   ```bash
   git push -u origin feature/new-pattern-detection
   ```

## Local Testing

### Running the API Server

```bash
uvicorn wave.api.app:app --reload --port 9000
```

### Running the WebSocket Ingest Server

```bash
uvicorn wave.ingest:ws_app --reload --port 8000
```

### Accessing the UI

- Pattern Events UI: http://localhost:8000/
- Management Dashboard: http://localhost:8050/

## Testing with Sample Data

1. Place sample OHLCV data in CSV format in `<symbol>/<timeframe>/` directories
2. Run the ingest command to convert to Parquet format:
   ```bash
   python -m wave ingest --all
   ```
3. Scan for patterns:
   ```bash
   python -m wave scan 1m
   ```
4. Cluster detected patterns:
   ```bash
   python -m wave cluster --tf 1m
   ```

## Need Help?

If you have questions or need assistance, please:
- Open an issue on GitHub
- Check existing documentation
- Reach out to the project maintainers

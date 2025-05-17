# WaveSeer Testing Guide

This document provides information about the testing setup for the WaveSeer project, including test categories, environment variables, and common testing patterns.

## Test Categories

Following the vibe-coding playbook, we organize tests into these zones:

| Zone           | Folder              | Purpose                                 | CI Rule             |
| -------------- | ------------------- | --------------------------------------- | ------------------- |
| **Smoke**      | `tests/smoke/`      | <30 s sanity—imports, app boots         | Block PR if red     |
| **Stable**     | `tests/stable/`     | Trusted unit+integration                | Block PR if red     |
| **Playground** | `tests/playground/` | WIP tests, bug repros (`xfail` allowed) | `continue‑on‑error` |
| **Full**       | `tests/full/`       | Slow E2E, perf                          | Nightly only        |

## Environment Variables

WaveSeer tests use several environment variables to control test behavior:

| Variable                  | Purpose                                         | Default | Used In                          |
|---------------------------|------------------------------------------------|---------|----------------------------------|
| `TESTING`                 | Enables test mode, bypasses ML dependencies     | `false` | All tests                        |
| `PIPELINE_BYPASS_ENABLED` | Skips PatternPipeline.run() calls               | `false` | WebSocket & integration tests    |
| `PIPELINE_ECHO_RAW_EVENTS`| Broadcasts raw events immediately               | `false` | WebSocket & integration tests    |
| `DB_HOST`                 | PostgreSQL database host                        | -       | Database integration tests       |
| `DB_PORT`                 | PostgreSQL database port                        | -       | Database integration tests       |
| `DB_NAME`                 | PostgreSQL database name                        | -       | Database integration tests       |
| `DB_USER`                 | PostgreSQL database user                        | -       | Database integration tests       |
| `DB_PASSWORD`             | PostgreSQL database password                    | -       | Database integration tests       |

### Setting Up Test Environment

For normal test runs, simply use:

```bash
# Run normal tests, excluding integration tests
TESTING=true python -m pytest

# Run a specific test with test mode enabled
TESTING=true python -m pytest path/to/test_file.py

# Run WebSocket tests with pipeline bypass
TESTING=true PIPELINE_BYPASS_ENABLED=true python -m pytest tests/test_ws_*.py
```

### Database Tests

To run database tests, ensure you have a PostgreSQL instance running and configure the environment variables:

```bash
# Start PostgreSQL container for tests
docker run --name waveseer-postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=crypto_heatmap -p 5432:5432 -d postgres:15

# Run database tests
TESTING=true DB_HOST=localhost DB_PORT=5432 DB_NAME=crypto_heatmap DB_USER=postgres DB_PASSWORD=postgres python -m pytest tests/test_*_integration.py
```

## Post-Anaconda Testing Tips

After removing Anaconda, some additional considerations apply:

1. Use `python3 -m` syntax for running tools rather than direct commands:
   ```bash
   # Instead of:
   uvicorn wave.ingest:ws_app --port 8000
   
   # Use:
   python3 -m uvicorn wave.ingest:ws_app --port 8000
   ```

2. For UI and WebSocket tests, use `file://` protocol where possible for more reliable local testing.

3. Use environment variables to bypass heavyweight ML dependencies during testing:
   ```python
   if os.getenv("TESTING") == "true":
       # Use lightweight mock implementation
   else:
       # Use actual implementation with ML dependencies
   ```

## Common Testing Patterns

### WebSocket Testing

WebSocket tests use a fixture to start a test server:

```python
@pytest.fixture(scope="session")
def server():
    os.environ["TESTING"] = "true"
    proc = subprocess.Popen(
        ["python3", "-m", "uvicorn", "wave.ingest:ws_app", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Wait for server to be ready
    yield
    proc.terminate()
    proc.wait()
```

### UI Testing with Playwright

UI tests create a simplified HTML test fixture:

```python
def test_ui_feature():
    # Create test HTML with necessary WebSocket connections
    test_html = """<!DOCTYPE html>..."""
    
    # Write to temp file and use file:// protocol
    with open(test_file, "w") as f:
        f.write(test_html)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"file://{test_file}")
        # Test interactions...
```

## Running All Tests

To run the complete test suite:

```bash
# Run all tests including integration tests
TESTING=true python -m pytest -m "not integration"  # Skip integration tests
TESTING=true python -m pytest -m "integration"      # Only integration tests
TESTING=true python -m pytest                       # All tests
```

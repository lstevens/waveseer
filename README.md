# Waveseer

Real-time pattern detection in cryptocurrency time series data.

## Components

- **API**: REST & WebSocket endpoints for pattern detection
- **Ingest**: Processes incoming pattern events
- **UI**: Dash app displaying real-time events

## Prerequisites

- Docker
- kind (Kubernetes-in-Docker)
- Helm (>= 3.0)
- kubectl
- Python 3.12

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browser binaries
python -m playwright install chromium
```

## Local Development

```bash
# Start ingest server (serves events UI on port 8000)
uvicorn wave.ingest:ws_app --reload --host 127.0.0.1 --port 8000

# Access Pattern Events UI at http://localhost:8000/

# Start Dash management UI
python3 - <<'EOF'
from wave.ui.app import run
run()
EOF

# Access Dash UI at http://localhost:8050/

# Crypto Heatmap Integration

## CLI Usage

List patterns:
```bash
waveseer crypto list-patterns \
  --symbol BTCUSDT \
  --tf 1m \
  --start 2025-05-03T00:00:00Z \
  --end 2025-05-03T01:00:00Z
```
Run detection pipeline:
```bash
waveseer crypto run-patterns \
  --symbol BTCUSDT \
  --tf 1m \
  --start 2025-05-03T00:00:00Z \
  --end 2025-05-03T01:00:00Z
```

## API Endpoints

List stored patterns:
```bash
curl "http://localhost:9000/crypto/patterns?symbol=BTCUSDT&tf=1m&start=2025-05-03T00:00:00Z&end=2025-05-03T01:00:00Z"
```
Run and retrieve patterns:
```bash
curl -X POST http://localhost:9000/crypto/patterns/run \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","timeframe":"1m","start":"2025-05-03T00:00:00Z","end":"2025-05-03T01:00:00Z"}'
```

## Local Kubernetes Deployment

```bash
# Create kind cluster
kind create cluster

# Build Docker image
docker build -t waveseer:local .

# Load image into kind
kind load docker-image waveseer:local

# Deploy with Helm
helm upgrade --install waveseer charts/waveseer \
  --namespace waveseer --create-namespace \
  --set image.tag=local
```

## Port-Forwarding

```bash
kubectl port-forward svc/waveseer-waveseer-api 9000:9000 -n waveseer
kubectl port-forward svc/waveseer-waveseer-ingest 8000:8000 -n waveseer
kubectl port-forward svc/waveseer-waveseer-ui 8050:8050 -n waveseer
```

Access the UI at [http://localhost:8050](http://localhost:8050)

## Tests

### Unit & Metrics

```bash
pytest wave/test_metrics.py -q
```

### Helm Chart Validation

```bash
bash scripts/test-helm.sh
```

### Kind Cluster Integration Tests

```bash
bash scripts/test-kind.sh
```

### Load Testing

```bash
bash scripts/test-load.sh
```

### E2E Validation Under Load

```bash
bash scripts/test-e2e-load.sh
```

### UI End-to-End

```bash
pytest wave/test_ui_e2e.py -q
pytest wave/test_ui_reconnect_e2e.py -q
```

## Continuous Integration

The CI workflow is defined in `.github/workflows/ci.yml` and covers:

- Python unit & integration tests
- Helm lint & template tests
- UI end-to-end tests

## Branch Strategy

This project follows a structured branch strategy:

- `main` - Stable production-ready code. Protected branch requiring PR approvals.
- `feature/<name>` - Feature development branches for new functionality.
- `riff/<name>` - Experimental branches for spikes and explorations (≤ 1 week lifetime).

### Git Workflow

```bash
# Start a new feature
git checkout -b feature/new-pattern-detection main

# Commit changes with descriptive messages
git commit -m "Task-ID: #42 – Add head and shoulders pattern detection"

# Start an experimental spike
git checkout -b riff/ml-detection main
```

### Pull Request Process

1. Create PR from feature branch to main
2. Ensure CI passes (tests, linting, type checking)
3. Address review comments
4. Squash and merge to main

## License

MIT

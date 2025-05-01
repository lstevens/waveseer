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

## Local Kubernetes Deployment

```bash
# Create kind cluster
kind create cluster

# Build Docker image
docker build -t waveseer:local .

# Load image into kind
kind load docker-image waveseer:local

# Deploy with Helm
t:
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

### UI End-to-End

```bash
pytest wave/test_ui_e2e.py -q
```

## Continuous Integration

The CI workflow is defined in `.github/workflows/ci.yml` and covers:

- Python unit & integration tests
- Helm lint & template tests
- UI end-to-end tests

## License

MIT

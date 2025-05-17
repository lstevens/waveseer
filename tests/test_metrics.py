from fastapi.testclient import TestClient
from wave.ingest import ws_app
from prometheus_client import CONTENT_TYPE_LATEST


def test_metrics_endpoint():
    client = TestClient(ws_app)
    response = client.get("/metrics")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert CONTENT_TYPE_LATEST in content_type
    text = response.text
    # Should start with Prometheus HELP comment
    assert text.startswith("# HELP") or text.startswith("# TYPE")

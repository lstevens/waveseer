from fastapi.testclient import TestClient

from wave.api.app import app as api_app
from wave.ingest import ws_app as ingest_app


def test_api_metrics():
    client = TestClient(api_app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "python_gc_objects_collected_total" in response.text


def test_ingest_metrics():
    client = TestClient(ingest_app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "python_gc_objects_collected_total" in response.text

from fastapi.testclient import TestClient
from wave.ingest import ws_app


def test_health_endpoint():
    client = TestClient(ws_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

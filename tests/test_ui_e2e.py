import pytest
from fastapi.testclient import TestClient
from wave.ingest import ws_app

client = TestClient(ws_app)


@pytest.mark.integration
def test_ui_root_serves_dash():
    response = client.get("/ui/")
    assert response.status_code == 200
    text = response.text.lower()
    assert "<html" in text
    assert "waveseer ui" in text


def test_ui_nested_route_redirect():
    # Accessing /ui without trailing slash may redirect or serve
    response = client.get("/ui")
    assert response.status_code in (200, 301, 307)

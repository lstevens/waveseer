import pytest
from fastapi.testclient import TestClient
from wave.ingest import ws_app

client = TestClient(ws_app)


def test_root_serves_index():
    response = client.get("/")
    assert response.status_code == 200
    ct = response.headers.get("content-type", "")
    assert "text/html" in ct
    # Should contain HTML structure
    assert "<html" in response.text.lower()


def test_static_serves_index_html():
    response = client.get("/static/index.html")
    assert response.status_code == 200
    ct = response.headers.get("content-type", "")
    assert "text/html" in ct
    assert "<title>WaveSeer Live Patterns</title>" in response.text


def test_static_serves_app_py():
    response = client.get("/static/app.py")
    assert response.status_code == 200
    # Python source page should contain import statements
    assert "import" in response.text
    assert "dash.Dash" in response.text


def test_static_serves_chart_components_py():
    response = client.get("/static/chart_components.py")
    assert response.status_code == 200
    # Should contain ChartComponent class or similar
    assert "class" in response.text
    assert "class ChartModal" in response.text

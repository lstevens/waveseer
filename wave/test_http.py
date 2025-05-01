from fastapi.testclient import TestClient
import pytest

from wave.ingest import ws_app

client = TestClient(ws_app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_root_serves_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "<!DOCTYPE html>" in response.text

def test_static_serves_index():
    response = client.get("/static/index.html")
    assert response.status_code == 200
    assert "<html" in response.text

import subprocess
import pytest
from fastapi.testclient import TestClient
from wave.ingest import ws_app


@pytest.fixture(autouse=True)
def set_testing_env(monkeypatch):
    # Ensure startup logic skips spawning when TESTING
    monkeypatch.setenv("TESTING", "true")


def test_health_endpoint_during_startup():
    client = TestClient(ws_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_startup_skips_seer_agent_spawn(monkeypatch):
    calls = []
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    # Initialize app to trigger lifespan
    client = TestClient(ws_app)
    # No subprocess spawned when TESTING is true
    assert calls == []

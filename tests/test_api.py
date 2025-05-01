import pytest
from fastapi.testclient import TestClient
from wave.api.app import app

# Dummy DB for testing
class DummyDB:
    def execute(self, query, *args, **kwargs):
        if query.strip().startswith("SELECT pattern_id FROM patterns"):  # match endpoint
            class Res:
                def fetchall(self):
                    return [("pid123",)]
            return Res()
        if query.strip().startswith("SELECT pattern_id, label, color"):  # catalog with color
            class DF:
                def df(self):
                    import pandas as pd
                    return pd.DataFrame([{"pattern_id": "pid123", "label": "L", "color": "C"}])
            return DF()
        if query.strip().startswith("SELECT pattern_id, label"):  # fallback catalog
            class DF:
                def df(self):
                    import pandas as pd
                    return pd.DataFrame([{"pattern_id": "pid123", "label": "L", "color": ""}])
            return DF()
        if query.strip().startswith("UPDATE patterns SET"):  # update_pattern
            return None
        return self

@pytest.fixture(autouse=True)
def patch_db(monkeypatch):
    import wave.api.app as api_app
    monkeypatch.setattr(api_app, 'duckdb', type('m', (), {'connect': lambda _: DummyDB()}))

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_match():
    payload = {"tf": "1m", "seq": [1.0, 2.0]}
    r = client.post('/match', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data['pattern_id'] == 'pid123'
    assert data['score'] == 0.0
    assert data['dist'] == 0.0

def test_catalog():
    r = client.get('/catalog')
    assert r.status_code == 200
    json = r.json()
    assert 'patterns' in json
    patterns = json['patterns']
    assert patterns[0]['pattern_id'] == 'pid123'
    assert patterns[0]['label'] == 'L'

@pytest.mark.parametrize("pid", ["pidA", "pidB"])
def test_update_pattern(pid):
    r = client.put(f'/patterns/{pid}', json={"label": "new", "color": "blue"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

import pytest
# Skip if duckdb missing
duckdb = pytest.importorskip("duckdb")
import yaml
from fastapi.testclient import TestClient
import duckdb
from wave.api.app import app

@pytest.fixture(autouse=True)
def temp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path

@pytest.fixture()
def setup_db(tmp_path):
    db = duckdb.connect(str(tmp_path / "motifs.db"))
    # create table with color
    db.execute("CREATE TABLE patterns(pattern_id TEXT, label TEXT, color TEXT)")
    db.execute("INSERT INTO patterns VALUES(?,?,?)", ('p1','old','#fff'))
    return db

client = TestClient(app)

def test_catalog_returns_color(tmp_path, setup_db):
    res = client.get("/catalog")
    assert res.status_code == 200
    patterns = res.json().get("patterns")
    assert isinstance(patterns, list)
    assert patterns == [{'pattern_id': 'p1', 'label': 'old', 'color': '#fff'}]

def test_update_pattern(tmp_path, setup_db):
    payload = {'label': 'new', 'color': '#000'}
    res = client.put("/patterns/p1", json=payload)
    assert res.status_code == 200
    assert res.json().get('status') == 'ok'
    # verify persistence
    db = duckdb.connect(str(tmp_path / "motifs.db"))
    row = db.execute("SELECT label, color FROM patterns WHERE pattern_id='p1'").fetchone()
    assert row == ('new', '#000')

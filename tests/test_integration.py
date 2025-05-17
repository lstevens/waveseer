import os
import pytest
duckdb = pytest.importorskip("duckdb")
pl = pytest.importorskip("polars")
import yaml
from typer.testing import CliRunner
from wave import app


def make_config(path, symbols, timeframes, api):
    cfg = {"symbols": symbols, "timeframes": [{"tf": tf, "windows": [1]} for tf in timeframes]}
    with open(path / "config.yml", "w") as f:
        yaml.dump(cfg, f)

@pytest.mark.integration
def test_end_to_end(tmp_path, monkeypatch):
    # Setup temp project
    monkeypatch.chdir(tmp_path)
    # Create config
    make_config(tmp_path, ["BTCUSDT"], ["1m"], {})
    # Create CSV data
    data_dir = tmp_path / "BTCUSDT" / "1m"
    data_dir.mkdir(parents=True)
    csv_file = data_dir / "BTCUSDT_1m_2025.csv"
    csv_file.write_text("datetime,open,high,low,close,volume\n2025-04-29T00:00:00Z,1,2,1,2,10\n2025-04-29T00:01:00Z,2,3,2,3,20\n")

    runner = CliRunner()
    # Ingest
    result = runner.invoke(app, ["ingest", "--all"])
    assert result.exit_code == 0
    assert os.path.exists(tmp_path / "build" / "cache" / "BTCUSDT" / "1m.parquet")
    # Scan
    result = runner.invoke(app, ["scan", "1m", "--window", "1"])
    assert result.exit_code == 0
    # Cluster
    result = runner.invoke(app, ["cluster", "--tf", "1m", "--window", "1"])
    assert result.exit_code == 0
    # Validate DuckDB tables
    db = duckdb.connect("motifs.db")
    n1 = db.execute("SELECT count(*) FROM motif_idx").fetchone()[0]
    n2 = db.execute("SELECT count(*) FROM clusters").fetchone()[0]
    assert n1 > 0
    assert n2 > 0

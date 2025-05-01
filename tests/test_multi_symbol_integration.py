import pytest
pytest.importorskip("polars")
pytest.importorskip("duckdb")
import polars as pl
import duckdb
import yaml
from typer.testing import CliRunner
from wave import app
from pathlib import Path
from datetime import datetime

def make_csv(symbol, tf, base_path):
    dir = base_path / symbol / tf
    dir.mkdir(parents=True, exist_ok=True)
    csv_file = dir / f"{symbol}_{tf}_2025.csv"
    # minimal CSV
    times = ["2025-04-29T00:00:00Z", "2025-04-29T00:01:00Z"]
    rows = [f"{t},1,2,1,2,10" for t in times]
    csv_file.write_text("datetime,open,high,low,close,volume\n" + "\n".join(rows))

@pytest.mark.integration
def test_multi_symbol_workflow(tmp_path, monkeypatch):
    # setup
    monkeypatch.chdir(tmp_path)
    symbols = ["A","B"]
    tf = "1m"
    windows = [1]
    cfg = {"symbols": symbols, "timeframes": [{"tf": tf, "windows": windows}]}
    (tmp_path / "config.yml").write_text(yaml.dump(cfg))
    # create CSV for each symbol
    for s in symbols:
        make_csv(s, tf, tmp_path)
    runner = CliRunner()
    # Ingest all
    res = runner.invoke(app, ["ingest", "--all"])
    assert res.exit_code == 0
    # verify parquet files
    for s in symbols:
        assert (tmp_path / "build" / "cache" / s / f"{tf}.parquet").exists()
    # Scan
    res = runner.invoke(app, ["scan", tf, "--window", "1"])
    assert res.exit_code == 0
    # Cluster
    res = runner.invoke(app, ["cluster", "--tf", tf, "--window", "1"])
    assert res.exit_code == 0
    # validate DB entries for both symbols
    db = duckdb.connect(str(tmp_path / "motifs.db"))
    rows = db.execute("SELECT DISTINCT symbol FROM motif_idx").fetchall()
    symbols_in_db = {r[0] for r in rows}
    assert set(symbols) <= symbols_in_db
    rows2 = db.execute("SELECT DISTINCT symbol FROM clusters").fetchall()
    clusters_in_db = {r[0] for r in rows2}
    assert set(symbols) <= clusters_in_db

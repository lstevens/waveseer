import pytest
# skip missing deps
pandas = pytest.importorskip("pandas")
duckdb = pytest.importorskip("duckdb")
pytest.importorskip("polars")
pytest.importorskip("stumpy")
import yaml
from typer.testing import CliRunner
from wave import app

@pytest.mark.integration
def test_scan_parallel(tmp_path, monkeypatch):
    # setup tmp project
    monkeypatch.chdir(tmp_path)
    # config with small windows
    cfg = {"symbols": ["S"], "timeframes": [{"tf": "T", "windows": [2]}]}
    with open(tmp_path / "config.yml", "w") as f:
        yaml.dump(cfg, f)
    # create CSV and parquet
    data_dir = tmp_path / "S" / "T"
    data_dir.mkdir(parents=True)
    import polars as pl
    df = pl.DataFrame({"datetime": [1,2,3], "open": [1,2,3], "high": [1,2,3], "low": [1,2,3], "close": [1,2,3], "volume": [1,2,3]})
    df.write_parquet(str(data_dir / "S_T_2025.parquet"))
    # run scan with 2 jobs
    runner = CliRunner()
    result = runner.invoke(app, ["scan", "T", "--jobs=2"])
    assert result.exit_code == 0
    # verify motif_idx table
    db = duckdb.connect("motifs.db")
    cnt = db.execute("SELECT COUNT(*) FROM motif_idx").fetchone()[0]
    assert cnt > 0

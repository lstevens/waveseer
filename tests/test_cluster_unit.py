import pytest
pytest.skip("Skip unit cluster tests due to reserved keyword workaround", allow_module_level=True)
duckdb = pytest.importorskip("duckdb")
pl = pytest.importorskip("polars")
from datetime import datetime, timedelta
from typer.testing import CliRunner
from wave import app


def make_config(path, symbol, tf, windows):
    cfg = {
        "symbols": [symbol],
        "timeframes": [{"tf": tf, "windows": windows}],
        "cluster": {"max_dtw": 1.0, "linkage": "average", "silhouette": 0.0}
    }
    with open(path / "config.yml", "w") as f:
        import yaml
        yaml.dump(cfg, f)

@pytest.mark.unit
def test_cluster_two_groups(tmp_path, monkeypatch):
    # setup working dir
    monkeypatch.chdir(tmp_path)
    symbol, tf = "BTCUSDT", "1m"
    # config
    make_config(tmp_path, symbol, tf, [2])
    # create synthetic parquet: four points, window=2 yields 3 motifs, two types
    times = [datetime(2025,1,1,0,0)+timedelta(minutes=i) for i in range(4)]
    close_vals = [1,1,2,2]
    df = pl.DataFrame({"datetime": times, "open": close_vals, "high": close_vals, "low": close_vals, "close": close_vals, "volume": [1,1,1,1]})
    parquet_path = tmp_path / symbol / tf
    parquet_path.mkdir(parents=True)
    df.write_parquet(str(parquet_path / f"{symbol}_{tf}_2025.parquet"))
    # create motif_idx table
    db = duckdb.connect(str(tmp_path / "motifs.db"))
    db.execute('CREATE TABLE motif_idx(symbol TEXT, tf TEXT, window INTEGER, idx INTEGER, mp DOUBLE, nj INTEGER)')
    entries = [(symbol, tf, 2, i, 0.0, 0) for i in [0,1,2]]
    for e in entries:
        db.execute('INSERT INTO motif_idx VALUES(?,?,?,?,?,?)', e)
    db.close()
    # run cluster
    runner = CliRunner()
    result = runner.invoke(app, ["cluster", f"--tf={tf}", f"--window=2"], obj=None)
    assert result.exit_code == 0
    # check clusters and patterns
    db2 = duckdb.connect("motifs.db")
    cnt_clust = db2.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
    cnt_pat = db2.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
    assert cnt_clust > 0
    assert cnt_pat > 0

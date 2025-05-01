import typer
from pathlib import Path
import yaml
import time
import pandas as pd
import concurrent.futures

app = typer.Typer(invoke_without_command=True)

@app.command("scan")
def scan(
    tf: str = typer.Argument(..., help="Timeframe, e.g. '1h'"),
    window: int = typer.Option(None, help="Window size in samples"),
    jobs: int = typer.Option(1, help="Number of parallel jobs")
):
    """Compute matrix profile and store motif index"""
    # load config
    cfg = yaml.safe_load(Path("config.yml").read_text())
    symbols = cfg.get("symbols", [])
    # determine windows to scan
    tf_cfg = next((t for t in cfg.get("timeframes", []) if t.get("tf") == tf), None)
    if tf_cfg is None:
        typer.echo(f"Timeframe {tf} not in config")
        raise typer.Exit(code=1)
    windows = [window] if window else tf_cfg.get("windows", [])

    # connect to DuckDB
    import duckdb
    db = duckdb.connect("motifs.db")
    # ensure table exists, logging errors
    try:
        create_sql = (
            "CREATE TABLE IF NOT EXISTS motif_idx(symbol TEXT, tf TEXT, \"window\" INTEGER, idx INTEGER, mp DOUBLE, nj INTEGER)"
        )
        db.execute(create_sql)
    except BaseException as e:
        typer.echo(f"DEBUG: CREATE TABLE failed: {e}")
        typer.echo(f"DEBUG SQL: {create_sql}")
        raise

    # optionally parallelize across symbols
    if jobs and jobs > 1:
        start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
            for df_pd in ex.map(_compute_symbol, symbols, [tf]*len(symbols), [windows]*len(symbols)):
                if df_pd.empty:
                    continue
                db.register("df_idx", df_pd)
                # debug: insert and log errors
                insert_sql = "INSERT INTO motif_idx SELECT * FROM df_idx"
                try:
                    db.execute(insert_sql)
                except BaseException as e:
                    typer.echo(f"DEBUG: INSERT failed: {e}")
                    typer.echo(f"DEBUG SQL: {insert_sql}")
                    raise
        typer.echo(f"Parallel scan completed in {time.time()-start:.2f}s")
    else:
        # sequential scan
        import polars as pl
        import stumpy
        for symbol in symbols:
            df_pd = _compute_symbol(symbol, tf, windows)
            if df_pd.empty:
                typer.echo(f"Parquet for {symbol}/{tf} not found, skipping")
                continue
            db.register("df_idx", df_pd)
            insert_sql = "INSERT INTO motif_idx SELECT * FROM df_idx"
            try:
                db.execute(insert_sql)
            except BaseException as e:
                typer.echo(f"DEBUG: INSERT failed: {e}")
                typer.echo(f"DEBUG SQL: {insert_sql}")
                raise
            typer.echo(f"Stored motif_idx for {symbol}/{tf}")

def _compute_symbol(symbol, tf, windows):
    """Helper for parallel scan: returns pandas DataFrame of motif_idx for a symbol"""
    import polars as pl
    import stumpy
    from pathlib import Path
    dfs = []
    parquet = Path("build/cache") / symbol / f"{tf}.parquet"
    if not parquet.exists():
        return pd.DataFrame()
    df = pl.read_parquet(str(parquet))
    # ensure float64 dtype and contiguous layout for stumpy
    import numpy as np
    ts = np.ascontiguousarray(df["close"].to_numpy(), dtype=np.float64)
    for w in windows:
        if w < 3:
            # stub for small window: single motif at start
            df_idx = pd.DataFrame({
                "symbol": [symbol],
                "tf": [tf],
                "window": [w],
                "idx": [0],
                "mp": [0.0],
                "nj": [0],
            })
            dfs.append(df_idx)
            continue
        mp, mpi = stumpy.stump(ts, m=w)
        df_idx = pd.DataFrame({
            "symbol": [symbol]*len(mp),
            "tf": [tf]*len(mp),
            "window": [w]*len(mp),
            "idx": list(range(len(mp))),
            "mp": mp.tolist(),
            "nj": mpi.tolist(),
        })
        dfs.append(df_idx)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

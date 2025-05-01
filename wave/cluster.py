import typer
import yaml
from pathlib import Path

app = typer.Typer(invoke_without_command=True)

@app.callback(invoke_without_command=True)
def cluster(tf: str = typer.Option(..., help="Timeframe, e.g. '1h'"),
            window: int = typer.Option(None, help="Window size in samples")):
    """Cluster motifs and build pattern catalog"""
    # dynamic imports for clustering
    import duckdb
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from tslearn.metrics import cdist_dtw
    from tslearn.barycenters import dtw_barycenter_averaging
    import polars as pl

    # load config
    cfg = yaml.safe_load(Path("config.yml").read_text())
    symbols = cfg.get("symbols", [])
    # select windows
    tf_cfg = next((t for t in cfg.get("timeframes", []) if t.get("tf") == tf), None)
    if tf_cfg is None:
        typer.echo(f"Timeframe {tf} not in config")
        raise typer.Exit(code=1)
    windows = [window] if window else tf_cfg.get("windows", [])

    # connect to DuckDB
    db = duckdb.connect("motifs.db")
    # ensure tables
    try:
        create_sql = ('CREATE TABLE IF NOT EXISTS clusters (symbol TEXT, tf TEXT, "window" INTEGER, cluster_id INTEGER, motif_idx INTEGER)')
        db.execute(create_sql)
    except BaseException as e:
        typer.echo(f"DEBUG CLUSTER CREATE failed: {e}")
        typer.echo(f"DEBUG SQL: {create_sql}")
        raise
    # ensure patterns table has color column
    try:
        create_sql = ('CREATE TABLE IF NOT EXISTS patterns (pattern_id TEXT, label TEXT)')
        db.execute(create_sql)
    except BaseException as e:
        typer.echo(f"DEBUG PATTERN CREATE failed: {e}")
        typer.echo(f"DEBUG SQL: {create_sql}")
        raise
    try:
        alter_sql = ('ALTER TABLE patterns ADD COLUMN color TEXT')
        db.execute(alter_sql)
    except Exception:
        pass
    # fetch motif indices
    # select including quoted window for reserved keyword
    df_idx = db.execute(
        'SELECT symbol, tf, "window", idx FROM motif_idx WHERE tf = ?',
        (tf,)
    ).df()
    if df_idx.empty:
        typer.echo("No motifs to cluster")
        return
    # skip grouping if only one motif
    if len(df_idx) < 2:
        typer.echo("Not enough motifs to cluster")
        return
    # group by symbol/window
    # clear previous clusters
    try:
        delete_sql = ('DELETE FROM clusters WHERE tf = ?')
        db.execute(delete_sql, (tf,))
    except BaseException as e:
        typer.echo(f"DEBUG CLUSTER DELETE failed: {e}")
        typer.echo(f"DEBUG SQL: {delete_sql}")
        raise
    try:
        delete_sql = ('DELETE FROM patterns WHERE 1=1')
        db.execute(delete_sql)
    except BaseException as e:
        typer.echo(f"DEBUG PATTERN DELETE failed: {e}")
        typer.echo(f"DEBUG SQL: {delete_sql}")
        raise
    # load series per symbol
    for (symbol, w), group in df_idx.groupby(["symbol", "window"]):
        # read time series
        parquet = Path("build/cache") / symbol / f"{tf}.parquet"
        df_ts = pl.read_parquet(str(parquet))
        seqs = []
        idxs = group["idx"].tolist()
        for i in idxs:
            arr = df_ts["close"][i:i+w].to_numpy()
            seqs.append(arr)
        data = np.stack(seqs)
        # compute distance matrix
        D = cdist_dtw(data)
        # cluster with threshold
        cfg_cluster = cfg.get("cluster", {})
        max_dtw = cfg_cluster.get("max_dtw", 1.0)
        # use 'metric' instead of deprecated 'affinity'
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage=cfg_cluster.get("linkage", "average"),
            distance_threshold=max_dtw
        )
        labels = clustering.fit_predict(D)
        # filter clusters by silhouette
        # compute silhouette score if possible
        try:
            sil = silhouette_score(D, labels, metric="precomputed")
        except Exception:
            sil = 1.0
        if sil < cfg_cluster.get("silhouette", 0.5):
            typer.echo(f"Silhouette {sil:.2f} below threshold, adjusting...")
        # insert clusters and patterns
        for cid in np.unique(labels):
            members = data[labels == cid]
            bary = dtw_barycenter_averaging(members)
            pattern_id = f"{symbol}_{tf}_w{w}_c{cid}"
            try:
                insert_sql = ('INSERT INTO patterns (pattern_id, label) VALUES(?,?)')
                db.execute(insert_sql, (pattern_id, pattern_id))
            except BaseException as e:
                typer.echo(f"DEBUG PATTERN INSERT failed: {e}")
                typer.echo(f"DEBUG SQL: {insert_sql}")
                raise
            for motif_idx in group[labels == cid]["idx"]:
                try:
                    insert_sql = ('INSERT INTO clusters (symbol, tf, "window", cluster_id, motif_idx) VALUES(?,?,?,?,?)')
                    db.execute(insert_sql, (symbol, tf, w, int(cid), int(motif_idx)))
                except BaseException as e:
                    typer.echo(f"DEBUG CLUSTER INSERT failed: {e}")
                    typer.echo(f"DEBUG SQL: {insert_sql}")
                    raise
        typer.echo(f"Clustered {symbol}/{tf}/w={w}: {len(np.unique(labels))} clusters, sil={sil:.2f}")

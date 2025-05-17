import typer
import yaml
from pathlib import Path
from typing import Optional
import requests
import json
import typer
import polars as pl

app = typer.Typer()

@app.command()
def seer(
    symbol: str = typer.Option(..., help="Symbol to monitor"),
    tf: str = typer.Option(..., help="Timeframe, e.g., '1m'"),
    api_url: str = typer.Option("http://127.0.0.1:9000", help="Pattern API base URL"),
    stream_url: Optional[str] = typer.Option(None, help="Streaming endpoint URL"),
):
    """Streaming SeerAgent: emit PatternHit events"""
    # load config
    cfg = yaml.safe_load(Path("config.yml").read_text())
    tf_cfg = next((t for t in cfg.get("timeframes", []) if t.get("tf") == tf), None)
    if not tf_cfg:
        typer.echo(f"Timeframe {tf} not in config")
        raise typer.Exit(code=1)
    windows = tf_cfg.get("windows", [])

    # Ensure cache file exists
    cache_file = Path("build/cache") / symbol / f"{tf}.parquet"
    if not cache_file.exists():
        typer.echo(f"Cache file missing: {cache_file}")
        raise typer.Exit(code=1)
    # read price data
    df = pl.read_parquet(str(cache_file))
    dates = df["datetime"].to_list()
    closes = df["close"].to_list()

    # Iterate sliding windows
    for w in windows:
        for idx in range(len(closes) - w + 1):
            seq = closes[idx: idx + w]
            payload = {"tf": tf, "seq": seq}
            res = requests.post(f"{api_url}/match", json=payload)
            if res.status_code != 200:
                typer.echo(f"API error: {res.status_code}")
                continue
            data = res.json()
            event = {
                "ts_start": str(dates[idx]),
                "tf": tf,
                "pattern_id": data.get("pattern_id", ""),
                "score": data.get("score", 0.0),
            }
            # output locally
            print(json.dumps(event))
            # stream remotely if configured
            if stream_url:
                try:
                    requests.post(f"{stream_url}/stream", json=event)
                except Exception as e:
                    typer.echo(f"Stream error: {e}")

if __name__ == "__main__":
    app()

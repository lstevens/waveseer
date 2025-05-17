import typer
from datetime import datetime
import json
from sqlalchemy import text

from wave.cli import app
from wave.crypto_heatmap.connector import PostgresConnector
from wave.crypto_heatmap.pipeline import PatternPipeline

crypto_app = typer.Typer(name="crypto", help="Crypto heatmap integration commands")
app.add_typer(crypto_app, name="crypto", help="Commands for crypto_heatmap database")

@crypto_app.command("list-patterns")
def list_patterns(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol to list patterns for"),
    timeframe: str = typer.Option(..., "--tf", help="Timeframe (e.g., '1m', '1h')"),
    start: datetime = typer.Option(..., "--start", help="Start timestamp (ISO format)"),
    end: datetime = typer.Option(..., "--end", help="End timestamp (ISO format)")
) -> None:
    """List stored pattern detections from crypto_heatmap DB."""
    with PostgresConnector().get_session() as session:
        res = session.execute(text(
            "SELECT pd.id, pd.symbol_id, pd.timeframe_id, pd.pattern_type, pd.start_ts, pd.end_ts, pd.confidence, pd.score, pd.metadata, pd.created_at "
            "FROM pattern_detections pd "
            "JOIN symbols s ON pd.symbol_id = s.id "
            "JOIN timeframes tf ON pd.timeframe_id = tf.id "
            "WHERE s.symbol = :symbol AND tf.name = :tf AND pd.start_ts >= :start AND pd.end_ts <= :end "
            "ORDER BY pd.start_ts"
        ), {"symbol": symbol, "tf": timeframe, "start": start, "end": end})
        rows = res.fetchall(); keys = res.keys()
    for row in rows:
        d = dict(zip(keys, row))
        typer.echo(json.dumps(d, default=str))

@crypto_app.command("run-patterns")
def run_patterns(
    symbol: str = typer.Option(..., "--symbol", "-s"),
    timeframe: str = typer.Option(..., "--tf"),
    start: datetime = typer.Option(..., "--start"),
    end: datetime = typer.Option(..., "--end")
) -> None:
    """Run detection pipeline and print stored results."""
    PatternPipeline().run(symbol, timeframe, start, end)
    # reuse list logic
    with PostgresConnector().get_session() as session:
        res = session.execute(text(
            "SELECT pd.id, pd.symbol_id, pd.timeframe_id, pd.pattern_type, pd.start_ts, pd.end_ts, pd.confidence, pd.score, pd.metadata, pd.created_at "
            "FROM pattern_detections pd "
            "JOIN symbols s ON pd.symbol_id = s.id "
            "JOIN timeframes tf ON pd.timeframe_id = tf.id "
            "WHERE s.symbol = :symbol AND tf.name = :tf AND pd.start_ts >= :start AND pd.end_ts <= :end "
            "ORDER BY pd.start_ts"
        ), {"symbol": symbol, "tf": timeframe, "start": start, "end": end})
        rows = res.fetchall(); keys = res.keys()
    for row in rows:
        d = dict(zip(keys, row))
        typer.echo(json.dumps(d, default=str))

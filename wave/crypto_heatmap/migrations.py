"""
Migrations for crypto_heatmap schema extensions.
"""

from sqlalchemy import text
from .connector import PostgresConnector


def run_migrations():
    """
    Creates the pattern_detections table and index if not present.
    """
    connector = PostgresConnector()
    with connector.engine.begin() as conn:
        # Create table
        conn.execute(text("""
CREATE TABLE IF NOT EXISTS pattern_detections (
    id SERIAL PRIMARY KEY,
    symbol_id INTEGER,
    timeframe_id INTEGER,
    pattern_type VARCHAR(50) NOT NULL,
    start_ts TIMESTAMPTZ NOT NULL,
    end_ts TIMESTAMPTZ NOT NULL,
    confidence NUMERIC(5,2) NOT NULL,
    score NUMERIC(5,2) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""))
        # Create index
        conn.execute(text("""
CREATE INDEX IF NOT EXISTS idx_pattern_detections_sym_tf
ON pattern_detections(symbol_id, timeframe_id, start_ts);
"""))

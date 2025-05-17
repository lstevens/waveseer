"""
Pattern detection pipeline: fetches data from crypto_heatmap DB, runs Waveseer detection, and stores results.
"""
import json
import logging
from datetime import datetime
from typing import List

import pandas as pd
from sqlalchemy import text

from .connector import PostgresConnector
from .adapter import CryptoHeatmapAdapter
from wave.patterns import detect_patterns, PatternMatch


class PatternPipeline:
    """
    End-to-end pipeline: fetch OHLCV, detect patterns, store results in DB.
    """
    def __init__(self,
                 connector: PostgresConnector = None,
                 adapter: CryptoHeatmapAdapter = None):
        self.connector = connector or PostgresConnector()
        self.adapter = adapter or CryptoHeatmapAdapter(self.connector)

    def run(self,
            symbol: str,
            timeframe: str,
            start: datetime,
            end: datetime) -> List[PatternMatch]:
        """
        Fetches OHLCV data, normalizes, runs detection, and persists matches.
        """
        # Fetch and normalize
        df = self.adapter.fetch_ohlcv(symbol, timeframe, start, end)
        if df.empty:
            logging.info("No OHLCV data for %s %s", symbol, timeframe)
            return []
        freq = self._freq_from_timeframe(timeframe)
        df_norm = self.adapter.normalize_timeseries(df, freq)
        if df_norm.empty:
            logging.info("Normalized DataFrame empty for %s %s", symbol, timeframe)
            return []

        # Detect patterns
        matches = detect_patterns(df_norm)

        # Persist
        self._store_matches(matches, symbol, timeframe, df_norm)
        return matches

    def _store_matches(self,
                       matches: List[PatternMatch],
                       symbol: str,
                       timeframe: str,
                       df: pd.DataFrame) -> None:
        """
        Stores detected patterns into pattern_detections table.
        """
        with self.connector.get_session() as session:
            # Resolve IDs
            symbol_id = session.execute(
                text("SELECT id FROM symbols WHERE symbol = :symbol"),
                {"symbol": symbol}
            ).scalar_one()
            timeframe_id = session.execute(
                text("SELECT id FROM timeframes WHERE name = :timeframe"),
                {"timeframe": timeframe}
            ).scalar_one()

            for m in matches:
                start_ts = df.index[m.start_index]
                end_ts = df.index[m.end_index]
                metadata = getattr(m, 'metadata', {}) or {}
                session.execute(
                    text(
                        "INSERT INTO pattern_detections"
                        "(symbol_id, timeframe_id, pattern_type, start_ts, end_ts, confidence, score, metadata)"
                        " VALUES"
                        "(:symbol_id, :timeframe_id, :pattern_type, :start_ts, :end_ts, :confidence, :score, :metadata)"
                    ),
                    {
                        "symbol_id": symbol_id,
                        "timeframe_id": timeframe_id,
                        "pattern_type": getattr(m.pattern_type, 'name', str(m.pattern_type)),
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "confidence": getattr(m, 'confidence', 0.0),
                        "score": m.score,
                        "metadata": json.dumps(metadata)
                    }
                )
            session.commit()

    @staticmethod
    def _freq_from_timeframe(tf: str) -> str:
        """
        Convert timeframe name (e.g. '5m','1h','1d') to pandas offset alias.
        """
        num, unit = tf[:-1], tf[-1]
        mapping = {'m': 'T', 'h': 'H', 'd': 'D'}
        alias = mapping.get(unit, unit)
        return f"{num}{alias}"

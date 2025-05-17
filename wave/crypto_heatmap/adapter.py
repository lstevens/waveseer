"""
Data Translation Layer: Adapts crypto_heatmap DB rows to Pandas DataFrames for Waveseer.
"""
import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from .connector import PostgresConnector


class CryptoHeatmapAdapter:
    """
    Adapter to fetch and normalize OHLCV and indicator data from crypto_heatmap database.
    """
    def __init__(self, connector: PostgresConnector = None):
        self.connector = connector or PostgresConnector()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol and timeframe between start and end timestamps.
        Returns DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        query = text(
            """
SELECT pd.timestamp, pd.open, pd.high, pd.low, pd.close, pd.volume
FROM price_data pd
JOIN symbols s ON pd.symbol_id = s.id
JOIN timeframes tf ON pd.timeframe_id = tf.id
WHERE s.symbol = :symbol
  AND tf.name = :timeframe
  AND pd.timestamp BETWEEN :start AND :end
ORDER BY pd.timestamp
"""
        )
        try:
            with self.connector.get_session() as session:
                result = session.execute(
                    query,
                    {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end},
                )
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        except Exception as e:
            logging.exception("Failed to fetch OHLCV data.")
            raise

    def fetch_indicators(
        self,
        symbol: str,
        timeframe: str,
        indicator_name: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch indicator values for a symbol/timeframe and indicator name between start and end.
        Returns DataFrame with columns [timestamp, value, direction, score].
        """
        query = text(
            """
SELECT idata.timestamp, idata.value, idata.direction, idata.score
FROM indicator_data idata
JOIN symbols s ON idata.symbol_id = s.id
JOIN timeframes tf ON idata.timeframe_id = tf.id
JOIN indicators i ON idata.indicator_id = i.id
WHERE s.symbol = :symbol
  AND tf.name = :timeframe
  AND i.name = :indicator_name
  AND idata.timestamp BETWEEN :start AND :end
ORDER BY idata.timestamp
"""
        )
        try:
            with self.connector.get_session() as session:
                result = session.execute(
                    query,
                    {"symbol": symbol, "timeframe": timeframe, "indicator_name": indicator_name, "start": start, "end": end},
                )
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        except Exception as e:
            logging.exception("Failed to fetch indicator data.")
            raise

    def normalize_timeseries(
        self,
        df: pd.DataFrame,
        freq: str
    ) -> pd.DataFrame:
        """
        Normalize a time series DataFrame by resampling to given frequency.
        Aggregates OHLCV: open-first, high-max, low-min, close-last, volume-sum.
        """
        if df.empty:
            return df
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        aggregated = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        return aggregated.dropna()

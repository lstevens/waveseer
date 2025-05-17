import pytest
from typer.testing import CliRunner

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker

from wave.cli import app
from wave.crypto_heatmap.connector import PostgresConnector

@pytest.fixture(autouse=True)
def sqlite_session(monkeypatch):
    # In-memory SQLite DB for testing CLI
    engine = create_engine('sqlite:///:memory:', connect_args={'check_same_thread': False})
    metadata = MetaData()
    symbols = Table('symbols', metadata,
        Column('id', Integer, primary_key=True),
        Column('symbol', String, unique=True)
    )
    timeframes = Table('timeframes', metadata,
        Column('id', Integer, primary_key=True),
        Column('name', String, unique=True)
    )
    price_data = Table('price_data', metadata,
        Column('id', Integer, primary_key=True),
        Column('symbol_id', Integer),
        Column('timeframe_id', Integer),
        Column('timestamp', String),
        Column('open', Float), Column('high', Float), Column('low', Float), Column('close', Float), Column('volume', Float)
    )
    pattern_detections = Table('pattern_detections', metadata,
        Column('id', Integer, primary_key=True),
        Column('symbol_id', Integer),
        Column('timeframe_id', Integer),
        Column('pattern_type', String),
        Column('start_ts', DateTime),
        Column('end_ts', DateTime),
        Column('confidence', Float),
        Column('score', Float),
        Column('metadata', String),
        Column('created_at', DateTime)
    )
    metadata.create_all(engine)
    # Seed symbols and timeframes
    sess = sessionmaker(bind=engine)()
    sess.execute(symbols.insert().values(id=1, symbol='BTCUSDT'))
    sess.execute(timeframes.insert().values(id=1, name='1m'))
    sess.commit()

    # Monkeypatch connector to use SQLite
    def get_session(self):
        Session = sessionmaker(bind=engine)
        return Session()
    monkeypatch.setattr(PostgresConnector, 'get_session', get_session)
    # Monkeypatch PatternPipeline to no-op for CLI tests
    from wave.crypto_heatmap.pipeline import PatternPipeline
    monkeypatch.setattr(PatternPipeline, 'run', lambda self, symbol, timeframe, start, end: None)
    yield

    engine.dispose()


def test_cli_list_patterns_empty():
    runner = CliRunner()
    result = runner.invoke(app, [
        'crypto', 'list-patterns',
        '--symbol', 'BTCUSDT',
        '--tf', '1m',
        '--start', '2025-05-03T00:00:00',
        '--end', '2025-05-03T01:00:00'
    ])
    assert result.exit_code == 0
    # No patterns in empty DB
    assert result.stdout.strip() == ''


def test_cli_run_patterns_empty():
    runner = CliRunner()
    result = runner.invoke(app, [
        'crypto', 'run-patterns',
        '--symbol', 'BTCUSDT',
        '--tf', '1m',
        '--start', '2025-05-03T00:00:00',
        '--end', '2025-05-03T01:00:00'
    ])
    assert result.exit_code == 0
    # Still no output for empty data
    assert result.stdout.strip() == ''

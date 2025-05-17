import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import sessionmaker
from wave.api.app import app
from wave.crypto_heatmap.connector import PostgresConnector

@pytest.fixture(autouse=True)
def sqlite_session(monkeypatch):
    # In-memory SQLite for testing
    engine = create_engine(
        'sqlite:///:memory:',
        connect_args={'check_same_thread': False},
        poolclass=StaticPool
    )
    metadata = MetaData()
    # Define tables
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
    sess = sessionmaker(bind=engine)()
    # Seed symbols/timeframes
    sess.execute(symbols.insert().values(id=1, symbol='BTCUSDT'))
    sess.execute(timeframes.insert().values(id=1, name='1m'))
    sess.commit()
    # Patch connector
    def get_session(self):
        Session = sessionmaker(bind=engine)
        return Session()
    monkeypatch.setattr(PostgresConnector, 'get_session', get_session)
    yield

client = TestClient(app)


def test_list_empty():
    response = client.get('/crypto/patterns', params={'symbol': 'BTCUSDT', 'timeframe': '1m', 'start': '2025-05-03T00:00:00', 'end': '2025-05-03T01:00:00'})
    assert response.status_code == 200
    assert response.json() == []


def test_run_empty():
    response = client.post('/crypto/patterns/run', json={'symbol': 'BTCUSDT', 'timeframe': '1m', 'start': '2025-05-03T00:00:00', 'end': '2025-05-03T01:00:00'})
    assert response.status_code == 200
    # No pattern matches expected
    assert response.json() == []

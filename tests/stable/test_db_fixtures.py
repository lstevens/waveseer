"""
Tests for database fixtures and isolation.

This module demonstrates how to use the database fixtures for isolated testing
following the Vibe-Coding Playbook's TDD approach.
"""

import pytest
import duckdb


def test_db_fixture_basics(test_db):
    """Test basic operations with the test_db fixture."""
    # Verify connection works
    result = test_db.execute("SELECT 1").fetchone()
    assert result[0] == 1
    
    # Verify schema is set up
    result = test_db.execute("SELECT COUNT(*) FROM patterns").fetchone()
    assert result[0] == 0
    
    # Insert and query data
    test_db.execute("""
        INSERT INTO patterns (id, symbol, timestamp, pattern_type, confidence)
        VALUES (1, 'BTC-USD', '2025-01-01 12:00:00', 'double_top', 0.95)
    """)
    
    result = test_db.execute("""
        SELECT * FROM patterns WHERE id = 1
    """).fetchone()
    
    assert result[0] == 1
    assert result[1] == 'BTC-USD'
    assert result[3] == 'double_top'
    assert result[4] == pytest.approx(0.95)  # Use approx for floating point comparison


def test_db_with_data_fixture(test_db_with_data):
    """Test the pre-seeded database fixture."""
    # Verify data is pre-loaded
    result = test_db_with_data.execute("SELECT COUNT(*) FROM patterns").fetchone()
    assert result[0] == 5
    
    result = test_db_with_data.execute("SELECT COUNT(*) FROM indicators").fetchone()
    assert result[0] == 10
    
    # Verify we can add more data
    test_db_with_data.execute("""
        INSERT INTO patterns (id, symbol, timestamp, pattern_type, confidence)
        VALUES (100, 'ETH-USD', '2025-02-01 12:00:00', 'head_shoulders', 0.85)
    """)
    
    result = test_db_with_data.execute("SELECT COUNT(*) FROM patterns").fetchone()
    assert result[0] == 6


def test_mock_duckdb_connection(mock_duckdb_connection):
    """Test the mock_duckdb_connection fixture that patches all DuckDB connections."""
    # This should create an in-memory database even though we specify a file
    conn = duckdb.connect("motifs.db")
    
    # Create a test table
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test VALUES (1, 'test')")
    
    # Query the data
    result = conn.execute("SELECT * FROM test").fetchall()
    assert len(result) == 1
    assert result[0][0] == 1
    assert result[0][1] == 'test'
    
    # Create another connection - should be a different database
    conn2 = duckdb.connect("motifs.db")
    
    # Table shouldn't exist in the second connection
    with pytest.raises(duckdb.CatalogException):
        conn2.execute("SELECT * FROM test")
    
    # Clean up
    conn.close()
    conn2.close()


def test_transaction_isolation(test_db):
    """Test that database changes are automatically rolled back between tests."""
    # Add some data
    test_db.execute("""
        INSERT INTO patterns (id, symbol, timestamp, pattern_type, confidence)
        VALUES (200, 'SOL-USD', '2025-03-01 12:00:00', 'channel', 0.75)
    """)
    
    # Verify it's there
    result = test_db.execute("SELECT COUNT(*) FROM patterns WHERE id = 200").fetchone()
    assert result[0] == 1


def test_transaction_isolation_verified(test_db):
    """Verify that data from the previous test is not present."""
    # Check for data from the previous test
    result = test_db.execute("SELECT COUNT(*) FROM patterns WHERE id = 200").fetchone()
    assert result[0] == 0


def test_db_monkeypatch_example(test_db, monkeypatch):
    """Example of how to use monkeypatch to replace a function that uses duckdb."""
    
    # Define a module-level namespace for testing
    class DBUtils:
        @staticmethod
        def fetch_patterns_for_symbol(symbol):
            """Function that connects to the real database."""
            conn = duckdb.connect("motifs.db")
            result = conn.execute(f"SELECT * FROM patterns WHERE symbol = '{symbol}'").fetchall()
            conn.close()
            return result
    
    def mock_fetch_patterns(symbol):
        """Mock function that would normally connect to the real database."""
        # In a real test, you might monkeypatch a function from your codebase
        # that connects to the database
        return [
            {"id": 1, "symbol": symbol, "pattern_type": "test_pattern", "confidence": 0.9}
        ]
    
    # Monkeypatch the method in our test class
    monkeypatch.setattr(DBUtils, "fetch_patterns_for_symbol", mock_fetch_patterns)
    
    # Test the monkeypatched function
    patterns = DBUtils.fetch_patterns_for_symbol("BTC-USD")
    assert len(patterns) == 1
    assert patterns[0]["symbol"] == "BTC-USD"
    assert patterns[0]["pattern_type"] == "test_pattern"

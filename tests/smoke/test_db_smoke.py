"""
Smoke test for database isolation.

This test verifies that the database fixtures are properly set up and
can be used to create isolated test databases.
"""

import duckdb


def test_db_smoke(test_db):
    """Basic smoke test for database functionality."""
    # Verify connection is working
    result = test_db.execute("SELECT 1").fetchone()
    assert result[0] == 1
    
    # Verify schema is set up
    result = test_db.execute("SHOW TABLES").fetchall()
    table_names = [row[0] for row in result]
    assert "patterns" in table_names
    assert "indicators" in table_names


def test_mock_duckdb_smoke(mock_duckdb_connection):
    """Verify that the global duckdb connection mocking works."""
    # Create a connection that should be in-memory regardless of path
    conn = duckdb.connect("motifs.db")
    
    # Verify we can execute a basic query
    result = conn.execute("SELECT 1").fetchone()
    assert result[0] == 1
    
    # Create a table and add data
    conn.execute("CREATE TABLE test_smoke (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test_smoke VALUES (1, 'smoke_test')")
    
    # Verify data was inserted
    result = conn.execute("SELECT * FROM test_smoke").fetchall()
    assert len(result) == 1
    assert result[0][0] == 1
    assert result[0][1] == 'smoke_test'
    
    # Clean up
    conn.close()

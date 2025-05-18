"""
Tests for database isolation and fixtures.

This module contains tests for the database isolation utilities, following the 
TDD approach outlined in the Vibe-Coding Playbook.
"""

import pytest
import duckdb
from pathlib import Path
from tests.db.fixtures import (
    get_test_db_path,
    test_db_connection,
    setup_test_schema,
    seed_test_data
)


def test_in_memory_db_path():
    """Test that in-memory path is correctly returned."""
    path = get_test_db_path(in_memory=True)
    assert path == ":memory:"


def test_tmp_path_db_path(tmp_path):
    """Test that tmp_path is correctly used."""
    path = get_test_db_path(in_memory=False, tmp_path=tmp_path)
    assert str(tmp_path) in path
    assert "test_motifs.db" in path


def test_temp_file_db_path():
    """Test that a temporary file path is returned when no tmp_path is provided."""
    path = get_test_db_path(in_memory=False)
    assert path.endswith(".db")
    assert Path(path).is_absolute()


def test_db_connection_in_memory():
    """Test creating and using an in-memory database connection."""
    with test_db_connection(in_memory=True) as conn:
        # Basic query should work
        result = conn.execute("SELECT 1").fetchall()
        assert result == [(1,)]


def test_db_connection_tmp_path(tmp_path):
    """Test creating and using a database in tmp_path."""
    db_path = tmp_path / "test_motifs.db"
    assert not db_path.exists()
    
    with test_db_connection(in_memory=False, tmp_path=tmp_path) as conn:
        # Basic query should work
        result = conn.execute("SELECT 1").fetchall()
        assert result == [(1,)]
    
    # The database file should exist after the connection is closed
    assert db_path.exists()


def test_db_schema_setup():
    """Test setting up the database schema."""
    with test_db_connection() as conn:
        # Initially, tables shouldn't exist
        with pytest.raises(duckdb.CatalogException):
            conn.execute("SELECT * FROM patterns")
        
        # Set up schema
        setup_test_schema(conn)
        
        # Now tables should exist
        result = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()
        assert result[0] == 0
        
        result = conn.execute("SELECT COUNT(*) FROM indicators").fetchone()
        assert result[0] == 0


def test_seed_test_data():
    """Test seeding test data."""
    with test_db_connection() as conn:
        setup_test_schema(conn)
        
        # Seed with default values
        result = seed_test_data(conn)
        assert result["patterns"] == 5
        assert result["indicators"] == 10
        
        # Verify data was inserted
        count = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        assert count == 5
        
        count = conn.execute("SELECT COUNT(*) FROM indicators").fetchone()[0]
        assert count == 10
        
        # Verify data matches expectations
        pattern = conn.execute("""
            SELECT * FROM patterns 
            WHERE id = 0
        """).fetchone()
        assert pattern[1] == "BTC-USD"
        assert "pattern_" in pattern[3]


def test_db_isolation():
    """Test that multiple database connections are isolated."""
    # Create and populate first database
    with test_db_connection() as conn1:
        setup_test_schema(conn1)
        seed_test_data(conn1, patterns=3)
        
        # Check data is there
        count = conn1.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        assert count == 3
        
        # Create second database and verify it's empty
        with test_db_connection() as conn2:
            setup_test_schema(conn2)
            
            # Should be empty
            count = conn2.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
            assert count == 0
            
            # Adding data to conn2 shouldn't affect conn1
            seed_test_data(conn2, patterns=2)
        
        # Original connection should still have original data
        count = conn1.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        assert count == 3

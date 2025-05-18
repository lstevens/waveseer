"""
Database fixtures and utilities for testing.

This module provides tools for creating isolated DuckDB instances for testing,
ensuring tests do not interfere with each other or depend on external state.
"""

import os
import tempfile
from pathlib import Path
from contextlib import contextmanager
import duckdb


def get_test_db_path(in_memory=True, tmp_path=None):
    """
    Get a path for a test database.
    
    Args:
        in_memory: If True, use in-memory database
        tmp_path: Optional path for temporary file-based database
        
    Returns:
        Path to use with duckdb.connect()
    """
    if in_memory:
        return ":memory:"
    
    if tmp_path:
        return str(tmp_path / "test_motifs.db")
    
    # Use a temporary file that will be automatically cleaned up
    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    return temp_file.name


@contextmanager
def test_db_connection(in_memory=True, tmp_path=None):
    """
    Context manager that provides an isolated DuckDB connection for testing.
    
    Args:
        in_memory: If True, use in-memory database
        tmp_path: Optional path for file-based database
        
    Yields:
        DuckDB connection object
    """
    db_path = get_test_db_path(in_memory, tmp_path)
    conn = duckdb.connect(db_path)
    
    try:
        yield conn
    finally:
        conn.close()
        
        # Clean up temporary file if created
        if not in_memory and not tmp_path and os.path.exists(db_path):
            os.unlink(db_path)


def setup_test_schema(conn):
    """
    Set up the database schema for testing.
    
    This function creates all tables needed for testing with appropriate
    schema matching the production database.
    
    Args:
        conn: DuckDB connection
    """
    # Create patterns table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER,
        symbol VARCHAR,
        timestamp TIMESTAMP,
        pattern_type VARCHAR,
        confidence FLOAT,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        metadata VARCHAR
    )
    """)
    
    # Create indicators table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS indicators (
        id INTEGER,
        symbol VARCHAR,
        timestamp TIMESTAMP,
        indicator_type VARCHAR,
        value FLOAT
    )
    """)


def seed_test_data(conn, patterns=5, indicators=10):
    """
    Seed the test database with sample data.
    
    Args:
        conn: DuckDB connection
        patterns: Number of pattern records to create
        indicators: Number of indicator records to create
        
    Returns:
        Dictionary with counts of records created
    """
    # Seed patterns
    for i in range(patterns):
        conn.execute("""
        INSERT INTO patterns VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            i, 
            f"BTC-USD", 
            f"2025-01-{i+1} 12:00:00",
            f"pattern_{i % 3}",
            0.5 + (i * 0.1),
            f"2025-01-{i+1} 11:30:00",
            f"2025-01-{i+1} 12:30:00",
            "{}"
        ])
    
    # Seed indicators
    for i in range(indicators):
        conn.execute("""
        INSERT INTO indicators VALUES (?, ?, ?, ?, ?)
        """, [
            i, 
            f"BTC-USD", 
            f"2025-01-{i+1} 12:00:00",
            f"indicator_{i % 4}",
            i * 1.5
        ])
    
    return {
        "patterns": patterns,
        "indicators": indicators
    }

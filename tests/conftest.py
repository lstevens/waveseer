# Ensure local 'wave' package takes precedence over stdlib
import sys
import pathlib
import os
import pytest
import duckdb

project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import database fixtures
from tests.db.fixtures import (
    test_db_connection,
    setup_test_schema,
    seed_test_data
)


@pytest.fixture(scope="function")
def test_db(tmp_path):
    """
    Fixture that provides an isolated DuckDB connection for testing.
    
    Uses an in-memory database by default for maximum isolation and speed.
    Sets up the basic schema but does not seed any data by default.
    
    Args:
        tmp_path: Pytest fixture for temporary directory
        
    Returns:
        DuckDB connection object with initialized schema
    """
    # Check if we should use a file-based DB instead of memory
    # This environment variable is helpful for debugging
    in_memory = os.environ.get("TEST_DB_USE_FILE", "false").lower() != "true"
    
    with test_db_connection(in_memory=in_memory, tmp_path=tmp_path) as conn:
        # Set up schema but don't seed data by default
        setup_test_schema(conn)
        yield conn


@pytest.fixture(scope="function")
def test_db_with_data(test_db):
    """
    Fixture that provides a DuckDB connection with pre-seeded test data.
    
    Builds on the test_db fixture but adds default test data.
    
    Args:
        test_db: Base database fixture
        
    Returns:
        DuckDB connection with schema and seeded data
    """
    # Seed with default data
    seed_test_data(test_db)
    return test_db


@pytest.fixture(scope="function")
def mock_duckdb_connection(monkeypatch):
    """
    Fixture that replaces all duckdb.connect() calls with in-memory test databases.
    
    This allows existing code that uses duckdb.connect() directly to
    automatically use isolated test databases without code changes.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    original_connect = duckdb.connect
    
    def mock_connect(database=None, *args, **kwargs):
        # Always use in-memory database for tests
        return original_connect(":memory:", *args, **kwargs)
    
    monkeypatch.setattr(duckdb, "connect", mock_connect)

"""
Connector module for crypto_heatmap PostgreSQL database.
"""
import logging
from contextlib import contextmanager

try:
    from pydantic import BaseSettings, Field
except Exception:
    from pydantic.v1 import BaseSettings, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, Session


class Settings(BaseSettings):
    """
    Pydantic settings for database connection.
    Reads from environment variables or .env file.
    """
    DB_HOST: str = Field(..., env="DB_HOST")
    DB_PORT: int = Field(..., env="DB_PORT")
    DB_NAME: str = Field(..., env="DB_NAME")
    DB_USER: str = Field(..., env="DB_USER")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class PostgresConnector:
    """
    Database connector using SQLAlchemy engine with connection pooling.
    """
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        url = (
            f"postgresql+psycopg2://{self.settings.DB_USER}:"
            f"{self.settings.DB_PASSWORD}@{self.settings.DB_HOST}:"
            f"{self.settings.DB_PORT}/{self.settings.DB_NAME}"
        )
        self.engine = create_engine(
            url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

    @contextmanager
    def get_session(self) -> Session:
        """
        Yields a SQLAlchemy Session and ensures it is closed after use.
        Usage:
            with connector.get_session() as session:
                session.execute(...)
        """
        session = self.SessionLocal()
        try:
            yield session
        except SQLAlchemyError:
            session.rollback()
            logging.exception("Session rollback because of exception.")
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """
        Tests the database connection by executing a simple query.
        Returns True if successful, False on failure.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as e:
            logging.error("Database connection failed", exc_info=e)
            return False

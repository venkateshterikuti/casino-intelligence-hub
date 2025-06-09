"""
Database connection manager for PostgreSQL.
Handles connection pooling, session management, and database operations.
"""

import logging
from contextlib import contextmanager
from typing import Optional, Generator
import pandas as pd
import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from config.database_config import db_config, POOL_CONFIG

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self):
        self.connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections and pools."""
        try:
            # Create SQLAlchemy engine
            self.engine = create_engine(
                db_config.connection_string,
                pool_size=POOL_CONFIG['pool_size'],
                max_overflow=POOL_CONFIG['max_overflow'],
                pool_timeout=POOL_CONFIG['pool_timeout'],
                pool_recycle=POOL_CONFIG['pool_recycle'],
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Create psycopg2 connection pool for raw connections
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=POOL_CONFIG['pool_size'],
                **db_config.connection_params
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Context manager for getting raw psycopg2 connections.
        
        Yields:
            psycopg2 connection object
        """
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for getting SQLAlchemy sessions.
        
        Yields:
            SQLAlchemy session object
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        try:
            with self.engine.connect() as connection:
                result = pd.read_sql(text(query), connection, params=params)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_sql(self, query: str, params: Optional[dict] = None) -> None:
        """
        Execute SQL query without returning results.
        
        Args:
            query: SQL query string
            params: Query parameters
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text(query), params or {})
                connection.commit()
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = self.execute_query("SELECT 1 as test")
            return len(result) == 1 and result.iloc[0]['test'] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def close_connections(self):
        """Close all database connections."""
        try:
            if self.connection_pool:
                self.connection_pool.closeall()
            if self.engine:
                self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

# Global database manager instance
db_manager = DatabaseManager()

# Alias for backward compatibility
DatabaseConnection = DatabaseManager 
"""
Database module for Casino Intelligence Hub.
Provides database connection management, data loading, and query execution utilities.
"""

from .connection import DatabaseManager, db_manager
from .data_loader import DataLoader
from .query_executor import QueryExecutor

__all__ = [
    'DatabaseManager',
    'db_manager',
    'DataLoader', 
    'QueryExecutor'
]

# Version info
__version__ = '1.0.0' 
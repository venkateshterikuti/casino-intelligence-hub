"""
Database configuration settings for the Casino Intelligence Hub.
Handles PostgreSQL connection parameters and database schema settings.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig:
    """Database configuration class for PostgreSQL connections."""
    
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', 5432))
        self.database = os.getenv('DB_NAME', 'casino_intelligence')
        self.username = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD', '')
        
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Return connection parameters as dictionary."""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.username,
            'password': self.password
        }

# Schema configuration
SCHEMA_CONFIG = {
    'main_schema': 'casino_intelligence',
    'tables': {
        'players': 'players',
        'games': 'games', 
        'transactions': 'transactions',
        'game_sessions': 'game_sessions',
        'player_churn_features': 'player_churn_features',
        'player_segments': 'player_segments',
        'anomaly_flags': 'anomaly_flags'
    }
}

# Connection pool settings
POOL_CONFIG = {
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'pool_recycle': 3600
}

# Default database instance
db_config = DatabaseConfig() 
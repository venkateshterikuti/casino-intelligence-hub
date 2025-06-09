"""
Query execution utilities for Casino Intelligence Hub.
Handles complex SQL queries, batch operations, and query optimization.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import time
from datetime import datetime, date, timedelta

from .connection import db_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class QueryExecutor:
    """
    Handles execution of complex SQL queries and batch operations.
    Provides utilities for feature engineering, analytics, and data validation.
    """
    
    def __init__(self, connection_manager=None):
        self.db_manager = connection_manager or db_manager
        self.query_cache = {}
    
    def load_sql_file(self, file_path: Union[str, Path]) -> str:
        """Load SQL query from file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"SQL file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return f.read()
    
    def execute_feature_engineering(self, start_date: date = None, end_date: date = None) -> Dict[str, Any]:
        """Execute feature engineering queries for ML models."""
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=90)
        
        logger.info(f"Executing feature engineering from {start_date} to {end_date}")
        
        # Calculate RFM scores
        rfm_query = """
        WITH player_metrics AS (
            SELECT 
                t.player_id,
                MAX(DATE(t.transaction_timestamp)) as last_transaction_date,
                COUNT(DISTINCT DATE(t.transaction_timestamp)) as frequency,
                SUM(CASE WHEN t.transaction_type = 'bet' THEN t.amount ELSE 0 END) as monetary
            FROM transactions t
            WHERE DATE(t.transaction_timestamp) BETWEEN %(start_date)s AND %(end_date)s
            GROUP BY t.player_id
        )
        SELECT 
            player_id,
            EXTRACT(DAYS FROM (%(end_date)s - last_transaction_date)) as recency_days,
            frequency,
            monetary,
            NTILE(5) OVER (ORDER BY EXTRACT(DAYS FROM (%(end_date)s - last_transaction_date)) DESC) as recency_score,
            NTILE(5) OVER (ORDER BY frequency ASC) as frequency_score,
            NTILE(5) OVER (ORDER BY monetary ASC) as monetary_score
        FROM player_metrics
        """
        
        rfm_df = self.db_manager.execute_query(rfm_query, {
            'start_date': start_date,
            'end_date': end_date
        })
        
        return {'rfm_features': len(rfm_df)}
    
    def get_kpi_metrics(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Calculate KPI metrics for dashboard."""
        kpi_query = """
        SELECT 
            COUNT(DISTINCT player_id) as active_players,
            COUNT(*) as total_transactions,
            SUM(CASE WHEN transaction_type = 'bet' THEN amount ELSE 0 END) as total_bets,
            SUM(CASE WHEN transaction_type = 'win' THEN amount ELSE 0 END) as total_wins
        FROM transactions
        WHERE DATE(transaction_timestamp) BETWEEN %(start_date)s AND %(end_date)s
        """
        
        result = self.db_manager.execute_query(kpi_query, {
            'start_date': start_date,
            'end_date': end_date
        })
        
        return result.iloc[0].to_dict() if len(result) > 0 else {}
    
    def execute_data_quality_checks(self) -> Dict[str, Any]:
        """Execute data quality checks."""
        checks = {}
        
        # Check for duplicates
        dup_query = "SELECT COUNT(*) - COUNT(DISTINCT transaction_id) as duplicates FROM transactions"
        result = self.db_manager.execute_query(dup_query)
        checks['transaction_duplicates'] = int(result.iloc[0, 0])
        
        # Check for missing values
        null_query = "SELECT COUNT(*) FROM transactions WHERE player_id IS NULL"
        result = self.db_manager.execute_query(null_query)
        checks['missing_player_ids'] = int(result.iloc[0, 0])
        
        return checks 
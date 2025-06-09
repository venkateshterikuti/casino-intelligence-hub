"""
Data loading utilities for Casino Intelligence Hub.
Handles loading data from various sources into PostgreSQL tables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime, date

from .connection import db_manager
from ..utils.logger import get_logger
from ..utils.helpers import clean_column_names, validate_data_types

logger = get_logger(__name__)

class DataLoader:
    """
    Handles loading data from various sources into PostgreSQL.
    Supports CSV files, DataFrames, and direct SQL inserts.
    """
    
    def __init__(self, connection_manager=None):
        self.db_manager = connection_manager or db_manager
        
    def load_csv_to_table(
        self, 
        csv_path: Union[str, Path], 
        table_name: str,
        schema: str = 'casino_intelligence',
        chunk_size: int = 10000,
        clean_columns: bool = True,
        if_exists: str = 'replace'
    ) -> Dict[str, Any]:
        """
        Load CSV file to PostgreSQL table.
        
        Args:
            csv_path: Path to CSV file
            table_name: Target table name
            schema: Database schema name
            chunk_size: Number of rows to process at once
            clean_columns: Whether to clean column names
            if_exists: What to do if table exists ('replace', 'append', 'fail')
            
        Returns:
            Dictionary with loading statistics
        """
        logger.info(f"Loading CSV {csv_path} to table {schema}.{table_name}")
        
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        stats = {
            'total_rows': 0,
            'chunks_processed': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        try:
            # Read CSV in chunks
            chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size)
            
            for chunk_num, chunk in enumerate(chunk_iter):
                try:
                    # Clean column names if requested
                    if clean_columns:
                        chunk.columns = clean_column_names(chunk.columns)
                    
                    # Load chunk to database
                    chunk.to_sql(
                        table_name,
                        self.db_manager.engine,
                        schema=schema,
                        if_exists=if_exists if chunk_num == 0 else 'append',
                        index=False,
                        method='multi'
                    )
                    
                    stats['total_rows'] += len(chunk)
                    stats['chunks_processed'] += 1
                    
                    if chunk_num % 10 == 0:
                        logger.info(f"Processed chunk {chunk_num + 1}, total rows: {stats['total_rows']:,}")
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_num}: {e}")
                    stats['errors'] += 1
            
            stats['end_time'] = datetime.now()
            stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
            
            logger.info(f"CSV loading completed: {stats['total_rows']:,} rows in {stats['duration']:.2f}s")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def load_dataframe_to_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str = 'casino_intelligence',
        if_exists: str = 'replace',
        clean_columns: bool = True
    ) -> bool:
        """
        Load pandas DataFrame to PostgreSQL table.
        
        Args:
            df: Source DataFrame
            table_name: Target table name
            schema: Database schema name
            if_exists: What to do if table exists
            clean_columns: Whether to clean column names
            
        Returns:
            Success status
        """
        logger.info(f"Loading DataFrame to table {schema}.{table_name} ({len(df):,} rows)")
        
        try:
            # Clean column names if requested
            if clean_columns:
                df.columns = clean_column_names(df.columns)
            
            # Load to database
            df.to_sql(
                table_name,
                self.db_manager.engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                method='multi'
            )
            
            logger.info(f"DataFrame loaded successfully: {len(df):,} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DataFrame: {e}")
            raise
    
    def load_ml_features(
        self,
        features_df: pd.DataFrame,
        table_name: str,
        feature_date: Optional[date] = None
    ) -> bool:
        """
        Load ML features with proper date handling.
        
        Args:
            features_df: DataFrame with ML features
            table_name: Target table name
            feature_date: Date for the features (defaults to today)
            
        Returns:
            Success status
        """
        if feature_date is None:
            feature_date = date.today()
        
        logger.info(f"Loading ML features to {table_name} for date {feature_date}")
        
        try:
            # Add feature date if not present
            if 'snapshot_date' not in features_df.columns and 'segmentation_date' not in features_df.columns:
                date_col = 'snapshot_date' if 'churn' in table_name.lower() else 'segmentation_date'
                features_df[date_col] = feature_date
            
            # Load to database
            self.load_dataframe_to_table(
                features_df,
                table_name,
                if_exists='append'  # Usually append for ML features
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ML features: {e}")
            raise
    
    def load_predictions(
        self,
        predictions: Union[pd.DataFrame, Dict[str, Any]],
        model_type: str,
        prediction_date: Optional[date] = None
    ) -> bool:
        """
        Load model predictions to appropriate tables.
        
        Args:
            predictions: Prediction results
            model_type: Type of model ('churn', 'segmentation', 'anomaly')
            prediction_date: Date of predictions
            
        Returns:
            Success status
        """
        if prediction_date is None:
            prediction_date = date.today()
        
        logger.info(f"Loading {model_type} predictions for date {prediction_date}")
        
        try:
            if model_type == 'churn':
                return self._load_churn_predictions(predictions, prediction_date)
            elif model_type == 'segmentation':
                return self._load_segmentation_predictions(predictions, prediction_date)
            elif model_type == 'anomaly':
                return self._load_anomaly_predictions(predictions, prediction_date)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to load {model_type} predictions: {e}")
            raise
    
    def _load_churn_predictions(self, predictions_df: pd.DataFrame, pred_date: date) -> bool:
        """Load churn predictions to player_churn_features table."""
        
        # Ensure required columns
        required_cols = ['player_id', 'churn_probability']
        if not all(col in predictions_df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Add prediction date
        predictions_df['prediction_date'] = pred_date
        
        # Update existing records or insert new ones
        for _, row in predictions_df.iterrows():
            update_sql = """
            UPDATE player_churn_features 
            SET churn_probability = %(prob)s,
                prediction_date = %(pred_date)s,
                updated_at = CURRENT_TIMESTAMP
            WHERE player_id = %(player_id)s 
            AND snapshot_date = (
                SELECT MAX(snapshot_date) 
                FROM player_churn_features 
                WHERE player_id = %(player_id)s
            )
            """
            
            self.db_manager.execute_sql(update_sql, {
                'prob': float(row['churn_probability']),
                'pred_date': pred_date,
                'player_id': row['player_id']
            })
        
        logger.info(f"Updated churn predictions for {len(predictions_df)} players")
        return True
    
    def _load_segmentation_predictions(self, predictions_df: pd.DataFrame, pred_date: date) -> bool:
        """Load segmentation results to player_segments table."""
        
        # Add segmentation date
        predictions_df['segmentation_date'] = pred_date
        
        # Load to table
        return self.load_dataframe_to_table(
            predictions_df,
            'player_segments',
            if_exists='append'
        )
    
    def _load_anomaly_predictions(self, anomalies_df: pd.DataFrame, pred_date: date) -> bool:
        """Load anomaly detections to anomaly_flags table."""
        
        # Add detection date
        anomalies_df['anomaly_timestamp'] = datetime.now()
        
        # Load to table
        return self.load_dataframe_to_table(
            anomalies_df,
            'anomaly_flags',
            if_exists='append'
        )
    
    def export_table_to_csv(
        self,
        table_name: str,
        output_path: Union[str, Path],
        schema: str = 'casino_intelligence',
        query: Optional[str] = None
    ) -> bool:
        """
        Export table or query results to CSV.
        
        Args:
            table_name: Source table name
            output_path: Output CSV file path
            schema: Database schema
            query: Custom query (if None, exports entire table)
            
        Returns:
            Success status
        """
        logger.info(f"Exporting {schema}.{table_name} to {output_path}")
        
        try:
            if query is None:
                query = f"SELECT * FROM {schema}.{table_name}"
            
            # Read data from database
            df = self.db_manager.execute_query(query)
            
            # Export to CSV
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(df):,} rows to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export table: {e}")
            raise
    
    def backup_table(
        self,
        table_name: str,
        backup_path: Union[str, Path],
        schema: str = 'casino_intelligence'
    ) -> bool:
        """
        Create a backup of a table as CSV.
        
        Args:
            table_name: Table to backup
            backup_path: Backup file path
            schema: Database schema
            
        Returns:
            Success status
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = Path(backup_path) / f"{table_name}_backup_{timestamp}.csv"
        
        return self.export_table_to_csv(table_name, backup_file, schema)
    
    def get_table_info(
        self,
        table_name: str,
        schema: str = 'casino_intelligence'
    ) -> Dict[str, Any]:
        """
        Get information about a table.
        
        Args:
            table_name: Table name
            schema: Database schema
            
        Returns:
            Table information dictionary
        """
        logger.info(f"Getting info for table {schema}.{table_name}")
        
        try:
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {schema}.{table_name}"
            count_result = self.db_manager.execute_query(count_query)
            row_count = count_result.iloc[0]['row_count']
            
            # Get column info
            column_query = """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """
            
            columns_df = self.db_manager.execute_query(column_query, (schema, table_name))
            
            # Get sample data
            sample_query = f"SELECT * FROM {schema}.{table_name} LIMIT 5"
            sample_df = self.db_manager.execute_query(sample_query)
            
            return {
                'table_name': table_name,
                'schema': schema,
                'row_count': int(row_count),
                'column_count': len(columns_df),
                'columns': columns_df.to_dict('records'),
                'sample_data': sample_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            raise 
"""
Utility helper functions for Casino Intelligence Hub.
Provides common data processing, validation, and transformation utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import re
from datetime import datetime, date, timedelta
import json
from pathlib import Path

def clean_column_names(columns: List[str]) -> List[str]:
    """
    Clean column names for database compatibility.
    
    Args:
        columns: List of column names to clean
        
    Returns:
        List of cleaned column names
    """
    cleaned = []
    for col in columns:
        # Convert to lowercase
        clean_col = str(col).lower()
        
        # Replace spaces and special characters with underscores
        clean_col = re.sub(r'[^a-z0-9_]', '_', clean_col)
        
        # Remove multiple consecutive underscores
        clean_col = re.sub(r'_+', '_', clean_col)
        
        # Remove leading/trailing underscores
        clean_col = clean_col.strip('_')
        
        # Ensure it doesn't start with a number
        if clean_col and clean_col[0].isdigit():
            clean_col = f'col_{clean_col}'
        
        # Handle empty names
        if not clean_col:
            clean_col = f'unnamed_column_{len(cleaned)}'
        
        cleaned.append(clean_col)
    
    return cleaned

def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate DataFrame column data types.
    
    Args:
        df: DataFrame to validate
        expected_types: Dictionary mapping column names to expected types
        
    Returns:
        Validation results
    """
    validation_results = {
        'valid': True,
        'issues': [],
        'type_mismatches': {}
    }
    
    for column, expected_type in expected_types.items():
        if column not in df.columns:
            validation_results['issues'].append(f"Missing column: {column}")
            validation_results['valid'] = False
            continue
        
        actual_type = str(df[column].dtype)
        
        # Check type compatibility
        type_compatible = False
        
        if expected_type in ['int', 'integer']:
            type_compatible = df[column].dtype in ['int64', 'int32', 'Int64']
        elif expected_type in ['float', 'numeric']:
            type_compatible = df[column].dtype in ['float64', 'float32', 'int64', 'int32']
        elif expected_type in ['str', 'string', 'text']:
            type_compatible = df[column].dtype == 'object' or df[column].dtype.name.startswith('string')
        elif expected_type in ['datetime', 'timestamp']:
            type_compatible = pd.api.types.is_datetime64_any_dtype(df[column])
        elif expected_type == 'bool':
            type_compatible = df[column].dtype == 'bool'
        
        if not type_compatible:
            validation_results['type_mismatches'][column] = {
                'expected': expected_type,
                'actual': actual_type
            }
            validation_results['valid'] = False
    
    return validation_results

def convert_data_types(df: pd.DataFrame, type_conversions: Dict[str, str]) -> pd.DataFrame:
    """
    Convert DataFrame column data types.
    
    Args:
        df: DataFrame to convert
        type_conversions: Dictionary mapping column names to target types
        
    Returns:
        DataFrame with converted types
    """
    df_converted = df.copy()
    
    for column, target_type in type_conversions.items():
        if column not in df_converted.columns:
            continue
        
        try:
            if target_type in ['int', 'integer']:
                # Handle NaN values for integer conversion
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                df_converted[column] = df_converted[column].astype('Int64')  # Nullable integer
            
            elif target_type in ['float', 'numeric']:
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
            
            elif target_type in ['str', 'string', 'text']:
                df_converted[column] = df_converted[column].astype(str)
            
            elif target_type in ['datetime', 'timestamp']:
                df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
            
            elif target_type == 'bool':
                # Convert various representations to boolean
                df_converted[column] = df_converted[column].map({
                    'true': True, 'false': False, 'True': True, 'False': False,
                    '1': True, '0': False, 1: True, 0: False,
                    'yes': True, 'no': False, 'Y': True, 'N': False
                })
            
        except Exception as e:
            print(f"Warning: Could not convert column {column} to {target_type}: {e}")
    
    return df_converted

def detect_outliers(
    series: pd.Series, 
    method: str = 'iqr',
    threshold: float = 1.5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect outliers in a pandas Series.
    
    Args:
        series: Data series to analyze
        method: Method to use ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (outlier_mask, outlier_info)
    """
    outlier_info = {'method': method, 'threshold': threshold}
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_info.update({
            'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
            'lower_bound': lower_bound, 'upper_bound': upper_bound
        })
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_mask = z_scores > threshold
        outlier_info.update({
            'mean': series.mean(), 'std': series.std(),
            'max_zscore': z_scores.max()
        })
    
    elif method == 'modified_zscore':
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        outlier_info.update({
            'median': median, 'mad': mad,
            'max_modified_zscore': np.abs(modified_z_scores).max()
        })
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    outlier_info['outlier_count'] = outlier_mask.sum()
    outlier_info['outlier_percentage'] = (outlier_mask.sum() / len(series)) * 100
    
    return outlier_mask, outlier_info

def calculate_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary of missing values in DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with missing values summary
    """
    missing_summary = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df)) * 100,
        'data_type': df.dtypes
    })
    
    missing_summary = missing_summary.sort_values('missing_percentage', ascending=False)
    missing_summary = missing_summary.reset_index(drop=True)
    
    return missing_summary

def create_date_features(
    df: pd.DataFrame, 
    date_column: str,
    features: List[str] = None
) -> pd.DataFrame:
    """
    Create date-based features from a datetime column.
    
    Args:
        df: DataFrame containing date column
        date_column: Name of the datetime column
        features: List of features to create
        
    Returns:
        DataFrame with additional date features
    """
    if features is None:
        features = ['year', 'month', 'day', 'dayofweek', 'hour', 'quarter']
    
    df_with_dates = df.copy()
    
    # Ensure column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_with_dates[date_column]):
        df_with_dates[date_column] = pd.to_datetime(df_with_dates[date_column])
    
    # Create features
    for feature in features:
        if feature == 'year':
            df_with_dates[f'{date_column}_year'] = df_with_dates[date_column].dt.year
        elif feature == 'month':
            df_with_dates[f'{date_column}_month'] = df_with_dates[date_column].dt.month
        elif feature == 'day':
            df_with_dates[f'{date_column}_day'] = df_with_dates[date_column].dt.day
        elif feature == 'dayofweek':
            df_with_dates[f'{date_column}_dayofweek'] = df_with_dates[date_column].dt.dayofweek
        elif feature == 'hour':
            df_with_dates[f'{date_column}_hour'] = df_with_dates[date_column].dt.hour
        elif feature == 'quarter':
            df_with_dates[f'{date_column}_quarter'] = df_with_dates[date_column].dt.quarter
        elif feature == 'is_weekend':
            df_with_dates[f'{date_column}_is_weekend'] = df_with_dates[date_column].dt.dayofweek.isin([5, 6])
        elif feature == 'is_month_start':
            df_with_dates[f'{date_column}_is_month_start'] = df_with_dates[date_column].dt.is_month_start
        elif feature == 'is_month_end':
            df_with_dates[f'{date_column}_is_month_end'] = df_with_dates[date_column].dt.is_month_end
    
    return df_with_dates

def calculate_rolling_features(
    df: pd.DataFrame,
    value_column: str,
    window_sizes: List[int] = [7, 30, 90],
    operations: List[str] = ['mean', 'sum', 'std']
) -> pd.DataFrame:
    """
    Calculate rolling window features.
    
    Args:
        df: DataFrame with time-ordered data
        value_column: Column to calculate rolling features for
        window_sizes: List of window sizes
        operations: List of operations to apply
        
    Returns:
        DataFrame with rolling features
    """
    df_rolling = df.copy()
    
    for window in window_sizes:
        for operation in operations:
            feature_name = f'{value_column}_rolling_{window}d_{operation}'
            
            if operation == 'mean':
                df_rolling[feature_name] = df_rolling[value_column].rolling(window=window).mean()
            elif operation == 'sum':
                df_rolling[feature_name] = df_rolling[value_column].rolling(window=window).sum()
            elif operation == 'std':
                df_rolling[feature_name] = df_rolling[value_column].rolling(window=window).std()
            elif operation == 'min':
                df_rolling[feature_name] = df_rolling[value_column].rolling(window=window).min()
            elif operation == 'max':
                df_rolling[feature_name] = df_rolling[value_column].rolling(window=window).max()
    
    return df_rolling

def encode_categorical_variables(
    df: pd.DataFrame,
    categorical_columns: List[str],
    encoding_type: str = 'onehot',
    max_categories: int = 20
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical variables.
    
    Args:
        df: DataFrame with categorical columns
        categorical_columns: List of categorical column names
        encoding_type: Type of encoding ('onehot', 'label', 'target')
        max_categories: Maximum number of categories for one-hot encoding
        
    Returns:
        Tuple of (encoded_dataframe, encoding_info)
    """
    df_encoded = df.copy()
    encoding_info = {}
    
    for column in categorical_columns:
        if column not in df_encoded.columns:
            continue
        
        unique_values = df_encoded[column].nunique()
        encoding_info[column] = {
            'unique_values': unique_values,
            'encoding_type': encoding_type
        }
        
        if encoding_type == 'onehot' and unique_values <= max_categories:
            # One-hot encoding
            dummies = pd.get_dummies(df_encoded[column], prefix=column, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(column, axis=1)
            encoding_info[column]['dummy_columns'] = list(dummies.columns)
        
        elif encoding_type == 'label':
            # Label encoding
            value_mapping = {value: idx for idx, value in enumerate(df_encoded[column].unique())}
            df_encoded[f'{column}_encoded'] = df_encoded[column].map(value_mapping)
            encoding_info[column]['value_mapping'] = value_mapping
        
        else:
            # Keep as is or handle high cardinality
            if unique_values > max_categories:
                # For high cardinality, keep top categories and group others
                top_categories = df_encoded[column].value_counts().head(max_categories).index
                df_encoded[f'{column}_grouped'] = df_encoded[column].apply(
                    lambda x: x if x in top_categories else 'Other'
                )
                encoding_info[column]['top_categories'] = list(top_categories)
    
    return df_encoded, encoding_info

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency values for display."""
    if pd.isna(amount):
        return "N/A"
    
    if currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'EUR':
        return f"€{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format percentage values for display."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"

def calculate_correlation_matrix(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df: DataFrame to analyze
        numeric_columns: List of numeric columns (if None, auto-detect)
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    correlation_matrix = df[numeric_columns].corr(method=method)
    return correlation_matrix

def save_config(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save configuration dictionary to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)

def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    """Split DataFrame into chunks."""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df.iloc[i:i + chunk_size])
    return chunks

def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive summary statistics for DataFrame."""
    summary = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum(),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary['categorical_summary'] = {}
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
    
    return summary 
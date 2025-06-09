"""
Logging utility for Casino Intelligence Hub.
Provides centralized logging configuration and utilities.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger


class CustomFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m'  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(CustomFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Optional specific log file for this logger
        
    Returns:
        Logger instance
    """
    # Get log level from environment or default to INFO
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Setup logging if not already configured
    if not logging.getLogger().handlers:
        default_log_file = None
        if not log_file:
            # Default log file path
            log_dir = Path(__file__).parent.parent.parent / "logs"
            default_log_file = log_dir / "application.log"
        
        setup_logging(
            log_level=log_level,
            log_file=str(default_log_file) if default_log_file else log_file,
            console_output=True
        )
    
    return logging.getLogger(name)


class DatabaseLogger:
    """Logger specifically for database operations."""
    
    def __init__(self):
        self.logger = get_logger('database', 'logs/database.log')
    
    def log_query(self, query: str, params: Optional[dict] = None, execution_time: Optional[float] = None):
        """Log SQL query execution."""
        message = f"Query executed: {query[:100]}{'...' if len(query) > 100 else ''}"
        if params:
            message += f" | Params: {params}"
        if execution_time:
            message += f" | Time: {execution_time:.3f}s"
        
        self.logger.info(message)
    
    def log_error(self, error: Exception, query: str = None):
        """Log database errors."""
        message = f"Database error: {str(error)}"
        if query:
            message += f" | Query: {query[:100]}{'...' if len(query) > 100 else ''}"
        
        self.logger.error(message)


class ModelLogger:
    """Logger specifically for machine learning operations."""
    
    def __init__(self):
        self.logger = get_logger('models', 'logs/model_training.log')
    
    def log_training_start(self, model_name: str, dataset_shape: tuple):
        """Log model training start."""
        self.logger.info(f"Starting training for {model_name} | Dataset shape: {dataset_shape}")
    
    def log_training_complete(self, model_name: str, metrics: dict, training_time: float):
        """Log model training completion."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Completed training for {model_name} | {metrics_str} | Time: {training_time:.2f}s")
    
    def log_prediction(self, model_name: str, input_shape: tuple, prediction_count: int):
        """Log model predictions."""
        self.logger.info(f"Predictions made with {model_name} | Input: {input_shape} | Predictions: {prediction_count}")


class ETLLogger:
    """Logger specifically for ETL operations."""
    
    def __init__(self):
        self.logger = get_logger('etl', 'logs/etl.log')
    
    def log_data_load(self, source: str, target: str, row_count: int):
        """Log data loading operations."""
        self.logger.info(f"Data loaded: {source} -> {target} | Rows: {row_count:,}")
    
    def log_transformation(self, operation: str, input_rows: int, output_rows: int):
        """Log data transformation operations."""
        self.logger.info(f"Transformation: {operation} | Input: {input_rows:,} -> Output: {output_rows:,}")
    
    def log_data_quality_issue(self, issue: str, affected_rows: int):
        """Log data quality issues."""
        self.logger.warning(f"Data quality issue: {issue} | Affected rows: {affected_rows:,}")


# Convenience functions
def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {str(e)}")
            raise
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Function {func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.3f} seconds: {str(e)}")
            raise
    
    return wrapper


# Initialize default logging on import
if not logging.getLogger().handlers:
    setup_logging()

# Export commonly used loggers
db_logger = DatabaseLogger()
model_logger = ModelLogger() 
etl_logger = ETLLogger() 
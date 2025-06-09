"""
Constants for Casino Intelligence Hub.
Defines project-wide constants, enums, and configuration values.
"""

from enum import Enum
from typing import Dict, List, Any

# ============================================================================
# DATABASE CONSTANTS
# ============================================================================

SCHEMA_NAME = 'casino_intelligence'

# Table names
TABLES = {
    'PLAYERS': 'players',
    'GAMES': 'games',
    'TRANSACTIONS': 'transactions',
    'GAME_SESSIONS': 'game_sessions',
    'PLAYER_CHURN_FEATURES': 'player_churn_features',
    'PLAYER_SEGMENTS': 'player_segments',
    'ANOMALY_FLAGS': 'anomaly_flags',
    'STAGING_RAW_DATA': 'staging_raw_data',
    'SYSTEM_LOG': 'system_log',
    'DATA_QUALITY_CHECKS': 'data_quality_checks'
}

# ============================================================================
# TRANSACTION TYPES
# ============================================================================

class TransactionType(Enum):
    DEPOSIT = 'deposit'
    WITHDRAWAL = 'withdrawal'
    BET = 'bet'
    WIN = 'win'
    BONUS = 'bonus'
    FREESPIN = 'freespin'
    REFUND = 'refund'

TRANSACTION_TYPES = [t.value for t in TransactionType]

# ============================================================================
# GAME TYPES
# ============================================================================

class GameType(Enum):
    SLOTS = 'slots'
    BLACKJACK = 'blackjack'
    ROULETTE = 'roulette'
    POKER = 'poker'
    BACCARAT = 'baccarat'
    OTHER = 'other'

GAME_TYPES = [g.value for g in GameType]

# ============================================================================
# VIP STATUS LEVELS
# ============================================================================

class VIPStatus(Enum):
    BRONZE = 'bronze'
    SILVER = 'silver'
    GOLD = 'gold'
    PLATINUM = 'platinum'
    DIAMOND = 'diamond'

VIP_LEVELS = [v.value for v in VIPStatus]

# ============================================================================
# RFM SEGMENTS
# ============================================================================

RFM_SEGMENTS = {
    'CHAMPIONS': 'Champions',
    'LOYAL_CUSTOMERS': 'Loyal Customers',
    'POTENTIAL_LOYALISTS': 'Potential Loyalists',
    'NEW_CUSTOMERS': 'New Customers',
    'PROMISING': 'Promising',
    'CUSTOMERS_NEEDING_ATTENTION': 'Customers Needing Attention',
    'ABOUT_TO_SLEEP': 'About to Sleep',
    'AT_RISK': 'At Risk',
    'CANNOT_LOSE_THEM': 'Cannot Lose Them',
    'HIBERNATING': 'Hibernating',
    'LOST': 'Lost'
}

# RFM segment definitions based on scores
RFM_SEGMENT_MAPPING = {
    '555': 'Champions',
    '554': 'Champions',
    '544': 'Champions',
    '545': 'Champions',
    '454': 'Champions',
    '455': 'Champions',
    '445': 'Champions',
    '543': 'Loyal Customers',
    '444': 'Loyal Customers',
    '435': 'Loyal Customers',
    '355': 'Loyal Customers',
    '354': 'Loyal Customers',
    '345': 'Loyal Customers',
    '344': 'Loyal Customers',
    '335': 'Loyal Customers',
    '534': 'Potential Loyalists',
    '343': 'Potential Loyalists',
    '334': 'Potential Loyalists',
    '325': 'Potential Loyalists',
    '324': 'Potential Loyalists',
    '512': 'New Customers',
    '511': 'New Customers',
    '422': 'New Customers',
    '421': 'New Customers',
    '412': 'New Customers',
    '411': 'New Customers',
    '311': 'New Customers',
    '512': 'Promising',
    '511': 'Promising',
    '332': 'Customers Needing Attention',
    '231': 'Customers Needing Attention',
    '241': 'Customers Needing Attention',
    '251': 'Customers Needing Attention',
    '155': 'Cannot Lose Them',
    '154': 'Cannot Lose Them',
    '144': 'Cannot Lose Them',
    '214': 'At Risk',
    '215': 'At Risk',
    '115': 'At Risk',
    '114': 'At Risk',
    '113': 'At Risk',
    '131': 'Hibernating',
    '141': 'Hibernating',
    '151': 'Hibernating',
    '111': 'Lost',
    '112': 'Lost',
    '121': 'Lost',
    '122': 'Lost',
    '123': 'Lost',
    '132': 'Lost'
}

# ============================================================================
# ANOMALY TYPES
# ============================================================================

class AnomalyType(Enum):
    HIGH_FREQUENCY_DEPOSITS = 'high_frequency_deposits'
    UNUSUAL_BET_PATTERN = 'unusual_bet_pattern'
    RAPID_LOSS_CHASING = 'rapid_loss_chasing'
    BONUS_ABUSE = 'bonus_abuse'
    SESSION_LENGTH_ANOMALY = 'session_length_anomaly'
    WIN_LOSS_ANOMALY = 'win_loss_anomaly'
    LARGE_TRANSACTION = 'large_transaction'
    VELOCITY_ANOMALY = 'velocity_anomaly'

ANOMALY_TYPES = [a.value for a in AnomalyType]

# ============================================================================
# SEVERITY LEVELS
# ============================================================================

class SeverityLevel(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

SEVERITY_LEVELS = [s.value for s in SeverityLevel]

# ============================================================================
# MACHINE LEARNING CONSTANTS
# ============================================================================

# Feature engineering windows
FEATURE_WINDOWS = {
    'SHORT_TERM': 7,    # 7 days
    'MEDIUM_TERM': 30,  # 30 days
    'LONG_TERM': 90     # 90 days
}

# Churn prediction thresholds
CHURN_THRESHOLDS = {
    'INACTIVITY_DAYS': 30,
    'HIGH_RISK_PROBABILITY': 0.7,
    'MEDIUM_RISK_PROBABILITY': 0.5,
    'LOW_RISK_PROBABILITY': 0.3
}

# Model performance thresholds
MODEL_PERFORMANCE_THRESHOLDS = {
    'MIN_AUC_SCORE': 0.75,
    'MIN_PRECISION': 0.70,
    'MIN_RECALL': 0.65,
    'MIN_F1_SCORE': 0.70
}

# ============================================================================
# DASHBOARD CONSTANTS
# ============================================================================

# Color schemes for visualizations
COLOR_SCHEMES = {
    'PRIMARY': '#1f77b4',
    'SECONDARY': '#ff7f0e',
    'SUCCESS': '#2ca02c',
    'WARNING': '#ffbb78',
    'DANGER': '#d62728',
    'INFO': '#17a2b8',
    'CHURN_HIGH': '#d62728',
    'CHURN_MEDIUM': '#ff7f0e',
    'CHURN_LOW': '#2ca02c'
}

# KPI display formats
KPI_FORMATS = {
    'CURRENCY': '${:,.2f}',
    'PERCENTAGE': '{:.2f}%',
    'COUNT': '{:,}',
    'RATIO': '{:.3f}'
}

# Chart types
CHART_TYPES = {
    'LINE': 'line',
    'BAR': 'bar',
    'PIE': 'pie',
    'SCATTER': 'scatter',
    'HISTOGRAM': 'histogram',
    'HEATMAP': 'heatmap'
}

# ============================================================================
# DATA VALIDATION CONSTANTS
# ============================================================================

# Data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    'MAX_MISSING_PERCENTAGE': 5.0,  # 5% max missing values
    'MAX_DUPLICATE_PERCENTAGE': 0.1,  # 0.1% max duplicates
    'MIN_UNIQUE_VALUES': 2,  # Minimum unique values for categorical columns
    'MAX_OUTLIER_PERCENTAGE': 2.0  # 2% max outliers
}

# Column name patterns
COLUMN_PATTERNS = {
    'PLAYER_ID': r'.*player.*id.*',
    'TRANSACTION_ID': r'.*transaction.*id.*',
    'GAME_ID': r'.*game.*id.*',
    'AMOUNT': r'.*amount.*|.*value.*|.*sum.*',
    'TIMESTAMP': r'.*time.*|.*date.*|.*timestamp.*'
}

# ============================================================================
# FILE PATHS AND DIRECTORIES
# ============================================================================

# Relative paths from project root
PATHS = {
    'DATA_RAW': 'data/raw',
    'DATA_PROCESSED': 'data/processed',
    'DATA_EXPORTS': 'data/exports',
    'MODELS': 'models',
    'LOGS': 'logs',
    'REPORTS': 'reports',
    'NOTEBOOKS': 'notebooks',
    'SQL_QUERIES': 'database/queries',
    'SQL_ETL': 'database/etl'
}

# File extensions
FILE_EXTENSIONS = {
    'CSV': '.csv',
    'JSON': '.json',
    'PICKLE': '.pkl',
    'PARQUET': '.parquet',
    'SQL': '.sql',
    'LOG': '.log'
}

# ============================================================================
# API AND EXTERNAL SERVICE CONSTANTS
# ============================================================================

# Rate limits for external APIs
RATE_LIMITS = {
    'DEFAULT_REQUESTS_PER_MINUTE': 60,
    'BATCH_SIZE': 1000,
    'RETRY_ATTEMPTS': 3,
    'RETRY_DELAY': 1.0  # seconds
}

# ============================================================================
# BUSINESS RULES AND THRESHOLDS
# ============================================================================

# Player categorization
PLAYER_CATEGORIES = {
    'HIGH_VALUE_THRESHOLD': 10000,  # $10,000+ lifetime value
    'MEDIUM_VALUE_THRESHOLD': 1000,  # $1,000+ lifetime value
    'ACTIVE_DAYS_THRESHOLD': 30,    # Active within 30 days
    'FREQUENT_PLAYER_SESSIONS': 10,  # 10+ sessions per month
    'SESSION_LENGTH_LONG': 3600,    # 60+ minutes
    'SESSION_LENGTH_SHORT': 300     # 5 minutes or less
}

# Risk assessment
RISK_THRESHOLDS = {
    'HIGH_DEPOSIT_FREQUENCY': 10,   # 10+ deposits per day
    'LARGE_SINGLE_DEPOSIT': 5000,  # $5,000+ single deposit
    'RAPID_BETTING': 100,           # 100+ bets per hour
    'LOSS_CHASING_MULTIPLIER': 3.0, # 3x average bet after loss
    'SESSION_LENGTH_WARNING': 8,    # 8+ hours continuous play
    'WITHDRAWAL_DELAY_DAYS': 7      # 7+ days to withdraw
}

# ============================================================================
# LOGGING LEVELS AND FORMATS
# ============================================================================

# Log levels
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# Log format strings
LOG_FORMATS = {
    'DETAILED': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    'SIMPLE': '%(asctime)s - %(levelname)s - %(message)s',
    'ETL': '%(asctime)s - ETL - %(levelname)s - %(message)s',
    'ML': '%(asctime)s - ML - %(levelname)s - %(message)s'
}

# ============================================================================
# VERSION AND METADATA
# ============================================================================

PROJECT_METADATA = {
    'NAME': 'Casino Intelligence Hub',
    'VERSION': '1.0.0',
    'DESCRIPTION': 'Analytics platform for casino player behavior analysis',
    'AUTHOR': 'Data Science Team',
    'LICENSE': 'MIT',
    'PYTHON_VERSION': '3.8+',
    'CREATED_DATE': '2024-01-01'
}

# Supported data formats
SUPPORTED_FORMATS = ['csv', 'json', 'parquet', 'excel']

# Database connection timeouts
DB_TIMEOUTS = {
    'CONNECTION_TIMEOUT': 30,
    'QUERY_TIMEOUT': 300,
    'BATCH_TIMEOUT': 900
} 
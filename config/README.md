# Configuration Module

This folder contains all configuration files for the Casino Intelligence Hub project.

## Files Overview

### `database_config.py`
- **Purpose**: PostgreSQL database connection configuration
- **Contains**: Connection strings, schema definitions, connection pool settings
- **Usage**: Used by all database operations throughout the project

### `ml_config.py`
- **Purpose**: Machine learning model configurations and hyperparameters
- **Contains**: Model settings for churn prediction, segmentation, and anomaly detection
- **Usage**: Referenced during model training, evaluation, and prediction

### `dashboard_config.py`
- **Purpose**: Dashboard and visualization settings
- **Contains**: Streamlit configuration, Power BI connection settings, chart configurations
- **Usage**: Used by dashboard applications and visualization modules

## Key Features

1. **Environment-based Configuration**: Uses `.env` file for sensitive information
2. **Centralized Settings**: All project configurations in one location
3. **Type Safety**: Uses Python typing for better code quality
4. **Extensible**: Easy to add new configurations as project grows

## Usage Example

```python
from config.database_config import db_config
from config.ml_config import CHURN_CONFIG

# Database connection
conn_string = db_config.connection_string

# Model parameters
churn_features = CHURN_CONFIG['feature_columns']
```

## Environment Variables

Make sure to set up your `.env` file based on `.env.example` with:
- Database credentials
- Model parameters
- Application settings 
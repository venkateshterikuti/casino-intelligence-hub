# Source Code Module

This folder contains the core Python source code for the Casino Intelligence Hub project.

## Module Structure

### `database/`
Database connectivity and data management:
- `connection.py` - PostgreSQL connection manager with pooling
- `data_loader.py` - Data loading utilities for CSV to PostgreSQL
- `query_executor.py` - SQL query execution wrapper

### `preprocessing/`
Data cleaning and feature engineering:
- `data_cleaner.py` - Data validation and cleaning functions
- `feature_engineer.py` - Feature creation and transformation logic
- `data_validator.py` - Data quality validation rules

### `models/`
Machine learning model implementations:
- `churn_predictor.py` - Churn prediction model (RandomForest, XGBoost, Logistic Regression)
- `player_segmentation.py` - K-means clustering for player segmentation
- `anomaly_detector.py` - Isolation Forest for anomaly detection
- `model_evaluator.py` - Model evaluation and metrics calculation

### `visualization/`
Data visualization and dashboard preparation:
- `eda_plots.py` - Exploratory data analysis visualizations
- `model_plots.py` - Model performance and interpretation plots
- `dashboard_data_prep.py` - Data preparation for dashboard consumption

### `utils/`
Common utilities and helper functions:
- `logger.py` - Logging configuration and utilities
- `helpers.py` - Common helper functions
- `constants.py` - Project constants and enumerations

## Key Design Principles

1. **Modular Architecture**: Each module has a specific responsibility
2. **Database-Centric**: Heavy use of SQL for data processing
3. **Configuration-Driven**: All settings externalized to config files
4. **Error Handling**: Comprehensive logging and error management
5. **Type Safety**: Python typing for better code quality

## Usage Patterns

### Database Operations
```python
from src.database.connection import db_manager

# Execute query
df = db_manager.execute_query("SELECT * FROM players LIMIT 10")

# Execute SQL command
db_manager.execute_sql("UPDATE players SET vip_status = 'Gold' WHERE player_id = %s", {'player_id': 123})
```

### Model Training
```python
from src.models.churn_predictor import ChurnPredictor

# Initialize and train model
churn_model = ChurnPredictor()
churn_model.train(feature_data, target_data)

# Make predictions
predictions = churn_model.predict(new_data)
```

### Data Processing
```python
from src.preprocessing.feature_engineer import FeatureEngineer

# Create features
feature_eng = FeatureEngineer()
features = feature_eng.create_churn_features(player_data)
```

## Testing

Each module includes comprehensive unit tests in the `tests/` directory. Run tests with:

```bash
pytest tests/
```

## Dependencies

All dependencies are managed in `requirements.txt`. Key libraries:
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **psycopg2**: PostgreSQL connectivity
- **sqlalchemy**: ORM and database abstraction 
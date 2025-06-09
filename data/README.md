# Data Directory

This folder contains all data files for the Casino Intelligence Hub project.

## Folder Structure

### `raw/`
Contains raw, unprocessed data files:
- **all_casino_details2.csv** - Primary dataset (1.12GB) from Kaggle
- **Original data format** - No modifications or cleaning applied

### `processed/`
Contains cleaned and transformed data:
- **cleaned_transactions.csv** - Cleaned transaction data
- **player_aggregations.csv** - Player-level aggregations
- **feature_datasets.csv** - Engineered features for ML models

### `exports/`
Contains exported data for external tools:
- **dashboard_data.csv** - Data prepared for Power BI/Streamlit
- **model_predictions.csv** - ML model outputs
- **segment_profiles.csv** - Player segmentation results

## Data Pipeline Flow

```
raw/ → ETL Processing → processed/ → Feature Engineering → ML Models → exports/
```

## File Naming Conventions

- **Timestamps**: YYYY-MM-DD format
- **Descriptive names**: Include purpose and date
- **Version control**: Use v1, v2 for iterations

Example: `player_features_2024-01-15_v2.csv`

## Data Quality Guidelines

1. **Raw Data**: Never modify files in `raw/` folder
2. **Processed Data**: Include metadata about transformations
3. **Exports**: Include creation timestamp and source information
4. **Backup**: Keep copies of critical processed datasets

## Security & Privacy

- **PII Protection**: Player data should be anonymized
- **Access Control**: Sensitive data requires proper permissions
- **Compliance**: Follow data retention and privacy policies

## Usage Instructions

### Loading Raw Data
```bash
# Place the main dataset file here:
data/raw/all_casino_details2.csv

# Run ETL pipeline:
python scripts/run_etl_pipeline.py
```

### Accessing Processed Data
```python
import pandas as pd

# Load processed data
df = pd.read_csv('data/processed/cleaned_transactions.csv')
```

### Exporting for Dashboard
```bash
# Generate dashboard exports
python scripts/export_dashboard_data.py
```

## Data Sources

### Primary Dataset
- **Source**: Kaggle - "Casino games (all_casino_details2.csv)"
- **Size**: 1.12 GB
- **Period**: November 2023 - February 2024
- **Description**: Player transaction and gaming activity data

### Derived Datasets
- **Player Features**: Engineered from raw transactions
- **Segmentation Data**: RFM analysis results
- **Churn Labels**: Historical churn indicators

## Monitoring & Maintenance

- **Daily**: Check data freshness and quality
- **Weekly**: Review processing logs for errors
- **Monthly**: Archive old processed files
- **Quarterly**: Data quality assessment and cleanup 
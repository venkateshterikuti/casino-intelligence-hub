# Casino Player Intelligence Hub

## Project Overview
A comprehensive data science project that analyzes casino player behavior to predict churn, segment players, and detect anomalies. This project demonstrates end-to-end data pipeline development using PostgreSQL, Python machine learning, and interactive dashboards.

## Business Objectives
- **Churn Prediction**: Identify players likely to stop playing within 30 days
- **Player Segmentation**: Group players based on behavior for targeted marketing
- **Anomaly Detection**: Flag suspicious activities and potential problem gambling

## Technology Stack
- **Database**: PostgreSQL 14+ (local data warehouse)
- **Backend**: Python 3.8+ with pandas, scikit-learn, XGBoost
- **Visualization**: Power BI Desktop + Streamlit (hybrid approach)
- **Development**: Jupyter Notebooks
- **Version Control**: Git

## Project Structure
```
├── config/          # Configuration files
├── data/            # Raw and processed data
├── database/        # Database setup and ETL scripts
├── src/             # Source code modules
├── notebooks/       # Jupyter analysis notebooks
├── dashboards/      # Dashboard files
├── models/          # Trained model artifacts
├── scripts/         # Automation scripts
├── tests/           # Unit tests
├── docs/            # Documentation
└── logs/            # Application logs
```

## Setup Instructions

### 1. Prerequisites
- PostgreSQL 14+ installed and running
- Python 3.8+
- Git

### 2. Installation
```bash
# Clone repository
git clone <repository-url>
cd casino-intelligence-hub

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your database credentials
```

### 3. Database Setup
```bash
# Run database setup
python scripts/setup_database.py

# Load data
python scripts/run_etl_pipeline.py
```

### 4. Model Training
```bash
# Train all models
python scripts/train_models.py
```

### 5. Dashboard Options

#### Option A: Power BI (Recommended for Business Users)
1. Install Power BI Desktop
2. Install PostgreSQL ODBC driver: `brew install psqlodbc`
3. Follow setup guide: `docs/power_bi_setup_guide.md`
4. Connect to your PostgreSQL database
5. Use pre-built DAX measures from `dashboards/powerbi/dax_measures.txt`

#### Option B: Streamlit (For Technical Users)
```bash
# Run Streamlit dashboard
streamlit run dashboards/streamlit/app.py
```

#### Hybrid Approach (Best of Both Worlds)
- Use **Power BI** for executive reporting and business stakeholders
- Use **Streamlit** for data exploration and technical analysis
- Both connect to the same PostgreSQL database

## Key Features
- **SQL-Heavy Processing**: Advanced PostgreSQL queries for feature engineering
- **Multiple ML Models**: Churn prediction, clustering, anomaly detection
- **Interactive Dashboards**: Real-time business intelligence
- **Automated Pipeline**: End-to-end data processing automation

## Data Schema
The project uses a star schema with:
- **Fact Tables**: transactions, game_sessions
- **Dimension Tables**: players, games
- **ML Tables**: player_churn_features, player_segments, anomaly_flags

## Model Performance
- **Churn Model**: Target AUC-ROC > 0.85
- **Segmentation**: Silhouette Score > 0.5
- **Anomaly Detection**: <5% false positive rate

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License
MIT License

## Contact
[Your Contact Information] 
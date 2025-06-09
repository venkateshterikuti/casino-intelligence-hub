"""
Machine Learning configuration settings for the Casino Intelligence Hub.
Contains model hyperparameters, feature engineering settings, and evaluation metrics.
"""

import os
from typing import Dict, Any, List

# Churn Prediction Configuration
CHURN_CONFIG = {
    'prediction_window_days': int(os.getenv('CHURN_PREDICTION_DAYS', 30)),
    'feature_window_days': 90,
    'models': {
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0],
            'max_iter': 1000,
            'random_state': 42
        },
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'random_state': 42
        }
    },
    'feature_columns': [
        'recency_days',
        'frequency_transactions_last30d',
        'monetary_total_bets_last30d',
        'avg_bet_amount_last90d',
        'days_since_last_deposit',
        'sessions_per_week_last4w',
        'change_in_bet_frequency_wow'
    ],
    'target_column': 'churn_label_next30d'
}

# Player Segmentation Configuration
SEGMENTATION_CONFIG = {
    'rfm_features': ['recency', 'frequency', 'monetary'],
    'behavioral_features': [
        'avg_session_duration',
        'preferred_game_type',
        'total_sessions',
        'avg_bet_amount',
        'win_loss_ratio'
    ],
    'clustering': {
        'kmeans': {
            'n_clusters_range': [3, 4, 5, 6, 7, 8],
            'random_state': 42,
            'max_iter': 300
        },
        'dbscan': {
            'eps': [0.3, 0.5, 0.7],
            'min_samples': [3, 5, 7]
        }
    },
    'scaler': 'StandardScaler'
}

# Anomaly Detection Configuration
ANOMALY_CONFIG = {
    'threshold': float(os.getenv('ANOMALY_THRESHOLD', 2.5)),
    'models': {
        'isolation_forest': {
            'contamination': 0.05,
            'random_state': 42,
            'n_estimators': 100
        },
        'one_class_svm': {
            'kernel': 'rbf',
            'gamma': 'scale',
            'nu': 0.05
        }
    },
    'features': [
        'transaction_amount',
        'transaction_frequency',
        'session_duration',
        'bet_amount_variance',
        'deposit_frequency'
    ]
}

# Model Evaluation Configuration
EVALUATION_CONFIG = {
    'cv_folds': 5,
    'test_size': 0.2,
    'random_state': 42,
    'scoring_metrics': {
        'churn': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'segmentation': ['silhouette_score', 'calinski_harabasz_score'],
        'anomaly': ['precision', 'recall', 'f1']
    }
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'time_windows': {
        'short_term': 7,    # days
        'medium_term': 30,  # days
        'long_term': 90     # days
    },
    'aggregation_functions': ['sum', 'mean', 'std', 'min', 'max', 'count'],
    'categorical_encoding': 'target',  # target, onehot, label
    'handle_missing': 'median'  # median, mean, mode, drop
}

# Model Persistence Configuration
MODEL_PATHS = {
    'churn_model': 'models/churn/churn_model.pkl',
    'churn_scaler': 'models/churn/churn_scaler.pkl',
    'segmentation_model': 'models/segmentation/kmeans_model.pkl',
    'segmentation_scaler': 'models/segmentation/scaler.pkl',
    'anomaly_model': 'models/anomaly/isolation_forest.pkl'
} 
"""
Churn Prediction Model for Casino Intelligence Hub.
Implements multiple algorithms for predicting player churn including
Logistic Regression, Random Forest, and XGBoost.
"""

import logging
import pickle
import json
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb

from config.ml_config import CHURN_CONFIG, EVALUATION_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ChurnPredictor:
    """
    Churn Prediction Model that supports multiple algorithms.
    Handles feature engineering, model training, evaluation, and prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = CHURN_CONFIG['feature_columns']
        self.target_column = CHURN_CONFIG['target_column']
        self.model_config = CHURN_CONFIG['models']
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable for modeling.
        
        Args:
            df: Input DataFrame with features and target
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Preparing features for churn prediction")
        
        # Extract features and target
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Log feature statistics
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        logger.info(f"Churn rate: {y.mean():.3f}")
        
        return X, y
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train Logistic Regression model."""
        logger.info("Training Logistic Regression model")
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Grid search for best parameters
        param_grid = {
            'C': self.model_config['logistic_regression']['C']
        }
        
        lr = LogisticRegression(
            max_iter=self.model_config['logistic_regression']['max_iter'],
            random_state=self.model_config['logistic_regression']['random_state']
        )
        
        grid_search = GridSearchCV(
            lr, param_grid, cv=EVALUATION_CONFIG['cv_folds'], 
            scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        self.models['logistic_regression'] = grid_search.best_estimator_
        self.scalers['logistic_regression'] = scaler
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train Random Forest model."""
        logger.info("Training Random Forest model")
        
        param_grid = {
            'n_estimators': self.model_config['random_forest']['n_estimators'],
            'max_depth': self.model_config['random_forest']['max_depth'],
            'min_samples_split': self.model_config['random_forest']['min_samples_split']
        }
        
        rf = RandomForestClassifier(
            random_state=self.model_config['random_forest']['random_state']
        )
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=EVALUATION_CONFIG['cv_folds'],
            scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['random_forest'] = grid_search.best_estimator_
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': dict(zip(
                self.feature_columns,
                grid_search.best_estimator_.feature_importances_
            ))
        }
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train XGBoost model."""
        logger.info("Training XGBoost model")
        
        param_grid = {
            'n_estimators': self.model_config['xgboost']['n_estimators'],
            'max_depth': self.model_config['xgboost']['max_depth'],
            'learning_rate': self.model_config['xgboost']['learning_rate']
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.model_config['xgboost']['random_state'],
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=EVALUATION_CONFIG['cv_folds'],
            scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['xgboost'] = grid_search.best_estimator_
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': dict(zip(
                self.feature_columns,
                grid_search.best_estimator_.feature_importances_
            ))
        }
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all churn prediction models and select the best one.
        
        Args:
            df: Training data with features and target
            
        Returns:
            Dictionary with training results for all models
        """
        logger.info("Starting churn model training")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=EVALUATION_CONFIG['test_size'],
            random_state=EVALUATION_CONFIG['random_state'],
            stratify=y
        )
        
        results = {}
        
        # Train each model
        try:
            results['logistic_regression'] = self.train_logistic_regression(X_train, y_train)
        except Exception as e:
            logger.error(f"Failed to train Logistic Regression: {e}")
            
        try:
            results['random_forest'] = self.train_random_forest(X_train, y_train)
        except Exception as e:
            logger.error(f"Failed to train Random Forest: {e}")
            
        try:
            results['xgboost'] = self.train_xgboost(X_train, y_train)
        except Exception as e:
            logger.error(f"Failed to train XGBoost: {e}")
        
        # Evaluate all models and select best
        best_score = 0
        for model_name, model_results in results.items():
            if model_results['best_score'] > best_score:
                best_score = model_results['best_score']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        # Evaluate on test set
        test_results = self.evaluate_models(X_test, y_test)
        results['test_evaluation'] = test_results
        
        logger.info(f"Best model: {self.best_model_name} with CV score: {best_score:.4f}")
        
        return results
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """Evaluate all trained models on test data."""
        results = {}
        
        for model_name in self.models.keys():
            model = self.models[model_name]
            
            # Get predictions
            if model_name == 'logistic_regression':
                scaler = self.scalers[model_name]
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'top_decile_lift': self._calculate_top_decile_lift(y_test, y_pred_proba)
            }
        
        return results
    
    def _calculate_top_decile_lift(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate top decile lift for churn prediction."""
        df_temp = pd.DataFrame({
            'true': y_true,
            'proba': y_pred_proba
        }).sort_values('proba', ascending=False)
        
        top_decile_size = len(df_temp) // 10
        top_decile_churners = df_temp.head(top_decile_size)['true'].sum()
        total_churners = df_temp['true'].sum()
        
        if total_churners == 0:
            return 0.0
            
        lift = (top_decile_churners / top_decile_size) / (total_churners / len(df_temp))
        return lift
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions using the best trained model.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        
        if self.best_model_name == 'logistic_regression':
            scaler = self.scalers[self.best_model_name]
            X_scaled = scaler.transform(X)
            predictions = self.best_model.predict(X_scaled)
            probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = self.best_model.predict(X)
            probabilities = self.best_model.predict_proba(X)[:, 1]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the best model."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        if hasattr(self.best_model, 'feature_importances_'):
            return dict(zip(self.feature_columns, self.best_model.feature_importances_))
        elif hasattr(self.best_model, 'coef_'):
            return dict(zip(self.feature_columns, abs(self.best_model.coef_[0])))
        else:
            return {}
    
    def save_model(self, model_path: str, scaler_path: str = None):
        """Save the best trained model."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        if scaler_path and self.best_model_name in self.scalers:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[self.best_model_name], f)
        
        # Save feature importance
        importance_path = model_path.replace('.pkl', '_feature_importance.json')
        with open(importance_path, 'w') as f:
            json.dump(self.get_feature_importance(), f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """Load a trained model."""
        with open(model_path, 'rb') as f:
            self.best_model = pickle.load(f)
        
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                self.scalers['loaded_model'] = scaler
        
        logger.info(f"Model loaded from {model_path}")

# Example usage
if __name__ == "__main__":
    # This would typically be called from a training script
    pass 
"""
Player segmentation model for Casino Intelligence Hub.
Implements RFM analysis and K-means clustering for player behavioral segmentation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime, date
import pickle
from pathlib import Path

from ..utils.logger import get_logger, ml_logger
from ..utils.helpers import save_config, load_config

logger = get_logger(__name__)

class PlayerSegmentation:
    """
    Player segmentation using RFM analysis and behavioral clustering.
    """
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []
        self.segment_profiles = {}
    
    def calculate_rfm_metrics(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each player.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            DataFrame with RFM metrics and segments
        """
        logger.info("Calculating RFM metrics...")
        
        # Filter relevant transactions
        relevant_transactions = transactions_df[
            transactions_df['transaction_type'].isin(['bet', 'deposit', 'withdrawal'])
        ].copy()
        
        # Calculate RFM components
        max_date = relevant_transactions['transaction_timestamp'].max()
        
        rfm_data = relevant_transactions.groupby('player_id').agg({
            'transaction_timestamp': 'max',  # For recency
            'transaction_id': 'count',       # For frequency
            'amount': 'sum'                  # For monetary
        }).reset_index()
        
        rfm_data.columns = ['player_id', 'last_transaction', 'frequency', 'monetary']
        
        # Calculate recency in days
        rfm_data['recency_days'] = (max_date - rfm_data['last_transaction']).dt.days
        
        # Calculate RFM scores (1-5 scale)
        rfm_data['recency_score'] = pd.qcut(
            rfm_data['recency_days'], 
            q=5, 
            labels=[5, 4, 3, 2, 1],  # Lower recency days = higher score
            duplicates='drop'
        ).astype(int)
        
        rfm_data['frequency_score'] = pd.qcut(
            rfm_data['frequency'], 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(int)
        
        rfm_data['monetary_score'] = pd.qcut(
            rfm_data['monetary'], 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        ).astype(int)
        
        # Create combined RFM score
        rfm_data['rfm_score'] = (
            rfm_data['recency_score'] + 
            rfm_data['frequency_score'] + 
            rfm_data['monetary_score']
        )
        
        # Assign segment labels based on RFM scores
        rfm_data['rfm_segment'] = rfm_data.apply(self._assign_rfm_segment, axis=1)
        
        logger.info(f"RFM metrics calculated for {len(rfm_data)} players")
        ml_logger.log_feature_engineering('RFM Metrics', len(rfm_data))
        
        return rfm_data
    
    def _assign_rfm_segment(self, row: pd.Series) -> str:
        """Assign RFM segment based on scores."""
        r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
        
        # High-value segments
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and (f >= 3 or m >= 3):
            return 'Potential Loyalists'
        
        # New customers
        elif r >= 4 and f <= 2:
            return 'New Customers'
        
        # At-risk segments
        elif r <= 2 and f >= 3 and m >= 3:
            return 'Cannot Lose Them'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 3 and f <= 2:
            return 'Hibernating'
        elif r <= 2 and f <= 2:
            return 'Lost'
        
        # Default
        else:
            return 'Others'
    
    def extract_behavioral_features(
        self, 
        transactions_df: pd.DataFrame,
        sessions_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract behavioral features for clustering.
        
        Args:
            transactions_df: Transaction data
            sessions_df: Session data (optional)
            
        Returns:
            DataFrame with behavioral features
        """
        logger.info("Extracting behavioral features...")
        
        # Transaction-based features
        behavioral_features = []
        
        for player_id in transactions_df['player_id'].unique():
            player_data = transactions_df[transactions_df['player_id'] == player_id]
            
            # Basic activity metrics
            total_transactions = len(player_data)
            unique_days = player_data['transaction_timestamp'].dt.date.nunique()
            
            # Transaction type distribution
            tx_types = player_data['transaction_type'].value_counts(normalize=True)
            bet_ratio = tx_types.get('bet', 0)
            deposit_ratio = tx_types.get('deposit', 0)
            withdrawal_ratio = tx_types.get('withdrawal', 0)
            
            # Amount statistics
            bet_data = player_data[player_data['transaction_type'] == 'bet']
            avg_bet = bet_data['amount'].mean() if len(bet_data) > 0 else 0
            std_bet = bet_data['amount'].std() if len(bet_data) > 0 else 0
            max_bet = bet_data['amount'].max() if len(bet_data) > 0 else 0
            
            # Temporal patterns
            hour_dist = player_data['transaction_timestamp'].dt.hour.value_counts(normalize=True)
            peak_hour_ratio = hour_dist.max() if len(hour_dist) > 0 else 0
            
            dow_dist = player_data['transaction_timestamp'].dt.dayofweek.value_counts(normalize=True)
            weekend_ratio = dow_dist.loc[[5, 6]].sum() if any(x in dow_dist.index for x in [5, 6]) else 0
            
            # Game diversity (if game_id available)
            game_diversity = 0
            if 'game_id' in player_data.columns:
                unique_games = player_data['game_id'].nunique()
                game_diversity = unique_games / total_transactions if total_transactions > 0 else 0
            
            features = {
                'player_id': player_id,
                'total_transactions': total_transactions,
                'unique_active_days': unique_days,
                'transactions_per_day': total_transactions / unique_days if unique_days > 0 else 0,
                'bet_ratio': bet_ratio,
                'deposit_ratio': deposit_ratio,
                'withdrawal_ratio': withdrawal_ratio,
                'avg_bet_amount': avg_bet,
                'std_bet_amount': std_bet,
                'max_bet_amount': max_bet,
                'bet_volatility': std_bet / avg_bet if avg_bet > 0 else 0,
                'peak_hour_ratio': peak_hour_ratio,
                'weekend_activity_ratio': weekend_ratio,
                'game_diversity': game_diversity
            }
            
            behavioral_features.append(features)
        
        behavioral_df = pd.DataFrame(behavioral_features)
        
        # Add session features if available
        if sessions_df is not None:
            session_features = self._extract_session_features(sessions_df)
            behavioral_df = behavioral_df.merge(session_features, on='player_id', how='left')
        
        logger.info(f"Behavioral features extracted for {len(behavioral_df)} players")
        return behavioral_df
    
    def _extract_session_features(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Extract session-based features."""
        session_agg = sessions_df.groupby('player_id').agg({
            'session_duration_seconds': ['mean', 'std', 'max', 'count'],
            'total_bets_in_session': ['mean', 'std'],
            'total_wins_in_session': ['mean', 'sum']
        }).reset_index()
        
        # Flatten column names
        session_agg.columns = [
            'player_id', 'avg_session_duration', 'std_session_duration',
            'max_session_duration', 'total_sessions',
            'avg_bets_per_session', 'std_bets_per_session',
            'avg_wins_per_session', 'total_wins'
        ]
        
        # Calculate derived metrics
        session_agg['session_efficiency'] = (
            session_agg['avg_bets_per_session'] / (session_agg['avg_session_duration'] / 60)
        ).fillna(0)  # Bets per minute
        
        return session_agg
    
    def perform_clustering(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform K-means clustering on behavioral features.
        
        Args:
            features_df: DataFrame with features for clustering
            
        Returns:
            DataFrame with cluster assignments
        """
        logger.info(f"Performing K-means clustering with {self.n_clusters} clusters")
        
        # Select numeric features for clustering
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        self.feature_columns = [col for col in numeric_features if col != 'player_id']
        
        # Prepare data for clustering
        clustering_data = features_df[self.feature_columns].fillna(0)
        
        # Scale features
        clustering_data_scaled = self.scaler.fit_transform(clustering_data)
        
        # Perform clustering
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        cluster_labels = self.kmeans_model.fit_predict(clustering_data_scaled)
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(clustering_data_scaled, cluster_labels)
        
        logger.info(f"Clustering completed - Silhouette Score: {silhouette_avg:.3f}")
        
        # Add cluster assignments
        result_df = features_df.copy()
        result_df['behavioral_cluster'] = cluster_labels
        
        # Store metrics
        self.clustering_metrics = {
            'silhouette_score': silhouette_avg,
            'n_clusters': self.n_clusters,
            'n_features': len(self.feature_columns),
            'n_samples': len(result_df)
        }
        
        ml_logger.log_model_training('K-Means Clustering', self.clustering_metrics)
        
        return result_df
    
    def create_segment_profiles(
        self, 
        rfm_df: pd.DataFrame, 
        behavioral_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create profiles for each segment.
        
        Args:
            rfm_df: RFM analysis results
            behavioral_df: Behavioral clustering results
            
        Returns:
            Dictionary with segment profiles
        """
        logger.info("Creating segment profiles...")
        
        # Merge RFM and behavioral data
        combined_df = rfm_df.merge(behavioral_df, on='player_id', how='inner')
        
        profiles = {}
        
        # RFM segment profiles
        for segment in combined_df['rfm_segment'].unique():
            segment_data = combined_df[combined_df['rfm_segment'] == segment]
            profiles[f'RFM_{segment}'] = self._calculate_profile_stats(segment_data)
        
        # Behavioral cluster profiles
        if 'behavioral_cluster' in combined_df.columns:
            for cluster in combined_df['behavioral_cluster'].unique():
                cluster_data = combined_df[combined_df['behavioral_cluster'] == cluster]
                profiles[f'Cluster_{cluster}'] = self._calculate_profile_stats(cluster_data)
        
        self.segment_profiles = profiles
        logger.info(f"Created profiles for {len(profiles)} segments")
        
        return profiles
    
    def _calculate_profile_stats(self, segment_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for a segment."""
        key_metrics = [
            'recency_days', 'frequency', 'monetary', 'rfm_score',
            'avg_bet_amount', 'total_transactions', 'weekend_activity_ratio'
        ]
        
        profile = {
            'size': len(segment_data),
            'percentage': len(segment_data) / len(segment_data) * 100,
            'metrics': {}
        }
        
        for metric in key_metrics:
            if metric in segment_data.columns:
                profile['metrics'][metric] = {
                    'mean': segment_data[metric].mean(),
                    'median': segment_data[metric].median(),
                    'std': segment_data[metric].std()
                }
        
        return profile
    
    def fit(
        self, 
        transactions_df: pd.DataFrame, 
        sessions_df: Optional[pd.DataFrame] = None
    ) -> 'PlayerSegmentation':
        """
        Fit the segmentation model.
        
        Args:
            transactions_df: Transaction data
            sessions_df: Session data (optional)
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting player segmentation model...")
        
        # Calculate RFM metrics
        rfm_data = self.calculate_rfm_metrics(transactions_df)
        
        # Extract behavioral features
        behavioral_data = self.extract_behavioral_features(transactions_df, sessions_df)
        
        # Perform clustering
        clustered_data = self.perform_clustering(behavioral_data)
        
        # Create segment profiles
        self.segment_profiles = self.create_segment_profiles(rfm_data, clustered_data)
        
        self.is_fitted = True
        logger.info("Player segmentation model fitted successfully")
        
        return self
    
    def predict(
        self, 
        transactions_df: pd.DataFrame,
        sessions_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Predict segments for new data.
        
        Args:
            transactions_df: Transaction data
            sessions_df: Session data (optional)
            
        Returns:
            DataFrame with segment predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info("Predicting player segments...")
        
        # Calculate RFM metrics
        rfm_predictions = self.calculate_rfm_metrics(transactions_df)
        
        # Extract behavioral features
        behavioral_features = self.extract_behavioral_features(transactions_df, sessions_df)
        
        # Predict clusters if model exists
        if self.kmeans_model is not None:
            feature_data = behavioral_features[self.feature_columns].fillna(0)
            feature_data_scaled = self.scaler.transform(feature_data)
            cluster_predictions = self.kmeans_model.predict(feature_data_scaled)
            behavioral_features['behavioral_cluster'] = cluster_predictions
        
        # Combine predictions
        predictions = rfm_predictions.merge(
            behavioral_features[['player_id', 'behavioral_cluster'] + 
                             [col for col in behavioral_features.columns 
                              if col not in ['player_id', 'behavioral_cluster']]], 
            on='player_id', 
            how='inner'
        )
        
        logger.info(f"Predicted segments for {len(predictions)} players")
        return predictions
    
    def save_model(self, model_path: str) -> None:
        """Save the fitted model."""
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn models
        if self.kmeans_model is not None:
            with open(model_path / 'kmeans_model.pkl', 'wb') as f:
                pickle.dump(self.kmeans_model, f)
        
        with open(model_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save model metadata
        model_info = {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'feature_columns': self.feature_columns,
            'segment_profiles': self.segment_profiles,
            'clustering_metrics': getattr(self, 'clustering_metrics', {}),
            'is_fitted': self.is_fitted,
            'created_at': datetime.now().isoformat()
        }
        
        save_config(model_info, model_path / 'model_info.json')
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> 'PlayerSegmentation':
        """Load a saved model."""
        model_path = Path(model_path)
        
        # Load model info
        model_info = load_config(model_path / 'model_info.json')
        
        self.n_clusters = model_info['n_clusters']
        self.random_state = model_info['random_state']
        self.feature_columns = model_info['feature_columns']
        self.segment_profiles = model_info['segment_profiles']
        self.clustering_metrics = model_info.get('clustering_metrics', {})
        self.is_fitted = model_info['is_fitted']
        
        # Load sklearn models
        kmeans_path = model_path / 'kmeans_model.pkl'
        if kmeans_path.exists():
            with open(kmeans_path, 'rb') as f:
                self.kmeans_model = pickle.load(f)
        
        with open(model_path / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return self 
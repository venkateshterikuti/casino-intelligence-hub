#!/usr/bin/env python3
"""
Churn Feature Engineering for Casino Intelligence Hub
Creates features for predicting 30-day player churn.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from database.connection import DatabaseConnection
from utils.logger import get_logger

logger = get_logger(__name__)

class ChurnFeatureEngineering:
    """Creates features for churn prediction model."""
    
    def __init__(self, prediction_window_days: int = 30):
        self.prediction_window = prediction_window_days
        self.db = DatabaseConnection()
        
    def create_churn_features(self, snapshot_date: str = None) -> pd.DataFrame:
        """
        Create comprehensive features for churn prediction.
        
        Args:
            snapshot_date: Date to create features as of (YYYY-MM-DD)
                         If None, uses latest date in data
        """
        logger.info("🔧 Starting churn feature engineering...")
        
        # Get snapshot date
        if snapshot_date is None:
            snapshot_date = self._get_latest_date()
            
        logger.info(f"📅 Creating features as of: {snapshot_date}")
        
        # Load raw data
        df = self._load_transaction_data()
        
        # Create features
        features = self._create_player_features(df, snapshot_date)
        
        # Create churn labels
        features = self._create_churn_labels(df, features, snapshot_date)
        
        # Save features to database
        self._save_features_to_db(features, snapshot_date)
        
        logger.info(f"✅ Feature engineering complete! Created {len(features)} player records")
        return features
    
    def _load_transaction_data(self) -> pd.DataFrame:
        """Load transaction data from database."""
        logger.info("📊 Loading transaction data...")
        
        query = """
        SELECT 
            user_id,
            calc_date,
            game_type,
            bet,
            won,
            profit
        FROM raw_casino_data
        ORDER BY user_id, calc_date
        """
        
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)
            
        df['calc_date'] = pd.to_datetime(df['calc_date'])
        logger.info(f"📈 Loaded {len(df):,} transactions for {df['user_id'].nunique():,} players")
        
        return df
    
    def _get_latest_date(self) -> str:
        """Get the latest date in the dataset."""
        query = "SELECT MAX(calc_date) as max_date FROM raw_casino_data"
        
        with self.db.get_connection() as conn:
            result = pd.read_sql(query, conn)
            
        return result['max_date'].iloc[0].strftime('%Y-%m-%d')
    
    def _create_player_features(self, df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
        """Create comprehensive player features."""
        logger.info("🛠️ Creating player features...")
        
        snapshot_dt = pd.to_datetime(snapshot_date)
        
        # Filter data up to snapshot date
        df_filtered = df[df['calc_date'] <= snapshot_dt].copy()
        
        # Create different time windows
        windows = {
            'last_7d': 7,
            'last_14d': 14,
            'last_30d': 30,
            'last_60d': 60,
            'last_90d': 90,
            'all_time': 999999
        }
        
        features_list = []
        
        for player_id in df_filtered['user_id'].unique():
            player_data = df_filtered[df_filtered['user_id'] == player_id]
            
            player_features = {'player_id': player_id, 'snapshot_date': snapshot_date}
            
            # Basic features
            player_features.update(self._basic_features(player_data, snapshot_dt))
            
            # Time-windowed features
            for window_name, days in windows.items():
                if days == 999999:  # All time
                    window_data = player_data
                else:
                    window_start = snapshot_dt - timedelta(days=days)
                    window_data = player_data[player_data['calc_date'] >= window_start]
                
                if len(window_data) > 0:
                    player_features.update(
                        self._windowed_features(window_data, window_name)
                    )
                else:
                    # Fill with zeros if no data in window
                    player_features.update(self._empty_window_features(window_name))
            
            # Advanced features
            player_features.update(self._advanced_features(player_data, snapshot_dt))
            
            features_list.append(player_features)
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)  # Fill any remaining NaN values
        
        logger.info(f"📊 Created {len(features_df.columns)} features for {len(features_df)} players")
        return features_df
    
    def _basic_features(self, player_data: pd.DataFrame, snapshot_dt: datetime) -> Dict:
        """Create basic player features."""
        if len(player_data) == 0:
            return {}
            
        return {
            'first_transaction_date': player_data['calc_date'].min(),
            'last_transaction_date': player_data['calc_date'].max(),
            'days_since_first_transaction': (snapshot_dt - player_data['calc_date'].min()).days,
            'days_since_last_transaction': (snapshot_dt - player_data['calc_date'].max()).days,
            'total_transactions': len(player_data),
            'total_bet_amount': player_data['bet'].sum(),
            'total_won_amount': player_data['won'].sum(),
            'total_profit_amount': player_data['profit'].sum(),
            'unique_game_types': player_data['game_type'].nunique(),
            'active_days': player_data['calc_date'].nunique()
        }
    
    def _windowed_features(self, window_data: pd.DataFrame, window_name: str) -> Dict:
        """Create features for a specific time window."""
        if len(window_data) == 0:
            return self._empty_window_features(window_name)
        
        prefix = f"{window_name}_"
        
        return {
            f"{prefix}transactions": len(window_data),
            f"{prefix}bet_amount": window_data['bet'].sum(),
            f"{prefix}won_amount": window_data['won'].sum(),
            f"{prefix}profit_amount": window_data['profit'].sum(),
            f"{prefix}avg_bet": window_data['bet'].mean(),
            f"{prefix}max_bet": window_data['bet'].max(),
            f"{prefix}min_bet": window_data['bet'].min(),
            f"{prefix}bet_std": window_data['bet'].std() if len(window_data) > 1 else 0,
            f"{prefix}active_days": window_data['calc_date'].nunique(),
            f"{prefix}game_types": window_data['game_type'].nunique(),
            f"{prefix}win_rate": (window_data['profit'] > 0).mean(),
            f"{prefix}avg_profit_per_transaction": window_data['profit'].mean(),
        }
    
    def _empty_window_features(self, window_name: str) -> Dict:
        """Create zero features for empty time windows."""
        prefix = f"{window_name}_"
        
        return {
            f"{prefix}transactions": 0,
            f"{prefix}bet_amount": 0,
            f"{prefix}won_amount": 0,
            f"{prefix}profit_amount": 0,
            f"{prefix}avg_bet": 0,
            f"{prefix}max_bet": 0,
            f"{prefix}min_bet": 0,
            f"{prefix}bet_std": 0,
            f"{prefix}active_days": 0,
            f"{prefix}game_types": 0,
            f"{prefix}win_rate": 0,
            f"{prefix}avg_profit_per_transaction": 0,
        }
    
    def _advanced_features(self, player_data: pd.DataFrame, snapshot_dt: datetime) -> Dict:
        """Create advanced behavioral features."""
        if len(player_data) < 2:
            return {
                'betting_trend_30d': 0,
                'activity_trend_30d': 0,
                'game_diversity_score': 0,
                'session_regularity': 0,
                'big_win_recency': 999,
                'volatility_score': 0
            }
        
        # Sort by date
        player_data = player_data.sort_values('calc_date')
        
        # Betting trend (last 30 days)
        last_30d = player_data[
            player_data['calc_date'] >= snapshot_dt - timedelta(days=30)
        ]
        
        betting_trend = 0
        if len(last_30d) >= 2:
            # Linear regression on daily bet amounts
            daily_bets = last_30d.groupby('calc_date')['bet'].sum().reset_index()
            if len(daily_bets) >= 2:
                x = np.arange(len(daily_bets))
                y = daily_bets['bet'].values
                slope = np.polyfit(x, y, 1)[0]
                betting_trend = slope
        
        # Activity trend
        last_30d_activity = last_30d['calc_date'].nunique() if len(last_30d) > 0 else 0
        prev_30d = player_data[
            (player_data['calc_date'] >= snapshot_dt - timedelta(days=60)) &
            (player_data['calc_date'] < snapshot_dt - timedelta(days=30))
        ]
        prev_30d_activity = prev_30d['calc_date'].nunique() if len(prev_30d) > 0 else 0
        
        activity_trend = last_30d_activity - prev_30d_activity
        
        # Game diversity score (Shannon entropy)
        game_counts = player_data['game_type'].value_counts()
        game_probs = game_counts / game_counts.sum()
        game_diversity = -sum(game_probs * np.log2(game_probs)) if len(game_probs) > 1 else 0
        
        # Session regularity (coefficient of variation of daily transactions)
        daily_tx = player_data.groupby('calc_date').size()
        session_regularity = daily_tx.std() / daily_tx.mean() if daily_tx.mean() > 0 else 0
        
        # Big win recency (days since profit > 10x average bet)
        avg_bet = player_data['bet'].mean()
        big_wins = player_data[player_data['profit'] > avg_bet * 10]
        big_win_recency = (snapshot_dt - big_wins['calc_date'].max()).days if len(big_wins) > 0 else 999
        
        # Volatility score (coefficient of variation of profits)
        volatility = player_data['profit'].std() / abs(player_data['profit'].mean()) if player_data['profit'].mean() != 0 else 0
        
        return {
            'betting_trend_30d': betting_trend,
            'activity_trend_30d': activity_trend,
            'game_diversity_score': game_diversity,
            'session_regularity': session_regularity,
            'big_win_recency': big_win_recency,
            'volatility_score': volatility
        }
    
    def _create_churn_labels(self, df: pd.DataFrame, features: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
        """Create churn labels for training."""
        logger.info("🏷️ Creating churn labels...")
        
        snapshot_dt = pd.to_datetime(snapshot_date)
        future_start = snapshot_dt + timedelta(days=1)
        future_end = snapshot_dt + timedelta(days=self.prediction_window)
        
        # Get future transactions for each player
        future_data = df[
            (df['calc_date'] > snapshot_dt) & 
            (df['calc_date'] <= future_end)
        ]
        
        # Players with future activity are not churned
        active_players = set(future_data['user_id'].unique())
        
        # Create churn labels
        features['churn_label_next30d'] = features['player_id'].apply(
            lambda x: 0 if x in active_players else 1
        )
        
        churn_rate = features['churn_label_next30d'].mean()
        logger.info(f"📊 Churn rate: {churn_rate:.2%} ({features['churn_label_next30d'].sum():,} churned players)")
        
        return features
    
    def _save_features_to_db(self, features: pd.DataFrame, snapshot_date: str):
        """Save features to database."""
        logger.info("💾 Saving features to database...")
        
        # Prepare data for insertion
        features_clean = features.copy()
        
        # Convert datetime columns to strings
        datetime_columns = ['first_transaction_date', 'last_transaction_date']
        for col in datetime_columns:
            if col in features_clean.columns:
                features_clean[col] = features_clean[col].astype(str)
        
        # Insert/update features in database
        with self.db.get_connection() as conn:
            # Delete existing features for this snapshot date
            delete_query = """
            DELETE FROM player_churn_features 
            WHERE snapshot_date = %s
            """
            conn.execute(delete_query, (snapshot_date,))
            
            # Insert new features
            features_clean.to_sql(
                'player_churn_features',
                conn,
                if_exists='append',
                index=False,
                method='multi'
            )
            
        logger.info("✅ Features saved to player_churn_features table")

def main():
    """Main function to run feature engineering."""
    print("🔧 Casino Churn Feature Engineering")
    print("=" * 50)
    
    try:
        engineer = ChurnFeatureEngineering()
        
        # Create features for latest date
        features = engineer.create_churn_features()
        
        print(f"✅ Feature engineering completed!")
        print(f"📊 Created features for {len(features):,} players")
        print(f"🏷️ Churn rate: {features['churn_label_next30d'].mean():.2%}")
        print()
        print("📈 Feature summary:")
        print(f"   - Total features: {len(features.columns)}")
        print(f"   - Players: {len(features):,}")
        print(f"   - Date range: {features['snapshot_date'].iloc[0]}")
        print()
        print("🎯 Next steps:")
        print("   1. python src/models/churn_predictor.py")
        print("   2. python src/models/evaluate_churn_model.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
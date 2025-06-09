#!/usr/bin/env python3
"""
Load the actual casino dataset (all_casino_details2.csv) into PostgreSQL.
Handles the 1.6M+ rows with proper chunking and data type conversion.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.database.data_loader import DataLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_date_format(date_str):
    """Convert date from M/D/YYYY format to YYYY-MM-DD."""
    try:
        # Handle different date formats
        if pd.isna(date_str):
            return None
        return pd.to_datetime(date_str, format='%m/%d/%Y').strftime('%Y-%m-%d')
    except:
        try:
            return pd.to_datetime(date_str).strftime('%Y-%m-%d')
        except:
            return None

def load_actual_casino_data():
    """Load the actual casino dataset into PostgreSQL."""
    
    print("🎰 Loading ACTUAL Casino Dataset")
    print("=" * 50)
    
    csv_file = 'data/raw/all_casino_details2.csv'
    
    try:
        # First, get basic info about the dataset
        print("📊 Analyzing dataset...")
        sample_df = pd.read_csv(csv_file, nrows=1000)
        print(f"✅ Dataset columns: {list(sample_df.columns)}")
        print(f"✅ Sample shape: {sample_df.shape}")
        print(f"✅ Game types: {sample_df['GAME_TYPE'].unique()}")
        
        # Estimate total rows
        total_rows = sum(1 for line in open(csv_file)) - 1  # -1 for header
        print(f"✅ Estimated total rows: {total_rows:,}")
        
        # Load data in chunks to handle large file
        chunk_size = 10000
        loader = DataLoader()
        
        print(f"\n📥 Loading data in chunks of {chunk_size:,} rows...")
        
        total_loaded = 0
        chunks_processed = 0
        
        # Read and process the CSV in chunks
        for chunk_num, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
            
            # Clean column names to match database schema
            chunk.columns = [col.lower() for col in chunk.columns]
            
            # Convert date format from M/D/YYYY to YYYY-MM-DD
            print(f"Processing chunk {chunk_num + 1}...")
            chunk['calc_date'] = pd.to_datetime(chunk['calc_date'], format='%m/%d/%Y')
            
            # Ensure data types match schema
            chunk['user_id'] = chunk['user_id'].astype(int)
            chunk['bet'] = pd.to_numeric(chunk['bet'], errors='coerce')
            chunk['won'] = pd.to_numeric(chunk['won'], errors='coerce')
            chunk['profit'] = pd.to_numeric(chunk['profit'], errors='coerce')
            
            # Handle any NaN values
            chunk = chunk.dropna()
            
            # Load chunk to database
            if chunk_num == 0:
                # First chunk - replace existing data
                if_exists = 'replace'
            else:
                # Subsequent chunks - append
                if_exists = 'append'
            
            try:
                chunk.to_sql(
                    'raw_casino_data',
                    loader.db_manager.engine,
                    if_exists=if_exists,
                    index=False,
                    method='multi'
                )
                
                total_loaded += len(chunk)
                chunks_processed += 1
                
                # Progress update every 10 chunks
                if chunk_num % 10 == 0:
                    print(f"   ✅ Processed {chunks_processed} chunks, loaded {total_loaded:,} rows")
                    
            except Exception as e:
                print(f"   ❌ Error loading chunk {chunk_num}: {e}")
                continue
        
        print(f"\n🎉 Data loading completed!")
        print(f"📊 Total rows loaded: {total_loaded:,}")
        print(f"📦 Chunks processed: {chunks_processed}")
        
        return True, total_loaded
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False, 0

def verify_data_load():
    """Verify the data was loaded correctly."""
    print("\n🔍 Verifying data load...")
    
    try:
        loader = DataLoader()
        
        # Get basic statistics
        result = loader.db_manager.execute_query("""
            SELECT 
                COUNT(*) as total_rows,
                MIN(calc_date) as min_date,
                MAX(calc_date) as max_date,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT game_type) as unique_game_types,
                COUNT(DISTINCT game_name) as unique_games,
                SUM(bet) as total_bet_volume,
                SUM(won) as total_won_volume,
                SUM(profit) as total_profit
            FROM raw_casino_data
        """)
        
        stats = result.iloc[0]
        
        print("\n📈 DATABASE STATISTICS:")
        print(f"   Total Rows: {stats['total_rows']:,}")
        print(f"   Date Range: {stats['min_date']} to {stats['max_date']}")
        print(f"   Unique Users: {stats['unique_users']:,}")
        print(f"   Game Types: {stats['unique_game_types']}")
        print(f"   Unique Games: {stats['unique_games']}")
        print(f"   Total Bet Volume: ${stats['total_bet_volume']:,.2f}")
        print(f"   Total Won Volume: ${stats['total_won_volume']:,.2f}")
        print(f"   Total Profit: ${stats['total_profit']:,.2f}")
        
        # Get game type breakdown
        game_types = loader.db_manager.execute_query("""
            SELECT 
                game_type,
                COUNT(*) as transactions,
                COUNT(DISTINCT user_id) as unique_players,
                SUM(bet) as total_bet,
                SUM(profit) as total_profit
            FROM raw_casino_data 
            GROUP BY game_type 
            ORDER BY transactions DESC
        """)
        
        print("\n🎮 GAME TYPE BREAKDOWN:")
        for _, row in game_types.iterrows():
            print(f"   {row['game_type']}: {row['transactions']:,} transactions, "
                  f"{row['unique_players']:,} players, "
                  f"${row['total_bet']:,.2f} bet volume")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying data: {e}")
        return False

if __name__ == "__main__":
    print("🎰 CASINO INTELLIGENCE HUB - REAL DATA LOADER")
    print("=" * 60)
    
    # Load the actual data
    success, rows_loaded = load_actual_casino_data()
    
    if success:
        # Verify the load
        verify_success = verify_data_load()
        
        if verify_success:
            print("\n✅ SUCCESS! Your actual casino data is now loaded and ready for analysis!")
            print("\n🚀 Next steps:")
            print("   1. Run the dashboard: python start_dashboard.py")
            print("   2. Connect Power BI to see your real data")
            print("   3. Train ML models on your actual data")
        else:
            print("\n⚠️  Data loaded but verification failed. Check the database.")
    else:
        print("\n❌ Data loading failed. Please check the file and database connection.") 
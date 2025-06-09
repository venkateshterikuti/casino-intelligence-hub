#!/usr/bin/env python3
"""
Database setup automation script for Casino Intelligence Hub.
Creates database, schema, tables, and indexes automatically.
"""

import os
import sys
import logging
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.database_config import db_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseSetup:
    """Handles automated database setup and initialization."""
    
    def __init__(self):
        self.db_config = db_config
        self.setup_scripts_path = Path(__file__).parent.parent / "database" / "setup"
        
    def create_database_if_not_exists(self):
        """Create the database if it doesn't exist."""
        logger.info("Checking if database exists...")
        
        # Connect to postgres database to create our target database
        conn_params = self.db_config.connection_params.copy()
        target_db = conn_params.pop('database')
        conn_params['database'] = 'postgres'
        
        try:
            conn = psycopg2.connect(**conn_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (target_db,)
            )
            
            if cursor.fetchone():
                logger.info(f"Database '{target_db}' already exists")
            else:
                # Create database
                cursor.execute(f'CREATE DATABASE "{target_db}"')
                logger.info(f"Database '{target_db}' created successfully")
                
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    def execute_sql_file(self, file_path: Path):
        """Execute SQL commands from a file."""
        logger.info(f"Executing SQL file: {file_path.name}")
        
        try:
            with psycopg2.connect(**self.db_config.connection_params) as conn:
                with conn.cursor() as cursor:
                    with open(file_path, 'r') as f:
                        sql_content = f.read()
                    
                    # Split by semicolon and execute each statement
                    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                    
                    for statement in statements:
                        if statement:
                            cursor.execute(statement)
                    
                    conn.commit()
            
            logger.info(f"Successfully executed {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to execute {file_path.name}: {e}")
            raise
    
    def run_setup_scripts(self):
        """Run all setup scripts in order."""
        logger.info("Running database setup scripts...")
        
        setup_files = [
            "01_create_database.sql",
            "02_create_schema.sql", 
            "03_create_tables.sql",
            "04_create_indexes.sql"
        ]
        
        for script_name in setup_files:
            script_path = self.setup_scripts_path / script_name
            
            if script_path.exists():
                self.execute_sql_file(script_path)
            else:
                logger.warning(f"Setup script not found: {script_name}")
    
    def test_connection(self):
        """Test database connection and basic functionality."""
        logger.info("Testing database connection...")
        
        try:
            with psycopg2.connect(**self.db_config.connection_params) as conn:
                with conn.cursor() as cursor:
                    # Test basic query
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    logger.info(f"PostgreSQL version: {version}")
                    
                    # Test schema
                    cursor.execute("""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name = 'casino_intelligence'
                    """)
                    
                    if cursor.fetchone():
                        logger.info("Schema 'casino_intelligence' found")
                    else:
                        logger.warning("Schema 'casino_intelligence' not found")
                    
                    # Test tables
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'casino_intelligence'
                    """)
                    
                    tables = [row[0] for row in cursor.fetchall()]
                    logger.info(f"Found tables: {', '.join(tables)}")
            
            logger.info("Database connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def setup_database(self):
        """Complete database setup process."""
        logger.info("Starting database setup...")
        
        try:
            # Step 1: Create database if needed
            self.create_database_if_not_exists()
            
            # Step 2: Run setup scripts
            self.run_setup_scripts()
            
            # Step 3: Test connection
            if self.test_connection():
                logger.info("Database setup completed successfully!")
                return True
            else:
                logger.error("Database setup completed but connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False

def main():
    """Main function to run database setup."""
    print("🎰 Casino Intelligence Hub - Database Setup")
    print("=" * 50)
    
    # Check if database configuration is available
    try:
        db_setup = DatabaseSetup()
        
        # Validate connection parameters
        if not all([
            db_setup.db_config.host,
            db_setup.db_config.username,
            db_setup.db_config.database
        ]):
            print("❌ Error: Database configuration incomplete")
            print("Please check your .env file and ensure all database parameters are set")
            return False
        
        print(f"📊 Target Database: {db_setup.db_config.database}")
        print(f"🖥️  Host: {db_setup.db_config.host}:{db_setup.db_config.port}")
        print(f"👤 User: {db_setup.db_config.username}")
        print()
        
        # Ask for confirmation
        response = input("Proceed with database setup? (y/N): ").lower().strip()
        if response != 'y':
            print("Setup cancelled")
            return False
        
        # Run setup
        success = db_setup.setup_database()
        
        if success:
            print("✅ Database setup completed successfully!")
            print()
            print("Next steps:")
            print("1. Place your data file in: data/raw/all_casino_details2.csv")
            print("2. Run: python scripts/run_etl_pipeline.py")
            print("3. Train models: python scripts/train_models.py")
        else:
            print("❌ Database setup failed. Check logs for details.")
            
        return success
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
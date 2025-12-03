"""
Data Loader for Vibration Analysis System

This module handles loading data from Databricks into SQLite cache.
It only focuses on data loading, not analysis functionality.
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import logging
from .VibExtractor import execute_query

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibrationDataLoader:
    """Loads vibration data from Databricks and stores in SQLite cache"""
    
    def __init__(self, cache_dir="app/data/cache"):
        """Initialize the data loader with cache directory"""
        self.cache_dir = cache_dir
        self.cache_db_path = os.path.join(cache_dir, "vibration_cache.db")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Create latest_vibration_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS latest_vibration_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                serialNumber INTEGER,
                EnqueuedTimeUtc TEXT,
                time TEXT,
                frequencies BLOB,
                fft_vel_d1 BLOB,
                fft_vel_d2 BLOB,
                fft_vel_d3 BLOB,
                rms_vel_d1 REAL,
                rms_vel_d2 REAL,
                rms_vel_d3 REAL,
                rms_acc_d1 REAL,
                rms_acc_d2 REAL,
                rms_acc_d3 REAL,
                rms_vel_fft1000_d1 REAL,
                rms_vel_fft1000_d2 REAL,
                rms_vel_fft1000_d3 REAL,
                rms_acc_fft1000_d1 REAL,
                rms_acc_fft1000_d2 REAL,
                rms_acc_fft1000_d3 REAL,
                rms_acc_fft_d1 REAL,
                rms_acc_fft_d2 REAL,
                rms_acc_fft_d3 REAL,
                cached_at TEXT,
                UNIQUE(serialNumber)
            )
        """)
        
        # Migrate existing tables to add missing columns
        self._migrate_schema(cursor)
        
        # Create equipment_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equipment_data (
                serialNumber INTEGER PRIMARY KEY,
                cached_at TEXT
            )
        """)
        
        # Create historical_data table for time series
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                serialNumber INTEGER,
                EnqueuedTimeUtc TEXT,
                mytimestamp TEXT,
                rms_acc_d1 REAL,
                rms_acc_d2 REAL,
                rms_acc_d3 REAL,
                rms_vel_d1 REAL,
                rms_vel_d2 REAL,
                rms_vel_d3 REAL,
                cached_at TEXT
            )
        """)
        
        # Create metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                table_name TEXT PRIMARY KEY,
                last_updated TEXT,
                record_count INTEGER
            )
        """)
        
        # Initialize metadata for all expected tables if they don't exist
        table_names = ['latest_vibration_data', 'equipment_data', 'historical_data']
        for table_name in table_names:
            cursor.execute("""
                INSERT OR IGNORE INTO cache_metadata (table_name, last_updated, record_count)
                VALUES (?, 'never', 0)
            """, (table_name,))
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def _migrate_schema(self, cursor):
        """Migrate existing database schema to add missing columns"""
        try:
            # Check if cached_at column exists in latest_vibration_data
            cursor.execute("PRAGMA table_info(latest_vibration_data)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'cached_at' not in columns:
                logger.info("Adding cached_at column to latest_vibration_data table")
                cursor.execute("ALTER TABLE latest_vibration_data ADD COLUMN cached_at TEXT")
            
            # Check equipment_data table
            cursor.execute("PRAGMA table_info(equipment_data)")
            eq_columns = [row[1] for row in cursor.fetchall()]
            
            if 'cached_at' not in eq_columns:
                # equipment_data table might not exist or might be missing cached_at
                cursor.execute("DROP TABLE IF EXISTS equipment_data")
                cursor.execute("""
                    CREATE TABLE equipment_data (
                        serialNumber INTEGER PRIMARY KEY,
                        cached_at TEXT
                    )
                """)
                logger.info("Recreated equipment_data table with cached_at column")
            
            # Check historical_data table
            cursor.execute("PRAGMA table_info(historical_data)")
            hist_columns = [row[1] for row in cursor.fetchall()]
            
            if 'cached_at' not in hist_columns:
                # historical_data table might not exist or might be missing cached_at
                cursor.execute("DROP TABLE IF EXISTS historical_data")
                cursor.execute("""
                    CREATE TABLE historical_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        serialNumber INTEGER,
                        EnqueuedTimeUtc TEXT,
                        mytimestamp TEXT,
                        rms_acc_d1 REAL,
                        rms_acc_d2 REAL,
                        rms_acc_d3 REAL,
                        rms_vel_d1 REAL,
                        rms_vel_d2 REAL,
                        rms_vel_d3 REAL,
                        cached_at TEXT
                    )
                """)
                logger.info("Recreated historical_data table with cached_at column")
                
        except Exception as e:
            logger.error(f"Error during schema migration: {e}")
            # Continue anyway - the tables will be created if they don't exist
    
    def load_latest_vibration_data(self):
        """Load latest vibration data from Databricks to cache"""
        logger.info("Loading latest vibration data from Databricks...")
        
        # Query to get all latest data (without serialNumber filter)
        query = """
        WITH RankedData AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY serialNumber ORDER BY EnqueuedTimeUtc DESC) AS rn
            FROM vibration.silver.ffts_waveforms
            WHERE rms_acc_d3 > 0.35
        )
        SELECT *
        FROM RankedData
        WHERE rn = 1;
        """
        
        try:
            df = execute_query(query)
            logger.info(f"Retrieved {len(df)} records from Databricks")
            
            if df.empty:
                logger.warning("No data retrieved from Databricks")
                return False
            
            # Store in cache
            self._store_latest_vibration_data(df)
            return True
            
        except Exception as e:
            logger.error(f"Error loading latest vibration data: {e}")
            return False
    
    def load_equipment_data(self):
        """Load equipment IDs from Databricks to cache"""
        logger.info("Loading equipment data from Databricks...")
        
        query = """
        WITH RankedData AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY serialNumber ORDER BY EnqueuedTimeUtc DESC) AS rn
            FROM vibration.silver.ffts_waveforms
            WHERE rms_acc_d3 > 0.35
        )
        SELECT distinct serialNumber
        FROM RankedData
        WHERE rn = 1;
        """
        
        try:
            df = execute_query(query)
            logger.info(f"Retrieved {len(df)} equipment records from Databricks")
            
            if df.empty:
                logger.warning("No equipment data retrieved from Databricks")
                return False
            
            # Store in cache
            self._store_equipment_data(df)
            return True
            
        except Exception as e:
            logger.error(f"Error loading equipment data: {e}")
            return False
    
    def load_historical_data(self):
        """Load historical time series data for all equipment from Databricks to cache"""
        logger.info("Loading historical data from Databricks...")
        
        # Query to get historical data for ALL equipment (no serialNumber filter)
        query = """
        SELECT EnqueuedTimeUtc,time as mytimestamp,
               serialNumber,rms_acc_d1,rms_acc_d2,rms_acc_d3,rms_vel_d1,rms_vel_d2,rms_vel_d3
        FROM vibration.silver.ffts_waveforms 
        ORDER BY serialNumber, time DESC
        """
        
        try:
            df = execute_query(query)
            logger.info(f"Retrieved {len(df)} historical records from Databricks")
            
            if df.empty:
                logger.warning("No historical data retrieved from Databricks")
                return False
            
            # Store in cache
            self._store_historical_data(df)
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
    
    def _store_latest_vibration_data(self, df):
        """Store latest vibration data in SQLite cache"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM latest_vibration_data")
        
        cached_at = datetime.now().isoformat()
        
        for _, row in df.iterrows():
            # Serialize numpy arrays as BLOB
            frequencies_blob = pickle.dumps(row.get('frequencies', np.array([])))
            fft_vel_d1_blob = pickle.dumps(row.get('fft_vel_d1', np.array([])))
            fft_vel_d2_blob = pickle.dumps(row.get('fft_vel_d2', np.array([])))
            fft_vel_d3_blob = pickle.dumps(row.get('fft_vel_d3', np.array([])))
            
            cursor.execute("""
                INSERT OR REPLACE INTO latest_vibration_data 
                (serialNumber, EnqueuedTimeUtc, time, frequencies, fft_vel_d1, fft_vel_d2, fft_vel_d3,
                 rms_vel_d1, rms_vel_d2, rms_vel_d3, rms_acc_d1, rms_acc_d2, rms_acc_d3,
                 rms_vel_fft1000_d1, rms_vel_fft1000_d2, rms_vel_fft1000_d3,
                 rms_acc_fft1000_d1, rms_acc_fft1000_d2, rms_acc_fft1000_d3,
                 rms_acc_fft_d1, rms_acc_fft_d2, rms_acc_fft_d3, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('serialNumber'), row.get('EnqueuedTimeUtc'), row.get('time'),
                frequencies_blob, fft_vel_d1_blob, fft_vel_d2_blob, fft_vel_d3_blob,
                row.get('rms_vel_d1'), row.get('rms_vel_d2'), row.get('rms_vel_d3'),
                row.get('rms_acc_d1'), row.get('rms_acc_d2'), row.get('rms_acc_d3'),
                row.get('rms_vel_fft1000_d1'), row.get('rms_vel_fft1000_d2'), row.get('rms_vel_fft1000_d3'),
                row.get('rms_acc_fft1000_d1'), row.get('rms_acc_fft1000_d2'), row.get('rms_acc_fft1000_d3'),
                row.get('rms_acc_fft_d1'), row.get('rms_acc_fft_d2'), row.get('rms_acc_fft_d3'),
                cached_at
            ))
        
        # Update metadata
        cursor.execute("""
            INSERT OR REPLACE INTO cache_metadata (table_name, last_updated, record_count)
            VALUES (?, ?, ?)
        """, ('latest_vibration_data', cached_at, len(df)))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(df)} latest vibration records in cache")
    
    def _store_equipment_data(self, df):
        """Store equipment data in SQLite cache"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM equipment_data")
        
        cached_at = datetime.now().isoformat()
        
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO equipment_data (serialNumber, cached_at)
                VALUES (?, ?)
            """, (int(row['serialNumber']), cached_at))
        
        # Update metadata
        cursor.execute("""
            INSERT OR REPLACE INTO cache_metadata (table_name, last_updated, record_count)
            VALUES (?, ?, ?)
        """, ('equipment_data', cached_at, len(df)))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(df)} equipment records in cache")
    
    def _store_historical_data(self, df):
        """Store historical data in SQLite cache"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM historical_data")
        
        cached_at = datetime.now().isoformat()
        
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO historical_data 
                (serialNumber, EnqueuedTimeUtc, mytimestamp, rms_acc_d1, rms_acc_d2, rms_acc_d3,
                 rms_vel_d1, rms_vel_d2, rms_vel_d3, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('serialNumber'), row.get('EnqueuedTimeUtc'), row.get('mytimestamp'),
                row.get('rms_acc_d1'), row.get('rms_acc_d2'), row.get('rms_acc_d3'),
                row.get('rms_vel_d1'), row.get('rms_vel_d2'), row.get('rms_vel_d3'),
                cached_at
            ))
        
        # Update metadata
        cursor.execute("""
            INSERT OR REPLACE INTO cache_metadata (table_name, last_updated, record_count)
            VALUES (?, ?, ?)
        """, ('historical_data', cached_at, len(df)))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(df)} historical records in cache")
    
    def load_all_data(self):
        """Load all data types from Databricks to cache"""
        logger.info("Loading all data from Databricks...")
        
        results = {}
        results['latest_vibration'] = self.load_latest_vibration_data()
        results['equipment'] = self.load_equipment_data()
        results['historical'] = self.load_historical_data()
        
        success_count = sum(results.values())
        logger.info(f"Data loading complete: {success_count}/3 operations successful")
        
        return results
    
    def get_cache_status(self):
        """Get status of cached data"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM cache_metadata")
        metadata = cursor.fetchall()
        
        status = {}
        for table_name, last_updated, record_count in metadata:
            # Calculate age from last_updated
            if last_updated == 'never':
                age = 'N/A'
            else:
                try:
                    from datetime import datetime
                    updated_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    current_time = datetime.now(updated_time.tzinfo) if updated_time.tzinfo else datetime.now()
                    age_delta = current_time - updated_time
                    
                    if age_delta.days > 0:
                        age = f"{age_delta.days} day{'s' if age_delta.days > 1 else ''}"
                    elif age_delta.seconds > 3600:
                        hours = age_delta.seconds // 3600
                        age = f"{hours} hour{'s' if hours > 1 else ''}"
                    else:
                        minutes = age_delta.seconds // 60
                        age = f"{minutes} minute{'s' if minutes > 1 else ''}"
                except:
                    age = 'Unknown'
            
            status[table_name] = {
                'last_updated': last_updated,
                'record_count': record_count,
                'age': age
            }
        
        conn.close()
        return status
    
    def clear_cache(self):
        """Clear all cached data"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        tables = ['latest_vibration_data', 'equipment_data', 'historical_data', 'cache_metadata']
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
        
        conn.commit()
        conn.close()
        logger.info("Cache cleared successfully")
    
    def recreate_database(self):
        """Recreate database with correct schema (use if migration fails)"""
        try:
            # Close any existing connections
            if os.path.exists(self.cache_db_path):
                os.remove(self.cache_db_path)
                logger.info("Removed old database file")
            
            # Recreate with correct schema
            self._init_database()
            logger.info("Database recreated successfully with correct schema")
            return True
            
        except Exception as e:
            logger.error(f"Error recreating database: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    loader = VibrationDataLoader()
    results = loader.load_all_data()
    print("Loading results:", results)
    print("Cache status:", loader.get_cache_status())

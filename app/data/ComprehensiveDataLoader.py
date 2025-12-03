"""
Comprehensive DataLoader with support for all vibration analysis columns
"""

import sqlite3
import pickle
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
from app.data.VibExtractor import execute_query

logger = logging.getLogger(__name__)

class ComprehensiveVibrationDataLoader:
    """Enhanced data loader that preserves ALL columns for analysis functions"""
    
    def __init__(self, cache_db_path=None):
        if cache_db_path is None:
            # Use consistent default cache path
            cache_dir = os.path.join(os.path.expanduser('~'), '.vibration_cache')
            os.makedirs(cache_dir, exist_ok=True)
            cache_db_path = os.path.join(cache_dir, 'comprehensive_cache.db')
        self.cache_db_path = cache_db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the comprehensive database schema"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        try:
            # Create comprehensive latest_vibration_data table with ALL columns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS latest_vibration_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    SequenceNumber TEXT,
                    Offset TEXT,
                    EnqueuedTimeUtc TEXT,
                    SystemProperties TEXT,
                    Properties TEXT,
                    _rescued_data TEXT,
                    source TEXT,
                    type TEXT,
                    time TEXT,
                    specversion TEXT,
                    datacontenttype TEXT,
                    subject TEXT,
                    serialNumber INTEGER,
                    sampleRate REAL,
                    gwSerial TEXT,
                    timestamp TEXT,
                    data_type TEXT,
                    data_dataType TEXT,
                    direction1 BLOB,
                    direction2 BLOB,
                    direction3 BLOB,
                    event_time_ts TEXT,
                    data_timestamp_ts TEXT,
                    dat_vel_d1 BLOB,
                    dat_vel_d2 BLOB,
                    dat_vel_d3 BLOB,
                    fft_acc_d1 BLOB,
                    fft_acc_d2 BLOB,
                    fft_acc_d3 BLOB,
                    fft_vel_d1 BLOB,
                    fft_vel_d2 BLOB,
                    fft_vel_d3 BLOB,
                    frequencies BLOB,
                    rms_acc_d1 REAL,
                    rms_acc_fft_d1 REAL,
                    rms_acc_fft1000_d1 REAL,
                    rms_acc_d2 REAL,
                    rms_acc_fft_d2 REAL,
                    rms_acc_fft1000_d2 REAL,
                    rms_acc_d3 REAL,
                    rms_acc_fft_d3 REAL,
                    rms_acc_fft1000_d3 REAL,
                    rms_vel_d1 REAL,
                    rms_vel_fft_d1 REAL,
                    rms_vel_fft1000_d1 REAL,
                    rms_vel_d2 REAL,
                    rms_vel_fft_d2 REAL,
                    rms_vel_fft1000_d2 REAL,
                    rms_vel_d3 REAL,
                    rms_vel_fft_d3 REAL,
                    rms_vel_fft1000_d3 REAL,
                    cached_at TEXT,
                    UNIQUE(serialNumber)
                )
            """)
            
            # Create historical_data table (optimized for RMS trends - no waveform data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    serialNumber INTEGER,
                    time TEXT,
                    rms_vel_d1 REAL,
                    rms_vel_d2 REAL,
                    rms_vel_d3 REAL,
                    rms_acc_d1 REAL,
                    rms_acc_d2 REAL,
                    rms_acc_d3 REAL,
                    cached_at TEXT,
                    UNIQUE(serialNumber, time)
                )
            """)
            
            # Create historical_waveforms table (for full waveform data when needed)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_waveforms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    SequenceNumber TEXT,
                    Offset TEXT,
                    EnqueuedTimeUtc TEXT,
                    SystemProperties TEXT,
                    Properties TEXT,
                    _rescued_data TEXT,
                    source TEXT,
                    type TEXT,
                    time TEXT,
                    specversion TEXT,
                    datacontenttype TEXT,
                    subject TEXT,
                    serialNumber INTEGER,
                    sampleRate REAL,
                    gwSerial TEXT,
                    timestamp TEXT,
                    mytimestamp TEXT,
                    data_type TEXT,
                    data_dataType TEXT,
                    direction1 BLOB,
                    direction2 BLOB,
                    direction3 BLOB,
                    event_time_ts TEXT,
                    data_timestamp_ts TEXT,
                    dat_vel_d1 BLOB,
                    dat_vel_d2 BLOB,
                    dat_vel_d3 BLOB,
                    fft_acc_d1 BLOB,
                    fft_acc_d2 BLOB,
                    fft_acc_d3 BLOB,
                    fft_vel_d1 BLOB,
                    fft_vel_d2 BLOB,
                    fft_vel_d3 BLOB,
                    frequencies BLOB,
                    rms_acc_d1 REAL,
                    rms_acc_fft_d1 REAL,
                    rms_acc_fft1000_d1 REAL,
                    rms_acc_d2 REAL,
                    rms_acc_fft_d2 REAL,
                    rms_acc_fft1000_d2 REAL,
                    rms_acc_d3 REAL,
                    rms_acc_fft_d3 REAL,
                    rms_acc_fft1000_d3 REAL,
                    rms_vel_d1 REAL,
                    rms_vel_fft_d1 REAL,
                    rms_vel_fft1000_d1 REAL,
                    rms_vel_d2 REAL,
                    rms_vel_fft_d2 REAL,
                    rms_vel_fft1000_d2 REAL,
                    rms_vel_d3 REAL,
                    rms_vel_fft_d3 REAL,
                    rms_vel_fft1000_d3 REAL,
                    cached_at TEXT
                )
            """)
            
            # Create equipment_data table  
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equipment_data (
                    serialNumber INTEGER PRIMARY KEY,
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
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def load_latest_vibration_data(self):
        """Load latest vibration data from Databricks to cache"""
        logger.info("Loading latest vibration data from Databricks...")
        
        try:
            df = execute_query("""
                WITH RankedData AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY serialNumber ORDER BY EnqueuedTimeUtc DESC) AS rn
                    FROM vibration.silver.ffts_waveforms
                    WHERE rms_acc_d3 > 0.35
                )
                SELECT SequenceNumber, Offset, EnqueuedTimeUtc, SystemProperties, Properties, _rescued_data,
                       source, type, time, specversion, datacontenttype, subject, serialNumber, sampleRate,
                       gwSerial, timestamp, data_type, data_dataType, direction1, direction2, direction3,
                       event_time_ts, data_timestamp_ts, dat_vel_d1, dat_vel_d2, dat_vel_d3, fft_acc_d1,
                       fft_acc_d2, fft_acc_d3, fft_vel_d1, fft_vel_d2, fft_vel_d3, frequencies,
                       rms_acc_d1, rms_acc_fft_d1, rms_acc_fft1000_d1, rms_acc_d2, rms_acc_fft_d2,
                       rms_acc_fft1000_d2, rms_acc_d3, rms_acc_fft_d3, rms_acc_fft1000_d3,
                       rms_vel_d1, rms_vel_fft_d1, rms_vel_fft1000_d1, rms_vel_d2, rms_vel_fft_d2,
                       rms_vel_fft1000_d2, rms_vel_d3, rms_vel_fft_d3, rms_vel_fft1000_d3
                FROM RankedData
                WHERE rn = 1;
            """)
            
            logger.info(f"Retrieved {len(df)} records from Databricks")
            
            if df.empty:
                logger.warning("No latest vibration data retrieved from Databricks")
                return False
            
            # Store in cache
            self._store_comprehensive_data(df, 'latest_vibration_data')
            return True
            
        except Exception as e:
            logger.error(f"Error loading latest vibration data: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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
    
    def _store_historical_rms_data(self, df):
        """Store historical RMS trend data efficiently (no waveforms)"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM historical_data")
        
        cached_at = datetime.now().isoformat()
        
        # Insert historical RMS data (much simpler than full waveform data)
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO historical_data 
                (serialNumber, time, rms_vel_d1, rms_vel_d2, rms_vel_d3, 
                 rms_acc_d1, rms_acc_d2, rms_acc_d3, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(row['serialNumber']), row['time'],
                float(row['rms_vel_d1']), float(row['rms_vel_d2']), float(row['rms_vel_d3']),
                float(row['rms_acc_d1']), float(row['rms_acc_d2']), float(row['rms_acc_d3']),
                cached_at
            ))
        
        conn.commit()
        conn.close()

    def _store_comprehensive_data(self, df, table_name):
        """Store comprehensive data with all columns preserved"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute(f"DELETE FROM {table_name}")
        
        cached_at = datetime.now().isoformat()
        
        # Columns to store as BLOB (numpy arrays)
        blob_columns = [
            "direction1",
            "direction2",
            "direction3",
            "dat_vel_d1",
            "dat_vel_d2",
            "dat_vel_d3",
            "fft_acc_d1",
            "fft_acc_d2",
            "fft_acc_d3",
            "fft_vel_d1",
            "fft_vel_d2",
            "fft_vel_d3",
            "frequencies",
        ]

        def _normalize_array(val):
            """Convert various representations into a numpy float32 array."""
            if val is None:
                return None

            # Already a numpy array or list/tuple
            if isinstance(val, (np.ndarray, list, tuple)):
                try:
                    return np.asarray(val, dtype=np.float32)
                except Exception:
                    return None

            # String-encoded list: "[1, 2, 3]" or "1,2,3"
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    return None
                import json, ast
                # Try JSON or Python literal first
                for parser in (json.loads, ast.literal_eval):
                    try:
                        parsed = parser(s)
                        return np.asarray(parsed, dtype=np.float32)
                    except Exception:
                        pass
                # Fallback: loose "1,2,3" style
                try:
                    s2 = s.strip("[]")
                    toks = [t for t in s2.replace("\n", " ").split(",") if t.strip()]
                    return np.asarray([float(t) for t in toks], dtype=np.float32)
                except Exception:
                    return None

            # Anything array-like
            if hasattr(val, "__array__"):
                try:
                    return np.asarray(val, dtype=np.float32)
                except Exception:
                    return None

            return None  # Unknown type

        for _, row in df.iterrows():
            # Prepare data - serialize arrays to blobs, handle missing columns
            data_dict = {}
            all_columns = [col for col in df.columns if col != "id"]

            values = []
            for col in all_columns:
                if col in df.columns:
                    val = row.get(col)
                    if col in blob_columns and val is not None:
                        arr = _normalize_array(val)
                        if arr is not None:
                            values.append(pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL))
                        else:
                            values.append(None)
                    elif col in ["serialNumber", "SequenceNumber"] and val is not None:
                        values.append(int(val))
                    elif col in ["SystemProperties", "Properties"] and val is not None:
                        import json
                        try:
                            values.append(
                                json.dumps(val) if isinstance(val, (list, dict)) else str(val)
                            )
                        except Exception:
                            values.append(str(val))
                    elif hasattr(val, "dtype") and "int" in str(val.dtype):
                        values.append(int(val))
                    elif hasattr(val, "dtype") and "float" in str(val.dtype):
                        values.append(float(val))
                    elif str(type(val)).startswith("<class 'pandas._libs.tslibs"):
                        values.append(str(val))
                    elif hasattr(val, "dtype"):
                        # Numpy scalar vs array
                        if hasattr(val, "size") and val.size == 1:
                            try:
                                values.append(val.item())
                            except Exception:
                                values.append(
                                    float(val)
                                    if "float" in str(val.dtype)
                                    else int(val)
                                )
                        elif hasattr(val, "size") and val.size > 1:
                            # Arrays not listed in blob_columns → pickle anyway
                            try:
                                values.append(
                                    pickle.dumps(
                                        np.asarray(val, dtype=np.float32),
                                        protocol=pickle.HIGHEST_PROTOCOL,
                                    )
                                )
                            except Exception:
                                values.append(str(val))
                        else:
                            try:
                                values.append(val.item())
                            except Exception:
                                values.append(str(val))
                    elif hasattr(val, "__array__") or "numpy" in str(type(val)):
                        try:
                            arr = np.asarray(val, dtype=np.float32)
                            values.append(pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL))
                        except Exception:
                            values.append(str(val))
                    else:
                        # Primitive or fallback to string
                        if isinstance(val, (int, float, str, bytes, type(None))):
                            values.append(val)
                        else:
                            values.append(str(val))
                else:
                    values.append(None)

            # Add cached_at timestamp
            values.append(cached_at)

            # Build insert statement
            placeholders = ", ".join(["?" for _ in range(len(all_columns) + 1)])  # +1 for cached_at
            column_names = ", ".join(all_columns + ["cached_at"])

            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {table_name} ({column_names})
                VALUES ({placeholders})
            """,
                values,
            )

        # Update metadata
        cursor.execute(
            """
            INSERT OR REPLACE INTO cache_metadata (table_name, last_updated, record_count)
            VALUES (?, ?, ?)
        """,
            (table_name, cached_at, len(df)),
        )

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(df)} records in {table_name}")

    
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
    
    def get_cached_data(self, table_name, serialNumber=None):
        """Retrieve cached data and reconstruct numpy arrays"""
        conn = sqlite3.connect(self.cache_db_path)
        
        if serialNumber:
            query = f"SELECT * FROM {table_name} WHERE serialNumber = ?"
            df = pd.read_sql_query(query, conn, params=(serialNumber,))
        else:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        if df.empty:
            return df
        
        # Reconstruct BLOB columns back to numpy arrays
        blob_columns = ['direction1', 'direction2', 'direction3', 'dat_vel_d1', 'dat_vel_d2', 'dat_vel_d3', 
                       'fft_acc_d1', 'fft_acc_d2', 'fft_acc_d3', 'fft_vel_d1', 'fft_vel_d2', 'fft_vel_d3', 'frequencies']
        
        for col in blob_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: pickle.loads(x) if x is not None else None)
        
        return df
    
    def load_historical_data(self):
        """Load historical vibration data from Databricks to cache"""
        logger.info("Loading historical data from Databricks...")
        
        # Query that gets historical RMS trend data (no waveforms) for trend analysis
        # This is optimized for historical trend plotting - gets recent data for all serialNumbers
        query = """
        SELECT serialNumber, time,
               rms_vel_d1, rms_vel_d2, rms_vel_d3,
               rms_acc_d1, rms_acc_d2, rms_acc_d3
        FROM vibration.silver.ffts_waveforms 
        WHERE rms_acc_d3 > 0.35
        ORDER BY serialNumber, time DESC
        """
        
        try:
            df = execute_query(query)
            logger.info(f"Retrieved {len(df)} historical records from Databricks")
            
            if df.empty:
                logger.warning("No historical data retrieved from Databricks")
                return False
            
            # Store in optimized historical_data table (RMS trends only)
            self._store_historical_rms_data(df)
            
            logger.info(f"Stored {len(df)} historical trend records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

    def load_all_data(self):
        """Load all data types from Databricks"""
        logger.info("Loading all data from Databricks...")
        
        results = {
            'latest_vibration': self.load_latest_vibration_data(),
            'equipment': self.load_equipment_data(),
            'historical': self.load_historical_data()
        }
        
        success_count = sum(results.values())
        total_count = len(results)
        
        logger.info(f"Data loading complete: {success_count}/{total_count} operations successful")
        return results

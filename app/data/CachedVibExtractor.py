"""
Enhanced VibExtractor with Local SQLite Caching

This module provides the same interface as the original VibExtractor but uses
local SQLite cache when available, falling back to Databricks when needed.

Key features:
- Drop-in replacement for original VibExtractor
- Automatic fallback to Databricks if cache is empty
- Maintains all original functionality and analysis methods
- Supports both cached and live data queries
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime

# Import original VibExtractor for fallback and analysis methods
from . import VibExtractor as OriginalVibExtractor

logger = logging.getLogger(__name__)

class CachedVibExtractor:
    """Enhanced VibExtractor that uses local SQLite cache with Databricks fallback."""
    
    def __init__(self, db_path="app/data/cache/vibration_cache.db", use_cache=True):
        """Initialize the cached extractor."""
        self.db_path = db_path
        self.use_cache = use_cache
        self.cache_available = self._check_cache_availability()
        
        # Import all static data and methods from original VibExtractor
        self.info_dat = OriginalVibExtractor.info_dat
        self.sensor_naming_map = OriginalVibExtractor.sensor_naming_map
        self.mechanical_fault_rms = OriginalVibExtractor.mechanical_fault_rms
        self.mechanical_fault_peak = OriginalVibExtractor.mechanical_fault_peak
        self.bearing_fault_rms = OriginalVibExtractor.bearing_fault_rms
        self.get_bearing_fault_rms = OriginalVibExtractor.get_bearing_fault_rms
        
        # Import all analysis methods
        self.check_unbalance = OriginalVibExtractor.check_unbalance
        self.check_loosenes = OriginalVibExtractor.check_loosenes
        self.check_misalignment = OriginalVibExtractor.check_misalignment
        self.check_bearing_rms = OriginalVibExtractor.check_bearing_rms
        self.get_metrics = OriginalVibExtractor.get_metrics
        self.detect_faults = OriginalVibExtractor.detect_faults
        
    def _check_cache_availability(self):
        """Check if cache database exists and has data."""
        if not self.use_cache:
            return False
            
        try:
            if not os.path.exists(self.db_path):
                logger.info("Cache database does not exist")
                return False
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM latest_vibration_data")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    logger.info(f"Cache available with {count} records")
                    return True
                else:
                    logger.info("Cache database exists but is empty")
                    return False
                    
        except Exception as e:
            logger.warning(f"Error checking cache availability: {e}")
            return False
            
    def deserialize_array(self, blob_data):
        """Deserialize numpy array from database blob."""
        if blob_data is None:
            return None
        try:
            return pickle.loads(blob_data)
        except Exception as e:
            logger.warning(f"Error deserializing array: {e}")
            return None
            
    def get_latest_vibration_data(self, serialNumber=None, use_cache=None):
        """Get latest vibration data with cache support."""
        # Override use_cache parameter if provided
        if use_cache is None:
            use_cache = self.use_cache
            
        # Try cache first if available and requested
        if use_cache and self.cache_available:
            try:
                return self._get_latest_from_cache(serialNumber)
            except Exception as e:
                logger.warning(f"Cache query failed, falling back to Databricks: {e}")
                
        # Fallback to original Databricks query
        logger.info("Using Databricks fallback for latest vibration data")
        return OriginalVibExtractor.get_latest_vibration_data(serialNumber)
        
    def _get_latest_from_cache(self, serialNumber=None):
        """Get latest vibration data from local cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if serialNumber:
                    query = "SELECT * FROM latest_vibration_data WHERE serialNumber = ?"
                    df = pd.read_sql_query(query, conn, params=(serialNumber,))
                else:
                    query = "SELECT * FROM latest_vibration_data"
                    df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    logger.warning(f"No cached data found for serialNumber: {serialNumber}")
                    return df
                
                # Deserialize array columns
                array_columns = ['frequencies', 'fft_vel_d1', 'fft_vel_d2', 'fft_vel_d3',
                               'fft_acc_d1', 'fft_acc_d2', 'fft_acc_d3',
                               'direction1', 'direction2', 'direction3']
                
                for col in array_columns:
                    if col in df.columns:
                        df[col] = df[col].apply(self.deserialize_array)
                
                logger.info(f"Retrieved {len(df)} records from cache")
                return df
                
        except Exception as e:
            logger.error(f"Error querying cache: {e}")
            raise
            
    def get_equipment_id(self, use_cache=None):
        """Get equipment IDs with cache support."""
        # Override use_cache parameter if provided
        if use_cache is None:
            use_cache = self.use_cache
            
        # Try cache first if available and requested
        if use_cache and self.cache_available:
            try:
                return self._get_equipment_from_cache()
            except Exception as e:
                logger.warning(f"Cache query failed, falling back to Databricks: {e}")
                
        # Fallback to original Databricks query
        logger.info("Using Databricks fallback for equipment data")
        return OriginalVibExtractor.get_equipment_id()
        
    def _get_equipment_from_cache(self):
        """Get equipment IDs from local cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT DISTINCT serialNumber FROM latest_vibration_data"
                df = pd.read_sql_query(query, conn)
                
                logger.info(f"Retrieved {len(df)} equipment records from cache")
                return df
                
        except Exception as e:
            logger.error(f"Error querying equipment from cache: {e}")
            raise
            
    def execute_query(self, query, use_cache=None):
        """Execute query with cache support for historical data."""
        # Override use_cache parameter if provided
        if use_cache is None:
            use_cache = self.use_cache
            
        # Check if this is a historical time series query
        if "rms_acc_d1,rms_acc_d2,rms_acc_d3,rms_vel_d1,rms_vel_d2,rms_vel_d3" in query and use_cache and self.cache_available:
            try:
                return self._execute_historical_query_from_cache(query)
            except Exception as e:
                logger.warning(f"Cache historical query failed, falling back to Databricks: {e}")
                
        # Fallback to original Databricks query
        logger.info("Using Databricks fallback for query execution")
        return OriginalVibExtractor.execute_query(query)
        
    def _execute_historical_query_from_cache(self, original_query):
        """Execute historical time series query from cache."""
        try:
            # Extract serialNumber from query if present
            serialNumber = None
            if "serialNumber=" in original_query:
                # Parse serialNumber from query
                import re
                match = re.search(r'serialNumber\s*=\s*(\d+)', original_query)
                if match:
                    serialNumber = int(match.group(1))
            
            with sqlite3.connect(self.db_path) as conn:
                if serialNumber:
                    query = """
                        SELECT EnqueuedTimeUtc, mytimestamp as time,
                               serialNumber, rms_acc_d1, rms_acc_d2, rms_acc_d3,
                               rms_vel_d1, rms_vel_d2, rms_vel_d3
                        FROM historical_time_series 
                        WHERE serialNumber = ?
                        ORDER BY mytimestamp DESC
                    """
                    df = pd.read_sql_query(query, conn, params=(serialNumber,))
                else:
                    query = """
                        SELECT EnqueuedTimeUtc, mytimestamp as time,
                               serialNumber, rms_acc_d1, rms_acc_d2, rms_acc_d3,
                               rms_vel_d1, rms_vel_d2, rms_vel_d3
                        FROM historical_time_series 
                        ORDER BY serialNumber, mytimestamp DESC
                    """
                    df = pd.read_sql_query(query, conn)
                
                logger.info(f"Retrieved {len(df)} historical records from cache")
                return df
                
        except Exception as e:
            logger.error(f"Error executing historical query from cache: {e}")
            raise
            
    def get_cache_stats(self):
        """Get cache statistics."""
        if not self.cache_available:
            return {
                "cache_available": False,
                "latest_records": 0,
                "historical_records": 0,
                "equipment_records": 0,
                "cache_size_mb": 0,
                "last_updated": None
            }
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get record counts
                cursor.execute("SELECT COUNT(*) FROM latest_vibration_data")
                latest_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM historical_time_series")
                historical_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM equipment_metadata")
                equipment_count = cursor.fetchone()[0]
                
                # Get last update time
                cursor.execute("SELECT MAX(last_loaded) FROM load_status")
                last_updated = cursor.fetchone()[0]
                
                # Get database size
                db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
                
                return {
                    "cache_available": True,
                    "latest_records": latest_count,
                    "historical_records": historical_count,
                    "equipment_records": equipment_count,
                    "cache_size_mb": round(db_size, 2),
                    "last_updated": last_updated
                }
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"cache_available": False, "error": str(e)}


# Create a global instance that can be imported
cached_vibextractor = CachedVibExtractor()

# Export all functions and data from the cached instance for backward compatibility
get_latest_vibration_data = cached_vibextractor.get_latest_vibration_data
get_equipment_id = cached_vibextractor.get_equipment_id
execute_query = cached_vibextractor.execute_query

# Export static data
info_dat = cached_vibextractor.info_dat
sensor_naming_map = cached_vibextractor.sensor_naming_map
mechanical_fault_rms = cached_vibextractor.mechanical_fault_rms
mechanical_fault_peak = cached_vibextractor.mechanical_fault_peak
bearing_fault_rms = cached_vibextractor.bearing_fault_rms
get_bearing_fault_rms = cached_vibextractor.get_bearing_fault_rms

# Export analysis functions
check_unbalance = cached_vibextractor.check_unbalance
check_loosenes = cached_vibextractor.check_loosenes
check_misalignment = cached_vibextractor.check_misalignment
check_bearing_rms = cached_vibextractor.check_bearing_rms
get_metrics = cached_vibextractor.get_metrics
detect_faults = cached_vibextractor.detect_faults

# Export the cached extractor instance
vibextractor = cached_vibextractor

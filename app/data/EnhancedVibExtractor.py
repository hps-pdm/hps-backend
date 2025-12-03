"""
Enhanced VibExtractor with Caching Support

This module wraps the original VibExtractor to add caching capabilities
while preserving ALL original functionality including analysis methods,
static data, and interpolation functions.
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import logging
from .forecasting import ForecastRequest, build_forecast


# Import ALL functionality from original VibExtractor
from .VibExtractor import (
    # Original data functions
    execute_query,
    get_latest_vibration_data as original_get_latest_vibration_data,
    get_equipment_id as original_get_equipment_id,
    
    # Analysis functions - PRESERVE ALL
    check_unbalance,
    check_loosenes,
    check_misalignment,
    check_bearing_rms,
    get_metrics,
    detect_faults,
    
    # Static data - PRESERVE ALL
    info_dat,
    sensor_naming_map,
    mechanical_fault_rms,
    mechanical_fault_peak,
    bearing_fault_rms,
    get_bearing_fault_rms
)

# Import comprehensive data loader
from .ComprehensiveDataLoader import ComprehensiveVibrationDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedVibExtractor:
    """
    Enhanced VibExtractor with caching support.
    
    Preserves ALL original VibExtractor functionality while adding
    SQLite caching for improved performance.
    """
    
    
    def __init__(self, cache_ttl_hours=24):
        """Initialize Enhanced VibExtractor with caching"""
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Use custom cache directory if specified, otherwise default
        custom_cache_dir = os.environ.get('CACHE_DIR')
        if custom_cache_dir:
            self.cache_dir = custom_cache_dir
        else:
            self.cache_dir = os.path.join(os.path.expanduser('~'), '.vibration_cache')
        
        self.cache_db_path = os.path.join(self.cache_dir, 'comprehensive_cache.db')
        
        # Use custom TTL if specified
        self.cache_ttl_hours = int(os.environ.get('CACHE_TTL_HOURS', cache_ttl_hours))
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize comprehensive data loader
        self.data_loader = ComprehensiveVibrationDataLoader(self.cache_db_path)
        
        # PRESERVE ALL original static data and functions
        self.info_dat = info_dat
        self.sensor_naming_map = sensor_naming_map
        self.mechanical_fault_rms = mechanical_fault_rms
        self.mechanical_fault_peak = mechanical_fault_peak
        self.bearing_fault_rms = bearing_fault_rms
        self.get_bearing_fault_rms = get_bearing_fault_rms
        
        # PRESERVE ALL original analysis functions
        self.check_unbalance = check_unbalance
        self.check_loosenes = check_loosenes
        self.check_misalignment = check_misalignment
        self.check_bearing_rms = check_bearing_rms
        self.get_metrics = get_metrics
        self.detect_faults = detect_faults
        
        # PRESERVE original execute_query function
        self.execute_query = execute_query
    
    def _is_cache_fresh(self, table_name):
        """Check if cached data is still fresh"""
        if not os.path.exists(self.cache_db_path):
            return False
        
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT last_updated FROM cache_metadata WHERE table_name = ?
            """, (table_name,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return False
            
            last_updated = datetime.fromisoformat(result[0])
            cutoff_time = datetime.now() - timedelta(hours=self.cache_ttl_hours)
            
            return last_updated > cutoff_time
            
        except Exception as e:
            logger.error(f"Error checking cache freshness: {e}")
            return False
    
    def _load_from_cache(self, table_name, serialNumber=None):
        """Load data from SQLite cache"""
        if not os.path.exists(self.cache_db_path):
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.cache_db_path)
            
            if table_name == 'latest_vibration_data':
                if serialNumber:
                    query = "SELECT * FROM latest_vibration_data WHERE serialNumber = ?"
                    df = pd.read_sql_query(query, conn, params=(serialNumber,))
                else:
                    df = pd.read_sql_query("SELECT * FROM latest_vibration_data", conn)
                
                # Deserialize BLOB columns back to numpy arrays
                if not df.empty:
                    for col in ['frequencies', 'fft_vel_d1', 'fft_vel_d2', 'fft_vel_d3']:
                        if col in df.columns:
                            df[col] = df[col].apply(lambda x: pickle.loads(x) if x else np.array([]))
            
            elif table_name == 'equipment_data':
                df = pd.read_sql_query("SELECT serialNumber FROM equipment_data", conn)
            
            elif table_name == 'historical_data':
                if serialNumber:
                    # Load optimized historical RMS trend data (alias time as mytimestamp for compatibility)
                    query = """SELECT serialNumber, time as mytimestamp,
                               rms_vel_d1, rms_vel_d2, rms_vel_d3,
                               rms_acc_d1, rms_acc_d2, rms_acc_d3,
                               cached_at 
                               FROM historical_data 
                               WHERE serialNumber = ? 
                               ORDER BY time DESC"""
                    df = pd.read_sql_query(query, conn, params=(serialNumber,))
                else:
                    query = """SELECT serialNumber, time as mytimestamp,
                               rms_vel_d1, rms_vel_d2, rms_vel_d3,
                               rms_acc_d1, rms_acc_d2, rms_acc_d3,
                               cached_at 
                               FROM historical_data 
                               ORDER BY serialNumber, time DESC"""
                    df = pd.read_sql_query(query, conn)
            
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return pd.DataFrame()
    
    def get_latest_vibration_data(self, serialNumber=None, use_cache=True, force_refresh=False):
        """
        Get latest vibration data with caching support.
        
        Args:
            serialNumber: Optional serial number filter
            use_cache: Whether to use cache (default True)
            force_refresh: Force refresh cache from Databricks (default False)
        """
        if force_refresh:
            logger.info("Force refresh requested - updating cache from Databricks...")
            self.data_loader.load_latest_vibration_data()
        
        if use_cache and self._is_cache_fresh('latest_vibration_data'):
            logger.info("Loading latest vibration data from cache...")
            df = self.data_loader.get_cached_data('latest_vibration_data', serialNumber)
            if not df.empty:
                return df
            logger.warning("Cache miss or empty, attempting to refresh cache...")
        
        # Cache is empty or stale - try to refresh it
        if use_cache:
            logger.info("Refreshing cache from Databricks...")
            refresh_success = self.data_loader.load_latest_vibration_data()
            
            if refresh_success:
                df = self.data_loader.get_cached_data('latest_vibration_data', serialNumber)
                if not df.empty:
                    return df
        
        # Fallback to original function
        logger.info("Loading latest vibration data directly from Databricks...")
        return original_get_latest_vibration_data(serialNumber)
    
    def get_equipment_id(self, use_cache=True, force_refresh=False):
        """
        Get equipment IDs with caching support.
        
        Args:
            use_cache: Whether to use cache (default True)
            force_refresh: Force refresh cache from Databricks (default False)
        """
        if force_refresh:
            logger.info("Force refresh requested - updating cache from Databricks...")
            self.data_loader.load_equipment_data()
        
        if use_cache and self._is_cache_fresh('equipment_data'):
            logger.info("Loading equipment data from cache...")
            df = self.data_loader.get_cached_data('equipment_data')
            if not df.empty:
                return df
            logger.warning("Cache miss or empty, attempting to refresh cache...")
        
        # Cache is empty or stale - try to refresh it
        if use_cache:
            logger.info("Refreshing cache from Databricks...")
            refresh_success = self.data_loader.load_equipment_data()
            
            if refresh_success:
                df = self.data_loader.get_cached_data('equipment_data')
                if not df.empty:
                    return df
        
        # Fallback to original function
        logger.info("Loading equipment data directly from Databricks...")
        return original_get_equipment_id()
    
    def get_historical_data(self, serialNumber, use_cache=True, force_refresh=False):
        """
        Get historical data with caching support.
        
        This replaces the execute_query call in callbacks.py for historical data.
        
        Args:
            serialNumber: Equipment serial number
            use_cache: Whether to use cache (default True)
            force_refresh: Force refresh cache from Databricks (default False)
        """
        # Validate serialNumber parameter
        if serialNumber is None:
            logger.warning("serialNumber is None, cannot query historical data")
            import pandas as pd
            return pd.DataFrame()
            
        if force_refresh:
            logger.info("Force refresh requested - updating cache from Databricks...")
            self._refresh_cache('historical_data')
        
        if use_cache and self._is_cache_fresh('historical_data'):
            logger.info(f"Loading historical data for {serialNumber} from cache...")
            df = self._load_from_cache('historical_data', serialNumber)
            if not df.empty:
                return df
            logger.warning("Cache miss or empty, attempting to refresh cache...")
        
        # Cache is empty or stale - try to refresh it
        if use_cache:
            logger.info("Refreshing cache from Databricks...")
            refresh_success = self._refresh_cache('historical_data')
            
            if refresh_success:
                df = self._load_from_cache('historical_data', serialNumber)
                if not df.empty:
                    return df
        
        # Fallback to original execute_query
        logger.info(f"Loading historical data for {serialNumber} directly from Databricks...")
        
        # Validate serialNumber parameter
        if serialNumber is None:
            logger.warning("serialNumber is None, cannot query historical data")
            import pandas as pd
            return pd.DataFrame()
            
        query = """
        SELECT EnqueuedTimeUtc,time as mytimestamp,
               serialNumber,rms_acc_d1,rms_acc_d2,rms_acc_d3,rms_vel_d1,rms_vel_d2,rms_vel_d3
        FROM vibration.silver.ffts_waveforms 
        WHERE serialNumber={eqp} 
        ORDER BY time DESC
        """.format(eqp=serialNumber)
        
        return self.execute_query(query)
    
    def get_cache_stats(self):
        """Get cache statistics"""
        if not os.path.exists(self.cache_db_path):
            return {
                'cache_exists': False,
                'record_count': 0,
                'cache_size_mb': 0,
                'is_fresh': False
            }
        
        try:
            # Get file size
            file_size = os.path.getsize(self.cache_db_path)
            
            # Get record counts
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM latest_vibration_data")
            record_count = cursor.fetchone()[0]
            
            # Check if fresh
            is_fresh = self._is_cache_fresh('latest_vibration_data')
            
            conn.close()
            
            return {
                'cache_exists': True,
                'record_count': record_count,
                'cache_size_mb': round(file_size / (1024 * 1024), 2),
                'is_fresh': is_fresh
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'cache_exists': False,
                'record_count': 0,
                'cache_size_mb': 0,
                'is_fresh': False
            }
    
    def _refresh_cache(self, table_name):
        """Refresh specific cache table from Databricks"""
        try:
            # Use the comprehensive data loader that's already initialized
            if table_name == 'latest_vibration_data':
                success = self.data_loader.load_latest_vibration_data()
            elif table_name == 'equipment_data':
                success = self.data_loader.load_equipment_data()
            elif table_name == 'historical_data':
                success = self.data_loader.load_historical_data()
            else:
                logger.error(f"Unknown table name: {table_name}")
                return False
            
            if success:
                logger.info(f"Successfully refreshed cache for {table_name}")
            else:
                logger.error(f"Failed to refresh cache for {table_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error refreshing cache for {table_name}: {e}")
            return False
    
    def refresh_all_caches(self):
        """Refresh all cache tables from Databricks"""
        logger.info("Refreshing all caches from Databricks...")
        
        try:
            results = self.data_loader.load_all_data()
            
            success_count = sum(results.values())
            total_count = len(results)
            
            logger.info(f"Cache refresh complete: {success_count}/{total_count} operations successful")
            
            return {
                'success': success_count == total_count,
                'results': results,
                'success_count': success_count,
                'total_count': total_count
            }
            
        except Exception as e:
            logger.error(f"Error refreshing all caches: {e}")
            return {
                'success': False,
                'error': str(e),
                'success_count': 0,
                'total_count': 3
            }
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            from .DataLoader import VibrationDataLoader
            loader = VibrationDataLoader(cache_dir=self.cache_dir)
            loader.clear_cache()
            logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
            
            
    def get_rms_forecast(
        self,
        serialNumber,
        metric: str = "rms_vel",
        direction: str = "d1",
        horizon_days: int = 14,
        model: str = "lgbm", #"linear","arima"
        use_cache: bool = True,
        force_refresh: bool = False,
    ):
        """
        Build RMS trend forecast (linear or ARIMA) for a given equipment.

        - Uses get_historical_data() (so it uses your SQLite cache)
        - Aggregates per-day
        - Returns a JSON-ready dict (history + forecast band)
        """
        import pandas as pd

        if serialNumber is None:
            return {"error": "serialNumber is required"}

        hist_df: pd.DataFrame = self.get_historical_data(
            serialNumber, use_cache=use_cache, force_refresh=force_refresh
        )

        if hist_df is None or hist_df.empty:
            return {"error": f"No historical data for serialNumber={serialNumber}"}

        req = ForecastRequest(
            sn=int(serialNumber),
            metric=metric,       # type: ignore[arg-type]
            direction=direction, # type: ignore[arg-type]
            horizon_days=horizon_days,
            model=model,         # type: ignore[arg-type]
        )

        try:
            return build_forecast(hist_df, req)
        except Exception as exc:
            return {
                "error": f"Forecast failed: {exc}",
                "sn": serialNumber,
            }


# Create a global instance that can be imported and used like the original VibExtractor
enhanced_vibextractor = EnhancedVibExtractor()

# Export the enhanced instance so it can be imported directly
vibration_extractor = enhanced_vibextractor

# Export all functions at module level for compatibility
get_latest_vibration_data = enhanced_vibextractor.get_latest_vibration_data
get_equipment_id = enhanced_vibextractor.get_equipment_id
get_historical_data = enhanced_vibextractor.get_historical_data
execute_query = enhanced_vibextractor.execute_query

# Export all analysis functions at module level
check_unbalance = enhanced_vibextractor.check_unbalance
check_loosenes = enhanced_vibextractor.check_loosenes  
check_misalignment = enhanced_vibextractor.check_misalignment
check_bearing_rms = enhanced_vibextractor.check_bearing_rms
get_metrics = enhanced_vibextractor.get_metrics
detect_faults = enhanced_vibextractor.detect_faults

# Export all static data at module level
info_dat = enhanced_vibextractor.info_dat
sensor_naming_map = enhanced_vibextractor.sensor_naming_map
mechanical_fault_rms = enhanced_vibextractor.mechanical_fault_rms
mechanical_fault_peak = enhanced_vibextractor.mechanical_fault_peak
bearing_fault_rms = enhanced_vibextractor.bearing_fault_rms
get_bearing_fault_rms = enhanced_vibextractor.get_bearing_fault_rms

# Export cache management functions at module level
refresh_all_caches = enhanced_vibextractor.refresh_all_caches
clear_cache = enhanced_vibextractor.clear_cache
get_cache_stats = enhanced_vibextractor.get_cache_stats
get_rms_forecast = enhanced_vibextractor.get_rms_forecast

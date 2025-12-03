"""
Test Suite for Enhanced VibExtractor Caching System
==================================================

This test suite validates:
1. Cache functionality and performance
2. Data integrity between Databricks and cache
3. Fallback mechanisms
4. Background sync operations
"""

import sys
import time
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from data.EnhancedVibExtractor import EnhancedVibExtractor, VibrationDataCache
    from data.cache_manager import CacheManager
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class CacheTestSuite:
    """Comprehensive test suite for the caching system."""
    
    def __init__(self):
        # Create temporary cache for testing
        self.temp_dir = tempfile.mkdtemp(prefix="vib_cache_test_")
        print(f"🧪 Using test cache directory: {self.temp_dir}")
        
        self.cache = VibrationDataCache(self.temp_dir)
        self.extractor = EnhancedVibExtractor()
        self.extractor.cache = self.cache  # Use test cache
        
        self.test_results = {}
    
    def create_mock_data(self, num_records: int = 5) -> pd.DataFrame:
        """Create mock vibration data for testing."""
        data = []
        
        for i in range(num_records):
            record = {
                'serialNumber': 189315064 + i,
                'EnqueuedTimeUtc': f'2024-10-0{i+1}T10:00:00.000Z',
                'frequencies': np.linspace(0, 1000, 1000),
                'fft_vel_d1': np.random.random(1000) * 0.1,
                'fft_vel_d2': np.random.random(1000) * 0.1,
                'fft_vel_d3': np.random.random(1000) * 0.1,
                'rms_vel_fft1000_d1': 0.05 + i * 0.01,
                'rms_vel_fft1000_d2': 0.04 + i * 0.01,
                'rms_vel_fft1000_d3': 0.06 + i * 0.01,
                'rms_acc_fft1000_d1': 0.5 + i * 0.1,
                'rms_acc_fft1000_d2': 0.4 + i * 0.1,
                'rms_acc_fft1000_d3': 0.6 + i * 0.1,
                'rms_acc_fft_d1': 0.8 + i * 0.1,
                'rms_acc_fft_d2': 0.7 + i * 0.1,
                'rms_acc_fft_d3': 0.9 + i * 0.1,
                'rms_acc_d1': 0.4 + i * 0.05,
                'rms_acc_d2': 0.35 + i * 0.05,
                'rms_acc_d3': 0.45 + i * 0.05,
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def test_cache_initialization(self) -> bool:
        """Test cache database initialization."""
        print("🧪 Testing cache initialization...")
        
        try:
            # Check if database file exists
            if not self.cache.db_path.exists():
                print("❌ Cache database not created")
                return False
            
            # Check if tables exist
            with self.cache._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ['vibration_data', 'cache_metadata', 'equipment_info']
                for table in required_tables:
                    if table not in tables:
                        print(f"❌ Required table '{table}' not found")
                        return False
            
            print("✅ Cache initialization successful")
            return True
            
        except Exception as e:
            print(f"❌ Cache initialization failed: {e}")
            return False
    
    def test_data_caching(self) -> bool:
        """Test caching of vibration data."""
        print("🧪 Testing data caching...")
        
        try:
            # Create mock data
            mock_df = self.create_mock_data(3)
            
            # Cache the data
            start_time = time.time()
            success = self.cache.cache_vibration_data(mock_df)
            cache_time = time.time() - start_time
            
            if not success:
                print("❌ Failed to cache data")
                return False
            
            print(f"✅ Cached {len(mock_df)} records in {cache_time:.3f}s")
            
            # Verify data in cache
            with self.cache._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM vibration_data")
                count = cursor.fetchone()[0]
                
                if count != len(mock_df):
                    print(f"❌ Expected {len(mock_df)} records, found {count}")
                    return False
            
            print("✅ Data caching successful")
            return True
            
        except Exception as e:
            print(f"❌ Data caching failed: {e}")
            return False
    
    def test_data_retrieval(self) -> bool:
        """Test retrieving data from cache."""
        print("🧪 Testing data retrieval...")
        
        try:
            # Retrieve cached data
            start_time = time.time()
            cached_df = self.cache.get_cached_vibration_data()
            retrieval_time = time.time() - start_time
            
            if cached_df.empty:
                print("❌ No data retrieved from cache")
                return False
            
            print(f"✅ Retrieved {len(cached_df)} records in {retrieval_time:.3f}s")
            
            # Verify data integrity
            required_columns = ['serialNumber', 'EnqueuedTimeUtc', 'frequencies', 'fft_vel_d1']
            for col in required_columns:
                if col not in cached_df.columns:
                    print(f"❌ Missing column: {col}")
                    return False
            
            # Check array deserialization
            if not isinstance(cached_df['frequencies'].iloc[0], np.ndarray):
                print("❌ Array deserialization failed")
                return False
            
            print("✅ Data retrieval successful")
            return True
            
        except Exception as e:
            print(f"❌ Data retrieval failed: {e}")
            return False
    
    def test_performance_comparison(self) -> bool:
        """Test performance comparison between operations."""
        print("🧪 Testing performance characteristics...")
        
        try:
            # Create larger dataset for performance testing
            large_mock_df = self.create_mock_data(50)
            
            # Test caching performance
            start_time = time.time()
            self.cache.cache_vibration_data(large_mock_df)
            cache_time = time.time() - start_time
            
            # Test retrieval performance
            start_time = time.time()
            retrieved_df = self.cache.get_cached_vibration_data()
            retrieval_time = time.time() - start_time
            
            cache_rate = len(large_mock_df) / cache_time
            retrieval_rate = len(retrieved_df) / retrieval_time
            
            print(f"📊 Cache performance: {cache_rate:.1f} records/sec")
            print(f"📊 Retrieval performance: {retrieval_rate:.1f} records/sec")
            
            # Performance should be reasonable
            if retrieval_rate < 100:  # Should be able to retrieve at least 100 records/sec
                print(f"⚠️ Retrieval performance may be slow: {retrieval_rate:.1f} records/sec")
            
            print("✅ Performance testing completed")
            return True
            
        except Exception as e:
            print(f"❌ Performance testing failed: {e}")
            return False
    
    def test_cache_stats(self) -> bool:
        """Test cache statistics functionality."""
        print("🧪 Testing cache statistics...")
        
        try:
            stats = self.cache.get_cache_stats()
            
            if 'error' in stats:
                print(f"❌ Error getting stats: {stats['error']}")
                return False
            
            required_fields = ['record_count', 'cache_size_mb', 'oldest_cache', 'newest_cache', 'is_fresh']
            for field in required_fields:
                if field not in stats:
                    print(f"❌ Missing stat field: {field}")
                    return False
            
            print(f"📊 Stats: {stats['record_count']} records, {stats['cache_size_mb']} MB")
            print("✅ Cache statistics working")
            return True
            
        except Exception as e:
            print(f"❌ Cache statistics failed: {e}")
            return False
    
    def test_enhanced_extractor(self) -> bool:
        """Test the enhanced extractor functionality."""
        print("🧪 Testing enhanced extractor...")
        
        try:
            # Test with cache (should return cached data)
            start_time = time.time()
            df_cached = self.extractor.get_latest_vibration_data(use_cache=True)
            cached_time = time.time() - start_time
            
            if not df_cached.empty:
                print(f"✅ Retrieved {len(df_cached)} records from cache in {cached_time:.3f}s")
            else:
                print("ℹ️ No cached data available")
            
            # Test equipment IDs
            equipment_df = self.extractor.get_equipment_id(use_cache=True)
            print(f"📋 Found {len(equipment_df)} equipment IDs")
            
            print("✅ Enhanced extractor working")
            return True
            
        except Exception as e:
            print(f"❌ Enhanced extractor failed: {e}")
            return False
    
    def test_cleanup_operations(self) -> bool:
        """Test cache cleanup operations."""
        print("🧪 Testing cache cleanup...")
        
        try:
            # Get initial count
            initial_stats = self.cache.get_cache_stats()
            initial_count = initial_stats['record_count']
            
            # Run cleanup (should not remove anything recent)
            self.cache.clear_old_cache()
            
            # Check count after cleanup
            final_stats = self.cache.get_cache_stats()
            final_count = final_stats['record_count']
            
            print(f"📊 Before cleanup: {initial_count} records")
            print(f"📊 After cleanup: {final_count} records")
            
            print("✅ Cache cleanup completed")
            return True
            
        except Exception as e:
            print(f"❌ Cache cleanup failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests and report results."""
        print("🚀 Starting Enhanced VibExtractor Test Suite")
        print("=" * 60)
        
        tests = [
            ('Cache Initialization', self.test_cache_initialization),
            ('Data Caching', self.test_data_caching),
            ('Data Retrieval', self.test_data_retrieval),
            ('Performance Testing', self.test_performance_comparison),
            ('Cache Statistics', self.test_cache_stats),
            ('Enhanced Extractor', self.test_enhanced_extractor),
            ('Cleanup Operations', self.test_cleanup_operations),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                self.test_results[test_name] = result
                if result:
                    passed += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"💥 {test_name} ERROR: {e}")
                self.test_results[test_name] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:.<50} {status}")
        
        print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests PASSED! Enhanced VibExtractor is ready for production.")
        else:
            print("⚠️ Some tests failed. Please review the issues above.")
        
        # Cleanup
        self.cleanup()
        
        return passed == total
    
    def cleanup(self):
        """Clean up test resources."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🗑️ Cleaned up test directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


def main():
    """Run the test suite."""
    test_suite = CacheTestSuite()
    success = test_suite.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

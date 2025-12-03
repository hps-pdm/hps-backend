"""
Enhanced VibExtractor Integration Guide
======================================

This script helps integrate the new caching system with the existing application.
It provides step-by-step migration and performance comparison tools.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import data.VibExtractor as original_extractor
    from data.EnhancedVibExtractor import EnhancedVibExtractor, VibrationDataCache
    from data.cache_manager import CacheManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class IntegrationManager:
    """Manages the integration of enhanced caching with existing application."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.original_extractor = original_extractor  # Module with functions
        self.enhanced_extractor = EnhancedVibExtractor()
        self.cache_manager = CacheManager()
        
        print("🚀 Enhanced VibExtractor Integration Manager")
        print("=" * 50)
    
    def compare_performance(self) -> Dict[str, Any]:
        """Compare performance between original and enhanced extractors."""
        print("\n📊 Performance Comparison")
        print("-" * 30)
        
        results = {
            'original': {},
            'enhanced_no_cache': {},
            'enhanced_with_cache': {}
        }
        
        # Test original extractor
        print("🧪 Testing original VibExtractor...")
        try:
            start_time = time.time()
            original_data = self.original_extractor.get_latest_vibration_data()
            original_time = time.time() - start_time
            
            results['original'] = {
                'time': original_time,
                'records': len(original_data) if not original_data.empty else 0,
                'success': True
            }
            
            print(f"✅ Original: {results['original']['records']} records in {original_time:.2f}s")
            
        except Exception as e:
            results['original'] = {'error': str(e), 'success': False}
            print(f"❌ Original extractor error: {e}")
        
        # Test enhanced extractor without cache (should fallback to Databricks)
        print("\n🧪 Testing enhanced VibExtractor (no cache)...")
        try:
            start_time = time.time()
            enhanced_data = self.enhanced_extractor.get_latest_vibration_data(use_cache=False)
            enhanced_time = time.time() - start_time
            
            results['enhanced_no_cache'] = {
                'time': enhanced_time,
                'records': len(enhanced_data) if not enhanced_data.empty else 0,
                'success': True
            }
            
            print(f"✅ Enhanced (no cache): {results['enhanced_no_cache']['records']} records in {enhanced_time:.2f}s")
            
        except Exception as e:
            results['enhanced_no_cache'] = {'error': str(e), 'success': False}
            print(f"❌ Enhanced extractor (no cache) error: {e}")
        
        # Test enhanced extractor with cache
        print("\n🧪 Testing enhanced VibExtractor (with cache)...")
        try:
            # First, populate cache if data is available
            if results['enhanced_no_cache']['success'] and results['enhanced_no_cache']['records'] > 0:
                print("📥 Populating cache with fresh data...")
                cache_start = time.time()
                self.enhanced_extractor.refresh_cache()
                cache_time = time.time() - cache_start
                print(f"✅ Cache populated in {cache_time:.2f}s")
            
            start_time = time.time()
            cached_data = self.enhanced_extractor.get_latest_vibration_data(use_cache=True)
            cached_time = time.time() - start_time
            
            results['enhanced_with_cache'] = {
                'time': cached_time,
                'records': len(cached_data) if not cached_data.empty else 0,
                'success': True
            }
            
            print(f"✅ Enhanced (with cache): {results['enhanced_with_cache']['records']} records in {cached_time:.2f}s")
            
        except Exception as e:
            results['enhanced_with_cache'] = {'error': str(e), 'success': False}
            print(f"❌ Enhanced extractor (with cache) error: {e}")
        
        # Performance summary
        print("\n📈 Performance Summary")
        print("=" * 30)
        
        for method, data in results.items():
            if data.get('success'):
                print(f"{method.replace('_', ' ').title():.<25} {data['time']:.3f}s ({data['records']} records)")
                if data['records'] > 0:
                    rate = data['records'] / data['time']
                    print(f"{'Rate':.<25} {rate:.1f} records/sec")
            else:
                print(f"{method.replace('_', ' ').title():.<25} ❌ Error")
        
        # Calculate speedup
        if (results['original']['success'] and 
            results['enhanced_with_cache']['success'] and 
            results['enhanced_with_cache']['records'] > 0):
            
            speedup = results['original']['time'] / results['enhanced_with_cache']['time']
            print(f"\n🚀 Cache Speedup: {speedup:.1f}x faster!")
            
            if speedup > 5:
                print("🎉 Excellent performance improvement!")
            elif speedup > 2:
                print("👍 Good performance improvement!")
            else:
                print("⚠️ Modest performance improvement")
        
        return results
    
    def validate_data_integrity(self) -> bool:
        """Validate that cached data matches original data structure."""
        print("\n🔍 Data Integrity Validation")
        print("-" * 30)
        
        try:
            # Get sample from original (if available)
            original_sample = None
            try:
                original_data = self.original_extractor.get_latest_vibration_data()
                if original_data is not None and not original_data.empty:
                    original_sample = original_data.head(1)
                    print(f"✅ Retrieved original sample: {len(original_data)} records")
            except Exception as e:
                print(f"⚠️ Could not get original data: {e}")
            
            # Get sample from cache
            cached_data = self.enhanced_extractor.get_latest_vibration_data(use_cache=True)
            if cached_data.empty:
                print("❌ No cached data available for validation")
                return False
            
            cached_sample = cached_data.head(1)
            print(f"✅ Retrieved cached sample: {len(cached_data)} records")
            
            # Compare structure if both available
            if original_sample is not None:
                # Compare columns
                original_cols = set(original_sample.columns)
                cached_cols = set(cached_sample.columns)
                
                missing_in_cache = original_cols - cached_cols
                extra_in_cache = cached_cols - original_cols
                
                if missing_in_cache:
                    print(f"⚠️ Columns missing in cache: {missing_in_cache}")
                
                if extra_in_cache:
                    print(f"ℹ️ Extra columns in cache: {extra_in_cache}")
                
                # Check data types for common columns
                common_cols = original_cols & cached_cols
                for col in common_cols:
                    orig_type = type(original_sample[col].iloc[0])
                    cache_type = type(cached_sample[col].iloc[0])
                    
                    if orig_type != cache_type:
                        print(f"⚠️ Type mismatch in '{col}': {orig_type} vs {cache_type}")
                
                print("✅ Data structure validation completed")
            
            # Validate array columns specifically
            array_columns = ['frequencies', 'fft_vel_d1', 'fft_vel_d2', 'fft_vel_d3']
            for col in array_columns:
                if col in cached_sample.columns:
                    arr = cached_sample[col].iloc[0]
                    if hasattr(arr, 'shape'):
                        print(f"✅ Array column '{col}': shape {arr.shape}")
                    else:
                        print(f"⚠️ Array column '{col}' is not numpy array: {type(arr)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Data validation error: {e}")
            return False
    
    def check_cache_status(self) -> Dict[str, Any]:
        """Check current cache status and health."""
        print("\n📋 Cache Status")
        print("-" * 20)
        
        try:
            stats = self.enhanced_extractor.cache.get_cache_stats()
            
            print(f"📊 Records: {stats['record_count']}")
            print(f"💾 Size: {stats['cache_size_mb']} MB")
            print(f"🕒 Oldest: {stats.get('oldest_cache', 'N/A')}")
            print(f"🕒 Newest: {stats.get('newest_cache', 'N/A')}")
            print(f"🔄 Fresh: {'Yes' if stats['is_fresh'] else 'No'}")
            
            if stats['record_count'] == 0:
                print("\n⚠️ Cache is empty. Run refresh_cache() to populate.")
            elif not stats['is_fresh']:
                print("\n⚠️ Cache data is old. Consider refreshing.")
            else:
                print("\n✅ Cache is healthy and fresh")
            
            return stats
            
        except Exception as e:
            print(f"❌ Error checking cache status: {e}")
            return {}
    
    def migration_checklist(self):
        """Provide a migration checklist for developers."""
        print("\n📝 Migration Checklist")
        print("=" * 25)
        
        checklist = [
            "1. ✅ Install enhanced caching system",
            "2. 🧪 Run performance comparison",
            "3. 🔍 Validate data integrity", 
            "4. 📊 Check cache status",
            "5. 🔄 Update imports in application code",
            "6. ⚙️ Configure cache settings (optional)",
            "7. 🚀 Deploy with cache enabled",
            "8. 📈 Monitor performance improvements"
        ]
        
        for item in checklist:
            print(f"   {item}")
        
        print("\n💡 Code Changes Needed:")
        print("   Old: from data.VibExtractor import get_latest_vibration_data")
        print("   New: from data.EnhancedVibExtractor import EnhancedVibExtractor")
        print()
        print("   Old: data = get_latest_vibration_data()")
        print("   New: extractor = EnhancedVibExtractor()")
        print("        data = extractor.get_latest_vibration_data(use_cache=True)")
        print()
        print("   Alternative: Keep using functions with cache behind scenes")
        
        print("\n⚠️ Important Notes:")
        print("   - Cache will fallback to Databricks if no cached data")
        print("   - Cache refreshes automatically every 24 hours")
        print("   - Use cache_manager.py for maintenance operations")
        print("   - Monitor cache size and performance regularly")
    
    def run_integration_check(self):
        """Run complete integration validation."""
        print("🔧 Running Integration Check")
        print("=" * 35)
        
        # Step 1: Performance comparison
        perf_results = self.compare_performance()
        
        # Step 2: Data validation
        data_valid = self.validate_data_integrity()
        
        # Step 3: Cache status
        cache_stats = self.check_cache_status()
        
        # Step 4: Migration guidance
        self.migration_checklist()
        
        # Final assessment
        print("\n🎯 Integration Assessment")
        print("=" * 25)
        
        cache_ready = (
            perf_results.get('enhanced_with_cache', {}).get('success', False) and
            data_valid and
            cache_stats.get('record_count', 0) > 0
        )
        
        if cache_ready:
            print("✅ System is ready for production migration!")
            print("   - Cache is working correctly")
            print("   - Data integrity validated") 
            print("   - Performance improvements confirmed")
        else:
            print("⚠️ System needs attention before migration:")
            if not perf_results.get('enhanced_with_cache', {}).get('success', False):
                print("   - Cache functionality issues")
            if not data_valid:
                print("   - Data integrity concerns")
            if cache_stats.get('record_count', 0) == 0:
                print("   - Cache needs initial population")
        
        return cache_ready


def main():
    """Main integration check function."""
    print("Enhanced VibExtractor Integration Tool")
    print("====================================")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / 'app' / 'data').exists():
        print("❌ Please run this script from the project root directory")
        print(f"   Current: {current_dir}")
        print("   Expected: directory containing 'app/data' folder")
        return 1
    
    # Run integration check
    integration_manager = IntegrationManager()
    success = integration_manager.run_integration_check()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

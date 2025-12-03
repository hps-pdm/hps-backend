"""
Cache Management Utility for Enhanced VibExtractor
================================================

This utility provides tools for managing the local SQLite cache:
1. Manual cache updates from Databricks
2. Cache performance monitoring
3. Data migration and backup
4. Cache optimization and cleanup
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.EnhancedVibExtractor import EnhancedVibExtractor, VibrationDataCache

class CacheManager:
    """Utility class for managing vibration data cache."""
    
    def __init__(self):
        self.extractor = EnhancedVibExtractor()
        self.cache = self.extractor.cache
    
    def sync_from_databricks(self, force: bool = False):
        """Manually sync data from Databricks to local cache."""
        print("🔄 Starting manual cache sync from Databricks...")
        
        try:
            # Get fresh data from Databricks
            start_time = time.time()
            df = self.extractor.get_latest_vibration_data(use_cache=False)
            fetch_time = time.time() - start_time
            
            if df.empty:
                print("❌ No data retrieved from Databricks")
                return False
            
            print(f"📥 Fetched {len(df)} records in {fetch_time:.2f}s")
            
            # Cache the data
            start_time = time.time()
            success = self.cache.cache_vibration_data(df)
            cache_time = time.time() - start_time
            
            if success:
                print(f"✅ Cached data successfully in {cache_time:.2f}s")
                self.show_cache_stats()
                return True
            else:
                print("❌ Failed to cache data")
                return False
                
        except Exception as e:
            print(f"❌ Error during sync: {e}")
            return False
    
    def show_cache_stats(self):
        """Display detailed cache statistics."""
        print("\n📊 Cache Statistics:")
        print("=" * 50)
        
        stats = self.cache.get_cache_stats()
        if 'error' in stats:
            print(f"❌ Error getting stats: {stats['error']}")
            return
        
        print(f"📋 Records: {stats['record_count']:,}")
        print(f"💾 Size: {stats['cache_size_mb']} MB")
        print(f"🕐 Oldest: {stats['oldest_cache']}")
        print(f"🕑 Newest: {stats['newest_cache']}")
        print(f"✨ Fresh: {'Yes' if stats['is_fresh'] else 'No'}")
        
        # Performance comparison
        self._performance_comparison()
    
    def _performance_comparison(self):
        """Compare cache vs Databricks performance."""
        print("\n⚡ Performance Comparison:")
        print("-" * 30)
        
        try:
            # Test cache performance
            start_time = time.time()
            cached_df = self.cache.get_cached_vibration_data()
            cache_time = time.time() - start_time
            
            cache_records = len(cached_df)
            print(f"📋 Cache: {cache_records} records in {cache_time:.3f}s")
            
            if cache_records > 0:
                print(f"🚀 Cache speed: {cache_records/cache_time:.1f} records/second")
            
        except Exception as e:
            print(f"❌ Cache test failed: {e}")
    
    def cleanup_cache(self, days_old: int = 7):
        """Clean up old cache entries."""
        print(f"🗑️ Cleaning cache entries older than {days_old} days...")
        
        try:
            # Temporarily adjust TTL for cleanup
            original_ttl = self.cache.cache_ttl_hours
            self.cache.cache_ttl_hours = days_old * 24
            
            self.cache.clear_old_cache()
            
            # Restore original TTL
            self.cache.cache_ttl_hours = original_ttl
            
            print("✅ Cache cleanup completed")
            self.show_cache_stats()
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    
    def backup_cache(self, backup_path: str):
        """Create a backup of the cache database."""
        print(f"💾 Creating cache backup to {backup_path}...")
        
        try:
            import shutil
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(exist_ok=True)
            
            shutil.copy2(self.cache.db_path, backup_file)
            
            file_size = backup_file.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ Backup created successfully ({file_size:.2f} MB)")
            
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
    
    def restore_cache(self, backup_path: str):
        """Restore cache from backup."""
        print(f"📥 Restoring cache from {backup_path}...")
        
        try:
            import shutil
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                print(f"❌ Backup file not found: {backup_path}")
                return False
            
            # Create backup of current cache
            current_backup = self.cache.cache_dir / f"cache_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(self.cache.db_path, current_backup)
            print(f"📋 Current cache backed up to {current_backup}")
            
            # Restore from backup
            shutil.copy2(backup_file, self.cache.db_path)
            
            print("✅ Cache restored successfully")
            self.show_cache_stats()
            
            return True
            
        except Exception as e:
            print(f"❌ Error restoring cache: {e}")
            return False
    
    def validate_cache(self):
        """Validate cache integrity and consistency."""
        print("🔍 Validating cache integrity...")
        
        try:
            with self.cache._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check for duplicate records
                cursor.execute("""
                    SELECT serialNumber, EnqueuedTimeUtc, COUNT(*) as count
                    FROM vibration_data
                    GROUP BY serialNumber, EnqueuedTimeUtc
                    HAVING count > 1
                """)
                duplicates = cursor.fetchall()
                
                if duplicates:
                    print(f"⚠️ Found {len(duplicates)} duplicate records")
                    for dup in duplicates[:5]:  # Show first 5
                        print(f"   - Serial {dup[0]}, Time {dup[1]}: {dup[2]} copies")
                else:
                    print("✅ No duplicate records found")
                
                # Check for null critical fields
                cursor.execute("""
                    SELECT COUNT(*) FROM vibration_data 
                    WHERE serialNumber IS NULL OR EnqueuedTimeUtc IS NULL
                """)
                null_count = cursor.fetchone()[0]
                
                if null_count > 0:
                    print(f"⚠️ Found {null_count} records with null critical fields")
                else:
                    print("✅ All records have valid critical fields")
                
                # Check array data integrity
                cursor.execute("""
                    SELECT COUNT(*) FROM vibration_data 
                    WHERE frequencies IS NULL OR fft_vel_d1 IS NULL
                """)
                missing_arrays = cursor.fetchone()[0]
                
                if missing_arrays > 0:
                    print(f"⚠️ Found {missing_arrays} records with missing array data")
                else:
                    print("✅ All records have complete array data")
                
                print("✅ Cache validation completed")
                
        except Exception as e:
            print(f"❌ Error during validation: {e}")


def main():
    """Command-line interface for cache management."""
    parser = argparse.ArgumentParser(description="Manage VibExtractor cache")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Sync data from Databricks to cache')
    sync_parser.add_argument('--force', action='store_true', help='Force sync even if cache is fresh')
    
    # Stats command
    subparsers.add_parser('stats', help='Show cache statistics')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old cache entries')
    cleanup_parser.add_argument('--days', type=int, default=7, help='Remove entries older than N days')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create cache backup')
    backup_parser.add_argument('path', help='Backup file path')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore cache from backup')
    restore_parser.add_argument('path', help='Backup file path')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate cache integrity')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize cache manager
    manager = CacheManager()
    
    # Execute commands
    if args.command == 'sync':
        manager.sync_from_databricks(force=args.force)
    elif args.command == 'stats':
        manager.show_cache_stats()
    elif args.command == 'cleanup':
        manager.cleanup_cache(args.days)
    elif args.command == 'backup':
        manager.backup_cache(args.path)
    elif args.command == 'restore':
        manager.restore_cache(args.path)
    elif args.command == 'validate':
        manager.validate_cache()


if __name__ == "__main__":
    main()

"""
Unit tests for KV Cache Manager.

Tests specific scenarios, edge cases, and error conditions.
"""

import pytest
from datetime import datetime, timedelta
from mm_orch.optimization.kv_cache_manager import (
    KVCacheManager,
    CacheConfig,
    KVCache,
    CacheStats
)


class TestKVCacheManagerBasics:
    """Test basic cache manager functionality."""
    
    def test_initialization(self):
        """Test cache manager initialization."""
        config = CacheConfig(
            max_memory_mb=2048,
            eviction_policy="lru",
            enabled=True
        )
        manager = KVCacheManager(config)
        
        assert manager.config == config
        assert len(manager._cache) == 0
        assert manager._memory_used_mb == 0.0
        assert manager._total_hits == 0
        assert manager._total_misses == 0
    
    def test_store_and_retrieve_cache(self):
        """Test storing and retrieving cache."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Store cache
        manager.store_cache("conv1", "model1", "test_data", 100.0)
        
        # Retrieve cache
        cached = manager.get_cache("conv1", "model1")
        
        assert cached is not None
        assert cached.conversation_id == "conv1"
        assert cached.model_name == "model1"
        assert cached.cache_data == "test_data"
        assert cached.memory_mb == 100.0
        assert cached.access_count == 1  # One access from get_cache
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Try to retrieve non-existent cache
        cached = manager.get_cache("nonexistent", "model1")
        
        assert cached is None
        
        stats = manager.get_stats()
        assert stats.total_misses == 1
        assert stats.total_hits == 0
    
    def test_cache_disabled(self):
        """Test that cache operations are no-ops when disabled."""
        config = CacheConfig(enabled=False)
        manager = KVCacheManager(config)
        
        # Store should be no-op
        manager.store_cache("conv1", "model1", "data", 100.0)
        
        # Retrieve should return None
        cached = manager.get_cache("conv1", "model1")
        assert cached is None
        
        # Stats should show no activity
        stats = manager.get_stats()
        assert stats.total_entries == 0
        assert stats.memory_used_mb == 0.0


class TestLRUEviction:
    """Test LRU eviction behavior."""
    
    def test_eviction_on_memory_limit(self):
        """Test that eviction occurs when memory limit is reached."""
        config = CacheConfig(max_memory_mb=250, enabled=True)
        manager = KVCacheManager(config)
        
        # Add 3 entries of 100MB each (total 300MB > 250MB limit)
        manager.store_cache("conv1", "model1", "data1", 100.0)
        manager.store_cache("conv2", "model1", "data2", 100.0)
        manager.store_cache("conv3", "model1", "data3", 100.0)
        
        # Check that memory is within limit
        stats = manager.get_stats()
        assert stats.memory_used_mb <= 250
        assert stats.evictions > 0
        
        # First entry should be evicted (LRU)
        cached1 = manager.get_cache("conv1", "model1")
        assert cached1 is None, "Oldest entry should be evicted"
        
        # Later entries should still exist
        cached3 = manager.get_cache("conv3", "model1")
        assert cached3 is not None, "Recent entry should still exist"
    
    def test_lru_order_maintained(self):
        """Test that LRU order is maintained with accesses."""
        config = CacheConfig(max_memory_mb=250, enabled=True)
        manager = KVCacheManager(config)
        
        # Add 2 entries
        manager.store_cache("conv1", "model1", "data1", 100.0)
        manager.store_cache("conv2", "model1", "data2", 100.0)
        
        # Access conv1 to make it more recent
        manager.get_cache("conv1", "model1")
        
        # Add third entry that triggers eviction
        manager.store_cache("conv3", "model1", "data3", 100.0)
        
        # conv2 should be evicted (least recently used)
        cached2 = manager.get_cache("conv2", "model1")
        assert cached2 is None, "conv2 should be evicted as LRU"
        
        # conv1 should still exist (was accessed recently)
        cached1 = manager.get_cache("conv1", "model1")
        assert cached1 is not None, "conv1 should still exist"
    
    def test_manual_eviction(self):
        """Test manual eviction with evict_lru."""
        config = CacheConfig(max_memory_mb=250, enabled=True)
        manager = KVCacheManager(config)
        
        # Add entries that fill memory
        manager.store_cache("conv1", "model1", "data1", 100.0)
        manager.store_cache("conv2", "model1", "data2", 100.0)
        
        initial_memory = manager._memory_used_mb
        initial_entries = len(manager._cache)
        
        # Manually evict - this will evict if adding 100MB would exceed limit
        manager.evict_lru(100.0)
        
        # Check that memory was freed (since 200 + 100 > 250)
        assert manager._memory_used_mb < initial_memory
        assert len(manager._cache) < initial_entries
        assert manager._total_evictions > 0
    
    def test_eviction_from_empty_cache(self):
        """Test that eviction from empty cache doesn't crash."""
        config = CacheConfig(max_memory_mb=1000, enabled=True)
        manager = KVCacheManager(config)
        
        # Should not crash
        manager.evict_lru(100.0)
        
        stats = manager.get_stats()
        assert stats.evictions == 0


class TestConversationManagement:
    """Test conversation-level cache management."""
    
    def test_release_conversation(self):
        """Test releasing all caches for a conversation."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Add caches for multiple models in same conversation
        manager.store_cache("conv1", "model1", "data1", 100.0)
        manager.store_cache("conv1", "model2", "data2", 100.0)
        manager.store_cache("conv2", "model1", "data3", 100.0)
        
        initial_memory = manager._memory_used_mb
        
        # Release conv1
        manager.release_conversation("conv1")
        
        # Check that conv1 caches are gone
        assert manager.get_cache("conv1", "model1") is None
        assert manager.get_cache("conv1", "model2") is None
        
        # Check that conv2 cache still exists
        assert manager.get_cache("conv2", "model1") is not None
        
        # Check memory was freed
        assert manager._memory_used_mb == initial_memory - 200.0
    
    def test_release_nonexistent_conversation(self):
        """Test releasing a conversation that doesn't exist."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Should not crash
        manager.release_conversation("nonexistent")
        
        stats = manager.get_stats()
        assert stats.total_entries == 0


class TestCacheStatistics:
    """Test cache statistics tracking."""
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Store cache
        manager.store_cache("conv1", "model1", "data", 100.0)
        
        # Generate hits and misses
        manager.get_cache("conv1", "model1")  # Hit
        manager.get_cache("conv1", "model1")  # Hit
        manager.get_cache("conv2", "model1")  # Miss
        
        stats = manager.get_stats()
        assert stats.total_hits == 2
        assert stats.total_misses == 1
        assert stats.hit_rate == 2.0 / 3.0
    
    def test_stats_with_no_accesses(self):
        """Test stats when no cache accesses have occurred."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        stats = manager.get_stats()
        assert stats.hit_rate == 0.0
        assert stats.total_hits == 0
        assert stats.total_misses == 0
        assert stats.total_entries == 0
        assert stats.memory_used_mb == 0.0
        assert stats.evictions == 0
    
    def test_access_count_tracking(self):
        """Test that access count is tracked correctly."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        manager.store_cache("conv1", "model1", "data", 100.0)
        
        # Access multiple times
        for _ in range(5):
            cached = manager.get_cache("conv1", "model1")
        
        assert cached.access_count == 5


class TestCacheUpdates:
    """Test cache update behavior."""
    
    def test_update_existing_cache(self):
        """Test updating an existing cache entry."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Store initial cache
        manager.store_cache("conv1", "model1", "data1", 100.0)
        
        # Update with new data
        manager.store_cache("conv1", "model1", "data2", 150.0)
        
        # Retrieve and verify
        cached = manager.get_cache("conv1", "model1")
        assert cached.cache_data == "data2"
        assert cached.memory_mb == 150.0
        
        # Memory should be updated correctly
        stats = manager.get_stats()
        assert stats.memory_used_mb == 150.0
    
    def test_update_with_smaller_size(self):
        """Test updating cache with smaller memory size."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Store initial cache
        manager.store_cache("conv1", "model1", "data1", 200.0)
        
        # Update with smaller size
        manager.store_cache("conv1", "model1", "data2", 100.0)
        
        # Memory should decrease
        stats = manager.get_stats()
        assert stats.memory_used_mb == 100.0


class TestCacheClearing:
    """Test cache clearing functionality."""
    
    def test_clear_cache(self):
        """Test clearing all cache entries."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Add multiple entries
        manager.store_cache("conv1", "model1", "data1", 100.0)
        manager.store_cache("conv2", "model1", "data2", 100.0)
        manager.store_cache("conv3", "model1", "data3", 100.0)
        
        # Clear
        manager.clear()
        
        # Verify all cleared
        stats = manager.get_stats()
        assert stats.total_entries == 0
        assert stats.memory_used_mb == 0.0
        
        # Verify entries not retrievable
        assert manager.get_cache("conv1", "model1") is None
        assert manager.get_cache("conv2", "model1") is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_memory_cache(self):
        """Test cache with zero memory size."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Store cache with 0 memory
        manager.store_cache("conv1", "model1", "data", 0.0)
        
        # Should still be stored
        cached = manager.get_cache("conv1", "model1")
        assert cached is not None
        assert cached.memory_mb == 0.0
    
    def test_very_large_cache(self):
        """Test cache larger than max memory."""
        config = CacheConfig(max_memory_mb=100, enabled=True)
        manager = KVCacheManager(config)
        
        # Try to store cache larger than limit
        manager.store_cache("conv1", "model1", "data", 200.0)
        
        # Should trigger eviction but still store
        stats = manager.get_stats()
        assert stats.memory_used_mb <= 200.0  # Should be stored
    
    def test_multiple_models_same_conversation(self):
        """Test multiple models for same conversation."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Store caches for different models
        manager.store_cache("conv1", "model1", "data1", 100.0)
        manager.store_cache("conv1", "model2", "data2", 100.0)
        manager.store_cache("conv1", "model3", "data3", 100.0)
        
        # All should be retrievable
        assert manager.get_cache("conv1", "model1") is not None
        assert manager.get_cache("conv1", "model2") is not None
        assert manager.get_cache("conv1", "model3") is not None
        
        stats = manager.get_stats()
        assert stats.total_entries == 3
    
    def test_cache_key_generation(self):
        """Test that cache keys are unique per conversation and model."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Store different combinations
        manager.store_cache("conv1", "model1", "data1", 100.0)
        manager.store_cache("conv1", "model2", "data2", 100.0)
        manager.store_cache("conv2", "model1", "data3", 100.0)
        
        # Each should be separate
        cached1 = manager.get_cache("conv1", "model1")
        cached2 = manager.get_cache("conv1", "model2")
        cached3 = manager.get_cache("conv2", "model1")
        
        assert cached1.cache_data == "data1"
        assert cached2.cache_data == "data2"
        assert cached3.cache_data == "data3"

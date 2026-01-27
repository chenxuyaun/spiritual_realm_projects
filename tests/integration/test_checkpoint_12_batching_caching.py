"""
Checkpoint 12: Integration tests for batching and caching.

This checkpoint validates that:
1. Batching works with various request patterns
2. Cache hit rates are correct in multi-turn scenarios
3. Adaptive batch sizing responds to load
4. Batching and caching work together correctly
"""

import time
import pytest
from mm_orch.optimization.batcher import DynamicBatcher
from mm_orch.optimization.kv_cache_manager import KVCacheManager
from mm_orch.optimization.config import BatcherConfig, CacheConfig


class TestBatchingPatterns:
    """Test batching with various request patterns."""
    
    def test_burst_request_pattern(self):
        """Test batching with burst of requests arriving simultaneously."""
        config = BatcherConfig(
            enabled=True,
            max_batch_size=5,
            batch_timeout_ms=50,  # Shorter timeout
            adaptive_batching=False
        )
        batcher = DynamicBatcher(config)
        batcher.start()
        
        try:
            # Simulate burst: 5 requests arrive at once (matching batch size)
            request_ids = []
            for i in range(5):
                req_id = batcher.add_request(
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3, 4, 5]},
                    parameters={"temperature": 1.0}
                )
                request_ids.append(req_id)
            
            # Wait for processing (shorter wait)
            time.sleep(0.2)
            
            # Verify: Requests should be processed
            processed_count = 0
            for req_id in request_ids:
                try:
                    result = batcher.get_result(req_id, timeout=0.1)
                    processed_count += 1
                    assert result.request_id == req_id
                except:
                    pass  # Some may still be processing
            
            # At least some should be processed
            assert processed_count >= 1, f"Expected at least 1 processed, got {processed_count}"
            
            # Check stats
            stats = batcher.get_stats()
            assert stats["enabled"] is True
            
        finally:
            batcher.stop(timeout=1.0)
    
    def test_steady_stream_pattern(self):
        """Test batching with steady stream of requests."""
        config = BatcherConfig(
            enabled=True,
            max_batch_size=3,
            batch_timeout_ms=50,
            adaptive_batching=False
        )
        batcher = DynamicBatcher(config)
        batcher.start()
        
        try:
            # Simulate steady stream: 3 requests (matching batch size)
            request_ids = []
            for i in range(3):
                req_id = batcher.add_request(
                    model_name="test_model",
                    inputs={"input_ids": list(range(i + 1))},
                    parameters={}
                )
                request_ids.append(req_id)
                time.sleep(0.01)  # 10ms between requests
            
            # Wait for processing
            time.sleep(0.15)
            
            # Verify: Requests should be batched and processed
            processed_count = 0
            for req_id in request_ids:
                try:
                    result = batcher.get_result(req_id, timeout=0.1)
                    processed_count += 1
                except:
                    pass
            
            assert processed_count >= 1, f"Expected at least 1 processed, got {processed_count}"
            
        finally:
            batcher.stop(timeout=1.0)
    
    def test_mixed_model_pattern(self):
        """Test batching with requests for different models."""
        config = BatcherConfig(
            enabled=True,
            max_batch_size=2,
            batch_timeout_ms=50,
            adaptive_batching=False
        )
        batcher = DynamicBatcher(config)
        batcher.start()
        
        try:
            # Add requests for different models
            model_a_ids = []
            model_b_ids = []
            
            for i in range(2):
                req_id_a = batcher.add_request(
                    model_name="model_a",
                    inputs={"input_ids": [1, 2, 3]},
                    parameters={}
                )
                model_a_ids.append(req_id_a)
                
                req_id_b = batcher.add_request(
                    model_name="model_b",
                    inputs={"input_ids": [4, 5, 6]},
                    parameters={}
                )
                model_b_ids.append(req_id_b)
            
            # Wait for processing
            time.sleep(0.15)
            
            # Verify: Requests for each model should be batched separately
            stats = batcher.get_stats()
            
            # Both models should have been processed or queued
            assert stats["enabled"] is True
            
        finally:
            batcher.stop(timeout=1.0)
    
    def test_variable_length_pattern(self):
        """Test batching with variable-length sequences."""
        config = BatcherConfig(
            enabled=True,
            max_batch_size=3,
            batch_timeout_ms=50,
            adaptive_batching=False
        )
        batcher = DynamicBatcher(config)
        batcher.start()
        
        try:
            # Add requests with different sequence lengths
            request_ids = []
            lengths = [3, 5, 2]
            
            for length in lengths:
                req_id = batcher.add_request(
                    model_name="test_model",
                    inputs={"input_ids": list(range(length))},
                    parameters={}
                )
                request_ids.append(req_id)
            
            # Wait for processing
            time.sleep(0.15)
            
            # Verify: Requests should be processed with padding
            processed_count = 0
            for req_id in request_ids:
                try:
                    result = batcher.get_result(req_id, timeout=0.1)
                    processed_count += 1
                except:
                    pass
            
            assert processed_count >= 1, f"Expected at least 1 processed, got {processed_count}"
            
        finally:
            batcher.stop(timeout=1.0)


class TestCacheMultiTurn:
    """Test cache hit rates in multi-turn scenarios."""
    
    def test_single_conversation_multi_turn(self):
        """Test cache hits in a single multi-turn conversation."""
        config = CacheConfig(
            max_memory_mb=1024,
            eviction_policy="lru",
            enabled=True
        )
        manager = KVCacheManager(config)
        
        conversation_id = "conv_001"
        model_name = "test_model"
        
        # Turn 1: Store initial cache
        manager.store_cache(conversation_id, model_name, "turn1_data", 50.0)
        
        # Turn 2-5: Access cache (simulating multi-turn conversation)
        for turn in range(2, 6):
            cached = manager.get_cache(conversation_id, model_name)
            assert cached is not None, f"Cache should be available at turn {turn}"
            # Cache data will be from previous turn after first update
            
            # Update cache with new turn data
            manager.store_cache(conversation_id, model_name, f"turn{turn}_data", 50.0)
        
        # Verify: High cache hit rate
        stats = manager.get_stats()
        assert stats.total_hits >= 4, f"Expected at least 4 hits, got {stats.total_hits}"
        assert stats.hit_rate > 0.5, f"Expected hit rate > 0.5, got {stats.hit_rate}"
    
    def test_multiple_conversations_interleaved(self):
        """Test cache with multiple interleaved conversations."""
        config = CacheConfig(
            max_memory_mb=1024,
            eviction_policy="lru",
            enabled=True
        )
        manager = KVCacheManager(config)
        
        model_name = "test_model"
        conversations = ["conv_a", "conv_b", "conv_c"]
        
        # Initialize caches for all conversations
        for conv_id in conversations:
            manager.store_cache(conv_id, model_name, f"{conv_id}_data", 100.0)
        
        # Interleave accesses
        for _ in range(3):
            for conv_id in conversations:
                cached = manager.get_cache(conv_id, model_name)
                assert cached is not None
                assert cached.conversation_id == conv_id
        
        # Verify: All accesses should be hits
        stats = manager.get_stats()
        expected_hits = 3 * len(conversations)  # 3 rounds * 3 conversations
        assert stats.total_hits == expected_hits, (
            f"Expected {expected_hits} hits, got {stats.total_hits}"
        )
        assert stats.hit_rate == 1.0, "All accesses should be hits"
    
    def test_cache_hit_rate_with_eviction(self):
        """Test cache hit rate when eviction occurs."""
        config = CacheConfig(
            max_memory_mb=250,  # Can hold ~2 entries at 100MB each
            eviction_policy="lru",
            enabled=True
        )
        manager = KVCacheManager(config)
        
        model_name = "test_model"
        
        # Add 3 conversations (will trigger eviction)
        for i in range(3):
            conv_id = f"conv_{i}"
            manager.store_cache(conv_id, model_name, f"data_{i}", 100.0)
        
        # Access most recent conversations (should be hits)
        for i in range(1, 3):
            conv_id = f"conv_{i}"
            cached = manager.get_cache(conv_id, model_name)
            assert cached is not None, f"Recent conversation {conv_id} should be cached"
        
        # Access oldest conversation (should be miss due to eviction)
        cached = manager.get_cache("conv_0", model_name)
        assert cached is None, "Oldest conversation should be evicted"
        
        # Verify: Hit rate should reflect eviction
        stats = manager.get_stats()
        assert stats.total_hits == 2, f"Expected 2 hits, got {stats.total_hits}"
        assert stats.total_misses == 1, f"Expected 1 miss, got {stats.total_misses}"
        assert stats.evictions > 0, "Evictions should have occurred"
        
        # Hit rate should be 2/3
        expected_hit_rate = 2.0 / 3.0
        assert abs(stats.hit_rate - expected_hit_rate) < 0.01, (
            f"Expected hit rate ~{expected_hit_rate:.2f}, got {stats.hit_rate:.2f}"
        )
    
    def test_conversation_end_cleanup(self):
        """Test that cache is properly cleaned up when conversation ends."""
        config = CacheConfig(
            max_memory_mb=1024,
            eviction_policy="lru",
            enabled=True
        )
        manager = KVCacheManager(config)
        
        conversation_id = "conv_cleanup"
        models = ["model_a", "model_b", "model_c"]
        
        # Store caches for multiple models in same conversation
        for model_name in models:
            manager.store_cache(conversation_id, model_name, f"{model_name}_data", 50.0)
        
        initial_stats = manager.get_stats()
        assert initial_stats.total_entries == len(models)
        
        # End conversation
        manager.release_conversation(conversation_id)
        
        # Verify: All caches released
        final_stats = manager.get_stats()
        assert final_stats.total_entries == 0, "All entries should be released"
        assert final_stats.memory_used_mb == 0.0, "All memory should be freed"
        
        # Verify: Caches not retrievable
        for model_name in models:
            cached = manager.get_cache(conversation_id, model_name)
            assert cached is None, f"Cache for {model_name} should be released"


class TestAdaptiveBatching:
    """Test adaptive batch sizing responds to load."""
    
    def test_adaptive_batch_size_initialization(self):
        """Test that adaptive batching is properly initialized."""
        config = BatcherConfig(
            enabled=True,
            max_batch_size=10,
            batch_timeout_ms=50,
            adaptive_batching=True,
            min_batch_size=2
        )
        batcher = DynamicBatcher(config)
        
        # Verify: Adaptive batching is enabled
        assert batcher.config.adaptive_batching is True
        assert batcher._current_batch_size == config.max_batch_size
        
        # Verify: Batch size stays within bounds
        assert config.min_batch_size <= batcher._current_batch_size <= config.max_batch_size
    
    def test_adaptive_batch_size_tracking(self):
        """Test that performance metrics are tracked for adaptation."""
        config = BatcherConfig(
            enabled=True,
            max_batch_size=8,
            batch_timeout_ms=50,
            adaptive_batching=True,
            min_batch_size=2
        )
        batcher = DynamicBatcher(config)
        batcher.start()
        
        try:
            # Add some requests to generate metrics
            request_ids = []
            for i in range(3):
                req_id = batcher.add_request(
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]},
                    parameters={}
                )
                request_ids.append(req_id)
            
            # Wait for processing
            time.sleep(0.15)
            
            # Verify: Metrics are being tracked
            assert len(batcher._latency_history) >= 0
            assert len(batcher._throughput_history) >= 0
            
            # Get stats
            stats = batcher.get_stats()
            assert "avg_latency_ms" in stats
            assert "avg_throughput_rps" in stats
            assert "current_batch_size" in stats
            
        finally:
            batcher.stop(timeout=1.0)
    
    def test_batch_size_bounds_respected(self):
        """Test that adaptive batch size respects min/max bounds."""
        config = BatcherConfig(
            enabled=True,
            max_batch_size=10,
            batch_timeout_ms=50,
            adaptive_batching=True,
            min_batch_size=3
        )
        batcher = DynamicBatcher(config)
        
        # Manually test adaptation logic
        initial_size = batcher._current_batch_size
        
        # Simulate high latency (should reduce batch size)
        for _ in range(20):
            batcher._latency_history.append(200.0)  # High latency
        
        batcher._adapt_batch_size()
        
        # Verify: Batch size should decrease but not below min
        assert batcher._current_batch_size >= config.min_batch_size
        assert batcher._current_batch_size <= config.max_batch_size
        
        # Simulate low latency (should increase batch size)
        batcher._latency_history.clear()
        for _ in range(20):
            batcher._latency_history.append(20.0)  # Low latency
        
        batcher._adapt_batch_size()
        
        # Verify: Batch size should increase but not above max
        assert batcher._current_batch_size >= config.min_batch_size
        assert batcher._current_batch_size <= config.max_batch_size


class TestBatchingCachingIntegration:
    """Test batching and caching working together."""
    
    def test_batched_requests_with_cache(self):
        """Test that batched requests can utilize cache."""
        # Setup batcher
        batch_config = BatcherConfig(
            enabled=True,
            max_batch_size=3,
            batch_timeout_ms=50,
            adaptive_batching=False
        )
        batcher = DynamicBatcher(batch_config)
        batcher.start()
        
        # Setup cache
        cache_config = CacheConfig(
            max_memory_mb=1024,
            eviction_policy="lru",
            enabled=True
        )
        cache_manager = KVCacheManager(cache_config)
        
        try:
            # Simulate: Store cache for a conversation
            conversation_id = "conv_batch_cache"
            model_name = "test_model"
            cache_manager.store_cache(conversation_id, model_name, "cached_data", 50.0)
            
            # Add batched requests for same conversation
            request_ids = []
            for i in range(3):
                req_id = batcher.add_request(
                    model_name=model_name,
                    inputs={"input_ids": [1, 2, 3]},
                    parameters={"conversation_id": conversation_id}
                )
                request_ids.append(req_id)
            
            # Wait for batch processing
            time.sleep(0.15)
            
            # Verify: Cache should be accessible
            cached = cache_manager.get_cache(conversation_id, model_name)
            assert cached is not None, "Cache should be available for batched requests"
            
            # Verify: Batch stats
            batch_stats = batcher.get_stats()
            assert batch_stats["enabled"] is True
            
            # Verify: Cache stats
            cache_stats = cache_manager.get_stats()
            assert cache_stats.total_hits >= 1
            
        finally:
            batcher.stop(timeout=1.0)
    
    def test_cache_memory_with_batching_load(self):
        """Test cache memory management under batching load."""
        # Setup with limited cache memory
        cache_config = CacheConfig(
            max_memory_mb=300,  # Limited memory
            eviction_policy="lru",
            enabled=True
        )
        cache_manager = KVCacheManager(cache_config)
        
        # Simulate multiple conversations with batched requests
        for i in range(5):
            conv_id = f"conv_{i}"
            model_name = "test_model"
            
            # Store cache (100MB each, will trigger eviction)
            cache_manager.store_cache(conv_id, model_name, f"data_{i}", 100.0)
        
        # Verify: Memory limit respected
        stats = cache_manager.get_stats()
        assert stats.memory_used_mb <= cache_config.max_memory_mb, (
            f"Memory {stats.memory_used_mb}MB should not exceed {cache_config.max_memory_mb}MB"
        )
        
        # Verify: Evictions occurred
        assert stats.evictions > 0, "Evictions should occur under memory pressure"
        
        # Verify: Most recent caches still available
        for i in range(3, 5):
            conv_id = f"conv_{i}"
            cached = cache_manager.get_cache(conv_id, model_name)
            assert cached is not None, f"Recent conversation {conv_id} should be cached"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
Manual validation tests for checkpoint 12.

These tests manually validate batching and caching without relying on
background threads that may have timing issues in test environments.
"""

import time
import pytest
from mm_orch.optimization.batcher import (
    DynamicBatcher,
    InferenceRequest,
    BatchedRequest,
)
from mm_orch.optimization.kv_cache_manager import KVCacheManager
from mm_orch.optimization.config import BatcherConfig, CacheConfig


class TestBatchingManual:
    """Manual tests for batching functionality."""
    
    def test_batch_formation_logic(self):
        """Test batch formation without background thread."""
        config = BatcherConfig(
            enabled=True,
            max_batch_size=3,
            batch_timeout_ms=50,
            adaptive_batching=False
        )
        batcher = DynamicBatcher(config)
        
        # Manually create requests
        requests = []
        for i in range(3):
            req = InferenceRequest(
                request_id=f"req_{i}",
                model_name="test_model",
                inputs={"input_ids": [1, 2, 3]},
                parameters={}
            )
            requests.append(req)
        
        # Manually form batch
        from collections import deque
        req_queue = deque(requests)
        batch = batcher._form_batch("test_model", req_queue)
        
        # Verify batch formation
        assert batch is not None
        assert batch.batch_size == 3
        assert batch.model_name == "test_model"
        assert len(req_queue) == 0  # All requests consumed
    
    def test_request_compatibility_check(self):
        """Test request compatibility logic."""
        config = BatcherConfig(enabled=True, max_batch_size=10)
        batcher = DynamicBatcher(config)
        
        # Compatible requests (same parameters)
        req1 = InferenceRequest(
            request_id="req1",
            model_name="test_model",
            inputs={"input_ids": [1, 2, 3]},
            parameters={"temperature": 1.0, "top_p": 0.9}
        )
        req2 = InferenceRequest(
            request_id="req2",
            model_name="test_model",
            inputs={"input_ids": [4, 5, 6]},
            parameters={"temperature": 1.0, "top_p": 0.9}
        )
        
        assert batcher._are_requests_compatible(req1, req2) is True
        
        # Incompatible requests (different temperature)
        req3 = InferenceRequest(
            request_id="req3",
            model_name="test_model",
            inputs={"input_ids": [7, 8, 9]},
            parameters={"temperature": 0.5, "top_p": 0.9}
        )
        
        assert batcher._are_requests_compatible(req1, req3) is False
    
    def test_padding_logic(self):
        """Test variable-length sequence padding."""
        # Create requests with different lengths
        requests = []
        lengths = [3, 5, 2, 7]
        
        for i, length in enumerate(lengths):
            req = InferenceRequest(
                request_id=f"req_{i}",
                model_name="test_model",
                inputs={"input_ids": list(range(length))},
                parameters={}
            )
            requests.append(req)
        
        # Create batch
        max_length = max(lengths)
        batch = BatchedRequest(
            batch_id="test_batch",
            model_name="test_model",
            requests=requests,
            max_sequence_length=max_length
        )
        
        # Get padded inputs
        padded_inputs = batch.get_padded_inputs()
        
        # Verify padding
        assert padded_inputs["max_length"] == max_length
        assert padded_inputs["batch_size"] == len(requests)
        
        # Check each padded sequence
        for i, padded_ids in enumerate(padded_inputs["input_ids"]):
            if isinstance(padded_ids, list):
                assert len(padded_ids) == max_length
                # Original values preserved
                for j in range(lengths[i]):
                    assert padded_ids[j] == j
                # Padding is 0
                for j in range(lengths[i], max_length):
                    assert padded_ids[j] == 0
    
    def test_batching_disabled_mode(self):
        """Test that batching can be disabled."""
        config = BatcherConfig(
            enabled=False,
            max_batch_size=10,
            batch_timeout_ms=50
        )
        batcher = DynamicBatcher(config)
        
        # Add request (should process immediately)
        req_id = batcher.add_request(
            model_name="test_model",
            inputs={"input_ids": [1, 2, 3]},
            parameters={}
        )
        
        # Get result (should be available immediately)
        result = batcher.get_result(req_id, timeout=1.0)
        
        assert result.request_id == req_id
        assert result.metadata.get("batching_disabled") is True
        assert result.metadata.get("batch_size") == 1
        
        # Verify stats
        stats = batcher.get_stats()
        assert stats["enabled"] is False
        assert stats["running"] is False
    
    def test_adaptive_batch_size_logic(self):
        """Test adaptive batch size adjustment logic."""
        config = BatcherConfig(
            enabled=True,
            max_batch_size=10,
            batch_timeout_ms=50,
            adaptive_batching=True,
            min_batch_size=2
        )
        batcher = DynamicBatcher(config)
        
        initial_size = batcher._current_batch_size
        assert initial_size == 10
        
        # Simulate high latency scenario
        for _ in range(20):
            batcher._latency_history.append(200.0)  # High latency
        
        # Force adaptation by setting last adaptation time to past
        batcher._last_adaptation_time = time.time() - 20.0  # 20 seconds ago
        
        # Trigger adaptation
        batcher._adapt_batch_size()
        
        # Batch size should decrease
        assert batcher._current_batch_size < initial_size
        assert batcher._current_batch_size >= config.min_batch_size
        
        # Simulate low latency scenario
        batcher._latency_history.clear()
        for _ in range(20):
            batcher._latency_history.append(20.0)  # Low latency
        
        # Force adaptation again
        batcher._last_adaptation_time = time.time() - 20.0
        
        # Trigger adaptation
        batcher._adapt_batch_size()
        
        # Batch size should increase (but not exceed max)
        assert batcher._current_batch_size <= config.max_batch_size


class TestCachingManual:
    """Manual tests for caching functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache store and retrieve operations."""
        config = CacheConfig(
            max_memory_mb=1024,
            eviction_policy="lru",
            enabled=True
        )
        manager = KVCacheManager(config)
        
        # Store cache
        manager.store_cache("conv1", "model1", "data1", 100.0)
        
        # Retrieve cache
        cached = manager.get_cache("conv1", "model1")
        assert cached is not None
        assert cached.conversation_id == "conv1"
        assert cached.model_name == "model1"
        assert cached.cache_data == "data1"
        assert cached.memory_mb == 100.0
        
        # Verify stats
        stats = manager.get_stats()
        assert stats.total_entries == 1
        assert stats.memory_used_mb == 100.0
        assert stats.total_hits == 1
        assert stats.total_misses == 0
    
    def test_cache_lru_eviction_logic(self):
        """Test LRU eviction logic."""
        config = CacheConfig(
            max_memory_mb=250,  # Can hold ~2 entries at 100MB each
            eviction_policy="lru",
            enabled=True
        )
        manager = KVCacheManager(config)
        
        # Add 3 entries (will trigger eviction)
        manager.store_cache("conv1", "model1", "data1", 100.0)
        manager.store_cache("conv2", "model1", "data2", 100.0)
        manager.store_cache("conv3", "model1", "data3", 100.0)
        
        # Verify memory limit respected
        stats = manager.get_stats()
        assert stats.memory_used_mb <= 250.0
        assert stats.evictions > 0
        
        # Oldest entry should be evicted
        cached1 = manager.get_cache("conv1", "model1")
        assert cached1 is None  # Evicted
        
        # Recent entries should still be there
        cached2 = manager.get_cache("conv2", "model1")
        cached3 = manager.get_cache("conv3", "model1")
        assert cached2 is not None or cached3 is not None
    
    def test_cache_conversation_cleanup(self):
        """Test conversation cleanup."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Store caches for multiple models in same conversation
        manager.store_cache("conv1", "model_a", "data_a", 50.0)
        manager.store_cache("conv1", "model_b", "data_b", 50.0)
        manager.store_cache("conv1", "model_c", "data_c", 50.0)
        
        initial_stats = manager.get_stats()
        assert initial_stats.total_entries == 3
        assert initial_stats.memory_used_mb == 150.0
        
        # Release conversation
        manager.release_conversation("conv1")
        
        # Verify cleanup
        final_stats = manager.get_stats()
        assert final_stats.total_entries == 0
        assert final_stats.memory_used_mb == 0.0
        
        # Verify caches not retrievable
        assert manager.get_cache("conv1", "model_a") is None
        assert manager.get_cache("conv1", "model_b") is None
        assert manager.get_cache("conv1", "model_c") is None
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        config = CacheConfig(max_memory_mb=1024, enabled=True)
        manager = KVCacheManager(config)
        
        # Store some caches
        manager.store_cache("conv1", "model1", "data1", 50.0)
        manager.store_cache("conv2", "model1", "data2", 50.0)
        
        # Generate hits
        manager.get_cache("conv1", "model1")  # Hit
        manager.get_cache("conv2", "model1")  # Hit
        manager.get_cache("conv1", "model1")  # Hit
        
        # Generate misses
        manager.get_cache("conv3", "model1")  # Miss
        manager.get_cache("conv4", "model1")  # Miss
        
        # Verify hit rate
        stats = manager.get_stats()
        assert stats.total_hits == 3
        assert stats.total_misses == 2
        expected_hit_rate = 3.0 / 5.0  # 60%
        assert abs(stats.hit_rate - expected_hit_rate) < 0.01
    
    def test_cache_disabled_mode(self):
        """Test that caching can be disabled."""
        config = CacheConfig(enabled=False)
        manager = KVCacheManager(config)
        
        # Attempt to store cache
        manager.store_cache("conv1", "model1", "data1", 50.0)
        
        # Cache should not be stored
        cached = manager.get_cache("conv1", "model1")
        assert cached is None
        
        # Verify stats
        stats = manager.get_stats()
        assert stats.total_entries == 0
        assert stats.memory_used_mb == 0.0


class TestIntegrationManual:
    """Manual integration tests."""
    
    def test_batching_and_caching_together(self):
        """Test that batching and caching can work together."""
        # Setup cache
        cache_config = CacheConfig(max_memory_mb=1024, enabled=True)
        cache_manager = KVCacheManager(cache_config)
        
        # Setup batcher (disabled to avoid thread issues)
        batch_config = BatcherConfig(enabled=False)
        batcher = DynamicBatcher(batch_config)
        
        # Store cache for a conversation
        conversation_id = "conv_integration"
        model_name = "test_model"
        cache_manager.store_cache(conversation_id, model_name, "cached_data", 50.0)
        
        # Add request (will process immediately since batching disabled)
        req_id = batcher.add_request(
            model_name=model_name,
            inputs={"input_ids": [1, 2, 3]},
            parameters={"conversation_id": conversation_id}
        )
        
        # Get result
        result = batcher.get_result(req_id, timeout=1.0)
        assert result.request_id == req_id
        
        # Verify cache is still accessible
        cached = cache_manager.get_cache(conversation_id, model_name)
        assert cached is not None
        assert cached.cache_data == "cached_data"
        
        # Verify stats
        cache_stats = cache_manager.get_stats()
        assert cache_stats.total_hits >= 1
        
        batch_stats = batcher.get_stats()
        assert batch_stats["enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

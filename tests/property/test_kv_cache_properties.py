"""
Property-based tests for KV Cache Manager.

Tests universal properties of cache behavior across all valid inputs.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from mm_orch.optimization.kv_cache_manager import (
    KVCacheManager,
    CacheConfig,
    KVCache
)


# Strategies for generating test data
conversation_ids = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    min_codepoint=ord('a'),
    max_codepoint=ord('z')
))

model_names = st.sampled_from([
    "qwen-chat",
    "gpt2",
    "t5-small",
    "distilgpt2",
    "llama-7b"
])

cache_data = st.text(min_size=1, max_size=100)  # Simplified cache data
memory_sizes = st.floats(min_value=1.0, max_value=500.0)


# Feature: advanced-optimization-monitoring, Property 10: Cache is enabled for compatible models
@given(
    conversation_id=conversation_ids,
    model_name=model_names,
    cache_data_val=cache_data,
    memory_mb=memory_sizes
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_cache_enabled_for_compatible_models(
    conversation_id,
    model_name,
    cache_data_val,
    memory_mb
):
    """
    Property 10: Cache is enabled for compatible models.
    
    For any model that supports KV caching, the system should enable
    and utilize KV cache for inference operations.
    
    Validates: Requirements 7.1
    """
    # Setup: Create cache manager with caching enabled
    config = CacheConfig(
        max_memory_mb=4096,
        eviction_policy="lru",
        enabled=True
    )
    manager = KVCacheManager(config)
    
    # Execute: Store cache
    manager.store_cache(conversation_id, model_name, cache_data_val, memory_mb)
    
    # Verify: Cache should be stored and retrievable
    cached = manager.get_cache(conversation_id, model_name)
    assert cached is not None, "Cache should be enabled and store data"
    assert cached.conversation_id == conversation_id
    assert cached.model_name == model_name
    assert cached.cache_data == cache_data_val
    assert cached.memory_mb == memory_mb


# Feature: advanced-optimization-monitoring, Property 11: Multi-turn conversations reuse cache
@given(
    conversation_id=conversation_ids,
    model_name=model_names,
    cache_data_val=cache_data,
    memory_mb=memory_sizes,
    num_accesses=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_multi_turn_conversations_reuse_cache(
    conversation_id,
    model_name,
    cache_data_val,
    memory_mb,
    num_accesses
):
    """
    Property 11: Multi-turn conversations reuse cache.
    
    For any multi-turn conversation, subsequent turns should reuse
    cached key-value pairs from previous turns, resulting in cache hits.
    
    Validates: Requirements 7.2
    """
    # Setup: Create cache manager and store initial cache
    config = CacheConfig(max_memory_mb=4096, enabled=True)
    manager = KVCacheManager(config)
    
    manager.store_cache(conversation_id, model_name, cache_data_val, memory_mb)
    
    # Execute: Access cache multiple times (simulating multi-turn conversation)
    initial_stats = manager.get_stats()
    initial_hits = initial_stats.total_hits
    
    for _ in range(num_accesses):
        cached = manager.get_cache(conversation_id, model_name)
        assert cached is not None, "Cache should be available for reuse"
    
    # Verify: All accesses should be cache hits
    final_stats = manager.get_stats()
    new_hits = final_stats.total_hits - initial_hits
    
    assert new_hits == num_accesses, (
        f"All {num_accesses} accesses should be cache hits, got {new_hits}"
    )
    assert cached.access_count >= num_accesses, (
        "Access count should track multiple accesses"
    )


# Feature: advanced-optimization-monitoring, Property 12: LRU eviction occurs on memory overflow
@given(
    num_entries=st.integers(min_value=3, max_value=10),
    entry_size_mb=st.floats(min_value=100.0, max_value=200.0)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_lru_eviction_on_memory_overflow(num_entries, entry_size_mb):
    """
    Property 12: LRU eviction occurs on memory overflow.
    
    For any cache state where adding new entries would exceed memory limits,
    the least recently used entries should be evicted to make space.
    
    Validates: Requirements 7.3
    """
    # Setup: Create cache manager with limited memory
    max_memory = entry_size_mb * 2.5  # Can hold ~2-3 entries
    config = CacheConfig(max_memory_mb=max_memory, enabled=True)
    manager = KVCacheManager(config)
    
    # Execute: Add entries until memory is exceeded
    conversation_ids_list = [f"conv_{i}" for i in range(num_entries)]
    
    for i, conv_id in enumerate(conversation_ids_list):
        manager.store_cache(conv_id, "test-model", f"data_{i}", entry_size_mb)
    
    # Verify: Memory usage should not exceed limit
    stats = manager.get_stats()
    assert stats.memory_used_mb <= max_memory, (
        f"Memory usage {stats.memory_used_mb}MB should not exceed limit {max_memory}MB"
    )
    
    # Verify: Some evictions should have occurred
    if num_entries * entry_size_mb > max_memory:
        assert stats.evictions > 0, "Evictions should occur when memory limit exceeded"
    
    # Verify: Oldest entries should be evicted (LRU)
    # The first entries added should be evicted first
    entries = manager.get_cache_entries()
    entry_ids = [e.conversation_id for e in entries]
    
    # Later entries should still be in cache
    for conv_id in conversation_ids_list[-2:]:
        assert conv_id in entry_ids, f"Recent entry {conv_id} should still be in cache"


# Feature: advanced-optimization-monitoring, Property 13: Cache is released on conversation end
@given(
    conversation_id=conversation_ids,
    num_models=st.integers(min_value=1, max_value=5),
    memory_mb=memory_sizes
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_cache_released_on_conversation_end(
    conversation_id,
    num_models,
    memory_mb
):
    """
    Property 13: Cache is released on conversation end.
    
    For any conversation that ends, the associated cache memory should be
    released and available for other conversations.
    
    Validates: Requirements 7.4
    """
    # Setup: Create cache manager and store caches for multiple models
    config = CacheConfig(max_memory_mb=4096, enabled=True)
    manager = KVCacheManager(config)
    
    model_names_list = [f"model_{i}" for i in range(num_models)]
    
    for model_name in model_names_list:
        manager.store_cache(conversation_id, model_name, f"data_{model_name}", memory_mb)
    
    # Record initial state
    initial_stats = manager.get_stats()
    initial_entries = initial_stats.total_entries
    initial_memory = initial_stats.memory_used_mb
    
    # Execute: Release conversation
    manager.release_conversation(conversation_id)
    
    # Verify: All cache entries for conversation should be removed
    final_stats = manager.get_stats()
    
    assert final_stats.total_entries == initial_entries - num_models, (
        f"Should remove {num_models} entries, "
        f"had {initial_entries}, now {final_stats.total_entries}"
    )
    
    # Verify: Memory should be freed
    expected_freed = memory_mb * num_models
    actual_freed = initial_memory - final_stats.memory_used_mb
    
    # Allow small floating point tolerance
    assert abs(actual_freed - expected_freed) < 0.01, (
        f"Should free {expected_freed}MB, actually freed {actual_freed}MB"
    )
    
    # Verify: Cache entries should not be retrievable
    for model_name in model_names_list:
        cached = manager.get_cache(conversation_id, model_name)
        assert cached is None, (
            f"Cache for {model_name} should be released"
        )


# Feature: advanced-optimization-monitoring, Property 14: Cache hit rates are tracked
@given(
    num_stores=st.integers(min_value=1, max_value=10),
    num_hits=st.integers(min_value=0, max_value=10),
    num_misses=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_cache_hit_rates_tracked(num_stores, num_hits, num_misses):
    """
    Property 14: Cache hit rates are tracked.
    
    For any cache operations, hit rate statistics should be collected
    and available for monitoring queries.
    
    Validates: Requirements 7.5
    """
    assume(num_hits + num_misses > 0)  # Need at least one access
    
    # Setup: Create cache manager
    config = CacheConfig(max_memory_mb=4096, enabled=True)
    manager = KVCacheManager(config)
    
    # Execute: Store some caches
    for i in range(num_stores):
        manager.store_cache(f"conv_{i}", "test-model", f"data_{i}", 10.0)
    
    # Execute: Generate cache hits
    for i in range(min(num_hits, num_stores)):
        manager.get_cache(f"conv_{i}", "test-model")
    
    # Execute: Generate cache misses
    for i in range(num_misses):
        manager.get_cache(f"nonexistent_{i}", "test-model")
    
    # Verify: Statistics should be tracked
    stats = manager.get_stats()
    
    # Verify hit/miss counts
    expected_hits = min(num_hits, num_stores)
    assert stats.total_hits == expected_hits, (
        f"Should track {expected_hits} hits, got {stats.total_hits}"
    )
    assert stats.total_misses == num_misses, (
        f"Should track {num_misses} misses, got {stats.total_misses}"
    )
    
    # Verify hit rate calculation
    total_accesses = expected_hits + num_misses
    expected_hit_rate = expected_hits / total_accesses if total_accesses > 0 else 0.0
    
    assert abs(stats.hit_rate - expected_hit_rate) < 0.001, (
        f"Hit rate should be {expected_hit_rate:.3f}, got {stats.hit_rate:.3f}"
    )
    
    # Verify other stats are present
    assert stats.total_entries >= 0
    assert stats.memory_used_mb >= 0.0
    assert stats.evictions >= 0


# Additional property: Cache disabled mode
@given(
    conversation_id=conversation_ids,
    model_name=model_names,
    cache_data_val=cache_data,
    memory_mb=memory_sizes
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_cache_disabled_mode(
    conversation_id,
    model_name,
    cache_data_val,
    memory_mb
):
    """
    Property: When cache is disabled, no caching should occur.
    
    For any cache operation when caching is disabled, the operation
    should be a no-op and no cache should be stored or retrieved.
    """
    # Setup: Create cache manager with caching disabled
    config = CacheConfig(enabled=False)
    manager = KVCacheManager(config)
    
    # Execute: Attempt to store cache
    manager.store_cache(conversation_id, model_name, cache_data_val, memory_mb)
    
    # Verify: Cache should not be stored
    cached = manager.get_cache(conversation_id, model_name)
    assert cached is None, "Cache should not be stored when disabled"
    
    # Verify: No memory should be used
    stats = manager.get_stats()
    assert stats.memory_used_mb == 0.0, "No memory should be used when cache disabled"
    assert stats.total_entries == 0, "No entries should exist when cache disabled"

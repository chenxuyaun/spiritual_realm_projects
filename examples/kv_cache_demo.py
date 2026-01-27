"""
KV Cache Manager demonstration.

This example shows how to use the KV Cache Manager for optimizing
multi-turn conversation inference with transformer models.
"""

from mm_orch.optimization import KVCacheManager, CacheConfig


def main():
    """Demonstrate KV Cache Manager usage."""
    
    print("=" * 60)
    print("KV Cache Manager Demo")
    print("=" * 60)
    
    # 1. Initialize cache manager
    print("\n1. Initializing KV Cache Manager...")
    config = CacheConfig(
        max_memory_mb=1024,  # 1GB cache limit
        eviction_policy="lru",
        enabled=True
    )
    cache_manager = KVCacheManager(config)
    print(f"   Cache initialized with {config.max_memory_mb}MB limit")
    
    # 2. Simulate multi-turn conversation
    print("\n2. Simulating multi-turn conversation...")
    conversation_id = "user_123_session_456"
    model_name = "qwen-chat"
    
    # First turn - cache miss, store cache
    print(f"\n   Turn 1: Processing first message...")
    cached = cache_manager.get_cache(conversation_id, model_name)
    if cached is None:
        print("   - Cache miss (expected for first turn)")
        # Simulate storing KV cache after inference
        cache_data = {"keys": "...", "values": "..."}  # Simplified
        cache_manager.store_cache(conversation_id, model_name, cache_data, 150.0)
        print("   - Stored 150MB cache for future turns")
    
    # Second turn - cache hit, reuse cache
    print(f"\n   Turn 2: Processing second message...")
    cached = cache_manager.get_cache(conversation_id, model_name)
    if cached is not None:
        print(f"   - Cache hit! Reusing cached KV pairs")
        print(f"   - Access count: {cached.access_count}")
        print(f"   - Memory: {cached.memory_mb}MB")
    
    # Third turn - another cache hit
    print(f"\n   Turn 3: Processing third message...")
    cached = cache_manager.get_cache(conversation_id, model_name)
    if cached is not None:
        print(f"   - Cache hit! Reusing cached KV pairs")
        print(f"   - Access count: {cached.access_count}")
    
    # 3. Check statistics
    print("\n3. Cache Statistics:")
    stats = cache_manager.get_stats()
    print(f"   - Total entries: {stats.total_entries}")
    print(f"   - Memory used: {stats.memory_used_mb:.2f}MB")
    print(f"   - Hit rate: {stats.hit_rate:.2%}")
    print(f"   - Total hits: {stats.total_hits}")
    print(f"   - Total misses: {stats.total_misses}")
    print(f"   - Evictions: {stats.evictions}")
    
    # 4. Demonstrate LRU eviction
    print("\n4. Demonstrating LRU eviction...")
    print("   Adding multiple conversations to trigger eviction...")
    
    for i in range(5):
        conv_id = f"conversation_{i}"
        cache_manager.store_cache(conv_id, model_name, f"data_{i}", 200.0)
        print(f"   - Stored cache for {conv_id} (200MB)")
    
    stats = cache_manager.get_stats()
    print(f"\n   After adding 5 conversations:")
    print(f"   - Total entries: {stats.total_entries}")
    print(f"   - Memory used: {stats.memory_used_mb:.2f}MB / {config.max_memory_mb}MB")
    print(f"   - Evictions: {stats.evictions}")
    
    # 5. Release conversation cache
    print("\n5. Releasing conversation cache...")
    cache_manager.release_conversation(conversation_id)
    print(f"   Released all caches for {conversation_id}")
    
    stats = cache_manager.get_stats()
    print(f"   - Total entries: {stats.total_entries}")
    print(f"   - Memory used: {stats.memory_used_mb:.2f}MB")
    
    # 6. Multiple models per conversation
    print("\n6. Multiple models per conversation...")
    multi_conv_id = "multi_model_conversation"
    
    for model in ["qwen-chat", "gpt2", "t5-small"]:
        cache_manager.store_cache(multi_conv_id, model, f"data_{model}", 100.0)
        print(f"   - Stored cache for {model}")
    
    stats = cache_manager.get_stats()
    print(f"\n   Total entries: {stats.total_entries}")
    
    # Release all caches for this conversation
    cache_manager.release_conversation(multi_conv_id)
    print(f"   Released all {3} model caches for {multi_conv_id}")
    
    # 7. Final statistics
    print("\n7. Final Statistics:")
    stats = cache_manager.get_stats()
    print(f"   - Total entries: {stats.total_entries}")
    print(f"   - Memory used: {stats.memory_used_mb:.2f}MB")
    print(f"   - Hit rate: {stats.hit_rate:.2%}")
    print(f"   - Total hits: {stats.total_hits}")
    print(f"   - Total misses: {stats.total_misses}")
    print(f"   - Total evictions: {stats.evictions}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

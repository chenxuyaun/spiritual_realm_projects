"""
KV Cache Manager for transformer models.

This module implements a Key-Value cache manager with LRU eviction policy
for optimizing multi-turn conversation inference in transformer models.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime
from collections import OrderedDict
import logging

from mm_orch.optimization.config import CacheConfig

logger = logging.getLogger(__name__)


@dataclass
class KVCache:
    """Key-Value cache for a specific conversation and model."""

    conversation_id: str
    model_name: str
    cache_data: Any  # Actual cache tensors/data
    memory_mb: float
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0


@dataclass
class CacheStats:
    """Statistics for KV cache."""

    total_entries: int
    memory_used_mb: float
    hit_rate: float
    evictions: int
    total_hits: int = 0
    total_misses: int = 0


class KVCacheManager:
    """
    Manages key-value cache for transformer models with LRU eviction.

    Features:
    - Per-conversation cache isolation
    - LRU eviction when memory limit exceeded
    - Memory tracking and limits
    - Cache hit rate tracking
    - Automatic cleanup on conversation end
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize cache manager with configuration.

        Args:
            config: Cache configuration with memory limits and policies
        """
        self.config = config

        # Use OrderedDict for LRU tracking (maintains insertion order)
        self._cache: OrderedDict[str, KVCache] = OrderedDict()

        # Statistics tracking
        self._total_hits = 0
        self._total_misses = 0
        self._total_evictions = 0
        self._memory_used_mb = 0.0

        logger.info(
            f"KVCacheManager initialized with max_memory={config.max_memory_mb}MB, "
            f"eviction_policy={config.eviction_policy}, enabled={config.enabled}"
        )

    def _make_cache_key(self, conversation_id: str, model_name: str) -> str:
        """Create unique cache key from conversation and model."""
        return f"{conversation_id}:{model_name}"

    def get_cache(self, conversation_id: str, model_name: str) -> Optional[KVCache]:
        """
        Retrieve cached KV pairs for a conversation.

        Args:
            conversation_id: Unique conversation identifier
            model_name: Name of the model

        Returns:
            KVCache if found, None otherwise
        """
        if not self.config.enabled:
            return None

        cache_key = self._make_cache_key(conversation_id, model_name)

        if cache_key in self._cache:
            # Cache hit - update access time and move to end (most recent)
            cache_entry = self._cache[cache_key]
            cache_entry.last_accessed = datetime.now()
            cache_entry.access_count += 1

            # Move to end for LRU tracking
            self._cache.move_to_end(cache_key)

            self._total_hits += 1

            logger.debug(
                f"Cache hit for conversation={conversation_id}, model={model_name}, "
                f"access_count={cache_entry.access_count}"
            )

            return cache_entry
        else:
            # Cache miss
            self._total_misses += 1

            logger.debug(f"Cache miss for conversation={conversation_id}, model={model_name}")

            return None

    def store_cache(self, conversation_id: str, model_name: str, cache_data: Any, memory_mb: float):
        """
        Store KV cache for future use.

        Args:
            conversation_id: Unique conversation identifier
            model_name: Name of the model
            cache_data: Cache data to store (tensors, etc.)
            memory_mb: Memory size of cache in MB
        """
        if not self.config.enabled:
            return

        cache_key = self._make_cache_key(conversation_id, model_name)

        # Check if we need to evict to make space
        required_memory = memory_mb
        if cache_key in self._cache:
            # Updating existing cache - account for old memory
            old_cache = self._cache[cache_key]
            required_memory = memory_mb - old_cache.memory_mb

        # Evict if necessary
        if self._memory_used_mb + required_memory > self.config.max_memory_mb:
            self.evict_lru(required_memory)

        # Create or update cache entry
        cache_entry = KVCache(
            conversation_id=conversation_id,
            model_name=model_name,
            cache_data=cache_data,
            memory_mb=memory_mb,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
        )

        # Update memory tracking
        if cache_key in self._cache:
            old_memory = self._cache[cache_key].memory_mb
            self._memory_used_mb -= old_memory

        self._cache[cache_key] = cache_entry
        self._memory_used_mb += memory_mb

        # Move to end (most recent)
        self._cache.move_to_end(cache_key)

        logger.debug(
            f"Stored cache for conversation={conversation_id}, model={model_name}, "
            f"memory={memory_mb:.2f}MB, total_memory={self._memory_used_mb:.2f}MB"
        )

    def evict_lru(self, required_memory: float):
        """
        Evict least recently used caches to free memory.

        Args:
            required_memory: Amount of memory needed in MB
        """
        if not self._cache:
            logger.warning("Cannot evict from empty cache")
            return

        freed_memory = 0.0
        evicted_keys = []

        # Evict from front (least recently used) until we have enough space
        while (
            self._cache
            and self._memory_used_mb + required_memory > self.config.max_memory_mb
            and freed_memory < required_memory
        ):
            # Get least recently used (first item)
            cache_key, cache_entry = self._cache.popitem(last=False)

            freed_memory += cache_entry.memory_mb
            self._memory_used_mb -= cache_entry.memory_mb
            self._total_evictions += 1
            evicted_keys.append(cache_key)

            logger.debug(f"Evicted cache for {cache_key}, freed={cache_entry.memory_mb:.2f}MB")

        if evicted_keys:
            logger.info(
                f"LRU eviction: freed {freed_memory:.2f}MB by evicting {len(evicted_keys)} entries"
            )

    def release_conversation(self, conversation_id: str):
        """
        Release all cache entries for a conversation.

        Args:
            conversation_id: Conversation identifier to release
        """
        keys_to_remove = [
            key for key in self._cache.keys() if key.startswith(f"{conversation_id}:")
        ]

        freed_memory = 0.0
        for key in keys_to_remove:
            cache_entry = self._cache.pop(key)
            freed_memory += cache_entry.memory_mb
            self._memory_used_mb -= cache_entry.memory_mb

        if keys_to_remove:
            logger.info(
                f"Released {len(keys_to_remove)} cache entries for conversation={conversation_id}, "
                f"freed={freed_memory:.2f}MB"
            )

    def get_stats(self) -> CacheStats:
        """
        Return cache hit rate and memory usage statistics.

        Returns:
            CacheStats with current statistics
        """
        total_requests = self._total_hits + self._total_misses
        hit_rate = self._total_hits / total_requests if total_requests > 0 else 0.0

        return CacheStats(
            total_entries=len(self._cache),
            memory_used_mb=self._memory_used_mb,
            hit_rate=hit_rate,
            evictions=self._total_evictions,
            total_hits=self._total_hits,
            total_misses=self._total_misses,
        )

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._memory_used_mb = 0.0
        logger.info("Cache cleared")

    def get_cache_entries(self) -> List[KVCache]:
        """
        Get all cache entries (for testing/debugging).

        Returns:
            List of all cache entries
        """
        return list(self._cache.values())

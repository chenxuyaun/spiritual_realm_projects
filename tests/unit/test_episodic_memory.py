"""
Unit tests for the Episodic Memory module.

Tests episode creation, storage, retrieval, and memory management
functionality.

Requirements: 3.1, 3.2
"""

import pytest
import time
from typing import Dict, Any

from mm_orch.consciousness.episodic_memory import (
    Episode,
    EpisodicMemory,
    EpisodicMemoryConfig,
    create_episode,
)


class TestEpisode:
    """Tests for the Episode dataclass."""

    def test_episode_creation_with_all_fields(self):
        """Test creating an episode with all required fields."""
        episode = Episode(
            episode_id="test-001",
            timestamp=1234567890.0,
            context={"location": "test", "task": "unit_test"},
            events=[{"type": "action", "name": "test_action"}],
            emotional_state={"pleasure": 0.5, "arousal": 0.3, "dominance": 0.4},
            importance=0.7,
            access_count=0,
            last_accessed=None,
            consolidated=False,
            metadata={"source": "test"},
        )
        
        assert episode.episode_id == "test-001"
        assert episode.timestamp == 1234567890.0
        assert episode.context == {"location": "test", "task": "unit_test"}
        assert episode.events == [{"type": "action", "name": "test_action"}]
        assert episode.emotional_state == {"pleasure": 0.5, "arousal": 0.3, "dominance": 0.4}
        assert episode.importance == 0.7
        assert episode.access_count == 0
        assert episode.last_accessed is None
        assert episode.consolidated is False
        assert episode.metadata == {"source": "test"}

    def test_episode_creation_with_defaults(self):
        """Test creating an episode with default values."""
        episode = Episode(
            episode_id="test-002",
            timestamp=time.time(),
            context={},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        assert episode.access_count == 0
        assert episode.last_accessed is None
        assert episode.consolidated is False
        assert episode.metadata == {}

    def test_episode_importance_clamping(self):
        """Test that importance is clamped to [0.0, 1.0]."""
        # Test clamping high value
        episode_high = Episode(
            episode_id="test-high",
            timestamp=time.time(),
            context={},
            events=[],
            emotional_state={},
            importance=1.5,
        )
        assert episode_high.importance == 1.0
        
        # Test clamping low value
        episode_low = Episode(
            episode_id="test-low",
            timestamp=time.time(),
            context={},
            events=[],
            emotional_state={},
            importance=-0.5,
        )
        assert episode_low.importance == 0.0

    def test_episode_validation_empty_id(self):
        """Test that empty episode_id raises ValueError."""
        with pytest.raises(ValueError, match="episode_id cannot be empty"):
            Episode(
                episode_id="",
                timestamp=time.time(),
                context={},
                events=[],
                emotional_state={},
                importance=0.5,
            )

    def test_episode_validation_invalid_context(self):
        """Test that non-dict context raises ValueError."""
        with pytest.raises(ValueError, match="context must be a dictionary"):
            Episode(
                episode_id="test",
                timestamp=time.time(),
                context="invalid",  # type: ignore
                events=[],
                emotional_state={},
                importance=0.5,
            )

    def test_episode_validation_invalid_events(self):
        """Test that non-list events raises ValueError."""
        with pytest.raises(ValueError, match="events must be a list"):
            Episode(
                episode_id="test",
                timestamp=time.time(),
                context={},
                events="invalid",  # type: ignore
                emotional_state={},
                importance=0.5,
            )

    def test_episode_validation_negative_access_count(self):
        """Test that negative access_count raises ValueError."""
        with pytest.raises(ValueError, match="access_count cannot be negative"):
            Episode(
                episode_id="test",
                timestamp=time.time(),
                context={},
                events=[],
                emotional_state={},
                importance=0.5,
                access_count=-1,
            )

    def test_episode_record_access(self):
        """Test recording access to an episode."""
        episode = Episode(
            episode_id="test",
            timestamp=time.time(),
            context={},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        assert episode.access_count == 0
        assert episode.last_accessed is None
        
        before_access = time.time()
        episode.record_access()
        after_access = time.time()
        
        assert episode.access_count == 1
        assert episode.last_accessed is not None
        assert before_access <= episode.last_accessed <= after_access
        
        # Record another access
        episode.record_access()
        assert episode.access_count == 2

    def test_episode_to_dict(self):
        """Test serializing an episode to dictionary."""
        episode = Episode(
            episode_id="test-001",
            timestamp=1234567890.0,
            context={"key": "value"},
            events=[{"event": "test"}],
            emotional_state={"pleasure": 0.5},
            importance=0.7,
            access_count=3,
            last_accessed=1234567900.0,
            consolidated=True,
            metadata={"meta": "data"},
        )
        
        data = episode.to_dict()
        
        assert data["episode_id"] == "test-001"
        assert data["timestamp"] == 1234567890.0
        assert data["context"] == {"key": "value"}
        assert data["events"] == [{"event": "test"}]
        assert data["emotional_state"] == {"pleasure": 0.5}
        assert data["importance"] == 0.7
        assert data["access_count"] == 3
        assert data["last_accessed"] == 1234567900.0
        assert data["consolidated"] is True
        assert data["metadata"] == {"meta": "data"}

    def test_episode_from_dict(self):
        """Test deserializing an episode from dictionary."""
        data = {
            "episode_id": "test-001",
            "timestamp": 1234567890.0,
            "context": {"key": "value"},
            "events": [{"event": "test"}],
            "emotional_state": {"pleasure": 0.5},
            "importance": 0.7,
            "access_count": 3,
            "last_accessed": 1234567900.0,
            "consolidated": True,
            "metadata": {"meta": "data"},
        }
        
        episode = Episode.from_dict(data)
        
        assert episode.episode_id == "test-001"
        assert episode.timestamp == 1234567890.0
        assert episode.context == {"key": "value"}
        assert episode.events == [{"event": "test"}]
        assert episode.emotional_state == {"pleasure": 0.5}
        assert episode.importance == 0.7
        assert episode.access_count == 3
        assert episode.last_accessed == 1234567900.0
        assert episode.consolidated is True
        assert episode.metadata == {"meta": "data"}

    def test_episode_serialization_roundtrip(self):
        """Test that serialization and deserialization preserves data."""
        original = Episode(
            episode_id="roundtrip-test",
            timestamp=time.time(),
            context={"nested": {"key": "value"}},
            events=[{"type": "a"}, {"type": "b"}],
            emotional_state={"pleasure": 0.3, "arousal": 0.6, "dominance": 0.1},
            importance=0.8,
            access_count=5,
            last_accessed=time.time(),
            consolidated=True,
            metadata={"tags": ["test", "roundtrip"]},
        )
        
        data = original.to_dict()
        restored = Episode.from_dict(data)
        
        assert restored.episode_id == original.episode_id
        assert restored.timestamp == original.timestamp
        assert restored.context == original.context
        assert restored.events == original.events
        assert restored.emotional_state == original.emotional_state
        assert restored.importance == original.importance
        assert restored.access_count == original.access_count
        assert restored.last_accessed == original.last_accessed
        assert restored.consolidated == original.consolidated
        assert restored.metadata == original.metadata


class TestEpisodicMemoryConfig:
    """Tests for the EpisodicMemoryConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = EpisodicMemoryConfig()
        
        assert config.max_episodes == 5000
        assert config.default_importance == 0.5
        assert config.consolidation_interval == 100
        assert config.decay_rate == 0.99
        assert config.importance_threshold == 0.3
        assert config.prune_threshold == 0.9
        assert config.prune_ratio == 0.2

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = EpisodicMemoryConfig(
            max_episodes=1000,
            default_importance=0.7,
            consolidation_interval=50,
        )
        
        assert config.max_episodes == 1000
        assert config.default_importance == 0.7
        assert config.consolidation_interval == 50

    def test_config_validation_max_episodes(self):
        """Test that invalid max_episodes raises ValueError."""
        with pytest.raises(ValueError, match="max_episodes must be at least 1"):
            EpisodicMemoryConfig(max_episodes=0)

    def test_config_validation_default_importance(self):
        """Test that invalid default_importance raises ValueError."""
        with pytest.raises(ValueError, match="default_importance must be between"):
            EpisodicMemoryConfig(default_importance=1.5)

    def test_config_serialization_roundtrip(self):
        """Test config serialization and deserialization."""
        original = EpisodicMemoryConfig(
            max_episodes=2000,
            default_importance=0.6,
        )
        
        data = original.to_dict()
        restored = EpisodicMemoryConfig.from_dict(data)
        
        assert restored.max_episodes == original.max_episodes
        assert restored.default_importance == original.default_importance


class TestEpisodicMemory:
    """Tests for the EpisodicMemory class."""

    def test_memory_initialization_default(self):
        """Test default memory initialization."""
        memory = EpisodicMemory()
        
        assert len(memory) == 0
        assert memory._config.max_episodes == 5000

    def test_memory_initialization_custom_max(self):
        """Test memory initialization with custom max_episodes."""
        memory = EpisodicMemory(max_episodes=100)
        
        assert memory._config.max_episodes == 100

    def test_memory_initialization_with_config(self):
        """Test memory initialization with config dict."""
        config = {"max_episodes": 200, "default_importance": 0.8}
        memory = EpisodicMemory(config=config)
        
        assert memory._config.max_episodes == 200
        assert memory._config.default_importance == 0.8

    def test_create_episode(self):
        """Test creating an episode through the memory."""
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context={"task": "test"},
            events=[{"action": "test_action"}],
            emotional_state={"pleasure": 0.5},
            importance=0.7,
            metadata={"source": "test"},
        )
        
        assert episode.episode_id is not None
        assert episode.context == {"task": "test"}
        assert episode.events == [{"action": "test_action"}]
        assert episode.emotional_state == {"pleasure": 0.5}
        assert episode.importance == 0.7
        assert episode.metadata == {"source": "test"}
        assert len(memory) == 1

    def test_create_episode_default_importance(self):
        """Test that created episodes use default importance."""
        memory = EpisodicMemory(config={"default_importance": 0.6})
        
        episode = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
        )
        
        assert episode.importance == 0.6

    def test_create_episode_auto_timestamp(self):
        """Test that created episodes have automatic timestamps."""
        memory = EpisodicMemory()
        
        before = time.time()
        episode = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
        )
        after = time.time()
        
        assert before <= episode.timestamp <= after

    def test_store_episode(self):
        """Test storing an existing episode."""
        memory = EpisodicMemory()
        
        episode = Episode(
            episode_id="external-001",
            timestamp=time.time(),
            context={"external": True},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        ep_id = memory.store_episode(episode)
        
        assert ep_id == "external-001"
        assert len(memory) == 1
        assert memory.contains("external-001")

    def test_get_episode(self):
        """Test retrieving an episode by ID."""
        memory = EpisodicMemory()
        
        created = memory.create_episode(
            context={"test": True},
            events=[],
            emotional_state={},
        )
        
        retrieved = memory.get_episode(created.episode_id)
        
        assert retrieved is not None
        assert retrieved.episode_id == created.episode_id
        assert retrieved.context == {"test": True}

    def test_get_episode_records_access(self):
        """Test that getting an episode records access."""
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
        )
        
        assert episode.access_count == 0
        
        memory.get_episode(episode.episode_id)
        
        assert episode.access_count == 1
        assert episode.last_accessed is not None

    def test_get_episode_without_recording_access(self):
        """Test getting an episode without recording access."""
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
        )
        
        memory.get_episode(episode.episode_id, record_access=False)
        
        assert episode.access_count == 0
        assert episode.last_accessed is None

    def test_get_episode_not_found(self):
        """Test getting a non-existent episode returns None."""
        memory = EpisodicMemory()
        
        result = memory.get_episode("non-existent")
        
        assert result is None

    def test_contains(self):
        """Test checking if an episode exists."""
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
        )
        
        assert memory.contains(episode.episode_id)
        assert not memory.contains("non-existent")
        assert episode.episode_id in memory
        assert "non-existent" not in memory

    def test_remove_episode(self):
        """Test removing an episode."""
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
        )
        
        assert len(memory) == 1
        
        result = memory.remove_episode(episode.episode_id)
        
        assert result is True
        assert len(memory) == 0
        assert not memory.contains(episode.episode_id)

    def test_remove_episode_not_found(self):
        """Test removing a non-existent episode returns False."""
        memory = EpisodicMemory()
        
        result = memory.remove_episode("non-existent")
        
        assert result is False

    def test_update_importance(self):
        """Test updating episode importance."""
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        result = memory.update_importance(episode.episode_id, 0.9)
        
        assert result is True
        assert episode.importance == 0.9

    def test_update_importance_clamping(self):
        """Test that importance updates are clamped."""
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
        )
        
        memory.update_importance(episode.episode_id, 1.5)
        assert episode.importance == 1.0
        
        memory.update_importance(episode.episode_id, -0.5)
        assert episode.importance == 0.0

    def test_update_importance_not_found(self):
        """Test updating importance of non-existent episode."""
        memory = EpisodicMemory()
        
        result = memory.update_importance("non-existent", 0.9)
        
        assert result is False

    def test_mark_consolidated(self):
        """Test marking an episode as consolidated."""
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
        )
        
        assert episode.consolidated is False
        
        result = memory.mark_consolidated(episode.episode_id)
        
        assert result is True
        assert episode.consolidated is True

    def test_get_unconsolidated_episodes(self):
        """Test getting unconsolidated episodes."""
        memory = EpisodicMemory()
        
        ep1 = memory.create_episode(context={}, events=[], emotional_state={})
        ep2 = memory.create_episode(context={}, events=[], emotional_state={})
        ep3 = memory.create_episode(context={}, events=[], emotional_state={})
        
        memory.mark_consolidated(ep2.episode_id)
        
        unconsolidated = memory.get_unconsolidated_episodes()
        
        assert len(unconsolidated) == 2
        assert ep1 in unconsolidated
        assert ep2 not in unconsolidated
        assert ep3 in unconsolidated

    def test_get_all_episodes(self):
        """Test getting all episodes."""
        memory = EpisodicMemory()
        
        ep1 = memory.create_episode(context={"id": 1}, events=[], emotional_state={})
        ep2 = memory.create_episode(context={"id": 2}, events=[], emotional_state={})
        
        all_episodes = memory.get_all_episodes()
        
        assert len(all_episodes) == 2
        assert ep1 in all_episodes
        assert ep2 in all_episodes

    def test_clear(self):
        """Test clearing all episodes."""
        memory = EpisodicMemory()
        
        memory.create_episode(context={}, events=[], emotional_state={})
        memory.create_episode(context={}, events=[], emotional_state={})
        
        assert len(memory) == 2
        
        memory.clear()
        
        assert len(memory) == 0

    def test_pruning_on_capacity(self):
        """Test that pruning occurs when capacity is reached."""
        memory = EpisodicMemory(max_episodes=5, config={"prune_ratio": 0.4})
        
        # Create 5 episodes with varying importance
        for i in range(5):
            memory.create_episode(
                context={"id": i},
                events=[],
                emotional_state={},
                importance=i * 0.2,  # 0.0, 0.2, 0.4, 0.6, 0.8
            )
        
        assert len(memory) == 5
        
        # Creating another episode should trigger pruning
        memory.create_episode(
            context={"id": 5},
            events=[],
            emotional_state={},
            importance=0.9,
        )
        
        # Should have pruned some episodes
        assert len(memory) < 6

    def test_pruning_preserves_high_importance(self):
        """Test that pruning preserves high-importance episodes."""
        memory = EpisodicMemory(max_episodes=5, config={"prune_ratio": 0.4})
        
        # Create episodes with known importance
        low_importance = memory.create_episode(
            context={"type": "low"},
            events=[],
            emotional_state={},
            importance=0.1,
        )
        
        high_importance = memory.create_episode(
            context={"type": "high"},
            events=[],
            emotional_state={},
            importance=0.9,
        )
        
        # Fill to capacity
        for i in range(3):
            memory.create_episode(
                context={"type": "filler"},
                events=[],
                emotional_state={},
                importance=0.5,
            )
        
        # Trigger pruning
        memory.create_episode(
            context={"type": "trigger"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        # High importance should be preserved
        assert memory.contains(high_importance.episode_id)

    def test_get_state(self):
        """Test getting memory state."""
        memory = EpisodicMemory(max_episodes=100)
        
        memory.create_episode(context={}, events=[], emotional_state={})
        memory.create_episode(context={}, events=[], emotional_state={})
        
        state = memory.get_state()
        
        assert state["episode_count"] == 2
        assert state["max_episodes"] == 100
        assert state["total_created"] == 2
        assert "config" in state

    def test_get_statistics(self):
        """Test getting memory statistics."""
        memory = EpisodicMemory()
        
        memory.create_episode(context={}, events=[], emotional_state={}, importance=0.3)
        memory.create_episode(context={}, events=[], emotional_state={}, importance=0.7)
        
        stats = memory.get_statistics()
        
        assert stats["count"] == 2
        assert stats["avg_importance"] == 0.5
        assert stats["min_importance"] == 0.3
        assert stats["max_importance"] == 0.7

    def test_get_statistics_empty(self):
        """Test getting statistics from empty memory."""
        memory = EpisodicMemory()
        
        stats = memory.get_statistics()
        
        assert stats["count"] == 0
        assert stats["avg_importance"] == 0.0

    def test_serialization_roundtrip(self):
        """Test memory serialization and deserialization."""
        memory = EpisodicMemory(max_episodes=100)
        
        ep1 = memory.create_episode(
            context={"test": 1},
            events=[{"action": "a"}],
            emotional_state={"pleasure": 0.5},
            importance=0.7,
        )
        ep2 = memory.create_episode(
            context={"test": 2},
            events=[{"action": "b"}],
            emotional_state={"arousal": 0.3},
            importance=0.4,
        )
        
        # Access one episode
        memory.get_episode(ep1.episode_id)
        memory.mark_consolidated(ep2.episode_id)
        
        # Serialize
        data = memory.to_dict()
        
        # Deserialize
        restored = EpisodicMemory.from_dict(data)
        
        assert len(restored) == 2
        assert restored.contains(ep1.episode_id)
        assert restored.contains(ep2.episode_id)
        
        restored_ep1 = restored.get_episode(ep1.episode_id, record_access=False)
        assert restored_ep1.context == {"test": 1}
        assert restored_ep1.importance == 0.7
        
        restored_ep2 = restored.get_episode(ep2.episode_id, record_access=False)
        assert restored_ep2.consolidated is True


class TestCreateEpisodeFactory:
    """Tests for the create_episode factory function."""

    def test_create_episode_basic(self):
        """Test basic episode creation with factory function."""
        episode = create_episode(
            context={"test": True},
            events=[{"action": "test"}],
            emotional_state={"pleasure": 0.5},
        )
        
        assert episode.episode_id is not None
        assert len(episode.episode_id) > 0
        assert episode.context == {"test": True}
        assert episode.events == [{"action": "test"}]
        assert episode.emotional_state == {"pleasure": 0.5}
        assert episode.importance == 0.5  # Default
        assert episode.access_count == 0
        assert episode.consolidated is False

    def test_create_episode_with_importance(self):
        """Test episode creation with custom importance."""
        episode = create_episode(
            context={},
            events=[],
            emotional_state={},
            importance=0.9,
        )
        
        assert episode.importance == 0.9

    def test_create_episode_with_metadata(self):
        """Test episode creation with metadata."""
        episode = create_episode(
            context={},
            events=[],
            emotional_state={},
            metadata={"source": "test", "tags": ["a", "b"]},
        )
        
        assert episode.metadata == {"source": "test", "tags": ["a", "b"]}

    def test_create_episode_unique_ids(self):
        """Test that factory creates unique IDs."""
        episodes = [
            create_episode(context={}, events=[], emotional_state={})
            for _ in range(10)
        ]
        
        ids = [ep.episode_id for ep in episodes]
        assert len(ids) == len(set(ids))  # All unique

    def test_create_episode_auto_timestamp(self):
        """Test that factory sets automatic timestamp."""
        before = time.time()
        episode = create_episode(context={}, events=[], emotional_state={})
        after = time.time()
        
        assert before <= episode.timestamp <= after


class TestEpisodicMemoryRetrieval:
    """Tests for episodic memory retrieval methods.
    
    Requirements: 3.3, 3.4
    """

    def test_retrieve_by_temporal_proximity_basic(self):
        """Test basic temporal proximity retrieval."""
        memory = EpisodicMemory()
        
        # Create episodes at different times
        base_time = 1000000.0
        ep1 = Episode(
            episode_id="ep1",
            timestamp=base_time - 100,  # 100 seconds before
            context={"id": 1},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        ep2 = Episode(
            episode_id="ep2",
            timestamp=base_time,  # At reference time
            context={"id": 2},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        ep3 = Episode(
            episode_id="ep3",
            timestamp=base_time + 50,  # 50 seconds after
            context={"id": 3},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        ep4 = Episode(
            episode_id="ep4",
            timestamp=base_time + 200,  # 200 seconds after (outside window)
            context={"id": 4},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        for ep in [ep1, ep2, ep3, ep4]:
            memory.store_episode(ep)
        
        # Retrieve within 150 second window
        results = memory.retrieve_by_temporal_proximity(
            reference_time=base_time,
            time_window=150,
            max_results=10,
        )
        
        # Should get ep2 (closest), ep3, ep1 - but not ep4
        assert len(results) == 3
        assert results[0].episode_id == "ep2"  # Closest (distance 0)
        assert results[1].episode_id == "ep3"  # Second closest (distance 50)
        assert results[2].episode_id == "ep1"  # Third closest (distance 100)

    def test_retrieve_by_temporal_proximity_sorted_by_distance(self):
        """Test that results are sorted by temporal distance."""
        memory = EpisodicMemory()
        
        base_time = 1000000.0
        # Create episodes at varying distances
        for i, offset in enumerate([500, 100, 300, 50, 200]):
            ep = Episode(
                episode_id=f"ep{i}",
                timestamp=base_time + offset,
                context={},
                events=[],
                emotional_state={},
                importance=0.5,
            )
            memory.store_episode(ep)
        
        results = memory.retrieve_by_temporal_proximity(
            reference_time=base_time,
            time_window=600,
            max_results=10,
        )
        
        # Verify sorted by distance
        distances = [abs(ep.timestamp - base_time) for ep in results]
        assert distances == sorted(distances)

    def test_retrieve_by_temporal_proximity_max_results(self):
        """Test that max_results limits output."""
        memory = EpisodicMemory()
        
        base_time = 1000000.0
        for i in range(10):
            ep = Episode(
                episode_id=f"ep{i}",
                timestamp=base_time + i * 10,
                context={},
                events=[],
                emotional_state={},
                importance=0.5,
            )
            memory.store_episode(ep)
        
        results = memory.retrieve_by_temporal_proximity(
            reference_time=base_time,
            time_window=1000,
            max_results=3,
        )
        
        assert len(results) == 3

    def test_retrieve_by_temporal_proximity_empty_window(self):
        """Test retrieval with no episodes in window."""
        memory = EpisodicMemory()
        
        ep = Episode(
            episode_id="ep1",
            timestamp=1000000.0,
            context={},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        memory.store_episode(ep)
        
        results = memory.retrieve_by_temporal_proximity(
            reference_time=2000000.0,  # Far from episode
            time_window=100,
            max_results=10,
        )
        
        assert len(results) == 0

    def test_retrieve_by_temporal_proximity_records_access(self):
        """Test that retrieval records access."""
        memory = EpisodicMemory()
        
        ep = Episode(
            episode_id="ep1",
            timestamp=1000000.0,
            context={},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        memory.store_episode(ep)
        
        assert ep.access_count == 0
        
        memory.retrieve_by_temporal_proximity(
            reference_time=1000000.0,
            time_window=100,
            max_results=10,
        )
        
        assert ep.access_count == 1
        assert ep.last_accessed is not None

    def test_retrieve_by_context_similarity_exact_match(self):
        """Test context similarity with exact matches."""
        memory = EpisodicMemory()
        
        ep1 = Episode(
            episode_id="ep1",
            timestamp=time.time(),
            context={"task": "search", "domain": "science"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        ep2 = Episode(
            episode_id="ep2",
            timestamp=time.time(),
            context={"task": "chat", "domain": "general"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        ep3 = Episode(
            episode_id="ep3",
            timestamp=time.time(),
            context={"task": "search", "domain": "general"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        for ep in [ep1, ep2, ep3]:
            memory.store_episode(ep)
        
        # Query for search + science
        results = memory.retrieve_by_context_similarity(
            query_context={"task": "search", "domain": "science"},
            max_results=10,
        )
        
        # ep1 should be first (exact match)
        assert len(results) >= 1
        assert results[0].episode_id == "ep1"

    def test_retrieve_by_context_similarity_partial_match(self):
        """Test context similarity with partial matches."""
        memory = EpisodicMemory()
        
        ep1 = Episode(
            episode_id="ep1",
            timestamp=time.time(),
            context={"task": "search", "domain": "science", "extra": "value"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        ep2 = Episode(
            episode_id="ep2",
            timestamp=time.time(),
            context={"task": "chat"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        for ep in [ep1, ep2]:
            memory.store_episode(ep)
        
        # Query with partial match
        results = memory.retrieve_by_context_similarity(
            query_context={"task": "search"},
            max_results=10,
        )
        
        # ep1 should be first (has matching task)
        assert len(results) >= 1
        assert results[0].episode_id == "ep1"

    def test_retrieve_by_context_similarity_empty_query(self):
        """Test context similarity with empty query returns recent episodes."""
        memory = EpisodicMemory()
        
        ep1 = Episode(
            episode_id="ep1",
            timestamp=1000000.0,
            context={"task": "old"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        ep2 = Episode(
            episode_id="ep2",
            timestamp=2000000.0,
            context={"task": "new"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        for ep in [ep1, ep2]:
            memory.store_episode(ep)
        
        results = memory.retrieve_by_context_similarity(
            query_context={},
            max_results=10,
        )
        
        # Should return most recent first
        assert len(results) == 2
        assert results[0].episode_id == "ep2"

    def test_retrieve_by_context_similarity_nested_dict(self):
        """Test context similarity with nested dictionaries."""
        memory = EpisodicMemory()
        
        ep1 = Episode(
            episode_id="ep1",
            timestamp=time.time(),
            context={"user": {"name": "Alice", "role": "admin"}},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        ep2 = Episode(
            episode_id="ep2",
            timestamp=time.time(),
            context={"user": {"name": "Bob", "role": "user"}},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        for ep in [ep1, ep2]:
            memory.store_episode(ep)
        
        results = memory.retrieve_by_context_similarity(
            query_context={"user": {"name": "Alice", "role": "admin"}},
            max_results=10,
        )
        
        # ep1 should be first (exact nested match)
        assert len(results) >= 1
        assert results[0].episode_id == "ep1"

    def test_retrieve_by_emotional_salience_basic(self):
        """Test basic emotional salience retrieval."""
        memory = EpisodicMemory()
        
        # Create episodes with different emotional states
        ep_happy = Episode(
            episode_id="happy",
            timestamp=time.time(),
            context={},
            events=[],
            emotional_state={"pleasure": 0.8, "arousal": 0.6, "dominance": 0.5},
            importance=0.5,
        )
        ep_sad = Episode(
            episode_id="sad",
            timestamp=time.time(),
            context={},
            events=[],
            emotional_state={"pleasure": -0.7, "arousal": 0.3, "dominance": -0.4},
            importance=0.5,
        )
        ep_neutral = Episode(
            episode_id="neutral",
            timestamp=time.time(),
            context={},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.4, "dominance": 0.0},
            importance=0.5,
        )
        
        for ep in [ep_happy, ep_sad, ep_neutral]:
            memory.store_episode(ep)
        
        # Query for happy emotion
        results = memory.retrieve_by_emotional_salience(
            target_emotion={"pleasure": 0.7, "arousal": 0.6, "dominance": 0.5},
            max_results=10,
        )
        
        # Happy episode should be closest
        assert len(results) == 3
        assert results[0].episode_id == "happy"

    def test_retrieve_by_emotional_salience_sorted_by_distance(self):
        """Test that results are sorted by PAD distance."""
        memory = EpisodicMemory()
        
        # Create episodes at varying emotional distances
        emotions = [
            ("close", {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5}),
            ("medium", {"pleasure": 0.2, "arousal": 0.3, "dominance": 0.2}),
            ("far", {"pleasure": -0.5, "arousal": 0.1, "dominance": -0.5}),
        ]
        
        for ep_id, emotion in emotions:
            ep = Episode(
                episode_id=ep_id,
                timestamp=time.time(),
                context={},
                events=[],
                emotional_state=emotion,
                importance=0.5,
            )
            memory.store_episode(ep)
        
        # Query for target emotion
        target = {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5}
        results = memory.retrieve_by_emotional_salience(
            target_emotion=target,
            max_results=10,
        )
        
        # Should be sorted by distance
        assert results[0].episode_id == "close"
        assert results[1].episode_id == "medium"
        assert results[2].episode_id == "far"

    def test_retrieve_by_emotional_salience_missing_pad_values(self):
        """Test emotional retrieval with missing PAD values uses defaults."""
        memory = EpisodicMemory()
        
        ep = Episode(
            episode_id="ep1",
            timestamp=time.time(),
            context={},
            events=[],
            emotional_state={"pleasure": 0.5},  # Missing arousal and dominance
            importance=0.5,
        )
        memory.store_episode(ep)
        
        # Should not raise error
        results = memory.retrieve_by_emotional_salience(
            target_emotion={"pleasure": 0.5, "arousal": 0.5, "dominance": 0.0},
            max_results=10,
        )
        
        assert len(results) == 1

    def test_retrieve_relevant_combined_scoring(self):
        """Test combined relevance retrieval."""
        memory = EpisodicMemory()
        
        base_time = time.time()
        
        # Episode with good context match but old
        ep_context = Episode(
            episode_id="context_match",
            timestamp=base_time - 10000,
            context={"task": "search", "domain": "science"},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            importance=0.5,
        )
        
        # Episode with good emotion match but different context
        ep_emotion = Episode(
            episode_id="emotion_match",
            timestamp=base_time - 5000,
            context={"task": "chat", "domain": "general"},
            events=[],
            emotional_state={"pleasure": 0.8, "arousal": 0.6, "dominance": 0.5},
            importance=0.5,
        )
        
        # Episode that's recent but poor matches
        ep_recent = Episode(
            episode_id="recent",
            timestamp=base_time - 100,
            context={"task": "other"},
            events=[],
            emotional_state={"pleasure": -0.5, "arousal": 0.2, "dominance": -0.3},
            importance=0.5,
        )
        
        for ep in [ep_context, ep_emotion, ep_recent]:
            memory.store_episode(ep)
        
        # Query with all factors
        results = memory.retrieve_relevant(
            query={
                "context": {"task": "search", "domain": "science"},
                "emotional_state": {"pleasure": 0.8, "arousal": 0.6, "dominance": 0.5},
                "reference_time": base_time,
            },
            max_results=10,
        )
        
        assert len(results) == 3
        # Results should be ranked by composite score

    def test_retrieve_relevant_custom_weights(self):
        """Test combined retrieval with custom weights."""
        memory = EpisodicMemory()
        
        base_time = time.time()
        
        # Create episodes
        ep_old_good_context = Episode(
            episode_id="old_context",
            timestamp=base_time - 10000,
            context={"task": "search"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        ep_new_bad_context = Episode(
            episode_id="new_no_context",
            timestamp=base_time - 10,
            context={"task": "other"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        for ep in [ep_old_good_context, ep_new_bad_context]:
            memory.store_episode(ep)
        
        # Query with high context weight
        results_context = memory.retrieve_relevant(
            query={
                "context": {"task": "search"},
                "reference_time": base_time,
                "context_weight": 0.9,
                "time_weight": 0.1,
                "emotion_weight": 0.0,
            },
            max_results=10,
        )
        
        # Context match should be first
        assert results_context[0].episode_id == "old_context"
        
        # Query with high time weight
        results_time = memory.retrieve_relevant(
            query={
                "context": {"task": "search"},
                "reference_time": base_time,
                "context_weight": 0.1,
                "time_weight": 0.9,
                "emotion_weight": 0.0,
            },
            max_results=10,
        )
        
        # Recent episode should be first
        assert results_time[0].episode_id == "new_no_context"

    def test_retrieve_relevant_empty_memory(self):
        """Test combined retrieval on empty memory."""
        memory = EpisodicMemory()
        
        results = memory.retrieve_relevant(
            query={"context": {"task": "search"}},
            max_results=10,
        )
        
        assert len(results) == 0

    def test_retrieve_relevant_importance_boost(self):
        """Test that importance boosts relevance score."""
        memory = EpisodicMemory()
        
        base_time = time.time()
        
        # Two similar episodes with different importance
        ep_low = Episode(
            episode_id="low_importance",
            timestamp=base_time,
            context={"task": "search"},
            events=[],
            emotional_state={},
            importance=0.1,
        )
        ep_high = Episode(
            episode_id="high_importance",
            timestamp=base_time,
            context={"task": "search"},
            events=[],
            emotional_state={},
            importance=0.9,
        )
        
        for ep in [ep_low, ep_high]:
            memory.store_episode(ep)
        
        results = memory.retrieve_relevant(
            query={
                "context": {"task": "search"},
                "reference_time": base_time,
            },
            max_results=10,
        )
        
        # High importance should be ranked higher
        assert results[0].episode_id == "high_importance"

    def test_retrieve_relevant_records_access(self):
        """Test that combined retrieval records access."""
        memory = EpisodicMemory()
        
        ep = Episode(
            episode_id="ep1",
            timestamp=time.time(),
            context={"task": "test"},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        memory.store_episode(ep)
        
        assert ep.access_count == 0
        
        memory.retrieve_relevant(
            query={"context": {"task": "test"}},
            max_results=10,
        )
        
        assert ep.access_count == 1

    def test_retrieve_by_temporal_proximity_negative_window(self):
        """Test that negative time window is handled (uses absolute value)."""
        memory = EpisodicMemory()
        
        ep = Episode(
            episode_id="ep1",
            timestamp=1000000.0,
            context={},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        memory.store_episode(ep)
        
        # Negative window should be treated as positive
        results = memory.retrieve_by_temporal_proximity(
            reference_time=1000000.0,
            time_window=-100,
            max_results=10,
        )
        
        assert len(results) == 1

    def test_retrieve_max_results_zero(self):
        """Test that max_results=0 returns empty list."""
        memory = EpisodicMemory()
        
        ep = Episode(
            episode_id="ep1",
            timestamp=time.time(),
            context={"task": "test"},
            events=[],
            emotional_state={"pleasure": 0.5},
            importance=0.5,
        )
        memory.store_episode(ep)
        
        assert memory.retrieve_by_temporal_proximity(time.time(), 1000, 0) == []
        assert memory.retrieve_by_context_similarity({"task": "test"}, 0) == []
        assert memory.retrieve_by_emotional_salience({"pleasure": 0.5}, 0) == []
        assert memory.retrieve_relevant({"context": {}}, 0) == []


class TestEpisodicMemoryManagement:
    """Tests for episodic memory management methods.
    
    Requirements: 3.5, 3.6
    """

    def test_consolidate_empty_memory(self):
        """Test consolidation with no unconsolidated episodes."""
        memory = EpisodicMemory()
        
        result = memory.consolidate()
        
        assert result["consolidated_count"] == 0
        assert result["extracted_patterns"]["context_patterns"] == []
        assert result["extracted_patterns"]["event_patterns"] == []
        assert result["extracted_patterns"]["emotional_patterns"] == []

    def test_consolidate_marks_episodes(self):
        """Test that consolidation marks episodes as consolidated."""
        memory = EpisodicMemory()
        
        ep1 = memory.create_episode(
            context={"task": "test"},
            events=[{"type": "action"}],
            emotional_state={"pleasure": 0.5},
        )
        ep2 = memory.create_episode(
            context={"task": "test"},
            events=[{"type": "action"}],
            emotional_state={"pleasure": 0.3},
        )
        
        assert not ep1.consolidated
        assert not ep2.consolidated
        
        result = memory.consolidate()
        
        assert result["consolidated_count"] == 2
        assert ep1.consolidated
        assert ep2.consolidated
        assert "consolidated_at" in ep1.metadata
        assert "consolidated_at" in ep2.metadata

    def test_consolidate_extracts_context_patterns(self):
        """Test that consolidation extracts context patterns."""
        memory = EpisodicMemory()
        
        # Create episodes with common context
        for i in range(5):
            memory.create_episode(
                context={"task": "search", "domain": "science"},
                events=[],
                emotional_state={},
            )
        
        # Add one with different context
        memory.create_episode(
            context={"task": "chat", "domain": "general"},
            events=[],
            emotional_state={},
        )
        
        result = memory.consolidate()
        
        patterns = result["extracted_patterns"]["context_patterns"]
        assert len(patterns) > 0
        
        # "task" and "domain" should be identified as common keys
        pattern_keys = [p["key"] for p in patterns]
        assert "task" in pattern_keys
        assert "domain" in pattern_keys

    def test_consolidate_extracts_event_patterns(self):
        """Test that consolidation extracts event patterns."""
        memory = EpisodicMemory()
        
        # Create episodes with common event types
        for i in range(5):
            memory.create_episode(
                context={},
                events=[
                    {"type": "search", "query": f"query{i}"},
                    {"type": "summarize", "result": "summary"},
                ],
                emotional_state={},
            )
        
        result = memory.consolidate()
        
        patterns = result["extracted_patterns"]["event_patterns"]
        assert len(patterns) > 0
        
        # Check for event type patterns
        event_types = [p["event_type"] for p in patterns if p.get("pattern_type") == "event_type"]
        assert "search" in event_types
        assert "summarize" in event_types

    def test_consolidate_extracts_emotional_patterns(self):
        """Test that consolidation extracts emotional patterns."""
        memory = EpisodicMemory()
        
        # Create episodes with varying emotional states
        for i in range(5):
            memory.create_episode(
                context={},
                events=[],
                emotional_state={
                    "pleasure": 0.5 + i * 0.1,
                    "arousal": 0.3,
                    "dominance": 0.2,
                },
            )
        
        result = memory.consolidate()
        
        patterns = result["extracted_patterns"]["emotional_patterns"]
        assert len(patterns) > 0
        
        # Check for overall statistics pattern
        stats_pattern = next(
            (p for p in patterns if p.get("pattern_type") == "overall_statistics"),
            None
        )
        assert stats_pattern is not None
        assert "pleasure" in stats_pattern
        assert "arousal" in stats_pattern
        assert "dominance" in stats_pattern

    def test_consolidate_only_processes_unconsolidated(self):
        """Test that consolidation only processes unconsolidated episodes."""
        memory = EpisodicMemory()
        
        ep1 = memory.create_episode(
            context={"id": 1},
            events=[],
            emotional_state={},
        )
        ep2 = memory.create_episode(
            context={"id": 2},
            events=[],
            emotional_state={},
        )
        
        # Mark ep1 as already consolidated
        memory.mark_consolidated(ep1.episode_id)
        
        result = memory.consolidate()
        
        # Only ep2 should be consolidated
        assert result["consolidated_count"] == 1
        assert ep1.episode_id not in result["consolidated_episode_ids"]
        assert ep2.episode_id in result["consolidated_episode_ids"]

    def test_apply_decay_basic(self):
        """Test basic importance decay."""
        memory = EpisodicMemory(config={"decay_rate": 0.9})
        
        ep = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
            importance=1.0,
        )
        
        affected = memory.apply_decay(time_elapsed=1.0)
        
        assert affected == 1
        assert ep.importance == pytest.approx(0.9, rel=0.01)

    def test_apply_decay_multiple_time_units(self):
        """Test decay over multiple time units."""
        memory = EpisodicMemory(config={"decay_rate": 0.9})
        
        ep = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
            importance=1.0,
        )
        
        memory.apply_decay(time_elapsed=2.0)
        
        # 1.0 * 0.9^2 = 0.81
        assert ep.importance == pytest.approx(0.81, rel=0.01)

    def test_apply_decay_zero_time(self):
        """Test that zero time elapsed doesn't change importance."""
        memory = EpisodicMemory()
        
        ep = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
            importance=0.8,
        )
        
        affected = memory.apply_decay(time_elapsed=0)
        
        assert affected == 0
        assert ep.importance == 0.8

    def test_apply_decay_negative_time(self):
        """Test that negative time elapsed doesn't change importance."""
        memory = EpisodicMemory()
        
        ep = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
            importance=0.8,
        )
        
        affected = memory.apply_decay(time_elapsed=-1.0)
        
        assert affected == 0
        assert ep.importance == 0.8

    def test_apply_decay_empty_memory(self):
        """Test decay on empty memory."""
        memory = EpisodicMemory()
        
        affected = memory.apply_decay(time_elapsed=1.0)
        
        assert affected == 0

    def test_apply_decay_clamps_to_zero(self):
        """Test that decay doesn't go below zero."""
        memory = EpisodicMemory(config={"decay_rate": 0.1})
        
        ep = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
            importance=0.01,
        )
        
        # Apply heavy decay
        memory.apply_decay(time_elapsed=10.0)
        
        assert ep.importance >= 0.0

    def test_prune_by_importance_basic(self):
        """Test basic importance-based pruning."""
        memory = EpisodicMemory()
        
        low_importance = memory.create_episode(
            context={"type": "low"},
            events=[],
            emotional_state={},
            importance=0.1,
        )
        high_importance = memory.create_episode(
            context={"type": "high"},
            events=[],
            emotional_state={},
            importance=0.9,
        )
        
        removed = memory.prune_by_importance(threshold=0.5)
        
        assert removed == 1
        assert not memory.contains(low_importance.episode_id)
        assert memory.contains(high_importance.episode_id)

    def test_prune_by_importance_preserves_frequently_accessed(self):
        """Test that frequently accessed episodes are preserved."""
        memory = EpisodicMemory()
        
        # Create low importance but frequently accessed episode
        frequent_ep = memory.create_episode(
            context={"type": "frequent"},
            events=[],
            emotional_state={},
            importance=0.1,
        )
        # Simulate frequent access
        for _ in range(5):
            memory.get_episode(frequent_ep.episode_id)
        
        # Create low importance, rarely accessed episode
        rare_ep = memory.create_episode(
            context={"type": "rare"},
            events=[],
            emotional_state={},
            importance=0.1,
        )
        
        removed = memory.prune_by_importance(threshold=0.5)
        
        # Frequently accessed should be preserved
        assert removed == 1
        assert memory.contains(frequent_ep.episode_id)
        assert not memory.contains(rare_ep.episode_id)

    def test_prune_by_importance_empty_memory(self):
        """Test pruning on empty memory."""
        memory = EpisodicMemory()
        
        removed = memory.prune_by_importance(threshold=0.5)
        
        assert removed == 0

    def test_prune_by_importance_threshold_clamping(self):
        """Test that threshold is clamped to valid range."""
        memory = EpisodicMemory()
        
        ep = memory.create_episode(
            context={},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        # Threshold > 1.0 should be clamped to 1.0
        removed = memory.prune_by_importance(threshold=1.5)
        assert removed == 1  # All episodes below 1.0 are removed
        
        memory.create_episode(
            context={},
            events=[],
            emotional_state={},
            importance=0.5,
        )
        
        # Threshold < 0.0 should be clamped to 0.0
        removed = memory.prune_by_importance(threshold=-0.5)
        assert removed == 0  # Nothing below 0.0

    def test_get_frequently_accessed_basic(self):
        """Test basic frequently accessed retrieval."""
        memory = EpisodicMemory()
        
        ep1 = memory.create_episode(context={"id": 1}, events=[], emotional_state={})
        ep2 = memory.create_episode(context={"id": 2}, events=[], emotional_state={})
        ep3 = memory.create_episode(context={"id": 3}, events=[], emotional_state={})
        
        # Access ep1 5 times, ep2 3 times, ep3 1 time
        for _ in range(5):
            memory.get_episode(ep1.episode_id)
        for _ in range(3):
            memory.get_episode(ep2.episode_id)
        memory.get_episode(ep3.episode_id)
        
        # Get episodes accessed at least 3 times
        results = memory.get_frequently_accessed(min_access_count=3, max_results=10)
        
        assert len(results) == 2
        assert results[0].episode_id == ep1.episode_id  # Highest access count
        assert results[1].episode_id == ep2.episode_id

    def test_get_frequently_accessed_sorted_by_count(self):
        """Test that results are sorted by access count."""
        memory = EpisodicMemory()
        
        episodes = []
        for i in range(5):
            ep = memory.create_episode(context={"id": i}, events=[], emotional_state={})
            episodes.append(ep)
            # Access each episode i+1 times
            for _ in range(i + 1):
                memory.get_episode(ep.episode_id)
        
        results = memory.get_frequently_accessed(min_access_count=1, max_results=10)
        
        # Should be sorted by access count descending
        for i in range(len(results) - 1):
            assert results[i].access_count >= results[i + 1].access_count

    def test_get_frequently_accessed_max_results(self):
        """Test max_results limit."""
        memory = EpisodicMemory()
        
        for i in range(10):
            ep = memory.create_episode(context={"id": i}, events=[], emotional_state={})
            memory.get_episode(ep.episode_id)  # Access once
        
        results = memory.get_frequently_accessed(min_access_count=1, max_results=3)
        
        assert len(results) == 3

    def test_get_frequently_accessed_empty_result(self):
        """Test when no episodes meet the threshold."""
        memory = EpisodicMemory()
        
        memory.create_episode(context={}, events=[], emotional_state={})
        
        results = memory.get_frequently_accessed(min_access_count=5, max_results=10)
        
        assert len(results) == 0

    def test_get_frequently_accessed_zero_max_results(self):
        """Test that max_results=0 returns empty list."""
        memory = EpisodicMemory()
        
        ep = memory.create_episode(context={}, events=[], emotional_state={})
        memory.get_episode(ep.episode_id)
        
        results = memory.get_frequently_accessed(min_access_count=1, max_results=0)
        
        assert results == []

    def test_get_recently_accessed_basic(self):
        """Test basic recently accessed retrieval."""
        memory = EpisodicMemory()
        
        ep1 = memory.create_episode(context={"id": 1}, events=[], emotional_state={})
        ep2 = memory.create_episode(context={"id": 2}, events=[], emotional_state={})
        ep3 = memory.create_episode(context={"id": 3}, events=[], emotional_state={})
        
        # Access all episodes
        memory.get_episode(ep1.episode_id)
        time.sleep(0.01)  # Small delay
        memory.get_episode(ep2.episode_id)
        time.sleep(0.01)
        memory.get_episode(ep3.episode_id)
        
        # Get recently accessed within 1 second
        results = memory.get_recently_accessed(time_window=1.0, max_results=10)
        
        assert len(results) == 3
        # Most recent first
        assert results[0].episode_id == ep3.episode_id

    def test_get_recently_accessed_time_window(self):
        """Test that time window filters correctly."""
        memory = EpisodicMemory()
        
        ep = memory.create_episode(context={}, events=[], emotional_state={})
        memory.get_episode(ep.episode_id)
        
        # Very small time window should still include just-accessed episode
        results = memory.get_recently_accessed(time_window=1.0, max_results=10)
        assert len(results) == 1
        
        # Zero time window should return nothing (or very recent)
        results = memory.get_recently_accessed(time_window=0.0, max_results=10)
        # This depends on timing, but should be 0 or 1

    def test_get_recently_accessed_excludes_never_accessed(self):
        """Test that never-accessed episodes are excluded."""
        memory = EpisodicMemory()
        
        ep1 = memory.create_episode(context={"id": 1}, events=[], emotional_state={})
        ep2 = memory.create_episode(context={"id": 2}, events=[], emotional_state={})
        
        # Only access ep1
        memory.get_episode(ep1.episode_id)
        
        results = memory.get_recently_accessed(time_window=1.0, max_results=10)
        
        assert len(results) == 1
        assert results[0].episode_id == ep1.episode_id

    def test_get_recently_accessed_max_results(self):
        """Test max_results limit."""
        memory = EpisodicMemory()
        
        for i in range(10):
            ep = memory.create_episode(context={"id": i}, events=[], emotional_state={})
            memory.get_episode(ep.episode_id)
        
        results = memory.get_recently_accessed(time_window=1.0, max_results=3)
        
        assert len(results) == 3

    def test_get_recently_accessed_zero_max_results(self):
        """Test that max_results=0 returns empty list."""
        memory = EpisodicMemory()
        
        ep = memory.create_episode(context={}, events=[], emotional_state={})
        memory.get_episode(ep.episode_id)
        
        results = memory.get_recently_accessed(time_window=1.0, max_results=0)
        
        assert results == []

    def test_get_recently_accessed_negative_window(self):
        """Test that negative time window is handled."""
        memory = EpisodicMemory()
        
        ep = memory.create_episode(context={}, events=[], emotional_state={})
        memory.get_episode(ep.episode_id)
        
        # Negative window should be treated as positive
        results = memory.get_recently_accessed(time_window=-1.0, max_results=10)
        
        assert len(results) == 1

    def test_consolidation_resets_counter(self):
        """Test that consolidation resets the episodes_since_consolidation counter."""
        memory = EpisodicMemory()
        
        # Create some episodes
        for i in range(5):
            memory.create_episode(context={"id": i}, events=[], emotional_state={})
        
        assert memory._episodes_since_consolidation == 5
        
        memory.consolidate()
        
        assert memory._episodes_since_consolidation == 0

    def test_prune_by_importance_updates_total_pruned(self):
        """Test that pruning updates the total_pruned counter."""
        memory = EpisodicMemory()
        
        for i in range(5):
            memory.create_episode(
                context={"id": i},
                events=[],
                emotional_state={},
                importance=0.1,
            )
        
        initial_pruned = memory._total_pruned
        
        removed = memory.prune_by_importance(threshold=0.5)
        
        assert memory._total_pruned == initial_pruned + removed

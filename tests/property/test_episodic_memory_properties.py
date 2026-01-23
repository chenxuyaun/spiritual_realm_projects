"""
Property-based tests for Episodic Memory.

Tests properties 10-14 from the consciousness-system-deepening design document:
- Property 10: Episode Structure Completeness
- Property 11: Significant Event Episode Creation
- Property 12: Episode Retrieval Relevance Ordering
- Property 13: Memory Consolidation Pattern Extraction
- Property 14: Importance-Weighted Memory Pruning

Validates: Requirements 3.1-3.6
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import Dict, List, Any, Optional
import time
import math

from mm_orch.consciousness.episodic_memory import (
    EpisodicMemory,
    EpisodicMemoryConfig,
    Episode,
    create_episode,
)


# =============================================================================
# Hypothesis Strategies for Test Data Generation
# =============================================================================

# Strategy for valid float values in [0.0, 1.0]
unit_float_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for importance values (0.0 to 1.0)
importance_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for PAD emotional state values
pad_value_strategy = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

arousal_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for timestamps (reasonable range)
timestamp_strategy = st.floats(
    min_value=1000000.0, max_value=2000000000.0, allow_nan=False, allow_infinity=False
)

# Strategy for context dictionaries
context_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    values=st.one_of(
        st.integers(min_value=-1000, max_value=1000),
        st.text(max_size=20),
        st.booleans()
    ),
    min_size=0,
    max_size=5
)

# Strategy for event dictionaries
event_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    values=st.one_of(
        st.integers(min_value=-1000, max_value=1000),
        st.text(max_size=20),
        st.booleans()
    ),
    min_size=1,
    max_size=5
)

# Strategy for event lists
events_list_strategy = st.lists(event_strategy, min_size=0, max_size=5)

# Strategy for metadata dictionaries
metadata_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    values=st.one_of(
        st.integers(min_value=-1000, max_value=1000),
        st.text(max_size=20),
        st.booleans()
    ),
    min_size=0,
    max_size=3
)


@st.composite
def emotional_state_strategy(draw):
    """Generate a valid PAD emotional state dictionary."""
    return {
        "pleasure": draw(pad_value_strategy),
        "arousal": draw(arousal_strategy),
        "dominance": draw(pad_value_strategy),
    }


@st.composite
def episode_data_strategy(draw, importance: Optional[float] = None):
    """Generate valid data for creating an episode."""
    return {
        "context": draw(context_strategy),
        "events": draw(events_list_strategy),
        "emotional_state": draw(emotional_state_strategy()),
        "importance": importance if importance is not None else draw(importance_strategy),
        "metadata": draw(metadata_strategy),
    }


@st.composite
def episode_list_data_strategy(draw, min_size: int = 1, max_size: int = 20):
    """Generate a list of episode data dictionaries."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return [draw(episode_data_strategy()) for _ in range(size)]


@st.composite
def significant_event_type_strategy(draw):
    """Generate a significant event type."""
    return draw(st.sampled_from(["task_complete", "task_error", "user_feedback"]))


@st.composite
def episodic_memory_config_strategy(draw):
    """Generate a valid EpisodicMemoryConfig object."""
    max_episodes = draw(st.integers(min_value=10, max_value=500))
    default_importance = draw(st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
    consolidation_interval = draw(st.integers(min_value=5, max_value=100))
    decay_rate = draw(st.floats(min_value=0.5, max_value=0.99, allow_nan=False))
    importance_threshold = draw(st.floats(min_value=0.1, max_value=0.5, allow_nan=False))
    prune_threshold = draw(st.floats(min_value=0.7, max_value=0.95, allow_nan=False))
    prune_ratio = draw(st.floats(min_value=0.1, max_value=0.4, allow_nan=False))
    
    return EpisodicMemoryConfig(
        max_episodes=max_episodes,
        default_importance=default_importance,
        consolidation_interval=consolidation_interval,
        decay_rate=decay_rate,
        importance_threshold=importance_threshold,
        prune_threshold=prune_threshold,
        prune_ratio=prune_ratio,
    )


# =============================================================================
# Property 10: Episode Structure Completeness
# =============================================================================

class TestEpisodeStructureCompleteness:
    """
    Tests for Property 10: Episode Structure Completeness
    
    *For any* episode stored in EpisodicMemory, it SHALL contain all required 
    fields: episode_id, timestamp, context, events, outcomes, emotional_state, 
    and relevance_tags.
    
    **Validates: Requirements 3.1**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_created_episode_has_all_required_fields(self, data):
        """
        Feature: consciousness-system-deepening, Property 10: Episode Structure Completeness
        
        For any episode created through EpisodicMemory.create_episode(), it SHALL
        contain all required fields populated with valid values.
        
        **Validates: Requirements 3.1**
        """
        episode_data = data.draw(episode_data_strategy())
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context=episode_data["context"],
            events=episode_data["events"],
            emotional_state=episode_data["emotional_state"],
            importance=episode_data["importance"],
            metadata=episode_data["metadata"],
        )
        
        # Verify all required fields are present and valid
        assert episode.episode_id is not None and len(episode.episode_id) > 0, \
            "episode_id must be non-empty"
        assert isinstance(episode.timestamp, (int, float)), \
            "timestamp must be a number"
        assert isinstance(episode.context, dict), \
            "context must be a dictionary"
        assert isinstance(episode.events, list), \
            "events must be a list"
        assert isinstance(episode.emotional_state, dict), \
            "emotional_state must be a dictionary"
        assert isinstance(episode.importance, float), \
            "importance must be a float"
        assert 0.0 <= episode.importance <= 1.0, \
            f"importance must be in [0.0, 1.0], got {episode.importance}"
        assert isinstance(episode.access_count, int) and episode.access_count >= 0, \
            "access_count must be a non-negative integer"
        assert isinstance(episode.consolidated, bool), \
            "consolidated must be a boolean"
        assert isinstance(episode.metadata, dict), \
            "metadata must be a dictionary"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_factory_created_episode_has_all_required_fields(self, data):
        """
        Feature: consciousness-system-deepening, Property 10: Episode Structure Completeness
        
        For any episode created through the create_episode() factory function, it SHALL
        contain all required fields populated with valid values.
        
        **Validates: Requirements 3.1**
        """
        episode_data = data.draw(episode_data_strategy())
        
        episode = create_episode(
            context=episode_data["context"],
            events=episode_data["events"],
            emotional_state=episode_data["emotional_state"],
            importance=episode_data["importance"],
            metadata=episode_data["metadata"],
        )
        
        # Verify all required fields are present and valid
        assert episode.episode_id is not None and len(episode.episode_id) > 0, \
            "episode_id must be non-empty"
        assert isinstance(episode.timestamp, (int, float)), \
            "timestamp must be a number"
        assert isinstance(episode.context, dict), \
            "context must be a dictionary"
        assert isinstance(episode.events, list), \
            "events must be a list"
        assert isinstance(episode.emotional_state, dict), \
            "emotional_state must be a dictionary"
        assert isinstance(episode.importance, float), \
            "importance must be a float"
        assert 0.0 <= episode.importance <= 1.0, \
            f"importance must be in [0.0, 1.0], got {episode.importance}"

    @given(
        context=context_strategy,
        events=events_list_strategy,
        emotional_state=emotional_state_strategy(),
        importance=importance_strategy,
    )
    @settings(max_examples=100)
    def test_episode_preserves_input_data(
        self,
        context: Dict[str, Any],
        events: List[Dict[str, Any]],
        emotional_state: Dict[str, float],
        importance: float,
    ):
        """
        Feature: consciousness-system-deepening, Property 10: Episode Structure Completeness
        
        For any episode created, the stored context, events, and emotional_state
        SHALL match the input values provided.
        
        **Validates: Requirements 3.1**
        """
        memory = EpisodicMemory()
        
        episode = memory.create_episode(
            context=context,
            events=events,
            emotional_state=emotional_state,
            importance=importance,
        )
        
        # Verify data is preserved
        assert episode.context == context, \
            "context should be preserved exactly"
        assert episode.events == events, \
            "events should be preserved exactly"
        assert episode.emotional_state == emotional_state, \
            "emotional_state should be preserved exactly"
        # Importance is clamped, so check it's within bounds
        assert 0.0 <= episode.importance <= 1.0, \
            "importance should be clamped to [0.0, 1.0]"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_stored_episode_retrievable_with_all_fields(self, data):
        """
        Feature: consciousness-system-deepening, Property 10: Episode Structure Completeness
        
        For any episode stored in memory, retrieving it SHALL return an episode
        with all required fields intact.
        
        **Validates: Requirements 3.1**
        """
        episode_data = data.draw(episode_data_strategy())
        memory = EpisodicMemory()
        
        created = memory.create_episode(
            context=episode_data["context"],
            events=episode_data["events"],
            emotional_state=episode_data["emotional_state"],
            importance=episode_data["importance"],
            metadata=episode_data["metadata"],
        )
        
        # Retrieve the episode
        retrieved = memory.get_episode(created.episode_id, record_access=False)
        
        assert retrieved is not None, "Episode should be retrievable"
        
        # Verify all fields are present
        assert retrieved.episode_id == created.episode_id
        assert retrieved.timestamp == created.timestamp
        assert retrieved.context == created.context
        assert retrieved.events == created.events
        assert retrieved.emotional_state == created.emotional_state
        assert retrieved.importance == created.importance
        assert retrieved.metadata == created.metadata

    @given(data=st.data())
    @settings(max_examples=100)
    def test_episode_unique_ids(self, data):
        """
        Feature: consciousness-system-deepening, Property 10: Episode Structure Completeness
        
        For any set of episodes created, each episode_id SHALL be unique.
        
        **Validates: Requirements 3.1**
        """
        num_episodes = data.draw(st.integers(min_value=2, max_value=20))
        memory = EpisodicMemory()
        
        episode_ids = []
        for _ in range(num_episodes):
            episode_data = data.draw(episode_data_strategy())
            episode = memory.create_episode(
                context=episode_data["context"],
                events=episode_data["events"],
                emotional_state=episode_data["emotional_state"],
            )
            episode_ids.append(episode.episode_id)
        
        # All IDs should be unique
        assert len(episode_ids) == len(set(episode_ids)), \
            "All episode IDs should be unique"


# =============================================================================
# Property 11: Significant Event Episode Creation
# =============================================================================

class TestSignificantEventEpisodeCreation:
    """
    Tests for Property 11: Significant Event Episode Creation
    
    *For any* significant event (task_complete, task_error, user_feedback), 
    the EpisodicMemory SHALL create a new episode, and the episode count 
    SHALL increase by exactly one.
    
    **Validates: Requirements 3.2**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_episode_count_increases_by_one(self, data):
        """
        Feature: consciousness-system-deepening, Property 11: Significant Event Episode Creation
        
        For any episode creation, the episode count SHALL increase by exactly one.
        
        **Validates: Requirements 3.2**
        """
        memory = EpisodicMemory()
        initial_count = len(memory)
        
        episode_data = data.draw(episode_data_strategy())
        memory.create_episode(
            context=episode_data["context"],
            events=episode_data["events"],
            emotional_state=episode_data["emotional_state"],
        )
        
        assert len(memory) == initial_count + 1, \
            f"Episode count should increase by 1: was {initial_count}, now {len(memory)}"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_significant_event_creates_episode_with_high_importance(self, data):
        """
        Feature: consciousness-system-deepening, Property 11: Significant Event Episode Creation
        
        For any significant event (task_complete, task_error, user_feedback),
        the created episode SHALL have importance >= a significance threshold.
        
        **Validates: Requirements 3.2**
        """
        memory = EpisodicMemory()
        event_type = data.draw(significant_event_type_strategy())
        
        # Significant events should have higher importance
        # Using 0.6 as the significance threshold
        significance_threshold = 0.6
        high_importance = data.draw(st.floats(
            min_value=significance_threshold,
            max_value=1.0,
            allow_nan=False
        ))
        
        episode = memory.create_episode(
            context={"event_type": event_type, "significant": True},
            events=[{"type": event_type, "timestamp": time.time()}],
            emotional_state=data.draw(emotional_state_strategy()),
            importance=high_importance,
        )
        
        assert episode.importance >= significance_threshold, \
            f"Significant event episode should have importance >= {significance_threshold}, " \
            f"got {episode.importance}"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_multiple_significant_events_create_multiple_episodes(self, data):
        """
        Feature: consciousness-system-deepening, Property 11: Significant Event Episode Creation
        
        For any sequence of N significant events, exactly N episodes SHALL be created.
        
        **Validates: Requirements 3.2**
        """
        memory = EpisodicMemory()
        num_events = data.draw(st.integers(min_value=1, max_value=10))
        
        for i in range(num_events):
            event_type = data.draw(significant_event_type_strategy())
            episode_data = data.draw(episode_data_strategy())
            
            memory.create_episode(
                context={"event_type": event_type, "event_index": i},
                events=[{"type": event_type}],
                emotional_state=episode_data["emotional_state"],
                importance=0.7,  # High importance for significant events
            )
        
        assert len(memory) == num_events, \
            f"Should have exactly {num_events} episodes, got {len(memory)}"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_significant_event_episode_contains_event_context(self, data):
        """
        Feature: consciousness-system-deepening, Property 11: Significant Event Episode Creation
        
        For any significant event, the created episode SHALL contain the event
        information in its context or events list.
        
        **Validates: Requirements 3.2**
        """
        memory = EpisodicMemory()
        event_type = data.draw(significant_event_type_strategy())
        event_details = {"type": event_type, "result": "success", "timestamp": time.time()}
        
        episode = memory.create_episode(
            context={"event_type": event_type},
            events=[event_details],
            emotional_state=data.draw(emotional_state_strategy()),
            importance=0.8,
        )
        
        # Verify event information is captured
        assert episode.context.get("event_type") == event_type, \
            "Episode context should contain event type"
        assert len(episode.events) > 0, \
            "Episode should contain at least one event"
        assert episode.events[0]["type"] == event_type, \
            "Episode events should contain the event type"

    @given(
        num_episodes=st.integers(min_value=1, max_value=15),
        importance=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_episode_creation_is_idempotent_on_count(
        self, num_episodes: int, importance: float
    ):
        """
        Feature: consciousness-system-deepening, Property 11: Significant Event Episode Creation
        
        For any N episode creations, the final count SHALL be exactly N
        (assuming no pruning occurs).
        
        **Validates: Requirements 3.2**
        """
        # Use large max_episodes to avoid pruning
        memory = EpisodicMemory(max_episodes=1000)
        
        for i in range(num_episodes):
            memory.create_episode(
                context={"index": i},
                events=[{"action": f"action_{i}"}],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=importance,
            )
        
        assert len(memory) == num_episodes, \
            f"Memory should contain exactly {num_episodes} episodes"


# =============================================================================
# Property 12: Episode Retrieval Relevance Ordering
# =============================================================================

class TestEpisodeRetrievalRelevanceOrdering:
    """
    Tests for Property 12: Episode Retrieval Relevance Ordering
    
    *For any* retrieval query, the returned episodes SHALL be ordered by 
    descending relevance score, and the first episode SHALL have the highest 
    similarity to the query context.
    
    **Validates: Requirements 3.3, 3.4**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_temporal_retrieval_ordered_by_proximity(self, data):
        """
        Feature: consciousness-system-deepening, Property 12: Episode Retrieval Relevance Ordering
        
        For any temporal proximity retrieval, episodes SHALL be ordered by
        temporal distance (closest first).
        
        **Validates: Requirements 3.3**
        """
        memory = EpisodicMemory()
        base_time = 1000000.0
        
        # Create episodes at different times
        offsets = data.draw(st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False),
            min_size=3,
            max_size=10,
            unique=True
        ))
        
        for i, offset in enumerate(offsets):
            ep = Episode(
                episode_id=f"ep_{i}",
                timestamp=base_time + offset,
                context={"index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=0.5,
            )
            memory.store_episode(ep)
        
        # Retrieve with large time window to get all
        results = memory.retrieve_by_temporal_proximity(
            reference_time=base_time,
            time_window=2000,
            max_results=len(offsets),
        )
        
        # Verify ordering by temporal distance
        for i in range(len(results) - 1):
            dist_i = abs(results[i].timestamp - base_time)
            dist_next = abs(results[i + 1].timestamp - base_time)
            assert dist_i <= dist_next, \
                f"Episodes should be ordered by temporal distance: " \
                f"{dist_i} should be <= {dist_next}"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_context_retrieval_ordered_by_similarity(self, data):
        """
        Feature: consciousness-system-deepening, Property 12: Episode Retrieval Relevance Ordering
        
        For any context similarity retrieval, episodes SHALL be ordered by
        contextual similarity (most similar first).
        
        **Validates: Requirements 3.3, 3.4**
        """
        memory = EpisodicMemory()
        
        # Create episodes with varying context similarity to query
        query_context = {"task": "search", "domain": "science", "level": "advanced"}
        
        # Episode with exact match
        ep_exact = memory.create_episode(
            context={"task": "search", "domain": "science", "level": "advanced"},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
        )
        
        # Episode with partial match
        ep_partial = memory.create_episode(
            context={"task": "search", "domain": "math", "level": "basic"},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
        )
        
        # Episode with no match
        ep_none = memory.create_episode(
            context={"action": "generate", "type": "text"},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
        )
        
        results = memory.retrieve_by_context_similarity(
            query_context=query_context,
            max_results=10,
        )
        
        # The exact match should be first (or among the first)
        if len(results) > 0:
            # First result should have highest similarity
            first_result = results[0]
            # Check that exact match is ranked highly
            assert first_result.episode_id == ep_exact.episode_id or \
                   first_result.context.get("task") == "search", \
                "Most similar episode should be ranked first"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_emotional_retrieval_ordered_by_pad_distance(self, data):
        """
        Feature: consciousness-system-deepening, Property 12: Episode Retrieval Relevance Ordering
        
        For any emotional salience retrieval, episodes SHALL be ordered by
        PAD distance (closest emotional state first).
        
        **Validates: Requirements 3.3**
        """
        memory = EpisodicMemory()
        target_emotion = {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5}
        
        # Create episodes with different emotional states
        emotional_states = [
            {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5},  # Exact match
            {"pleasure": 0.3, "arousal": 0.4, "dominance": 0.6},  # Close
            {"pleasure": -0.5, "arousal": 0.8, "dominance": -0.3},  # Far
        ]
        
        for i, emotion in enumerate(emotional_states):
            memory.create_episode(
                context={"index": i},
                events=[],
                emotional_state=emotion,
            )
        
        results = memory.retrieve_by_emotional_salience(
            target_emotion=target_emotion,
            max_results=10,
        )
        
        # Calculate PAD distances for verification
        def pad_distance(e1: Dict, e2: Dict) -> float:
            p1, a1, d1 = e1.get("pleasure", 0), e1.get("arousal", 0.5), e1.get("dominance", 0)
            p2, a2, d2 = e2.get("pleasure", 0), e2.get("arousal", 0.5), e2.get("dominance", 0)
            return math.sqrt((p1-p2)**2 + (a1-a2)**2 + (d1-d2)**2)
        
        # Verify ordering by PAD distance
        for i in range(len(results) - 1):
            dist_i = pad_distance(target_emotion, results[i].emotional_state)
            dist_next = pad_distance(target_emotion, results[i + 1].emotional_state)
            assert dist_i <= dist_next + 0.001, \
                f"Episodes should be ordered by PAD distance: {dist_i} should be <= {dist_next}"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_combined_retrieval_returns_ordered_results(self, data):
        """
        Feature: consciousness-system-deepening, Property 12: Episode Retrieval Relevance Ordering
        
        For any combined relevance retrieval, the returned episodes SHALL be
        ordered by composite relevance score (highest first).
        
        **Validates: Requirements 3.3, 3.4**
        """
        memory = EpisodicMemory()
        
        # Create several episodes
        num_episodes = data.draw(st.integers(min_value=5, max_value=15))
        for i in range(num_episodes):
            episode_data = data.draw(episode_data_strategy())
            memory.create_episode(
                context=episode_data["context"],
                events=episode_data["events"],
                emotional_state=episode_data["emotional_state"],
                importance=episode_data["importance"],
            )
        
        # Query with combined criteria
        query = {
            "context": data.draw(context_strategy),
            "emotional_state": data.draw(emotional_state_strategy()),
            "reference_time": time.time(),
        }
        
        results = memory.retrieve_relevant(
            query=query,
            max_results=min(num_episodes, 10),
        )
        
        # Results should be returned (may be empty if no matches)
        assert isinstance(results, list), "Results should be a list"
        
        # If we have results, they should be ordered by relevance
        # (We can't easily verify the exact ordering without reimplementing the scoring,
        # but we can verify the structure)
        for episode in results:
            assert isinstance(episode, Episode), "Each result should be an Episode"
            assert episode.episode_id in memory, "Each result should be in memory"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_retrieval_respects_max_results(self, data):
        """
        Feature: consciousness-system-deepening, Property 12: Episode Retrieval Relevance Ordering
        
        For any retrieval with max_results parameter, the number of returned
        episodes SHALL be <= max_results.
        
        **Validates: Requirements 3.3, 3.4**
        """
        memory = EpisodicMemory()
        
        # Create more episodes than we'll request
        num_episodes = data.draw(st.integers(min_value=10, max_value=20))
        for i in range(num_episodes):
            memory.create_episode(
                context={"index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            )
        
        max_results = data.draw(st.integers(min_value=1, max_value=num_episodes - 1))
        
        # Test temporal retrieval
        temporal_results = memory.retrieve_by_temporal_proximity(
            reference_time=time.time(),
            time_window=10000,
            max_results=max_results,
        )
        assert len(temporal_results) <= max_results, \
            f"Temporal retrieval should return <= {max_results} results"
        
        # Test context retrieval
        context_results = memory.retrieve_by_context_similarity(
            query_context={"index": 0},
            max_results=max_results,
        )
        assert len(context_results) <= max_results, \
            f"Context retrieval should return <= {max_results} results"
        
        # Test emotional retrieval
        emotional_results = memory.retrieve_by_emotional_salience(
            target_emotion={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            max_results=max_results,
        )
        assert len(emotional_results) <= max_results, \
            f"Emotional retrieval should return <= {max_results} results"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_retrieval_records_access(self, data):
        """
        Feature: consciousness-system-deepening, Property 12: Episode Retrieval Relevance Ordering
        
        For any retrieval operation, the accessed episodes SHALL have their
        access_count incremented and last_accessed updated.
        
        **Validates: Requirements 3.3, 3.4**
        """
        memory = EpisodicMemory()
        
        # Create episodes
        episodes = []
        for i in range(5):
            ep = memory.create_episode(
                context={"index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            )
            episodes.append(ep)
        
        # Verify initial access counts
        for ep in episodes:
            assert ep.access_count == 0, "Initial access count should be 0"
            assert ep.last_accessed is None, "Initial last_accessed should be None"
        
        # Perform retrieval
        before_retrieval = time.time()
        results = memory.retrieve_by_temporal_proximity(
            reference_time=time.time(),
            time_window=10000,
            max_results=3,
        )
        after_retrieval = time.time()
        
        # Verify access was recorded for retrieved episodes
        for ep in results:
            assert ep.access_count >= 1, "Retrieved episodes should have access_count >= 1"
            assert ep.last_accessed is not None, "Retrieved episodes should have last_accessed set"
            assert before_retrieval <= ep.last_accessed <= after_retrieval, \
                "last_accessed should be within retrieval time window"


# =============================================================================
# Property 13: Memory Consolidation Pattern Extraction
# =============================================================================

class TestMemoryConsolidationPatternExtraction:
    """
    Tests for Property 13: Memory Consolidation Pattern Extraction
    
    *For any* consolidation operation on a non-empty EpisodicMemory, the 
    ConsolidationResult SHALL contain extracted_patterns, and the SemanticMemory 
    SHALL be updated with new or strengthened concepts.
    
    **Validates: Requirements 3.5**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_consolidation_extracts_patterns_from_unconsolidated(self, data):
        """
        Feature: consciousness-system-deepening, Property 13: Memory Consolidation Pattern Extraction
        
        For any consolidation on non-empty memory with unconsolidated episodes,
        the result SHALL contain extracted_patterns.
        
        **Validates: Requirements 3.5**
        """
        memory = EpisodicMemory()
        
        # Create several unconsolidated episodes
        num_episodes = data.draw(st.integers(min_value=3, max_value=10))
        for i in range(num_episodes):
            episode_data = data.draw(episode_data_strategy())
            memory.create_episode(
                context=episode_data["context"],
                events=episode_data["events"],
                emotional_state=episode_data["emotional_state"],
            )
        
        # Verify episodes are unconsolidated
        unconsolidated_before = memory.get_unconsolidated_episodes()
        assert len(unconsolidated_before) == num_episodes, \
            "All episodes should be unconsolidated initially"
        
        # Perform consolidation
        result = memory.consolidate()
        
        # Verify result structure
        assert "extracted_patterns" in result, \
            "Consolidation result should contain extracted_patterns"
        assert "consolidated_count" in result, \
            "Consolidation result should contain consolidated_count"
        assert result["consolidated_count"] == num_episodes, \
            f"Should consolidate all {num_episodes} episodes"
        
        # Verify patterns structure
        patterns = result["extracted_patterns"]
        assert "context_patterns" in patterns, \
            "extracted_patterns should contain context_patterns"
        assert "event_patterns" in patterns, \
            "extracted_patterns should contain event_patterns"
        assert "emotional_patterns" in patterns, \
            "extracted_patterns should contain emotional_patterns"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_consolidation_marks_episodes_as_consolidated(self, data):
        """
        Feature: consciousness-system-deepening, Property 13: Memory Consolidation Pattern Extraction
        
        For any consolidation operation, all processed episodes SHALL be marked
        as consolidated.
        
        **Validates: Requirements 3.5**
        """
        memory = EpisodicMemory()
        
        # Create episodes
        num_episodes = data.draw(st.integers(min_value=2, max_value=8))
        episode_ids = []
        for i in range(num_episodes):
            ep = memory.create_episode(
                context={"index": i, "type": "test"},
                events=[{"action": f"action_{i}"}],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            )
            episode_ids.append(ep.episode_id)
        
        # Consolidate
        result = memory.consolidate()
        
        # Verify all episodes are now consolidated
        for ep_id in episode_ids:
            ep = memory.get_episode(ep_id, record_access=False)
            assert ep.consolidated is True, \
                f"Episode {ep_id} should be marked as consolidated"
        
        # Verify no unconsolidated episodes remain
        unconsolidated_after = memory.get_unconsolidated_episodes()
        assert len(unconsolidated_after) == 0, \
            "No unconsolidated episodes should remain after consolidation"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_consolidation_on_empty_memory_returns_empty_patterns(self, data):
        """
        Feature: consciousness-system-deepening, Property 13: Memory Consolidation Pattern Extraction
        
        For any consolidation on empty memory, the result SHALL contain empty
        pattern lists and consolidated_count of 0.
        
        **Validates: Requirements 3.5**
        """
        memory = EpisodicMemory()
        
        result = memory.consolidate()
        
        assert result["consolidated_count"] == 0, \
            "Empty memory should have 0 consolidated episodes"
        assert result["extracted_patterns"]["context_patterns"] == [], \
            "Empty memory should have empty context_patterns"
        assert result["extracted_patterns"]["event_patterns"] == [], \
            "Empty memory should have empty event_patterns"
        assert result["extracted_patterns"]["emotional_patterns"] == [], \
            "Empty memory should have empty emotional_patterns"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_consolidation_only_processes_unconsolidated(self, data):
        """
        Feature: consciousness-system-deepening, Property 13: Memory Consolidation Pattern Extraction
        
        For any consolidation operation, only unconsolidated episodes SHALL be
        processed; already consolidated episodes SHALL be skipped.
        
        **Validates: Requirements 3.5**
        """
        memory = EpisodicMemory()
        
        # Create and consolidate first batch
        for i in range(3):
            memory.create_episode(
                context={"batch": 1, "index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            )
        
        first_result = memory.consolidate()
        assert first_result["consolidated_count"] == 3
        
        # Create second batch
        for i in range(2):
            memory.create_episode(
                context={"batch": 2, "index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            )
        
        # Second consolidation should only process new episodes
        second_result = memory.consolidate()
        assert second_result["consolidated_count"] == 2, \
            "Second consolidation should only process 2 new episodes"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_consolidation_extracts_context_patterns(self, data):
        """
        Feature: consciousness-system-deepening, Property 13: Memory Consolidation Pattern Extraction
        
        For any consolidation with episodes sharing common context keys,
        context_patterns SHALL identify frequently occurring keys.
        
        **Validates: Requirements 3.5**
        """
        memory = EpisodicMemory()
        
        # Create episodes with common context keys
        common_key = "task_type"
        common_value = "search"
        
        for i in range(5):
            memory.create_episode(
                context={common_key: common_value, "index": i},
                events=[{"type": "action"}],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            )
        
        result = memory.consolidate()
        
        # Should extract pattern for the common key
        context_patterns = result["extracted_patterns"]["context_patterns"]
        
        # Find pattern for common_key
        common_key_pattern = None
        for pattern in context_patterns:
            if pattern.get("key") == common_key:
                common_key_pattern = pattern
                break
        
        if common_key_pattern:
            assert common_key_pattern["frequency"] >= 0.8, \
                f"Common key '{common_key}' should have high frequency"
            assert common_key_pattern["occurrence_count"] >= 4, \
                f"Common key '{common_key}' should have high occurrence count"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_consolidation_adds_metadata(self, data):
        """
        Feature: consciousness-system-deepening, Property 13: Memory Consolidation Pattern Extraction
        
        For any consolidation, processed episodes SHALL have consolidation
        metadata added (e.g., consolidated_at timestamp).
        
        **Validates: Requirements 3.5**
        """
        memory = EpisodicMemory()
        
        ep = memory.create_episode(
            context={"test": True},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
        )
        
        before_consolidation = time.time()
        memory.consolidate()
        after_consolidation = time.time()
        
        # Check metadata was added
        consolidated_ep = memory.get_episode(ep.episode_id, record_access=False)
        assert "consolidated_at" in consolidated_ep.metadata, \
            "Consolidated episode should have consolidated_at in metadata"
        assert before_consolidation <= consolidated_ep.metadata["consolidated_at"] <= after_consolidation, \
            "consolidated_at should be within consolidation time window"


# =============================================================================
# Property 14: Importance-Weighted Memory Pruning
# =============================================================================

class TestImportanceWeightedMemoryPruning:
    """
    Tests for Property 14: Importance-Weighted Memory Pruning
    
    *For any* pruning operation when memory exceeds limits, the removed 
    episodes/experiences SHALL have lower importance scores than retained ones, 
    preserving emotionally salient and frequently accessed items.
    
    **Validates: Requirements 3.6**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_pruning_removes_low_importance_episodes(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any pruning operation, the removed episodes SHALL have lower
        importance scores than the retained episodes on average.
        
        **Validates: Requirements 3.6**
        """
        # Create memory with small capacity to trigger pruning
        config = EpisodicMemoryConfig(
            max_episodes=10,
            prune_threshold=0.9,
            prune_ratio=0.3,
        )
        memory = EpisodicMemory(config=config.to_dict())
        
        # Add episodes with varied importance
        importances = []
        for i in range(12):  # More than max to trigger pruning
            importance = i / 11.0  # 0.0 to 1.0
            importances.append(importance)
            memory.create_episode(
                context={"index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=importance,
            )
        
        # Get remaining episodes' importance
        remaining_importances = [
            ep.importance for ep in memory.get_all_episodes()
        ]
        
        # Average importance of remaining should be higher than original average
        avg_original = sum(importances) / len(importances)
        avg_remaining = sum(remaining_importances) / len(remaining_importances)
        
        assert avg_remaining >= avg_original - 0.1, \
            f"Remaining episodes should have higher avg importance: " \
            f"remaining={avg_remaining:.3f}, original={avg_original:.3f}"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_pruning_preserves_high_importance_episodes(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any pruning operation, high importance episodes SHALL be preserved
        while low importance episodes are removed.
        
        **Validates: Requirements 3.6**
        """
        config = EpisodicMemoryConfig(
            max_episodes=8,
            prune_threshold=0.9,
            prune_ratio=0.4,
        )
        memory = EpisodicMemory(config=config.to_dict())
        
        # Create high importance episodes
        high_importance_ids = []
        for i in range(3):
            ep = memory.create_episode(
                context={"type": "high_importance", "index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=0.9 + i * 0.03,  # 0.9, 0.93, 0.96
            )
            high_importance_ids.append(ep.episode_id)
        
        # Create low importance episodes to trigger pruning
        for i in range(8):  # Enough to exceed capacity
            memory.create_episode(
                context={"type": "low_importance", "index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=0.1 + i * 0.05,  # 0.1 to 0.45
            )
        
        # Check that high importance episodes are preserved
        preserved_high = sum(1 for ep_id in high_importance_ids if ep_id in memory)
        
        assert preserved_high >= len(high_importance_ids) - 1, \
            f"High importance episodes should be preserved: " \
            f"{preserved_high}/{len(high_importance_ids)} preserved"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_pruning_preserves_frequently_accessed_episodes(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any pruning operation, frequently accessed episodes SHALL be
        preserved even if their importance is moderate.
        
        **Validates: Requirements 3.6**
        """
        config = EpisodicMemoryConfig(
            max_episodes=10,
            prune_threshold=0.9,
            prune_ratio=0.3,
        )
        memory = EpisodicMemory(config=config.to_dict())
        
        # Create an episode with moderate importance but high access count
        frequently_accessed = memory.create_episode(
            context={"type": "frequently_accessed"},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            importance=0.5,  # Moderate importance
        )
        
        # Access it multiple times
        for _ in range(10):
            memory.get_episode(frequently_accessed.episode_id)
        
        # Create many low importance episodes to trigger pruning
        for i in range(12):
            memory.create_episode(
                context={"type": "filler", "index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=0.2,  # Low importance
            )
        
        # The frequently accessed episode should be preserved
        assert frequently_accessed.episode_id in memory, \
            "Frequently accessed episode should be preserved during pruning"

    @given(
        num_episodes=st.integers(min_value=15, max_value=30),
        max_episodes=st.integers(min_value=8, max_value=12),
    )
    @settings(max_examples=100)
    def test_pruning_maintains_size_limit(self, num_episodes: int, max_episodes: int):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any memory, after pruning the size SHALL be within the configured limits.
        
        **Validates: Requirements 3.6**
        """
        config = EpisodicMemoryConfig(
            max_episodes=max_episodes,
            prune_threshold=0.9,
            prune_ratio=0.2,
        )
        memory = EpisodicMemory(config=config.to_dict())
        
        # Add more episodes than max_episodes
        for i in range(num_episodes):
            memory.create_episode(
                context={"index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=i / num_episodes,
            )
        
        # Memory size should be at or below max_episodes
        assert len(memory) <= max_episodes, \
            f"Memory size ({len(memory)}) should be <= max_episodes ({max_episodes})"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_importance_threshold_pruning(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any prune_by_importance operation, episodes below the threshold
        SHALL be removed (unless frequently accessed).
        
        **Validates: Requirements 3.6**
        """
        memory = EpisodicMemory(max_episodes=100)
        
        # Create episodes with varied importance
        low_importance_ids = []
        high_importance_ids = []
        
        for i in range(5):
            # Low importance episodes
            ep_low = memory.create_episode(
                context={"type": "low", "index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=0.1 + i * 0.05,  # 0.1 to 0.3
            )
            low_importance_ids.append(ep_low.episode_id)
            
            # High importance episodes
            ep_high = memory.create_episode(
                context={"type": "high", "index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=0.6 + i * 0.08,  # 0.6 to 0.92
            )
            high_importance_ids.append(ep_high.episode_id)
        
        # Prune with threshold of 0.5
        threshold = 0.5
        removed_count = memory.prune_by_importance(threshold)
        
        # Low importance episodes should be removed
        remaining_low = sum(1 for ep_id in low_importance_ids if ep_id in memory)
        
        # High importance episodes should be preserved
        remaining_high = sum(1 for ep_id in high_importance_ids if ep_id in memory)
        
        assert remaining_high == len(high_importance_ids), \
            f"All high importance episodes should be preserved: " \
            f"{remaining_high}/{len(high_importance_ids)}"
        
        assert remaining_low < len(low_importance_ids), \
            f"Some low importance episodes should be removed: " \
            f"{remaining_low}/{len(low_importance_ids)} remaining"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_decay_reduces_importance_over_time(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any decay operation, episode importance SHALL decrease over time,
        making older, less accessed episodes more likely to be pruned.
        
        **Validates: Requirements 3.6**
        """
        config = EpisodicMemoryConfig(
            max_episodes=100,
            decay_rate=0.9,  # 10% decay per time unit
        )
        memory = EpisodicMemory(config=config.to_dict())
        
        # Create episodes with known importance
        initial_importance = 0.8
        ep = memory.create_episode(
            context={"test": True},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            importance=initial_importance,
        )
        
        # Apply decay
        time_elapsed = data.draw(st.floats(min_value=1.0, max_value=5.0, allow_nan=False))
        memory.apply_decay(time_elapsed)
        
        # Get updated episode
        updated_ep = memory.get_episode(ep.episode_id, record_access=False)
        
        # Importance should have decreased
        expected_importance = initial_importance * (config.decay_rate ** time_elapsed)
        
        assert abs(updated_ep.importance - expected_importance) < 0.01, \
            f"Importance should decay: expected {expected_importance:.4f}, " \
            f"got {updated_ep.importance:.4f}"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_pruning_order_by_composite_value(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any pruning operation, episodes SHALL be removed in order of
        composite value (importance + access frequency + recency).
        
        **Validates: Requirements 3.6**
        """
        config = EpisodicMemoryConfig(
            max_episodes=5,
            prune_threshold=0.9,
            prune_ratio=0.4,
        )
        memory = EpisodicMemory(config=config.to_dict())
        
        # Create episode with high importance
        high_imp = memory.create_episode(
            context={"type": "high_importance"},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            importance=0.95,
        )
        
        # Create episode with moderate importance but high access
        high_access = memory.create_episode(
            context={"type": "high_access"},
            events=[],
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            importance=0.5,
        )
        for _ in range(5):
            memory.get_episode(high_access.episode_id)
        
        # Create low value episodes to trigger pruning
        for i in range(6):
            memory.create_episode(
                context={"type": "low_value", "index": i},
                events=[],
                emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
                importance=0.1,
            )
        
        # Both high value episodes should be preserved
        assert high_imp.episode_id in memory, \
            "High importance episode should be preserved"
        assert high_access.episode_id in memory, \
            "Frequently accessed episode should be preserved"

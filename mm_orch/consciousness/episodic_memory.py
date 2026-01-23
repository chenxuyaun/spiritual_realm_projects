"""
Episodic Memory module for consciousness system.

This module implements the EpisodicMemory system which stores specific experiences
as discrete episodes with temporal and contextual information. Episodes capture
the context, events, outcomes, and emotional state at the time of the experience.

The episodic memory supports:
- Episode creation with automatic ID and timestamp generation
- Storage with configurable limits
- Importance and access tracking
- Retrieval by temporal proximity, contextual similarity, and emotional salience
- Combined relevance-based retrieval
- Serialization/deserialization for persistence

Requirements: 3.1, 3.2, 3.3, 3.4
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import uuid
import math


@dataclass
class Episode:
    """
    Represents a discrete episode in memory.
    
    An episode captures a specific experience with all relevant contextual
    information, including the environmental context, sequence of events,
    emotional state, and metadata for memory management.
    
    Attributes:
        episode_id: Unique identifier for the episode.
        timestamp: When the episode was created (Unix timestamp).
        context: Environmental context at the time of the episode.
        events: Sequence of events that occurred during the episode.
        emotional_state: PAD (Pleasure-Arousal-Dominance) values at time of episode.
        importance: Significance score from 0.0 (low) to 1.0 (high).
        access_count: Number of times this episode has been retrieved.
        last_accessed: Timestamp of the last access (None if never accessed).
        consolidated: Whether patterns have been extracted from this episode.
        metadata: Additional information about the episode.
        
    Validates: Requirements 3.1
    """
    episode_id: str
    timestamp: float
    context: Dict[str, Any]
    events: List[Dict[str, Any]]
    emotional_state: Dict[str, float]
    importance: float
    access_count: int = 0
    last_accessed: Optional[float] = None
    consolidated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate episode fields after initialization."""
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty")
        
        if not isinstance(self.timestamp, (int, float)):
            raise ValueError("timestamp must be a number")
        
        if not isinstance(self.context, dict):
            raise ValueError("context must be a dictionary")
        
        if not isinstance(self.events, list):
            raise ValueError("events must be a list")
        
        if not isinstance(self.emotional_state, dict):
            raise ValueError("emotional_state must be a dictionary")
        
        if not isinstance(self.importance, (int, float)):
            raise ValueError("importance must be a number")
        
        # Clamp importance to valid range [0.0, 1.0]
        self.importance = max(0.0, min(1.0, float(self.importance)))
        
        if not isinstance(self.access_count, int):
            raise ValueError("access_count must be an integer")
        
        if self.access_count < 0:
            raise ValueError("access_count cannot be negative")
        
        if self.last_accessed is not None and not isinstance(self.last_accessed, (int, float)):
            raise ValueError("last_accessed must be a number or None")
        
        if not isinstance(self.consolidated, bool):
            raise ValueError("consolidated must be a boolean")
        
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def record_access(self) -> None:
        """Record that this episode was accessed."""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary representation for serialization."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "context": self.context.copy(),
            "events": [event.copy() if isinstance(event, dict) else event for event in self.events],
            "emotional_state": self.emotional_state.copy(),
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "consolidated": self.consolidated,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create an episode from dictionary representation."""
        return cls(
            episode_id=data["episode_id"],
            timestamp=data["timestamp"],
            context=data.get("context", {}),
            events=data.get("events", []),
            emotional_state=data.get("emotional_state", {}),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            consolidated=data.get("consolidated", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EpisodicMemoryConfig:
    """Configuration for the episodic memory system."""
    max_episodes: int = 5000
    default_importance: float = 0.5
    consolidation_interval: int = 100  # Episodes between consolidations
    decay_rate: float = 0.99  # Importance decay rate per time unit
    importance_threshold: float = 0.3  # Minimum importance to retain
    prune_threshold: float = 0.9  # Prune when memory reaches this fraction of max
    prune_ratio: float = 0.2  # Remove this fraction when pruning

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_episodes < 1:
            raise ValueError("max_episodes must be at least 1")
        if not (0.0 <= self.default_importance <= 1.0):
            raise ValueError("default_importance must be between 0.0 and 1.0")
        if self.consolidation_interval < 1:
            raise ValueError("consolidation_interval must be at least 1")
        if not (0.0 < self.decay_rate <= 1.0):
            raise ValueError("decay_rate must be between 0.0 (exclusive) and 1.0")
        if not (0.0 <= self.importance_threshold <= 1.0):
            raise ValueError("importance_threshold must be between 0.0 and 1.0")
        if not (0.0 < self.prune_threshold <= 1.0):
            raise ValueError("prune_threshold must be between 0.0 (exclusive) and 1.0")
        if not (0.0 < self.prune_ratio < 1.0):
            raise ValueError("prune_ratio must be between 0.0 and 1.0 (exclusive)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "max_episodes": self.max_episodes,
            "default_importance": self.default_importance,
            "consolidation_interval": self.consolidation_interval,
            "decay_rate": self.decay_rate,
            "importance_threshold": self.importance_threshold,
            "prune_threshold": self.prune_threshold,
            "prune_ratio": self.prune_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemoryConfig":
        """Create config from dictionary representation."""
        return cls(
            max_episodes=data.get("max_episodes", 5000),
            default_importance=data.get("default_importance", 0.5),
            consolidation_interval=data.get("consolidation_interval", 100),
            decay_rate=data.get("decay_rate", 0.99),
            importance_threshold=data.get("importance_threshold", 0.3),
            prune_threshold=data.get("prune_threshold", 0.9),
            prune_ratio=data.get("prune_ratio", 0.2),
        )


class EpisodicMemory:
    """
    Manages episodic memory storage and retrieval.
    
    This class implements the episodic memory system that stores specific
    experiences as discrete episodes. Each episode contains temporal and
    contextual information, emotional state, and importance scoring.
    
    The memory system supports:
    - Episode creation with automatic ID and timestamp
    - Storage with configurable limits
    - Importance and access tracking
    - Basic retrieval operations
    - Serialization/deserialization for persistence
    
    Requirements: 3.1, 3.2
    """

    def __init__(self, max_episodes: int = 5000, config: Optional[Dict[str, Any]] = None):
        """
        Initialize episodic memory.
        
        Args:
            max_episodes: Maximum number of episodes to store.
            config: Optional configuration dictionary.
        """
        if config is not None:
            self._config = EpisodicMemoryConfig.from_dict(config)
            # Override max_episodes if not provided in config
            if "max_episodes" not in config:
                self._config.max_episodes = max_episodes
        else:
            self._config = EpisodicMemoryConfig(max_episodes=max_episodes)
        
        # Main storage: episode_id -> Episode
        self._episodes: Dict[str, Episode] = {}
        
        # Statistics
        self._total_created: int = 0
        self._total_accessed: int = 0
        self._total_pruned: int = 0
        self._episodes_since_consolidation: int = 0
        self._initialized_at: float = time.time()

    def _generate_episode_id(self) -> str:
        """Generate a unique episode ID."""
        return str(uuid.uuid4())

    def create_episode(
        self,
        context: Dict[str, Any],
        events: List[Dict[str, Any]],
        emotional_state: Dict[str, float],
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """
        Create and store a new episode.
        
        This method creates a new episode with automatic ID and timestamp
        generation, then stores it in memory. If the memory is at capacity,
        pruning will be triggered automatically.
        
        Args:
            context: Environmental context at the time of the episode.
            events: Sequence of events that occurred during the episode.
            emotional_state: PAD values at time of episode.
            importance: Significance score (0.0-1.0). Defaults to config default.
            metadata: Additional information about the episode.
            
        Returns:
            The created Episode instance.
            
        Validates: Requirements 3.1, 3.2
        """
        # Check if we need to prune before adding
        if len(self._episodes) >= self._config.max_episodes:
            self._prune()
        
        # Create the episode
        episode = Episode(
            episode_id=self._generate_episode_id(),
            timestamp=time.time(),
            context=context if context is not None else {},
            events=events if events is not None else [],
            emotional_state=emotional_state if emotional_state is not None else {},
            importance=importance if importance is not None else self._config.default_importance,
            access_count=0,
            last_accessed=None,
            consolidated=False,
            metadata=metadata if metadata is not None else {},
        )
        
        # Store the episode
        self._episodes[episode.episode_id] = episode
        self._total_created += 1
        self._episodes_since_consolidation += 1
        
        return episode

    def store_episode(self, episode: Episode) -> str:
        """
        Store an existing episode.
        
        This method stores an episode that was created externally.
        If the memory is at capacity, pruning will be triggered automatically.
        
        Args:
            episode: The episode to store.
            
        Returns:
            The episode ID.
            
        Validates: Requirements 3.1
        """
        # Check if we need to prune before adding
        if len(self._episodes) >= self._config.max_episodes:
            self._prune()
        
        self._episodes[episode.episode_id] = episode
        self._total_created += 1
        self._episodes_since_consolidation += 1
        
        return episode.episode_id

    def get_episode(self, episode_id: str, record_access: bool = True) -> Optional[Episode]:
        """
        Get an episode by ID.
        
        Args:
            episode_id: ID of the episode to retrieve.
            record_access: Whether to record this access (updates access_count and last_accessed).
            
        Returns:
            The episode if found, None otherwise.
        """
        episode = self._episodes.get(episode_id)
        if episode is not None and record_access:
            episode.record_access()
            self._total_accessed += 1
        return episode

    def contains(self, episode_id: str) -> bool:
        """
        Check if an episode exists in memory.
        
        Args:
            episode_id: ID of the episode to check.
            
        Returns:
            True if the episode exists, False otherwise.
        """
        return episode_id in self._episodes

    def get_all_episodes(self) -> List[Episode]:
        """
        Get all episodes in memory.
        
        Returns:
            List of all episodes (not copies, be careful with modifications).
        """
        return list(self._episodes.values())

    def get_episode_count(self) -> int:
        """
        Get the number of episodes in memory.
        
        Returns:
            Number of episodes currently stored.
        """
        return len(self._episodes)

    def remove_episode(self, episode_id: str) -> bool:
        """
        Remove an episode from memory.
        
        Args:
            episode_id: ID of the episode to remove.
            
        Returns:
            True if the episode was removed, False if it didn't exist.
        """
        if episode_id in self._episodes:
            del self._episodes[episode_id]
            return True
        return False

    def update_importance(self, episode_id: str, importance: float) -> bool:
        """
        Update the importance score of an episode.
        
        Args:
            episode_id: ID of the episode to update.
            importance: New importance score (will be clamped to [0.0, 1.0]).
            
        Returns:
            True if the episode was updated, False if it didn't exist.
        """
        if episode_id not in self._episodes:
            return False
        
        # Clamp importance to valid range
        clamped_importance = max(0.0, min(1.0, float(importance)))
        self._episodes[episode_id].importance = clamped_importance
        return True

    def mark_consolidated(self, episode_id: str) -> bool:
        """
        Mark an episode as consolidated (patterns extracted).
        
        Args:
            episode_id: ID of the episode to mark.
            
        Returns:
            True if the episode was marked, False if it didn't exist.
        """
        if episode_id not in self._episodes:
            return False
        
        self._episodes[episode_id].consolidated = True
        return True

    def get_unconsolidated_episodes(self) -> List[Episode]:
        """
        Get all episodes that haven't been consolidated yet.
        
        Returns:
            List of unconsolidated episodes.
        """
        return [ep for ep in self._episodes.values() if not ep.consolidated]

    def retrieve_by_temporal_proximity(
        self,
        reference_time: float,
        time_window: float,
        max_results: int = 10,
    ) -> List[Episode]:
        """
        Retrieve episodes within a time window of the reference time.
        
        Returns episodes whose timestamps fall within [reference_time - time_window,
        reference_time + time_window], sorted by temporal proximity (closest first).
        
        Args:
            reference_time: The reference timestamp (Unix timestamp).
            time_window: The time window in seconds (episodes within +/- this range).
            max_results: Maximum number of episodes to return.
            
        Returns:
            List of episodes sorted by temporal proximity (closest to reference_time first).
            
        Validates: Requirements 3.3
        """
        if max_results <= 0:
            return []
        
        if time_window < 0:
            time_window = abs(time_window)
        
        # Calculate time bounds
        lower_bound = reference_time - time_window
        upper_bound = reference_time + time_window
        
        # Find episodes within the time window
        matching_episodes: List[Tuple[float, Episode]] = []
        for episode in self._episodes.values():
            if lower_bound <= episode.timestamp <= upper_bound:
                # Calculate temporal distance (absolute difference from reference)
                temporal_distance = abs(episode.timestamp - reference_time)
                matching_episodes.append((temporal_distance, episode))
        
        # Sort by temporal distance (closest first)
        matching_episodes.sort(key=lambda x: (x[0], x[1].episode_id))
        
        # Return top results, recording access
        results = []
        for _, episode in matching_episodes[:max_results]:
            episode.record_access()
            self._total_accessed += 1
            results.append(episode)
        
        return results

    def retrieve_by_context_similarity(
        self,
        query_context: Dict[str, Any],
        max_results: int = 10,
    ) -> List[Episode]:
        """
        Retrieve episodes with similar context using key/value matching.
        
        Calculates similarity based on matching keys and values between the
        query context and episode contexts. Episodes are sorted by similarity
        score (highest first).
        
        Args:
            query_context: The context to match against.
            max_results: Maximum number of episodes to return.
            
        Returns:
            List of episodes sorted by contextual similarity (most similar first).
            
        Validates: Requirements 3.3, 3.4
        """
        if max_results <= 0:
            return []
        
        if not query_context:
            # No query context - return most recent episodes
            sorted_episodes = sorted(
                self._episodes.values(),
                key=lambda e: e.timestamp,
                reverse=True
            )
            results = sorted_episodes[:max_results]
            for episode in results:
                episode.record_access()
                self._total_accessed += 1
            return results
        
        # Calculate similarity scores for all episodes
        scored_episodes: List[Tuple[float, Episode]] = []
        for episode in self._episodes.values():
            similarity = self._calculate_context_similarity(query_context, episode.context)
            if similarity > 0:
                scored_episodes.append((similarity, episode))
        
        # Sort by similarity (highest first), then by timestamp (most recent first)
        scored_episodes.sort(key=lambda x: (-x[0], -x[1].timestamp))
        
        # Return top results, recording access
        results = []
        for _, episode in scored_episodes[:max_results]:
            episode.record_access()
            self._total_accessed += 1
            results.append(episode)
        
        return results

    def _calculate_context_similarity(
        self,
        query_context: Dict[str, Any],
        episode_context: Dict[str, Any],
    ) -> float:
        """
        Calculate similarity between two contexts using key/value matching.
        
        The similarity score is based on:
        - Matching keys (weighted by whether values also match)
        - Exact value matches get higher weight than just key matches
        
        Args:
            query_context: The query context.
            episode_context: The episode's context.
            
        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not query_context or not episode_context:
            return 0.0
        
        query_keys = set(query_context.keys())
        episode_keys = set(episode_context.keys())
        
        # Find common keys
        common_keys = query_keys & episode_keys
        
        if not common_keys:
            return 0.0
        
        # Calculate weighted similarity
        total_score = 0.0
        max_possible_score = len(query_keys)
        
        for key in common_keys:
            query_value = query_context[key]
            episode_value = episode_context[key]
            
            # Exact match gets full score
            if query_value == episode_value:
                total_score += 1.0
            # Partial match for nested dicts
            elif isinstance(query_value, dict) and isinstance(episode_value, dict):
                nested_similarity = self._calculate_context_similarity(query_value, episode_value)
                total_score += nested_similarity * 0.8
            # Type match gets partial score
            elif type(query_value) == type(episode_value):
                # For strings, check if one contains the other
                if isinstance(query_value, str) and isinstance(episode_value, str):
                    if query_value.lower() in episode_value.lower() or episode_value.lower() in query_value.lower():
                        total_score += 0.5
                    else:
                        total_score += 0.2
                else:
                    total_score += 0.3
            else:
                # Key exists but different type
                total_score += 0.1
        
        return total_score / max_possible_score if max_possible_score > 0 else 0.0

    def retrieve_by_emotional_salience(
        self,
        target_emotion: Dict[str, float],
        max_results: int = 10,
    ) -> List[Episode]:
        """
        Retrieve episodes with similar emotional states using Euclidean distance in PAD space.
        
        Calculates the Euclidean distance between the target emotion (PAD values)
        and each episode's emotional state. Episodes are sorted by emotional
        similarity (closest first).
        
        Args:
            target_emotion: Target emotional state with PAD values 
                           (keys: 'pleasure', 'arousal', 'dominance').
            max_results: Maximum number of episodes to return.
            
        Returns:
            List of episodes sorted by emotional similarity (closest PAD distance first).
            
        Validates: Requirements 3.3
        """
        if max_results <= 0:
            return []
        
        # Calculate emotional distance for all episodes
        scored_episodes: List[Tuple[float, Episode]] = []
        for episode in self._episodes.values():
            distance = self._calculate_pad_distance(target_emotion, episode.emotional_state)
            scored_episodes.append((distance, episode))
        
        # Sort by distance (closest first), then by importance (highest first)
        scored_episodes.sort(key=lambda x: (x[0], -x[1].importance))
        
        # Return top results, recording access
        results = []
        for _, episode in scored_episodes[:max_results]:
            episode.record_access()
            self._total_accessed += 1
            results.append(episode)
        
        return results

    def _calculate_pad_distance(
        self,
        emotion1: Dict[str, float],
        emotion2: Dict[str, float],
    ) -> float:
        """
        Calculate Euclidean distance between two emotional states in PAD space.
        
        PAD dimensions:
        - Pleasure: -1.0 to 1.0
        - Arousal: 0.0 to 1.0
        - Dominance: -1.0 to 1.0
        
        Args:
            emotion1: First emotional state.
            emotion2: Second emotional state.
            
        Returns:
            Euclidean distance between the two states.
        """
        # Get PAD values with defaults
        p1 = emotion1.get("pleasure", 0.0)
        a1 = emotion1.get("arousal", 0.5)
        d1 = emotion1.get("dominance", 0.0)
        
        p2 = emotion2.get("pleasure", 0.0)
        a2 = emotion2.get("arousal", 0.5)
        d2 = emotion2.get("dominance", 0.0)
        
        # Calculate Euclidean distance
        distance = math.sqrt(
            (p1 - p2) ** 2 +
            (a1 - a2) ** 2 +
            (d1 - d2) ** 2
        )
        
        return distance

    def retrieve_relevant(
        self,
        query: Dict[str, Any],
        max_results: int = 10,
    ) -> List[Episode]:
        """
        Combined retrieval using multiple criteria with composite relevance scoring.
        
        This method combines temporal, contextual, and emotional factors to
        rank episodes by overall relevance. The query can include:
        - 'context': Dict for contextual similarity matching
        - 'emotional_state': Dict with PAD values for emotional similarity
        - 'reference_time': Float timestamp for temporal proximity
        - 'time_weight': Float weight for temporal factor (default 0.2)
        - 'context_weight': Float weight for context factor (default 0.5)
        - 'emotion_weight': Float weight for emotion factor (default 0.3)
        
        Args:
            query: Query parameters including context, emotional_state, and/or reference_time.
            max_results: Maximum number of episodes to return.
            
        Returns:
            List of episodes sorted by composite relevance score (highest first).
            
        Validates: Requirements 3.3, 3.4
        """
        if max_results <= 0:
            return []
        
        if not self._episodes:
            return []
        
        # Extract query parameters
        query_context = query.get("context", {})
        query_emotion = query.get("emotional_state", {})
        reference_time = query.get("reference_time", time.time())
        
        # Get weights (default to balanced weights)
        time_weight = query.get("time_weight", 0.2)
        context_weight = query.get("context_weight", 0.5)
        emotion_weight = query.get("emotion_weight", 0.3)
        
        # Normalize weights
        total_weight = time_weight + context_weight + emotion_weight
        if total_weight > 0:
            time_weight /= total_weight
            context_weight /= total_weight
            emotion_weight /= total_weight
        else:
            # Default equal weights if all zero
            time_weight = context_weight = emotion_weight = 1.0 / 3.0
        
        # Calculate relevance scores for all episodes
        scored_episodes: List[Tuple[float, Episode]] = []
        
        # Pre-calculate max values for normalization
        max_time_diff = self._get_max_time_difference(reference_time)
        max_pad_distance = math.sqrt(4 + 1 + 4)  # Max possible PAD distance
        
        for episode in self._episodes.values():
            # Calculate temporal score (closer = higher score)
            time_diff = abs(episode.timestamp - reference_time)
            if max_time_diff > 0:
                temporal_score = 1.0 - (time_diff / max_time_diff)
            else:
                temporal_score = 1.0
            
            # Calculate context similarity score
            if query_context:
                context_score = self._calculate_context_similarity(query_context, episode.context)
            else:
                context_score = 0.5  # Neutral if no context query
            
            # Calculate emotional similarity score (closer = higher score)
            if query_emotion:
                pad_distance = self._calculate_pad_distance(query_emotion, episode.emotional_state)
                emotion_score = 1.0 - (pad_distance / max_pad_distance)
            else:
                emotion_score = 0.5  # Neutral if no emotion query
            
            # Calculate composite relevance score
            relevance_score = (
                time_weight * temporal_score +
                context_weight * context_score +
                emotion_weight * emotion_score
            )
            
            # Boost by importance (slight factor)
            relevance_score *= (0.8 + 0.2 * episode.importance)
            
            scored_episodes.append((relevance_score, episode))
        
        # Sort by relevance score (highest first)
        scored_episodes.sort(key=lambda x: (-x[0], -x[1].timestamp))
        
        # Return top results, recording access
        results = []
        for _, episode in scored_episodes[:max_results]:
            episode.record_access()
            self._total_accessed += 1
            results.append(episode)
        
        return results

    def _get_max_time_difference(self, reference_time: float) -> float:
        """
        Get the maximum time difference between reference time and any episode.
        
        Args:
            reference_time: The reference timestamp.
            
        Returns:
            Maximum absolute time difference.
        """
        if not self._episodes:
            return 1.0  # Avoid division by zero
        
        max_diff = 0.0
        for episode in self._episodes.values():
            diff = abs(episode.timestamp - reference_time)
            if diff > max_diff:
                max_diff = diff
        
        return max_diff if max_diff > 0 else 1.0

    def _prune(self) -> int:
        """
        Remove low-importance episodes to maintain size limit.
        
        Uses importance-weighted pruning to retain the most valuable
        episodes. Episodes with lower importance and fewer accesses
        are more likely to be removed.
        
        Returns:
            Number of episodes removed.
            
        Validates: Requirements 3.6
        """
        current_size = len(self._episodes)
        if current_size == 0:
            return 0
        
        # Calculate target size after pruning
        target_size = int(self._config.max_episodes * (1 - self._config.prune_ratio))
        num_to_remove = max(0, current_size - target_size)
        
        if num_to_remove == 0:
            return 0
        
        # Calculate composite score for each episode
        # Higher score = more valuable = less likely to be removed
        def episode_value(ep: Episode) -> float:
            # Combine importance, access frequency, and recency
            recency_factor = 1.0
            if ep.last_accessed is not None:
                age = time.time() - ep.last_accessed
                recency_factor = 1.0 / (1.0 + age / 3600)  # Decay over hours
            
            access_factor = min(1.0, ep.access_count / 10.0)  # Cap at 10 accesses
            
            return (
                ep.importance * 0.5 +
                access_factor * 0.3 +
                recency_factor * 0.2
            )
        
        # Sort episodes by value (ascending) - lowest value first
        sorted_episodes = sorted(
            self._episodes.values(),
            key=lambda e: (episode_value(e), e.timestamp)
        )
        
        # Remove lowest value episodes
        removed_count = 0
        for ep in sorted_episodes[:num_to_remove]:
            del self._episodes[ep.episode_id]
            removed_count += 1
        
        self._total_pruned += removed_count
        return removed_count

    def consolidate(self) -> Dict[str, Any]:
        """
        Consolidate episodes by extracting patterns for semantic memory.
        
        This method processes unconsolidated episodes to extract:
        - Common context patterns (frequently occurring context keys/values)
        - Event sequence patterns (common event types and sequences)
        - Emotional state patterns (emotional trends and correlations)
        
        After pattern extraction, episodes are marked as consolidated.
        
        Returns:
            Dictionary containing:
            - extracted_patterns: Dict with context_patterns, event_patterns, emotional_patterns
            - consolidated_count: Number of episodes consolidated
            - statistics: Consolidation statistics
            
        Validates: Requirements 3.5
        """
        unconsolidated = self.get_unconsolidated_episodes()
        
        if not unconsolidated:
            return {
                "extracted_patterns": {
                    "context_patterns": [],
                    "event_patterns": [],
                    "emotional_patterns": [],
                },
                "consolidated_count": 0,
                "statistics": {
                    "total_episodes_processed": 0,
                    "unique_contexts": 0,
                    "unique_event_types": 0,
                },
            }
        
        # Extract context patterns
        context_patterns = self._extract_context_patterns(unconsolidated)
        
        # Extract event patterns
        event_patterns = self._extract_event_patterns(unconsolidated)
        
        # Extract emotional patterns
        emotional_patterns = self._extract_emotional_patterns(unconsolidated)
        
        # Mark episodes as consolidated
        consolidated_ids = []
        for episode in unconsolidated:
            episode.consolidated = True
            # Add consolidation metadata
            episode.metadata["consolidated_at"] = time.time()
            consolidated_ids.append(episode.episode_id)
        
        # Reset consolidation counter
        self._episodes_since_consolidation = 0
        
        return {
            "extracted_patterns": {
                "context_patterns": context_patterns,
                "event_patterns": event_patterns,
                "emotional_patterns": emotional_patterns,
            },
            "consolidated_count": len(consolidated_ids),
            "consolidated_episode_ids": consolidated_ids,
            "statistics": {
                "total_episodes_processed": len(unconsolidated),
                "unique_contexts": len(context_patterns),
                "unique_event_types": len(event_patterns),
                "emotional_clusters": len(emotional_patterns),
            },
        }

    def _extract_context_patterns(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """
        Extract common context patterns from episodes.
        
        Groups episodes by similar context keys and values to identify
        frequently occurring patterns.
        
        Args:
            episodes: List of episodes to analyze.
            
        Returns:
            List of context patterns with frequency information.
        """
        if not episodes:
            return []
        
        # Count context key occurrences
        key_counts: Dict[str, int] = {}
        key_value_counts: Dict[str, Dict[str, int]] = {}
        
        for episode in episodes:
            for key, value in episode.context.items():
                # Count key occurrences
                key_counts[key] = key_counts.get(key, 0) + 1
                
                # Count key-value pairs (for hashable values)
                if key not in key_value_counts:
                    key_value_counts[key] = {}
                
                # Convert value to string for counting
                value_str = str(value) if not isinstance(value, (str, int, float, bool)) else value
                value_key = str(value_str)
                key_value_counts[key][value_key] = key_value_counts[key].get(value_key, 0) + 1
        
        # Build patterns from frequent keys
        patterns = []
        total_episodes = len(episodes)
        
        for key, count in sorted(key_counts.items(), key=lambda x: -x[1]):
            frequency = count / total_episodes
            
            # Only include patterns that appear in at least 20% of episodes
            if frequency >= 0.2:
                # Find most common values for this key
                value_counts = key_value_counts.get(key, {})
                common_values = sorted(
                    value_counts.items(),
                    key=lambda x: -x[1]
                )[:5]  # Top 5 values
                
                patterns.append({
                    "key": key,
                    "frequency": frequency,
                    "occurrence_count": count,
                    "common_values": [
                        {"value": v, "count": c}
                        for v, c in common_values
                    ],
                })
        
        return patterns

    def _extract_event_patterns(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """
        Extract common event patterns from episodes.
        
        Identifies frequently occurring event types and sequences.
        
        Args:
            episodes: List of episodes to analyze.
            
        Returns:
            List of event patterns with frequency information.
        """
        if not episodes:
            return []
        
        # Count event type occurrences
        event_type_counts: Dict[str, int] = {}
        event_sequences: Dict[str, int] = {}
        
        for episode in episodes:
            # Track event types
            event_types_in_episode = []
            for event in episode.events:
                if isinstance(event, dict):
                    event_type = event.get("type", event.get("name", "unknown"))
                    event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
                    event_types_in_episode.append(event_type)
            
            # Track event sequences (pairs of consecutive events)
            for i in range(len(event_types_in_episode) - 1):
                sequence = f"{event_types_in_episode[i]} -> {event_types_in_episode[i+1]}"
                event_sequences[sequence] = event_sequences.get(sequence, 0) + 1
        
        # Build patterns
        patterns = []
        total_episodes = len(episodes)
        
        # Add event type patterns
        for event_type, count in sorted(event_type_counts.items(), key=lambda x: -x[1]):
            frequency = count / total_episodes
            if frequency >= 0.1:  # At least 10% occurrence
                patterns.append({
                    "pattern_type": "event_type",
                    "event_type": event_type,
                    "frequency": frequency,
                    "occurrence_count": count,
                })
        
        # Add sequence patterns
        for sequence, count in sorted(event_sequences.items(), key=lambda x: -x[1]):
            if count >= 2:  # At least 2 occurrences
                patterns.append({
                    "pattern_type": "sequence",
                    "sequence": sequence,
                    "occurrence_count": count,
                })
        
        return patterns

    def _extract_emotional_patterns(self, episodes: List[Episode]) -> List[Dict[str, Any]]:
        """
        Extract emotional state patterns from episodes.
        
        Identifies emotional trends and clusters in PAD space.
        
        Args:
            episodes: List of episodes to analyze.
            
        Returns:
            List of emotional patterns with statistics.
        """
        if not episodes:
            return []
        
        # Collect PAD values
        pleasures = []
        arousals = []
        dominances = []
        
        for episode in episodes:
            emotional_state = episode.emotional_state
            if emotional_state:
                pleasures.append(emotional_state.get("pleasure", 0.0))
                arousals.append(emotional_state.get("arousal", 0.5))
                dominances.append(emotional_state.get("dominance", 0.0))
        
        if not pleasures:
            return []
        
        # Calculate statistics
        def calc_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            
            mean = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            
            # Calculate standard deviation
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = math.sqrt(variance)
            
            return {
                "mean": mean,
                "min": min_val,
                "max": max_val,
                "std": std,
            }
        
        patterns = []
        
        # Overall emotional statistics
        patterns.append({
            "pattern_type": "overall_statistics",
            "pleasure": calc_stats(pleasures),
            "arousal": calc_stats(arousals),
            "dominance": calc_stats(dominances),
            "sample_count": len(pleasures),
        })
        
        # Identify emotional clusters (simple binning approach)
        # Positive vs negative pleasure
        positive_pleasure_count = sum(1 for p in pleasures if p > 0.2)
        negative_pleasure_count = sum(1 for p in pleasures if p < -0.2)
        neutral_pleasure_count = len(pleasures) - positive_pleasure_count - negative_pleasure_count
        
        patterns.append({
            "pattern_type": "pleasure_distribution",
            "positive_count": positive_pleasure_count,
            "negative_count": negative_pleasure_count,
            "neutral_count": neutral_pleasure_count,
            "positive_ratio": positive_pleasure_count / len(pleasures) if pleasures else 0,
            "negative_ratio": negative_pleasure_count / len(pleasures) if pleasures else 0,
        })
        
        # High vs low arousal
        high_arousal_count = sum(1 for a in arousals if a > 0.6)
        low_arousal_count = sum(1 for a in arousals if a < 0.4)
        
        patterns.append({
            "pattern_type": "arousal_distribution",
            "high_count": high_arousal_count,
            "low_count": low_arousal_count,
            "high_ratio": high_arousal_count / len(arousals) if arousals else 0,
            "low_ratio": low_arousal_count / len(arousals) if arousals else 0,
        })
        
        return patterns

    def apply_decay(self, time_elapsed: float) -> int:
        """
        Apply importance decay to all episodes based on time elapsed.
        
        Uses the configured decay_rate to reduce importance of episodes
        over time. The decay follows an exponential model:
        new_importance = importance * (decay_rate ^ time_elapsed)
        
        Args:
            time_elapsed: Time elapsed in arbitrary units (e.g., hours, days).
                         The decay_rate is applied per unit of time.
            
        Returns:
            Number of episodes affected (with importance changed).
            
        Validates: Requirements 3.6
        """
        if time_elapsed <= 0:
            return 0
        
        if not self._episodes:
            return 0
        
        affected_count = 0
        decay_factor = self._config.decay_rate ** time_elapsed
        
        for episode in self._episodes.values():
            old_importance = episode.importance
            new_importance = old_importance * decay_factor
            
            # Clamp to valid range
            new_importance = max(0.0, min(1.0, new_importance))
            
            if new_importance != old_importance:
                episode.importance = new_importance
                affected_count += 1
        
        return affected_count

    def prune_by_importance(self, threshold: float) -> int:
        """
        Remove episodes below the importance threshold.
        
        Preserves frequently accessed episodes (high access_count) even if
        their importance is below the threshold, as frequent access indicates
        ongoing relevance.
        
        Args:
            threshold: Minimum importance score to retain (0.0 to 1.0).
            
        Returns:
            Number of episodes removed.
            
        Validates: Requirements 3.6
        """
        if not self._episodes:
            return 0
        
        # Clamp threshold to valid range
        threshold = max(0.0, min(1.0, threshold))
        
        # Identify episodes to remove
        # Preserve episodes with high access count (frequently accessed)
        # Use access_count >= 3 as threshold for "frequently accessed"
        frequent_access_threshold = 3
        
        episodes_to_remove = []
        for episode_id, episode in self._episodes.items():
            if episode.importance < threshold:
                # Check if frequently accessed - preserve if so
                if episode.access_count < frequent_access_threshold:
                    episodes_to_remove.append(episode_id)
        
        # Remove identified episodes
        for episode_id in episodes_to_remove:
            del self._episodes[episode_id]
        
        removed_count = len(episodes_to_remove)
        self._total_pruned += removed_count
        
        return removed_count

    def get_frequently_accessed(
        self,
        min_access_count: int,
        max_results: int = 10,
    ) -> List[Episode]:
        """
        Return episodes that have been accessed at least min_access_count times.
        
        Results are sorted by access count (highest first).
        
        Args:
            min_access_count: Minimum number of accesses required.
            max_results: Maximum number of episodes to return.
            
        Returns:
            List of frequently accessed episodes, sorted by access count descending.
            
        Validates: Requirements 3.6
        """
        if max_results <= 0:
            return []
        
        if min_access_count < 0:
            min_access_count = 0
        
        # Filter episodes by access count
        frequent_episodes = [
            ep for ep in self._episodes.values()
            if ep.access_count >= min_access_count
        ]
        
        # Sort by access count (highest first), then by timestamp (most recent first)
        frequent_episodes.sort(
            key=lambda e: (-e.access_count, -e.timestamp)
        )
        
        return frequent_episodes[:max_results]

    def get_recently_accessed(
        self,
        time_window: float,
        max_results: int = 10,
    ) -> List[Episode]:
        """
        Return episodes accessed within the time window.
        
        Results are sorted by last_accessed (most recent first).
        
        Args:
            time_window: Time window in seconds from current time.
            max_results: Maximum number of episodes to return.
            
        Returns:
            List of recently accessed episodes, sorted by last_accessed descending.
            
        Validates: Requirements 3.6
        """
        if max_results <= 0:
            return []
        
        if time_window < 0:
            time_window = abs(time_window)
        
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        # Filter episodes by last_accessed time
        recent_episodes = [
            ep for ep in self._episodes.values()
            if ep.last_accessed is not None and ep.last_accessed >= cutoff_time
        ]
        
        # Sort by last_accessed (most recent first)
        recent_episodes.sort(
            key=lambda e: -e.last_accessed if e.last_accessed is not None else 0
        )
        
        return recent_episodes[:max_results]

    def clear(self) -> None:
        """Clear all episodes from memory."""
        self._episodes.clear()
        self._episodes_since_consolidation = 0

    def __len__(self) -> int:
        """Return the number of episodes in memory."""
        return len(self._episodes)

    def __contains__(self, episode_id: str) -> bool:
        """Check if an episode ID is in memory."""
        return episode_id in self._episodes

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the episodic memory system."""
        return {
            "episode_count": len(self._episodes),
            "max_episodes": self._config.max_episodes,
            "total_created": self._total_created,
            "total_accessed": self._total_accessed,
            "total_pruned": self._total_pruned,
            "episodes_since_consolidation": self._episodes_since_consolidation,
            "unconsolidated_count": len(self.get_unconsolidated_episodes()),
            "uptime": time.time() - self._initialized_at,
            "config": self._config.to_dict(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the memory."""
        if not self._episodes:
            return {
                "count": 0,
                "avg_importance": 0.0,
                "min_importance": 0.0,
                "max_importance": 0.0,
                "avg_access_count": 0.0,
                "consolidated_count": 0,
                "unconsolidated_count": 0,
            }
        
        importances = [ep.importance for ep in self._episodes.values()]
        access_counts = [ep.access_count for ep in self._episodes.values()]
        consolidated = sum(1 for ep in self._episodes.values() if ep.consolidated)
        
        return {
            "count": len(self._episodes),
            "avg_importance": sum(importances) / len(importances),
            "min_importance": min(importances),
            "max_importance": max(importances),
            "avg_access_count": sum(access_counts) / len(access_counts),
            "consolidated_count": consolidated,
            "unconsolidated_count": len(self._episodes) - consolidated,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the episodic memory to a dictionary."""
        return {
            "config": self._config.to_dict(),
            "episodes": {
                ep_id: ep.to_dict()
                for ep_id, ep in self._episodes.items()
            },
            "total_created": self._total_created,
            "total_accessed": self._total_accessed,
            "total_pruned": self._total_pruned,
            "episodes_since_consolidation": self._episodes_since_consolidation,
            "initialized_at": self._initialized_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        """Create an episodic memory from a dictionary."""
        config = data.get("config", {})
        memory = cls(config=config)
        
        # Restore episodes
        episodes_data = data.get("episodes", {})
        for ep_id, ep_data in episodes_data.items():
            ep = Episode.from_dict(ep_data)
            memory._episodes[ep_id] = ep
        
        # Restore statistics
        memory._total_created = data.get("total_created", len(memory._episodes))
        memory._total_accessed = data.get("total_accessed", 0)
        memory._total_pruned = data.get("total_pruned", 0)
        memory._episodes_since_consolidation = data.get("episodes_since_consolidation", 0)
        memory._initialized_at = data.get("initialized_at", time.time())
        
        return memory


def create_episode(
    context: Dict[str, Any],
    events: List[Dict[str, Any]],
    emotional_state: Dict[str, float],
    importance: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Episode:
    """
    Factory function to create an Episode with auto-generated ID and timestamp.
    
    This is a convenience function for creating episodes without needing
    an EpisodicMemory instance.
    
    Args:
        context: Environmental context at the time of the episode.
        events: Sequence of events that occurred during the episode.
        emotional_state: PAD values at time of episode.
        importance: Significance score (0.0-1.0). Defaults to 0.5.
        metadata: Additional information about the episode.
        
    Returns:
        A new Episode instance.
        
    Validates: Requirements 3.1
    """
    return Episode(
        episode_id=str(uuid.uuid4()),
        timestamp=time.time(),
        context=context if context is not None else {},
        events=events if events is not None else [],
        emotional_state=emotional_state if emotional_state is not None else {},
        importance=importance if importance is not None else 0.5,
        access_count=0,
        last_accessed=None,
        consolidated=False,
        metadata=metadata if metadata is not None else {},
    )

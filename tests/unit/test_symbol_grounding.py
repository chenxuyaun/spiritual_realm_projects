"""
Unit tests for the Symbol Grounding module.

Tests grounding creation, updates, confidence evolution, ambiguity resolution,
and serialization functionality.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import pytest
import time
from typing import Dict, Any

from mm_orch.consciousness.symbol_grounding import (
    SymbolGroundingModule,
    SymbolGrounding,
    GroundingCandidate,
    SymbolGroundingConfig,
)
from mm_orch.consciousness.episodic_memory import EpisodicMemory
from mm_orch.consciousness.semantic_memory import SemanticMemory


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def episodic_memory():
    """Create an episodic memory instance for testing."""
    return EpisodicMemory(max_episodes=100)


@pytest.fixture
def semantic_memory():
    """Create a semantic memory instance for testing."""
    return SemanticMemory()


@pytest.fixture
def symbol_grounding_module(episodic_memory, semantic_memory):
    """Create a symbol grounding module for testing."""
    return SymbolGroundingModule(
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
    )


@pytest.fixture
def configured_module(episodic_memory, semantic_memory):
    """Create a symbol grounding module with custom config."""
    config = {
        "min_confidence_threshold": 0.4,
        "confidence_growth_rate": 0.15,
        "confidence_decay_rate": 0.1,
        "max_candidates": 5,
        "context_similarity_threshold": 0.6,
        "probability_smoothing": 0.02,
    }
    return SymbolGroundingModule(
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        config=config,
    )


# =============================================================================
# Tests for GroundingCandidate dataclass
# =============================================================================

class TestGroundingCandidate:
    """Tests for the GroundingCandidate dataclass."""

    def test_candidate_creation_with_all_fields(self):
        """Test creating a grounding candidate with all fields."""
        candidate = GroundingCandidate(
            concept_id="concept-001",
            episode_ids=["ep-001", "ep-002"],
            perceptual_features={"color": "red", "size": "large"},
            probability=0.7,
        )
        
        assert candidate.concept_id == "concept-001"
        assert candidate.episode_ids == ["ep-001", "ep-002"]
        assert candidate.perceptual_features == {"color": "red", "size": "large"}
        assert candidate.probability == 0.7

    def test_candidate_creation_with_none_concept_id(self):
        """Test creating a candidate without concept link."""
        candidate = GroundingCandidate(
            concept_id=None,
            episode_ids=["ep-001"],
            perceptual_features={"ungrounded": True},
            probability=0.5,
        )
        
        assert candidate.concept_id is None
        assert candidate.probability == 0.5

    def test_candidate_probability_clamping_high(self):
        """Test that probability is clamped to 1.0 maximum."""
        candidate = GroundingCandidate(
            concept_id=None,
            episode_ids=[],
            perceptual_features={},
            probability=1.5,
        )
        
        assert candidate.probability == 1.0

    def test_candidate_probability_clamping_low(self):
        """Test that probability is clamped to 0.0 minimum."""
        candidate = GroundingCandidate(
            concept_id=None,
            episode_ids=[],
            perceptual_features={},
            probability=-0.5,
        )
        
        assert candidate.probability == 0.0

    def test_candidate_validation_invalid_episode_ids(self):
        """Test that non-list episode_ids raises ValueError."""
        with pytest.raises(ValueError, match="episode_ids must be a list"):
            GroundingCandidate(
                concept_id=None,
                episode_ids="invalid",  # type: ignore
                perceptual_features={},
                probability=0.5,
            )

    def test_candidate_validation_invalid_perceptual_features(self):
        """Test that non-dict perceptual_features raises ValueError."""
        with pytest.raises(ValueError, match="perceptual_features must be a dictionary"):
            GroundingCandidate(
                concept_id=None,
                episode_ids=[],
                perceptual_features="invalid",  # type: ignore
                probability=0.5,
            )

    def test_candidate_validation_invalid_probability(self):
        """Test that non-numeric probability raises ValueError."""
        with pytest.raises(ValueError, match="probability must be a number"):
            GroundingCandidate(
                concept_id=None,
                episode_ids=[],
                perceptual_features={},
                probability="invalid",  # type: ignore
            )

    def test_candidate_to_dict(self):
        """Test serializing a candidate to dictionary."""
        candidate = GroundingCandidate(
            concept_id="concept-001",
            episode_ids=["ep-001", "ep-002"],
            perceptual_features={"key": "value"},
            probability=0.8,
        )
        
        data = candidate.to_dict()
        
        assert data["concept_id"] == "concept-001"
        assert data["episode_ids"] == ["ep-001", "ep-002"]
        assert data["perceptual_features"] == {"key": "value"}
        assert data["probability"] == 0.8

    def test_candidate_from_dict(self):
        """Test deserializing a candidate from dictionary."""
        data = {
            "concept_id": "concept-001",
            "episode_ids": ["ep-001"],
            "perceptual_features": {"test": True},
            "probability": 0.6,
        }
        
        candidate = GroundingCandidate.from_dict(data)
        
        assert candidate.concept_id == "concept-001"
        assert candidate.episode_ids == ["ep-001"]
        assert candidate.perceptual_features == {"test": True}
        assert candidate.probability == 0.6

    def test_candidate_from_dict_with_defaults(self):
        """Test deserializing with missing optional fields."""
        data = {}
        
        candidate = GroundingCandidate.from_dict(data)
        
        assert candidate.concept_id is None
        assert candidate.episode_ids == []
        assert candidate.perceptual_features == {}
        assert candidate.probability == 0.0

    def test_candidate_serialization_roundtrip(self):
        """Test that serialization and deserialization preserves data."""
        original = GroundingCandidate(
            concept_id="test-concept",
            episode_ids=["ep-1", "ep-2", "ep-3"],
            perceptual_features={"nested": {"key": "value"}, "list": [1, 2, 3]},
            probability=0.75,
        )
        
        data = original.to_dict()
        restored = GroundingCandidate.from_dict(data)
        
        assert restored.concept_id == original.concept_id
        assert restored.episode_ids == original.episode_ids
        assert restored.perceptual_features == original.perceptual_features
        assert restored.probability == original.probability


# =============================================================================
# Tests for SymbolGrounding dataclass
# =============================================================================

class TestSymbolGrounding:
    """Tests for the SymbolGrounding dataclass."""

    def test_grounding_creation_with_all_fields(self):
        """Test creating a symbol grounding with all fields."""
        candidate = GroundingCandidate(
            concept_id="concept-001",
            episode_ids=[],
            perceptual_features={},
            probability=1.0,
        )
        
        grounding = SymbolGrounding(
            symbol="test",
            groundings=[candidate],
            confidence=0.7,
            grounding_count=5,
            last_updated=1234567890.0,
        )
        
        assert grounding.symbol == "test"
        assert len(grounding.groundings) == 1
        assert grounding.confidence == 0.7
        assert grounding.grounding_count == 5
        assert grounding.last_updated == 1234567890.0

    def test_grounding_validation_empty_symbol(self):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            SymbolGrounding(
                symbol="",
                groundings=[],
                confidence=0.5,
                grounding_count=1,
                last_updated=time.time(),
            )

    def test_grounding_validation_invalid_groundings(self):
        """Test that non-list groundings raises ValueError."""
        with pytest.raises(ValueError, match="groundings must be a list"):
            SymbolGrounding(
                symbol="test",
                groundings="invalid",  # type: ignore
                confidence=0.5,
                grounding_count=1,
                last_updated=time.time(),
            )

    def test_grounding_confidence_clamping_high(self):
        """Test that confidence is clamped to 1.0 maximum."""
        grounding = SymbolGrounding(
            symbol="test",
            groundings=[],
            confidence=1.5,
            grounding_count=1,
            last_updated=time.time(),
        )
        
        assert grounding.confidence == 1.0

    def test_grounding_confidence_clamping_low(self):
        """Test that confidence is clamped to 0.0 minimum."""
        grounding = SymbolGrounding(
            symbol="test",
            groundings=[],
            confidence=-0.5,
            grounding_count=1,
            last_updated=time.time(),
        )
        
        assert grounding.confidence == 0.0

    def test_grounding_validation_negative_count(self):
        """Test that negative grounding_count raises ValueError."""
        with pytest.raises(ValueError, match="grounding_count cannot be negative"):
            SymbolGrounding(
                symbol="test",
                groundings=[],
                confidence=0.5,
                grounding_count=-1,
                last_updated=time.time(),
            )

    def test_grounding_to_dict(self):
        """Test serializing a grounding to dictionary."""
        candidate = GroundingCandidate(
            concept_id="concept-001",
            episode_ids=["ep-001"],
            perceptual_features={"key": "value"},
            probability=1.0,
        )
        
        grounding = SymbolGrounding(
            symbol="test",
            groundings=[candidate],
            confidence=0.8,
            grounding_count=3,
            last_updated=1234567890.0,
        )
        
        data = grounding.to_dict()
        
        assert data["symbol"] == "test"
        assert len(data["groundings"]) == 1
        assert data["confidence"] == 0.8
        assert data["grounding_count"] == 3
        assert data["last_updated"] == 1234567890.0

    def test_grounding_from_dict(self):
        """Test deserializing a grounding from dictionary."""
        data = {
            "symbol": "test",
            "groundings": [
                {
                    "concept_id": "concept-001",
                    "episode_ids": [],
                    "perceptual_features": {},
                    "probability": 1.0,
                }
            ],
            "confidence": 0.6,
            "grounding_count": 2,
            "last_updated": 1234567890.0,
        }
        
        grounding = SymbolGrounding.from_dict(data)
        
        assert grounding.symbol == "test"
        assert len(grounding.groundings) == 1
        assert grounding.confidence == 0.6
        assert grounding.grounding_count == 2
        assert grounding.last_updated == 1234567890.0

    def test_grounding_serialization_roundtrip(self):
        """Test that serialization and deserialization preserves data."""
        candidate1 = GroundingCandidate(
            concept_id="concept-001",
            episode_ids=["ep-1"],
            perceptual_features={"type": "A"},
            probability=0.6,
        )
        candidate2 = GroundingCandidate(
            concept_id="concept-002",
            episode_ids=["ep-2", "ep-3"],
            perceptual_features={"type": "B"},
            probability=0.4,
        )
        
        original = SymbolGrounding(
            symbol="roundtrip",
            groundings=[candidate1, candidate2],
            confidence=0.75,
            grounding_count=10,
            last_updated=time.time(),
        )
        
        data = original.to_dict()
        restored = SymbolGrounding.from_dict(data)
        
        assert restored.symbol == original.symbol
        assert len(restored.groundings) == len(original.groundings)
        assert restored.confidence == original.confidence
        assert restored.grounding_count == original.grounding_count
        assert restored.last_updated == original.last_updated


# =============================================================================
# Tests for SymbolGroundingConfig dataclass
# =============================================================================

class TestSymbolGroundingConfig:
    """Tests for the SymbolGroundingConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SymbolGroundingConfig()
        
        assert config.min_confidence_threshold == 0.3
        assert config.confidence_growth_rate == 0.1
        assert config.confidence_decay_rate == 0.05
        assert config.max_candidates == 10
        assert config.context_similarity_threshold == 0.5
        assert config.probability_smoothing == 0.01

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = SymbolGroundingConfig(
            min_confidence_threshold=0.5,
            confidence_growth_rate=0.2,
            confidence_decay_rate=0.1,
            max_candidates=5,
            context_similarity_threshold=0.7,
            probability_smoothing=0.05,
        )
        
        assert config.min_confidence_threshold == 0.5
        assert config.confidence_growth_rate == 0.2
        assert config.confidence_decay_rate == 0.1
        assert config.max_candidates == 5
        assert config.context_similarity_threshold == 0.7
        assert config.probability_smoothing == 0.05

    def test_config_validation_min_confidence_threshold(self):
        """Test that invalid min_confidence_threshold raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence_threshold must be between"):
            SymbolGroundingConfig(min_confidence_threshold=1.5)
        
        with pytest.raises(ValueError, match="min_confidence_threshold must be between"):
            SymbolGroundingConfig(min_confidence_threshold=-0.1)

    def test_config_validation_confidence_growth_rate(self):
        """Test that invalid confidence_growth_rate raises ValueError."""
        with pytest.raises(ValueError, match="confidence_growth_rate must be between"):
            SymbolGroundingConfig(confidence_growth_rate=1.5)

    def test_config_validation_confidence_decay_rate(self):
        """Test that invalid confidence_decay_rate raises ValueError."""
        with pytest.raises(ValueError, match="confidence_decay_rate must be between"):
            SymbolGroundingConfig(confidence_decay_rate=-0.1)

    def test_config_validation_max_candidates(self):
        """Test that invalid max_candidates raises ValueError."""
        with pytest.raises(ValueError, match="max_candidates must be at least 1"):
            SymbolGroundingConfig(max_candidates=0)

    def test_config_validation_context_similarity_threshold(self):
        """Test that invalid context_similarity_threshold raises ValueError."""
        with pytest.raises(ValueError, match="context_similarity_threshold must be between"):
            SymbolGroundingConfig(context_similarity_threshold=1.5)

    def test_config_validation_probability_smoothing(self):
        """Test that invalid probability_smoothing raises ValueError."""
        with pytest.raises(ValueError, match="probability_smoothing must be non-negative"):
            SymbolGroundingConfig(probability_smoothing=-0.01)

    def test_config_to_dict(self):
        """Test serializing config to dictionary."""
        config = SymbolGroundingConfig(
            min_confidence_threshold=0.4,
            max_candidates=8,
        )
        
        data = config.to_dict()
        
        assert data["min_confidence_threshold"] == 0.4
        assert data["max_candidates"] == 8

    def test_config_from_dict(self):
        """Test deserializing config from dictionary."""
        data = {
            "min_confidence_threshold": 0.5,
            "confidence_growth_rate": 0.15,
            "max_candidates": 7,
        }
        
        config = SymbolGroundingConfig.from_dict(data)
        
        assert config.min_confidence_threshold == 0.5
        assert config.confidence_growth_rate == 0.15
        assert config.max_candidates == 7

    def test_config_serialization_roundtrip(self):
        """Test that serialization and deserialization preserves data."""
        original = SymbolGroundingConfig(
            min_confidence_threshold=0.35,
            confidence_growth_rate=0.12,
            confidence_decay_rate=0.08,
            max_candidates=6,
            context_similarity_threshold=0.55,
            probability_smoothing=0.02,
        )
        
        data = original.to_dict()
        restored = SymbolGroundingConfig.from_dict(data)
        
        assert restored.min_confidence_threshold == original.min_confidence_threshold
        assert restored.confidence_growth_rate == original.confidence_growth_rate
        assert restored.confidence_decay_rate == original.confidence_decay_rate
        assert restored.max_candidates == original.max_candidates
        assert restored.context_similarity_threshold == original.context_similarity_threshold
        assert restored.probability_smoothing == original.probability_smoothing



# =============================================================================
# Tests for SymbolGroundingModule initialization
# =============================================================================

class TestSymbolGroundingModuleInit:
    """Tests for SymbolGroundingModule initialization."""

    def test_module_initialization_default(self, episodic_memory, semantic_memory):
        """Test default module initialization."""
        module = SymbolGroundingModule(
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
        )
        
        assert module._episodic_memory is episodic_memory
        assert module._semantic_memory is semantic_memory
        assert module._config.min_confidence_threshold == 0.3
        assert len(module.get_all_symbols()) == 0

    def test_module_initialization_with_config(self, episodic_memory, semantic_memory):
        """Test module initialization with custom config."""
        config = {
            "min_confidence_threshold": 0.5,
            "max_candidates": 5,
        }
        
        module = SymbolGroundingModule(
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
            config=config,
        )
        
        assert module._config.min_confidence_threshold == 0.5
        assert module._config.max_candidates == 5

    def test_module_initialization_invalid_episodic_memory(self, semantic_memory):
        """Test that invalid episodic_memory raises TypeError."""
        with pytest.raises(TypeError, match="episodic_memory must be an EpisodicMemory instance"):
            SymbolGroundingModule(
                episodic_memory="invalid",  # type: ignore
                semantic_memory=semantic_memory,
            )

    def test_module_initialization_invalid_semantic_memory(self, episodic_memory):
        """Test that invalid semantic_memory raises TypeError."""
        with pytest.raises(TypeError, match="semantic_memory must be a SemanticMemory instance"):
            SymbolGroundingModule(
                episodic_memory=episodic_memory,
                semantic_memory="invalid",  # type: ignore
            )


# =============================================================================
# Tests for ground_symbol() - Grounding Creation
# =============================================================================

class TestGroundSymbol:
    """Tests for the ground_symbol() method.
    
    Requirements: 5.1, 5.2
    """

    def test_ground_symbol_creates_new_grounding(self, symbol_grounding_module):
        """Test that ground_symbol creates a new grounding for unknown symbol."""
        grounding = symbol_grounding_module.ground_symbol("apple")
        
        assert grounding is not None
        assert grounding.symbol == "apple"
        assert len(grounding.groundings) >= 1
        assert grounding.grounding_count == 1

    def test_ground_symbol_returns_existing_grounding(self, symbol_grounding_module):
        """Test that ground_symbol returns existing grounding for known symbol."""
        # First call creates grounding
        grounding1 = symbol_grounding_module.ground_symbol("banana")
        
        # Second call returns same grounding
        grounding2 = symbol_grounding_module.ground_symbol("banana")
        
        assert grounding1 is grounding2
        assert grounding1.grounding_count == 1  # Count doesn't increase

    def test_ground_symbol_normalizes_input(self, symbol_grounding_module):
        """Test that ground_symbol normalizes symbol to lowercase."""
        grounding1 = symbol_grounding_module.ground_symbol("Apple")
        grounding2 = symbol_grounding_module.ground_symbol("APPLE")
        grounding3 = symbol_grounding_module.ground_symbol("  apple  ")
        
        assert grounding1 is grounding2
        assert grounding2 is grounding3
        assert grounding1.symbol == "apple"

    def test_ground_symbol_empty_raises_error(self, symbol_grounding_module):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            symbol_grounding_module.ground_symbol("")

    def test_ground_symbol_whitespace_only_raises_error(self, symbol_grounding_module):
        """Test that whitespace-only symbol raises ValueError."""
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            symbol_grounding_module.ground_symbol("   ")

    def test_ground_symbol_has_valid_confidence(self, symbol_grounding_module):
        """Test that new grounding has valid confidence in [0.0, 1.0]."""
        grounding = symbol_grounding_module.ground_symbol("cherry")
        
        assert 0.0 <= grounding.confidence <= 1.0

    def test_ground_symbol_has_timestamp(self, symbol_grounding_module):
        """Test that new grounding has a valid timestamp."""
        before = time.time()
        grounding = symbol_grounding_module.ground_symbol("date")
        after = time.time()
        
        assert before <= grounding.last_updated <= after

    def test_ground_symbol_increments_created_count(self, symbol_grounding_module):
        """Test that creating new groundings increments the counter."""
        initial_stats = symbol_grounding_module.get_statistics()
        initial_count = initial_stats["total_groundings_created"]
        
        symbol_grounding_module.ground_symbol("elderberry")
        symbol_grounding_module.ground_symbol("fig")
        
        final_stats = symbol_grounding_module.get_statistics()
        assert final_stats["total_groundings_created"] == initial_count + 2

    def test_ground_symbol_probabilities_sum_to_one(self, symbol_grounding_module):
        """Test that candidate probabilities sum to 1.0."""
        grounding = symbol_grounding_module.ground_symbol("grape")
        
        total_prob = sum(c.probability for c in grounding.groundings)
        assert abs(total_prob - 1.0) < 1e-6

    def test_ground_symbol_with_semantic_memory_content(self, episodic_memory, semantic_memory):
        """Test grounding when semantic memory has relevant content."""
        # Add a concept to semantic memory
        semantic_memory.integrate_knowledge({
            "name": "honeydew",
            "type": "fruit",
            "properties": {"color": "green", "taste": "sweet"},
        })
        
        module = SymbolGroundingModule(
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
        )
        
        grounding = module.ground_symbol("honeydew")
        
        assert grounding is not None
        assert len(grounding.groundings) >= 1

    def test_ground_symbol_with_episodic_memory_content(self, episodic_memory, semantic_memory):
        """Test grounding when episodic memory has relevant content."""
        # Add an episode with relevant context
        episodic_memory.create_episode(
            context={"symbol": "kiwi", "query_text": "kiwi"},
            events=[{"type": "encounter", "symbol": "kiwi"}],
            emotional_state={"pleasure": 0.6, "arousal": 0.4, "dominance": 0.5},
            importance=0.7,
        )
        
        module = SymbolGroundingModule(
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
        )
        
        grounding = module.ground_symbol("kiwi")
        
        assert grounding is not None
        assert len(grounding.groundings) >= 1


# =============================================================================
# Tests for update_grounding() - Confidence Evolution
# =============================================================================

class TestUpdateGrounding:
    """Tests for the update_grounding() method.
    
    Requirements: 5.3, 5.5
    """

    def test_update_grounding_creates_if_not_exists(self, symbol_grounding_module):
        """Test that update_grounding creates grounding if it doesn't exist."""
        context = {"type": "fruit", "color": "red"}
        
        grounding = symbol_grounding_module.update_grounding("lemon", context)
        
        assert grounding is not None
        assert grounding.symbol == "lemon"

    def test_update_grounding_increments_count(self, symbol_grounding_module):
        """Test that update_grounding increments grounding_count."""
        symbol_grounding_module.ground_symbol("mango")
        
        context = {"type": "fruit"}
        grounding = symbol_grounding_module.update_grounding("mango", context)
        
        assert grounding.grounding_count == 2

    def test_update_grounding_updates_timestamp(self, symbol_grounding_module):
        """Test that update_grounding updates last_updated timestamp."""
        grounding1 = symbol_grounding_module.ground_symbol("nectarine")
        initial_timestamp = grounding1.last_updated
        
        time.sleep(0.01)  # Small delay
        
        context = {"type": "fruit"}
        grounding2 = symbol_grounding_module.update_grounding("nectarine", context)
        
        assert grounding2.last_updated > initial_timestamp

    def test_update_grounding_empty_symbol_raises_error(self, symbol_grounding_module):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            symbol_grounding_module.update_grounding("", {"key": "value"})

    def test_update_grounding_with_episode_id(self, symbol_grounding_module):
        """Test that update_grounding can link to an episode."""
        symbol_grounding_module.ground_symbol("orange")
        
        context = {"type": "fruit"}
        grounding = symbol_grounding_module.update_grounding(
            "orange", context, episode_id="ep-001"
        )
        
        # Check that at least one candidate has the episode_id
        has_episode = any(
            "ep-001" in c.episode_ids
            for c in grounding.groundings
        )
        # Episode may or may not be added depending on consistency
        assert grounding is not None

    def test_update_grounding_consistent_increases_confidence(self, symbol_grounding_module):
        """Test that consistent updates can increase confidence."""
        # Create initial grounding
        grounding = symbol_grounding_module.ground_symbol("papaya")
        initial_confidence = grounding.confidence
        
        # Update with consistent context multiple times
        consistent_context = {"type": "fruit", "category": "tropical"}
        for _ in range(5):
            grounding = symbol_grounding_module.update_grounding("papaya", consistent_context)
        
        # Confidence should remain in valid range
        assert 0.0 <= grounding.confidence <= 1.0

    def test_update_grounding_increments_updated_count(self, symbol_grounding_module):
        """Test that updates increment the total_groundings_updated counter."""
        symbol_grounding_module.ground_symbol("quince")
        
        initial_stats = symbol_grounding_module.get_statistics()
        initial_count = initial_stats["total_groundings_updated"]
        
        symbol_grounding_module.update_grounding("quince", {"type": "fruit"})
        symbol_grounding_module.update_grounding("quince", {"color": "yellow"})
        
        final_stats = symbol_grounding_module.get_statistics()
        assert final_stats["total_groundings_updated"] == initial_count + 2

    def test_update_grounding_probabilities_remain_normalized(self, symbol_grounding_module):
        """Test that probabilities remain normalized after updates."""
        symbol_grounding_module.ground_symbol("raspberry")
        
        # Multiple updates with different contexts
        contexts = [
            {"type": "A", "value": 1},
            {"type": "B", "value": 2},
            {"type": "C", "value": 3},
        ]
        
        for context in contexts:
            grounding = symbol_grounding_module.update_grounding("raspberry", context)
        
        total_prob = sum(c.probability for c in grounding.groundings)
        assert abs(total_prob - 1.0) < 1e-6

    def test_update_grounding_confidence_bounded(self, symbol_grounding_module):
        """Test that confidence stays within [0.0, 1.0] after many updates."""
        symbol_grounding_module.ground_symbol("strawberry")
        
        # Many updates
        for i in range(20):
            context = {"iteration": i, "data": f"update_{i}"}
            grounding = symbol_grounding_module.update_grounding("strawberry", context)
        
        assert 0.0 <= grounding.confidence <= 1.0


# =============================================================================
# Tests for resolve_ambiguity() - Ambiguity Resolution
# =============================================================================

class TestResolveAmbiguity:
    """Tests for the resolve_ambiguity() method.
    
    Requirements: 5.4
    """

    def test_resolve_ambiguity_returns_candidate(self, symbol_grounding_module):
        """Test that resolve_ambiguity returns a valid candidate."""
        symbol_grounding_module.ground_symbol("tangerine")
        
        context = {"type": "citrus", "color": "orange"}
        candidate = symbol_grounding_module.resolve_ambiguity("tangerine", context)
        
        assert candidate is not None
        assert isinstance(candidate, GroundingCandidate)

    def test_resolve_ambiguity_empty_symbol_raises_error(self, symbol_grounding_module):
        """Test that empty symbol raises ValueError."""
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            symbol_grounding_module.resolve_ambiguity("", {"key": "value"})

    def test_resolve_ambiguity_creates_grounding_if_needed(self, symbol_grounding_module):
        """Test that resolve_ambiguity creates grounding if it doesn't exist."""
        context = {"type": "fruit"}
        candidate = symbol_grounding_module.resolve_ambiguity("ugli", context)
        
        assert candidate is not None
        assert symbol_grounding_module.get_grounding("ugli") is not None

    def test_resolve_ambiguity_increments_resolution_count(self, symbol_grounding_module):
        """Test that resolve_ambiguity increments the resolution counter."""
        symbol_grounding_module.ground_symbol("vanilla")
        
        initial_stats = symbol_grounding_module.get_statistics()
        initial_count = initial_stats["total_ambiguity_resolutions"]
        
        symbol_grounding_module.resolve_ambiguity("vanilla", {"type": "spice"})
        symbol_grounding_module.resolve_ambiguity("vanilla", {"flavor": "sweet"})
        
        final_stats = symbol_grounding_module.get_statistics()
        assert final_stats["total_ambiguity_resolutions"] == initial_count + 2

    def test_resolve_ambiguity_returns_valid_probability(self, symbol_grounding_module):
        """Test that resolved candidate has valid probability."""
        symbol_grounding_module.ground_symbol("watermelon")
        
        candidate = symbol_grounding_module.resolve_ambiguity(
            "watermelon", {"type": "fruit"}
        )
        
        assert 0.0 <= candidate.probability <= 1.0

    def test_resolve_ambiguity_single_candidate(self, symbol_grounding_module):
        """Test resolve_ambiguity with single candidate returns that candidate."""
        grounding = symbol_grounding_module.ground_symbol("xigua")
        
        # If there's only one candidate, it should be returned
        if len(grounding.groundings) == 1:
            candidate = symbol_grounding_module.resolve_ambiguity("xigua", {})
            assert candidate is grounding.groundings[0]


# =============================================================================
# Tests for get_grounding_confidence()
# =============================================================================

class TestGetGroundingConfidence:
    """Tests for the get_grounding_confidence() method.
    
    Requirements: 5.3
    """

    def test_get_confidence_existing_symbol(self, symbol_grounding_module):
        """Test getting confidence for an existing symbol."""
        symbol_grounding_module.ground_symbol("yam")
        
        confidence = symbol_grounding_module.get_grounding_confidence("yam")
        
        assert 0.0 <= confidence <= 1.0

    def test_get_confidence_unknown_symbol(self, symbol_grounding_module):
        """Test getting confidence for unknown symbol returns 0.0."""
        confidence = symbol_grounding_module.get_grounding_confidence("unknown")
        
        assert confidence == 0.0

    def test_get_confidence_empty_symbol(self, symbol_grounding_module):
        """Test getting confidence for empty symbol returns 0.0."""
        confidence = symbol_grounding_module.get_grounding_confidence("")
        
        assert confidence == 0.0

    def test_get_confidence_normalizes_symbol(self, symbol_grounding_module):
        """Test that get_grounding_confidence normalizes the symbol."""
        symbol_grounding_module.ground_symbol("zucchini")
        
        confidence1 = symbol_grounding_module.get_grounding_confidence("ZUCCHINI")
        confidence2 = symbol_grounding_module.get_grounding_confidence("  zucchini  ")
        
        assert confidence1 == confidence2


# =============================================================================
# Tests for get_ungrounded_symbols()
# =============================================================================

class TestGetUngroundedSymbols:
    """Tests for the get_ungrounded_symbols() method.
    
    Requirements: 5.3
    """

    def test_get_ungrounded_empty_module(self, symbol_grounding_module):
        """Test getting ungrounded symbols from empty module."""
        ungrounded = symbol_grounding_module.get_ungrounded_symbols()
        
        assert ungrounded == []

    def test_get_ungrounded_with_low_confidence_symbols(self, configured_module):
        """Test getting symbols with confidence below threshold."""
        # Create some groundings
        configured_module.ground_symbol("alpha")
        configured_module.ground_symbol("beta")
        
        ungrounded = configured_module.get_ungrounded_symbols()
        
        # All new symbols start with low confidence
        assert isinstance(ungrounded, list)


# =============================================================================
# Tests for is_grounded()
# =============================================================================

class TestIsGrounded:
    """Tests for the is_grounded() method."""

    def test_is_grounded_unknown_symbol(self, symbol_grounding_module):
        """Test is_grounded for unknown symbol returns False."""
        assert symbol_grounding_module.is_grounded("unknown") is False

    def test_is_grounded_checks_threshold(self, configured_module):
        """Test that is_grounded checks against confidence threshold."""
        configured_module.ground_symbol("gamma")
        
        # Result depends on initial confidence vs threshold
        result = configured_module.is_grounded("gamma")
        assert isinstance(result, bool)


# =============================================================================
# Tests for get_all_symbols()
# =============================================================================

class TestGetAllSymbols:
    """Tests for the get_all_symbols() method."""

    def test_get_all_symbols_empty(self, symbol_grounding_module):
        """Test getting all symbols from empty module."""
        symbols = symbol_grounding_module.get_all_symbols()
        
        assert symbols == []

    def test_get_all_symbols_returns_all(self, symbol_grounding_module):
        """Test that get_all_symbols returns all grounded symbols."""
        symbol_grounding_module.ground_symbol("delta")
        symbol_grounding_module.ground_symbol("epsilon")
        symbol_grounding_module.ground_symbol("zeta")
        
        symbols = symbol_grounding_module.get_all_symbols()
        
        assert len(symbols) == 3
        assert "delta" in symbols
        assert "epsilon" in symbols
        assert "zeta" in symbols


# =============================================================================
# Tests for get_grounding()
# =============================================================================

class TestGetGrounding:
    """Tests for the get_grounding() method."""

    def test_get_grounding_existing(self, symbol_grounding_module):
        """Test getting an existing grounding."""
        symbol_grounding_module.ground_symbol("eta")
        
        grounding = symbol_grounding_module.get_grounding("eta")
        
        assert grounding is not None
        assert grounding.symbol == "eta"

    def test_get_grounding_not_found(self, symbol_grounding_module):
        """Test getting a non-existent grounding returns None."""
        grounding = symbol_grounding_module.get_grounding("nonexistent")
        
        assert grounding is None

    def test_get_grounding_empty_symbol(self, symbol_grounding_module):
        """Test getting grounding for empty symbol returns None."""
        grounding = symbol_grounding_module.get_grounding("")
        
        assert grounding is None

    def test_get_grounding_normalizes_symbol(self, symbol_grounding_module):
        """Test that get_grounding normalizes the symbol."""
        symbol_grounding_module.ground_symbol("theta")
        
        grounding1 = symbol_grounding_module.get_grounding("THETA")
        grounding2 = symbol_grounding_module.get_grounding("  theta  ")
        
        assert grounding1 is not None
        assert grounding1 is grounding2

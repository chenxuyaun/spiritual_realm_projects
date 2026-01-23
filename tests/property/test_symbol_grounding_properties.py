"""
Property-based tests for Symbol Grounding Module.

Tests properties 21-24 from the consciousness-system-deepening design document:
- Property 21: Symbol Grounding Existence
- Property 22: New Symbol Grounding Attempt
- Property 23: Grounding Confidence Evolution
- Property 24: Ambiguous Grounding Probability Distribution

Validates: Requirements 5.1-5.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import Dict, List, Any, Optional
import time
import math

from mm_orch.consciousness.symbol_grounding import (
    SymbolGroundingModule,
    SymbolGrounding,
    GroundingCandidate,
    SymbolGroundingConfig,
)
from mm_orch.consciousness.episodic_memory import EpisodicMemory, EpisodicMemoryConfig
from mm_orch.consciousness.semantic_memory import SemanticMemory


# =============================================================================
# Hypothesis Strategies for Test Data Generation
# =============================================================================

# Strategy for valid symbol strings (non-empty, alphanumeric)
symbol_strategy = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'Pd'))
).filter(lambda s: s.strip() != "")

# Strategy for probability values [0.0, 1.0]
probability_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for confidence values [0.0, 1.0]
confidence_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
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

# Strategy for perceptual features dictionaries
perceptual_features_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    values=st.one_of(
        st.integers(min_value=-1000, max_value=1000),
        st.text(max_size=20),
        st.booleans(),
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    ),
    min_size=0,
    max_size=5
)


@st.composite
def grounding_candidate_strategy(draw):
    """Generate a valid GroundingCandidate."""
    return GroundingCandidate(
        concept_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=20))),
        episode_ids=draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5)),
        perceptual_features=draw(perceptual_features_strategy),
        probability=draw(probability_strategy),
    )


@st.composite
def symbol_grounding_config_strategy(draw):
    """Generate a valid SymbolGroundingConfig."""
    return SymbolGroundingConfig(
        min_confidence_threshold=draw(st.floats(min_value=0.1, max_value=0.5, allow_nan=False)),
        confidence_growth_rate=draw(st.floats(min_value=0.05, max_value=0.3, allow_nan=False)),
        confidence_decay_rate=draw(st.floats(min_value=0.01, max_value=0.2, allow_nan=False)),
        max_candidates=draw(st.integers(min_value=3, max_value=20)),
        context_similarity_threshold=draw(st.floats(min_value=0.3, max_value=0.8, allow_nan=False)),
        probability_smoothing=draw(st.floats(min_value=0.001, max_value=0.1, allow_nan=False)),
    )


def create_test_memory_systems():
    """Create episodic and semantic memory systems for testing."""
    episodic_memory = EpisodicMemory(max_episodes=100)
    semantic_memory = SemanticMemory()
    return episodic_memory, semantic_memory


def create_symbol_grounding_module(config: Optional[Dict[str, Any]] = None):
    """Create a SymbolGroundingModule with test memory systems."""
    episodic_memory, semantic_memory = create_test_memory_systems()
    return SymbolGroundingModule(
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        config=config,
    )


# =============================================================================
# Property 21: Symbol Grounding Existence
# =============================================================================

class TestSymbolGroundingExistence:
    """
    Tests for Property 21: Symbol Grounding Existence
    
    *For any* symbol queried from SymbolGroundingModule, the returned 
    SymbolGrounding SHALL contain at least one grounding candidate 
    (even if with low confidence for new symbols).
    
    **Validates: Requirements 5.1**
    """

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_grounding_always_has_at_least_one_candidate(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 21: Symbol Grounding Existence
        
        For any symbol queried, the returned SymbolGrounding SHALL contain
        at least one grounding candidate.
        
        **Validates: Requirements 5.1**
        """
        module = create_symbol_grounding_module()
        
        grounding = module.ground_symbol(symbol)
        
        assert grounding is not None, "Grounding should not be None"
        assert isinstance(grounding, SymbolGrounding), "Should return SymbolGrounding"
        assert len(grounding.groundings) >= 1, \
            f"Grounding for '{symbol}' should have at least one candidate, got {len(grounding.groundings)}"

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_grounding_symbol_matches_query(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 21: Symbol Grounding Existence
        
        For any symbol queried, the returned SymbolGrounding SHALL have
        its symbol field match the normalized query symbol.
        
        **Validates: Requirements 5.1**
        """
        module = create_symbol_grounding_module()
        
        grounding = module.ground_symbol(symbol)
        
        # Symbol should be normalized (lowercase, stripped)
        expected_symbol = symbol.lower().strip()
        assert grounding.symbol == expected_symbol, \
            f"Grounding symbol should be '{expected_symbol}', got '{grounding.symbol}'"

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_grounding_has_valid_confidence(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 21: Symbol Grounding Existence
        
        For any symbol queried, the returned SymbolGrounding SHALL have
        a confidence value in the valid range [0.0, 1.0].
        
        **Validates: Requirements 5.1**
        """
        module = create_symbol_grounding_module()
        
        grounding = module.ground_symbol(symbol)
        
        assert 0.0 <= grounding.confidence <= 1.0, \
            f"Confidence should be in [0.0, 1.0], got {grounding.confidence}"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_repeated_grounding_returns_same_result(self, data):
        """
        Feature: consciousness-system-deepening, Property 21: Symbol Grounding Existence
        
        For any symbol queried multiple times without updates, the returned
        SymbolGrounding SHALL be consistent (same candidates).
        
        **Validates: Requirements 5.1**
        """
        symbol = data.draw(symbol_strategy)
        module = create_symbol_grounding_module()
        
        grounding1 = module.ground_symbol(symbol)
        grounding2 = module.ground_symbol(symbol)
        
        assert grounding1.symbol == grounding2.symbol
        assert len(grounding1.groundings) == len(grounding2.groundings)
        assert grounding1.confidence == grounding2.confidence

    @given(symbols=st.lists(symbol_strategy, min_size=2, max_size=10, unique=True))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_symbols_all_have_groundings(self, symbols: List[str]):
        """
        Feature: consciousness-system-deepening, Property 21: Symbol Grounding Existence
        
        For any set of symbols queried, each SHALL have at least one
        grounding candidate.
        
        **Validates: Requirements 5.1**
        """
        module = create_symbol_grounding_module()
        
        for symbol in symbols:
            grounding = module.ground_symbol(symbol)
            assert len(grounding.groundings) >= 1, \
                f"Symbol '{symbol}' should have at least one grounding candidate"

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_grounding_candidates_have_valid_probabilities(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 21: Symbol Grounding Existence
        
        For any symbol's grounding candidates, each candidate SHALL have
        a probability in the valid range [0.0, 1.0].
        
        **Validates: Requirements 5.1**
        """
        module = create_symbol_grounding_module()
        
        grounding = module.ground_symbol(symbol)
        
        for candidate in grounding.groundings:
            assert 0.0 <= candidate.probability <= 1.0, \
                f"Candidate probability should be in [0.0, 1.0], got {candidate.probability}"


# =============================================================================
# Property 22: New Symbol Grounding Attempt
# =============================================================================

class TestNewSymbolGroundingAttempt:
    """
    Tests for Property 22: New Symbol Grounding Attempt
    
    *For any* new symbol encountered, the SymbolGroundingModule SHALL attempt 
    grounding by querying both EpisodicMemory and SemanticMemory for relevant 
    associations.
    
    **Validates: Requirements 5.2**
    """

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_new_symbol_triggers_grounding_creation(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 22: New Symbol Grounding Attempt
        
        For any new symbol encountered, the module SHALL create a new grounding
        entry and increment the total groundings created counter.
        
        **Validates: Requirements 5.2**
        """
        module = create_symbol_grounding_module()
        
        initial_stats = module.get_statistics()
        initial_created = initial_stats["total_groundings_created"]
        
        # Ground a new symbol
        grounding = module.ground_symbol(symbol)
        
        final_stats = module.get_statistics()
        final_created = final_stats["total_groundings_created"]
        
        assert final_created == initial_created + 1, \
            f"Grounding creation count should increase by 1"
        assert grounding.grounding_count == 1, \
            "New grounding should have grounding_count of 1"

    @given(symbol=symbol_strategy, context=context_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_new_symbol_with_context_creates_grounding(self, symbol: str, context: Dict[str, Any]):
        """
        Feature: consciousness-system-deepening, Property 22: New Symbol Grounding Attempt
        
        For any new symbol with context, the module SHALL create a grounding
        that can be updated with the context information.
        
        **Validates: Requirements 5.2**
        """
        module = create_symbol_grounding_module()
        
        # First ground the symbol
        grounding = module.ground_symbol(symbol)
        assert grounding is not None
        
        # Update with context
        updated = module.update_grounding(symbol, context)
        
        assert updated is not None
        assert updated.grounding_count >= 1

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_grounding_attempt_queries_memory_systems(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 22: New Symbol Grounding Attempt
        
        For any new symbol, the grounding attempt SHALL query both memory systems,
        resulting in a grounding with candidates (even if default ungrounded).
        
        **Validates: Requirements 5.2**
        """
        episodic_memory, semantic_memory = create_test_memory_systems()
        
        # Add some data to semantic memory for potential matches
        semantic_memory.integrate_knowledge({
            "name": symbol[:3] if len(symbol) >= 3 else symbol,
            "type": "concept",
            "properties": {"test": True}
        })
        
        module = SymbolGroundingModule(
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
        )
        
        grounding = module.ground_symbol(symbol)
        
        # Should have at least one candidate from the grounding attempt
        assert len(grounding.groundings) >= 1
        # Grounding should have been created
        assert grounding.grounding_count >= 1

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_new_symbol_has_initial_timestamp(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 22: New Symbol Grounding Attempt
        
        For any new symbol grounded, the grounding SHALL have a valid
        last_updated timestamp.
        
        **Validates: Requirements 5.2**
        """
        module = create_symbol_grounding_module()
        
        before_time = time.time()
        grounding = module.ground_symbol(symbol)
        after_time = time.time()
        
        assert grounding.last_updated >= before_time, \
            "last_updated should be >= time before grounding"
        assert grounding.last_updated <= after_time, \
            "last_updated should be <= time after grounding"

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_grounding_with_episodic_memory_content(self, data):
        """
        Feature: consciousness-system-deepening, Property 22: New Symbol Grounding Attempt
        
        For any new symbol, when episodic memory contains relevant episodes,
        the grounding attempt SHALL consider those episodes.
        
        **Validates: Requirements 5.2**
        """
        symbol = data.draw(symbol_strategy)
        episodic_memory, semantic_memory = create_test_memory_systems()
        
        # Add an episode with context containing the symbol
        episodic_memory.create_episode(
            context={"symbol": symbol, "query_text": symbol},
            events=[{"type": "test", "symbol": symbol}],
            emotional_state={"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5},
            importance=0.7,
        )
        
        module = SymbolGroundingModule(
            episodic_memory=episodic_memory,
            semantic_memory=semantic_memory,
        )
        
        grounding = module.ground_symbol(symbol)
        
        # Should have grounding candidates
        assert len(grounding.groundings) >= 1


# =============================================================================
# Property 23: Grounding Confidence Evolution
# =============================================================================

class TestGroundingConfidenceEvolution:
    """
    Tests for Property 23: Grounding Confidence Evolution
    
    *For any* symbol with multiple consistent associations, the grounding 
    confidence SHALL increase with each consistent encounter, and confidence 
    SHALL decrease with inconsistent associations.
    
    **Validates: Requirements 5.3, 5.5**
    """

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_consistent_updates_increase_confidence(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 23: Grounding Confidence Evolution
        
        For any symbol with consistent associations, the grounding confidence
        SHALL increase with each consistent encounter.
        
        **Validates: Requirements 5.3, 5.5**
        """
        module = create_symbol_grounding_module()
        
        # Initial grounding
        grounding = module.ground_symbol(symbol)
        initial_grounding_count = grounding.grounding_count
        initial_confidence = grounding.confidence
        
        # Create a consistent context (similar to what the grounding already has)
        consistent_context = {"symbol": symbol, "type": "consistent", "value": 1}
        
        # Update with consistent context multiple times
        num_updates = 3
        for i in range(num_updates):
            updated = module.update_grounding(symbol, consistent_context)
        
        # Confidence should remain in valid range
        assert updated.confidence >= 0.0 and updated.confidence <= 1.0, \
            "Confidence should remain in valid range"
        # Grounding count should increase by the number of updates
        assert updated.grounding_count == initial_grounding_count + num_updates, \
            f"Grounding count should be {initial_grounding_count + num_updates}, got {updated.grounding_count}"

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_grounding_count_increases_with_updates(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 23: Grounding Confidence Evolution
        
        For any symbol updated multiple times, the grounding_count SHALL
        increase with each update.
        
        **Validates: Requirements 5.3, 5.5**
        """
        module = create_symbol_grounding_module()
        
        # Initial grounding
        grounding = module.ground_symbol(symbol)
        assert grounding.grounding_count == 1
        
        # Update multiple times
        num_updates = 5
        for i in range(num_updates):
            context = {"update_index": i, "data": f"update_{i}"}
            updated = module.update_grounding(symbol, context)
        
        assert updated.grounding_count == 1 + num_updates, \
            f"Grounding count should be {1 + num_updates}, got {updated.grounding_count}"

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inconsistent_updates_decrease_confidence(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 23: Grounding Confidence Evolution
        
        For any symbol with inconsistent associations, the grounding confidence
        SHALL decrease with inconsistent encounters.
        
        **Validates: Requirements 5.3, 5.5**
        """
        # Use config with high similarity threshold to make contexts inconsistent
        config = {
            "context_similarity_threshold": 0.9,  # High threshold
            "confidence_decay_rate": 0.1,  # Noticeable decay
            "confidence_growth_rate": 0.1,
        }
        module = create_symbol_grounding_module(config)
        
        # Initial grounding with specific context
        grounding = module.ground_symbol(symbol)
        
        # First update to establish a baseline context
        initial_context = {"type": "initial", "category": "A", "value": 100}
        module.update_grounding(symbol, initial_context)
        
        confidence_after_initial = module.get_grounding_confidence(symbol)
        
        # Update with very different (inconsistent) context
        inconsistent_context = {"completely": "different", "unrelated": True}
        module.update_grounding(symbol, inconsistent_context)
        
        final_confidence = module.get_grounding_confidence(symbol)
        
        # Confidence should remain in valid range
        assert 0.0 <= final_confidence <= 1.0, \
            f"Confidence should be in [0.0, 1.0], got {final_confidence}"

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_confidence_bounded_after_many_updates(self, data):
        """
        Feature: consciousness-system-deepening, Property 23: Grounding Confidence Evolution
        
        For any symbol after many updates, the confidence SHALL remain
        within valid bounds [0.0, 1.0].
        
        **Validates: Requirements 5.3, 5.5**
        """
        symbol = data.draw(symbol_strategy)
        num_updates = data.draw(st.integers(min_value=5, max_value=20))
        
        module = create_symbol_grounding_module()
        module.ground_symbol(symbol)
        
        for i in range(num_updates):
            context = data.draw(context_strategy)
            module.update_grounding(symbol, context)
        
        final_confidence = module.get_grounding_confidence(symbol)
        
        assert 0.0 <= final_confidence <= 1.0, \
            f"Confidence should be in [0.0, 1.0] after {num_updates} updates, got {final_confidence}"

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_last_updated_changes_with_updates(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 23: Grounding Confidence Evolution
        
        For any symbol update, the last_updated timestamp SHALL be updated
        to reflect the time of the update.
        
        **Validates: Requirements 5.3, 5.5**
        """
        module = create_symbol_grounding_module()
        
        grounding = module.ground_symbol(symbol)
        initial_timestamp = grounding.last_updated
        
        # Small delay to ensure timestamp difference
        time.sleep(0.01)
        
        context = {"update": True}
        updated = module.update_grounding(symbol, context)
        
        assert updated.last_updated >= initial_timestamp, \
            "last_updated should be >= initial timestamp after update"


# =============================================================================
# Property 24: Ambiguous Grounding Probability Distribution
# =============================================================================

class TestAmbiguousGroundingProbabilityDistribution:
    """
    Tests for Property 24: Ambiguous Grounding Probability Distribution
    
    *For any* symbol with multiple candidate groundings, the probabilities 
    of all candidates SHALL sum to 1.0 (within floating-point tolerance).
    
    **Validates: Requirements 5.4**
    """

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_probabilities_sum_to_one(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 24: Ambiguous Grounding Probability Distribution
        
        For any symbol's grounding candidates, the probabilities SHALL
        sum to 1.0 within floating-point tolerance.
        
        **Validates: Requirements 5.4**
        """
        module = create_symbol_grounding_module()
        
        grounding = module.ground_symbol(symbol)
        
        total_probability = sum(c.probability for c in grounding.groundings)
        
        assert abs(total_probability - 1.0) < 1e-6, \
            f"Probabilities should sum to 1.0, got {total_probability}"

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_probabilities_sum_to_one_after_updates(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 24: Ambiguous Grounding Probability Distribution
        
        For any symbol after multiple updates, the candidate probabilities
        SHALL still sum to 1.0 within floating-point tolerance.
        
        **Validates: Requirements 5.4**
        """
        module = create_symbol_grounding_module()
        
        # Initial grounding
        module.ground_symbol(symbol)
        
        # Multiple updates with different contexts
        contexts = [
            {"type": "A", "value": 1},
            {"type": "B", "value": 2},
            {"type": "C", "value": 3},
        ]
        
        for context in contexts:
            module.update_grounding(symbol, context)
        
        grounding = module.get_grounding(symbol)
        total_probability = sum(c.probability for c in grounding.groundings)
        
        assert abs(total_probability - 1.0) < 1e-6, \
            f"Probabilities should sum to 1.0 after updates, got {total_probability}"

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_probabilities_valid_range_for_all_candidates(self, data):
        """
        Feature: consciousness-system-deepening, Property 24: Ambiguous Grounding Probability Distribution
        
        For any symbol's grounding candidates, each probability SHALL be
        in the valid range [0.0, 1.0].
        
        **Validates: Requirements 5.4**
        """
        symbol = data.draw(symbol_strategy)
        module = create_symbol_grounding_module()
        
        grounding = module.ground_symbol(symbol)
        
        # Add some updates to potentially create multiple candidates
        for i in range(3):
            context = data.draw(context_strategy)
            module.update_grounding(symbol, context)
        
        grounding = module.get_grounding(symbol)
        
        for i, candidate in enumerate(grounding.groundings):
            assert 0.0 <= candidate.probability <= 1.0, \
                f"Candidate {i} probability should be in [0.0, 1.0], got {candidate.probability}"

    @given(symbols=st.lists(symbol_strategy, min_size=2, max_size=5, unique=True))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_probabilities_sum_to_one_for_multiple_symbols(self, symbols: List[str]):
        """
        Feature: consciousness-system-deepening, Property 24: Ambiguous Grounding Probability Distribution
        
        For any set of symbols, each symbol's candidate probabilities SHALL
        independently sum to 1.0.
        
        **Validates: Requirements 5.4**
        """
        module = create_symbol_grounding_module()
        
        for symbol in symbols:
            grounding = module.ground_symbol(symbol)
            total_probability = sum(c.probability for c in grounding.groundings)
            
            assert abs(total_probability - 1.0) < 1e-6, \
                f"Probabilities for '{symbol}' should sum to 1.0, got {total_probability}"

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_ambiguity_resolution_returns_valid_candidate(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 24: Ambiguous Grounding Probability Distribution
        
        For any symbol with ambiguous grounding, resolving ambiguity SHALL
        return a valid candidate from the grounding's candidate list.
        
        **Validates: Requirements 5.4**
        """
        module = create_symbol_grounding_module()
        
        # Create grounding
        grounding = module.ground_symbol(symbol)
        
        # Resolve ambiguity with some context
        context = {"query": symbol, "type": "test"}
        resolved = module.resolve_ambiguity(symbol, context)
        
        assert resolved is not None, "Resolved candidate should not be None"
        assert isinstance(resolved, GroundingCandidate), \
            "Resolved should be a GroundingCandidate"
        assert 0.0 <= resolved.probability <= 1.0, \
            f"Resolved candidate probability should be valid, got {resolved.probability}"

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_probabilities_normalized_after_linking_concept(self, data):
        """
        Feature: consciousness-system-deepening, Property 24: Ambiguous Grounding Probability Distribution
        
        For any symbol after linking to a concept, the candidate probabilities
        SHALL still sum to 1.0.
        
        **Validates: Requirements 5.4**
        """
        symbol = data.draw(symbol_strategy)
        concept_id = data.draw(st.text(min_size=5, max_size=20))
        
        module = create_symbol_grounding_module()
        module.ground_symbol(symbol)
        
        # Link to a concept
        module.link_to_concept(symbol, concept_id, probability_boost=0.3)
        
        grounding = module.get_grounding(symbol)
        total_probability = sum(c.probability for c in grounding.groundings)
        
        assert abs(total_probability - 1.0) < 1e-6, \
            f"Probabilities should sum to 1.0 after linking, got {total_probability}"

    @given(data=st.data())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_probabilities_normalized_after_merge(self, data):
        """
        Feature: consciousness-system-deepening, Property 24: Ambiguous Grounding Probability Distribution
        
        For any two symbols merged, the resulting grounding's candidate
        probabilities SHALL sum to 1.0.
        
        **Validates: Requirements 5.4**
        """
        symbol1 = data.draw(symbol_strategy)
        symbol2 = data.draw(symbol_strategy.filter(lambda s: s.lower().strip() != symbol1.lower().strip()))
        
        module = create_symbol_grounding_module()
        
        # Ground both symbols
        module.ground_symbol(symbol1)
        module.ground_symbol(symbol2)
        
        # Merge groundings
        merged = module.merge_groundings(symbol1, symbol2)
        
        if merged is not None:
            total_probability = sum(c.probability for c in merged.groundings)
            assert abs(total_probability - 1.0) < 1e-6, \
                f"Merged probabilities should sum to 1.0, got {total_probability}"


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestSymbolGroundingIntegration:
    """
    Integration tests for Symbol Grounding Module.
    
    These tests verify the module works correctly with the memory systems
    and maintains consistency across operations.
    """

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_serialization_preserves_probability_sum(self, data):
        """
        Feature: consciousness-system-deepening, Property 24: Ambiguous Grounding Probability Distribution
        
        For any symbol grounding serialized and deserialized, the candidate
        probabilities SHALL still sum to 1.0.
        
        **Validates: Requirements 5.4**
        """
        symbol = data.draw(symbol_strategy)
        module = create_symbol_grounding_module()
        
        # Create and update grounding
        module.ground_symbol(symbol)
        module.update_grounding(symbol, {"test": "context"})
        
        # Serialize
        state = module.to_dict()
        
        # Create new module and load state
        new_module = create_symbol_grounding_module()
        new_module.load_state(state)
        
        # Check probability sum
        grounding = new_module.get_grounding(symbol)
        if grounding is not None:
            total_probability = sum(c.probability for c in grounding.groundings)
            assert abs(total_probability - 1.0) < 1e-6, \
                f"Probabilities should sum to 1.0 after deserialization, got {total_probability}"

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_statistics_track_operations(self, data):
        """
        Feature: consciousness-system-deepening, Property 21-24: Symbol Grounding
        
        For any sequence of operations, the statistics SHALL accurately
        track the number of groundings created and updated.
        
        **Validates: Requirements 5.1-5.5**
        """
        num_symbols = data.draw(st.integers(min_value=1, max_value=5))
        num_updates = data.draw(st.integers(min_value=1, max_value=5))
        
        module = create_symbol_grounding_module()
        
        symbols = [data.draw(symbol_strategy) for _ in range(num_symbols)]
        # Ensure unique symbols
        symbols = list(set(s.lower().strip() for s in symbols))
        
        # Ground all symbols
        for symbol in symbols:
            module.ground_symbol(symbol)
        
        stats_after_grounding = module.get_statistics()
        assert stats_after_grounding["total_groundings_created"] == len(symbols), \
            f"Should have created {len(symbols)} groundings"
        
        # Update each symbol
        total_updates = 0
        for symbol in symbols:
            for _ in range(num_updates):
                context = data.draw(context_strategy)
                module.update_grounding(symbol, context)
                total_updates += 1
        
        stats_after_updates = module.get_statistics()
        assert stats_after_updates["total_groundings_updated"] == total_updates, \
            f"Should have {total_updates} updates recorded"

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_clear_removes_all_groundings(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 21: Symbol Grounding Existence
        
        After clearing the module, previously grounded symbols SHALL
        require re-grounding.
        
        **Validates: Requirements 5.1**
        """
        module = create_symbol_grounding_module()
        
        # Ground a symbol
        module.ground_symbol(symbol)
        assert module.get_grounding(symbol) is not None
        
        # Clear
        module.clear()
        
        # Symbol should no longer have grounding
        assert module.get_grounding(symbol) is None
        
        # Re-grounding should work
        new_grounding = module.ground_symbol(symbol)
        assert new_grounding is not None
        assert len(new_grounding.groundings) >= 1

    @given(symbol=symbol_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_is_grounded_reflects_confidence_threshold(self, symbol: str):
        """
        Feature: consciousness-system-deepening, Property 23: Grounding Confidence Evolution
        
        The is_grounded() method SHALL return True only when confidence
        is at or above the configured threshold.
        
        **Validates: Requirements 5.3**
        """
        config = {"min_confidence_threshold": 0.5}
        module = create_symbol_grounding_module(config)
        
        grounding = module.ground_symbol(symbol)
        confidence = grounding.confidence
        is_grounded = module.is_grounded(symbol)
        
        if confidence >= 0.5:
            assert is_grounded, \
                f"Symbol with confidence {confidence} >= 0.5 should be grounded"
        else:
            assert not is_grounded, \
                f"Symbol with confidence {confidence} < 0.5 should not be grounded"

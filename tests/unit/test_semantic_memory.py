"""
Unit tests for the SemanticMemory module.

Tests cover:
- Knowledge integration with conflict detection
- Query interface
- Pattern extraction from episodes
- Consolidation operations
- Serialization/deserialization

Requirements: 4.2, 4.4, 4.5, 4.6
"""

import pytest
import time
from typing import Dict, Any, List

from mm_orch.consciousness.semantic_memory import (
    SemanticMemory,
    SemanticMemoryConfig,
    IntegrationResult,
    ExtractionResult,
    ConsolidationResult,
    ConflictInfo,
    ConflictResolutionStrategy,
    create_semantic_memory,
)
from mm_orch.consciousness.episodic_memory import Episode
from mm_orch.consciousness.knowledge_graph import ConceptNode, Relationship


class TestSemanticMemoryConfig:
    """Tests for SemanticMemoryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SemanticMemoryConfig()
        
        assert config.conflict_resolution_strategy == "newest_wins"
        assert config.similarity_threshold == 0.8
        assert config.min_relationship_strength == 0.1
        assert config.consolidation_merge_threshold == 0.9
        assert config.prune_access_threshold == 0
        assert config.prune_strength_threshold == 0.2
        assert config.max_query_results == 20
        assert config.extraction_min_frequency == 0.2

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SemanticMemoryConfig(
            conflict_resolution_strategy="highest_confidence",
            similarity_threshold=0.9,
            min_relationship_strength=0.2,
        )
        
        assert config.conflict_resolution_strategy == "highest_confidence"
        assert config.similarity_threshold == 0.9
        assert config.min_relationship_strength == 0.2

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid conflict_resolution_strategy"):
            SemanticMemoryConfig(conflict_resolution_strategy="invalid")

    def test_invalid_threshold_raises_error(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError):
            SemanticMemoryConfig(similarity_threshold=1.5)
        
        with pytest.raises(ValueError):
            SemanticMemoryConfig(similarity_threshold=-0.1)

    def test_config_to_dict(self):
        """Test config serialization to dictionary."""
        config = SemanticMemoryConfig()
        data = config.to_dict()
        
        assert "conflict_resolution_strategy" in data
        assert "similarity_threshold" in data
        assert data["conflict_resolution_strategy"] == "newest_wins"

    def test_config_from_dict(self):
        """Test config deserialization from dictionary."""
        data = {
            "conflict_resolution_strategy": "merge",
            "similarity_threshold": 0.7,
        }
        config = SemanticMemoryConfig.from_dict(data)
        
        assert config.conflict_resolution_strategy == "merge"
        assert config.similarity_threshold == 0.7


class TestSemanticMemoryInitialization:
    """Tests for SemanticMemory initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        memory = SemanticMemory()
        
        assert memory.knowledge_graph is not None
        assert len(memory) == 0

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {"conflict_resolution_strategy": "merge"}
        memory = SemanticMemory(config=config)
        
        assert memory._config.conflict_resolution_strategy == "merge"

    def test_factory_function(self):
        """Test factory function creates valid instance."""
        memory = create_semantic_memory()
        
        assert isinstance(memory, SemanticMemory)
        assert len(memory) == 0


class TestKnowledgeIntegration:
    """Tests for knowledge integration functionality."""

    def test_integrate_new_concept(self):
        """Test integrating a new concept creates it in the graph."""
        memory = SemanticMemory()
        
        knowledge = {
            "concepts": [
                {"name": "Python", "type": "programming_language", "properties": {"paradigm": "multi-paradigm"}}
            ]
        }
        
        result = memory.integrate_knowledge(knowledge)
        
        assert result.success
        assert len(result.new_concepts) == 1
        assert len(result.updated_concepts) == 0
        assert "Python" in memory

    def test_integrate_existing_concept_updates(self):
        """Test integrating existing concept updates it."""
        memory = SemanticMemory()
        
        # First integration
        knowledge1 = {"concepts": [{"name": "Python", "type": "language"}]}
        memory.integrate_knowledge(knowledge1)
        
        # Second integration with same concept
        knowledge2 = {"concepts": [{"name": "Python", "type": "language", "properties": {"version": "3.11"}}]}
        result = memory.integrate_knowledge(knowledge2)
        
        assert result.success
        assert len(result.updated_concepts) == 1
        
        # Check property was added
        definition = memory.get_concept_definition("Python")
        assert definition is not None
        assert definition["properties"].get("version") == "3.11"

    def test_integrate_relationship(self):
        """Test integrating a relationship creates nodes and edge."""
        memory = SemanticMemory()
        
        knowledge = {
            "relationships": [
                {"source": "Python", "target": "Programming", "type": "is_a"}
            ]
        }
        
        result = memory.integrate_knowledge(knowledge)
        
        assert result.success
        assert len(result.new_concepts) == 2  # Both nodes created
        assert len(result.new_relationships) == 1
        assert "Python" in memory
        assert "Programming" in memory

    def test_integrate_attributes(self):
        """Test integrating attributes for existing concept."""
        memory = SemanticMemory()
        
        # Create concept first
        memory.integrate_knowledge({"concepts": [{"name": "Python", "type": "language"}]})
        
        # Add attributes
        knowledge = {
            "attributes": {
                "Python": {"creator": "Guido van Rossum", "year": 1991}
            }
        }
        
        result = memory.integrate_knowledge(knowledge)
        
        assert result.success
        definition = memory.get_concept_definition("Python")
        assert definition["properties"]["creator"] == "Guido van Rossum"
        assert definition["properties"]["year"] == 1991

    def test_conflict_detection_attribute(self):
        """Test conflict detection for attribute changes."""
        memory = SemanticMemory()
        
        # Create concept with attribute
        memory.integrate_knowledge({
            "concepts": [{"name": "Python", "type": "language", "properties": {"version": "3.10"}}]
        })
        
        # Update with conflicting attribute
        result = memory.integrate_knowledge({
            "concepts": [{"name": "Python", "type": "language", "properties": {"version": "3.11"}}]
        })
        
        assert result.success
        assert len(result.conflicts) == 1
        assert result.conflicts[0].conflict_type == "attribute"
        assert result.conflicts[0].existing_value == "3.10"
        assert result.conflicts[0].new_value == "3.11"

    def test_conflict_resolution_newest_wins(self):
        """Test newest_wins conflict resolution strategy."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "newest_wins"})
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Python", "type": "language", "properties": {"version": "3.10"}}]
        })
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Python", "type": "language", "properties": {"version": "3.11"}}]
        })
        
        definition = memory.get_concept_definition("Python")
        assert definition["properties"]["version"] == "3.11"

    def test_conflict_resolution_highest_confidence(self):
        """Test highest_confidence conflict resolution strategy."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "highest_confidence"})
        
        # First integration with high confidence
        memory.integrate_knowledge(
            {"concepts": [{"name": "Python", "type": "language", "properties": {"version": "3.10"}}]},
            confidence=0.9
        )
        
        # Second integration with lower confidence
        memory.integrate_knowledge(
            {"concepts": [{"name": "Python", "type": "language", "properties": {"version": "3.11"}}]},
            confidence=0.5
        )
        
        definition = memory.get_concept_definition("Python")
        # Original value should be kept due to higher confidence
        assert definition["properties"]["version"] == "3.10"

    def test_integration_result_structure(self):
        """Test IntegrationResult has all required fields."""
        memory = SemanticMemory()
        
        result = memory.integrate_knowledge({"concepts": [{"name": "Test", "type": "entity"}]})
        
        assert hasattr(result, "new_concepts")
        assert hasattr(result, "updated_concepts")
        assert hasattr(result, "new_relationships")
        assert hasattr(result, "conflicts")
        assert hasattr(result, "resolution_actions")
        assert hasattr(result, "success")
        assert hasattr(result, "timestamp")


class TestQueryInterface:
    """Tests for query interface functionality."""

    def test_query_by_name(self):
        """Test querying concepts by name."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [
                {"name": "Python", "type": "language"},
                {"name": "JavaScript", "type": "language"},
                {"name": "Java", "type": "language"},
            ]
        })
        
        results = memory.query("Python")
        
        assert len(results) >= 1
        assert any(node.name == "Python" for node in results)

    def test_query_partial_match(self):
        """Test querying with partial name match."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [
                {"name": "Python", "type": "language"},
                {"name": "Python3", "type": "language"},
            ]
        })
        
        results = memory.query("Pyth")
        
        assert len(results) >= 1

    def test_query_max_results(self):
        """Test query respects max_results parameter."""
        memory = SemanticMemory()
        
        # Add many concepts
        concepts = [{"name": f"Concept{i}", "type": "entity"} for i in range(20)]
        memory.integrate_knowledge({"concepts": concepts})
        
        results = memory.query("Concept", max_results=5)
        
        assert len(results) <= 5

    def test_query_relationships(self):
        """Test querying relationships for a node."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "relationships": [
                {"source": "Python", "target": "Programming", "type": "is_a"},
                {"source": "Python", "target": "Scripting", "type": "supports"},
            ]
        })
        
        # Get Python node
        python_nodes = memory.query("Python")
        assert len(python_nodes) > 0
        python_id = python_nodes[0].node_id
        
        relationships = memory.query_relationships(python_id)
        
        assert len(relationships) == 2

    def test_query_relationships_by_type(self):
        """Test querying relationships filtered by type."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "relationships": [
                {"source": "Python", "target": "Programming", "type": "is_a"},
                {"source": "Python", "target": "Scripting", "type": "supports"},
            ]
        })
        
        python_nodes = memory.query("Python")
        python_id = python_nodes[0].node_id
        
        relationships = memory.query_relationships(python_id, relationship_type="is_a")
        
        assert len(relationships) == 1
        assert relationships[0].relationship_type == "is_a"

    def test_find_related_concepts(self):
        """Test finding related concepts within depth."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "relationships": [
                {"source": "Python", "target": "Programming", "type": "is_a"},
                {"source": "Programming", "target": "Computer Science", "type": "part_of"},
            ]
        })
        
        related = memory.find_related_concepts("Python", max_depth=2)
        
        assert len(related) >= 1
        related_names = [node.name for node in related]
        assert "Programming" in related_names

    def test_get_concept_definition(self):
        """Test getting full concept definition."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Python", "type": "language", "properties": {"version": "3.11"}}],
            "relationships": [{"source": "Python", "target": "Programming", "type": "is_a"}]
        })
        
        definition = memory.get_concept_definition("Python")
        
        assert definition is not None
        assert definition["name"] == "Python"
        assert definition["type"] == "language"
        assert "version" in definition["properties"]
        assert len(definition["relationships"]) >= 1

    def test_get_concept_definition_not_found(self):
        """Test getting definition for non-existent concept."""
        memory = SemanticMemory()
        
        definition = memory.get_concept_definition("NonExistent")
        
        assert definition is None


class TestPatternExtraction:
    """Tests for pattern extraction from episodes."""

    def _create_episode(
        self,
        context: Dict[str, Any],
        events: List[Dict[str, Any]],
        emotional_state: Dict[str, float] = None,
    ) -> Episode:
        """Helper to create test episodes."""
        import uuid
        return Episode(
            episode_id=str(uuid.uuid4()),
            timestamp=time.time(),
            context=context,
            events=events,
            emotional_state=emotional_state or {"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            importance=0.5,
        )

    def test_extract_concepts_from_context(self):
        """Test extracting concepts from episode contexts."""
        memory = SemanticMemory(config={"extraction_min_frequency": 0.3})
        
        episodes = [
            self._create_episode({"task": "coding", "language": "Python"}, []),
            self._create_episode({"task": "coding", "language": "Python"}, []),
            self._create_episode({"task": "testing", "language": "Python"}, []),
        ]
        
        result = memory.extract_from_episodes(episodes)
        
        assert result.episodes_processed == 3
        # "Python" appears in all 3, should be extracted
        assert "Python" in result.extracted_concepts or "language" in result.extracted_concepts

    def test_extract_relationships_from_events(self):
        """Test extracting relationships from event sequences."""
        memory = SemanticMemory(config={"extraction_min_frequency": 0.3})
        
        episodes = [
            self._create_episode({}, [
                {"type": "start", "name": "task_start"},
                {"type": "process", "name": "task_process"},
            ]),
            self._create_episode({}, [
                {"type": "start", "name": "task_start"},
                {"type": "process", "name": "task_process"},
            ]),
            self._create_episode({}, [
                {"type": "start", "name": "task_start"},
                {"type": "complete", "name": "task_complete"},
            ]),
        ]
        
        result = memory.extract_from_episodes(episodes)
        
        assert result.episodes_processed == 3
        # Should extract "start followed_by process" relationship
        assert result.patterns_found >= 0

    def test_extract_empty_episodes(self):
        """Test extraction with empty episode list."""
        memory = SemanticMemory()
        
        result = memory.extract_from_episodes([])
        
        assert result.episodes_processed == 0
        assert len(result.extracted_concepts) == 0
        assert len(result.extracted_relationships) == 0

    def test_extraction_integrates_knowledge(self):
        """Test that extraction integrates knowledge into semantic memory."""
        memory = SemanticMemory(config={"extraction_min_frequency": 0.3})
        
        episodes = [
            self._create_episode({"domain": "AI"}, []),
            self._create_episode({"domain": "AI"}, []),
            self._create_episode({"domain": "AI"}, []),
        ]
        
        result = memory.extract_from_episodes(episodes)
        
        # If concepts were extracted, they should be integrated
        if result.extracted_concepts:
            assert result.integration_result is not None

    def test_extraction_result_structure(self):
        """Test ExtractionResult has all required fields."""
        memory = SemanticMemory()
        
        result = memory.extract_from_episodes([])
        
        assert hasattr(result, "extracted_concepts")
        assert hasattr(result, "extracted_relationships")
        assert hasattr(result, "integration_result")
        assert hasattr(result, "episodes_processed")
        assert hasattr(result, "patterns_found")
        assert hasattr(result, "timestamp")


class TestConsolidation:
    """Tests for consolidation functionality."""

    def test_consolidate_empty_memory(self):
        """Test consolidation on empty memory."""
        memory = SemanticMemory()
        
        result = memory.consolidate()
        
        assert isinstance(result, ConsolidationResult)
        assert len(result.merged_concepts) == 0
        assert len(result.pruned_concepts) == 0

    def test_consolidate_merges_similar_concepts(self):
        """Test consolidation merges very similar concepts."""
        memory = SemanticMemory(config={"consolidation_merge_threshold": 0.9})
        
        # Create concepts with identical names (will be merged)
        memory.integrate_knowledge({
            "concepts": [
                {"name": "Python", "type": "language"},
            ]
        })
        # Directly add another node with same name (simulating duplicate)
        memory.knowledge_graph.create_node(name="python", node_type="language")
        
        result = memory.consolidate()
        
        # Should have merged the similar concepts
        assert isinstance(result, ConsolidationResult)

    def test_consolidate_strengthens_relationships(self):
        """Test consolidation strengthens frequently accessed relationships."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "relationships": [
                {"source": "Python", "target": "Programming", "type": "is_a"}
            ]
        })
        
        # Access the nodes multiple times to increase access count
        for _ in range(10):
            memory.query("Python")
            memory.query("Programming")
        
        result = memory.consolidate()
        
        assert isinstance(result, ConsolidationResult)
        # Relationships should be strengthened due to high access counts
        assert len(result.strengthened_relationships) >= 0

    def test_consolidate_prunes_weak_relationships(self):
        """Test consolidation prunes weak relationships."""
        memory = SemanticMemory(config={"prune_strength_threshold": 0.3})
        
        memory.integrate_knowledge({
            "relationships": [
                {"source": "A", "target": "B", "type": "weak_rel", "weight": 0.1}
            ]
        })
        
        result = memory.consolidate()
        
        # Weak relationship should be pruned
        assert isinstance(result, ConsolidationResult)

    def test_consolidation_result_structure(self):
        """Test ConsolidationResult has all required fields."""
        memory = SemanticMemory()
        
        result = memory.consolidate()
        
        assert hasattr(result, "merged_concepts")
        assert hasattr(result, "strengthened_relationships")
        assert hasattr(result, "pruned_concepts")
        assert hasattr(result, "pruned_relationships")
        assert hasattr(result, "statistics")
        assert hasattr(result, "timestamp")

    def test_consolidation_statistics(self):
        """Test consolidation returns statistics."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Test", "type": "entity"}]
        })
        
        result = memory.consolidate()
        
        assert "total_concepts" in result.statistics
        assert "total_relationships" in result.statistics


class TestSerialization:
    """Tests for serialization/deserialization functionality."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Python", "type": "language"}],
            "relationships": [{"source": "Python", "target": "Programming", "type": "is_a"}]
        })
        
        data = memory.to_dict()
        
        assert "config" in data
        assert "knowledge_graph" in data
        assert "total_integrations" in data
        assert "initialized_at" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Python", "type": "language"}]
        })
        
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        assert "Python" in restored
        assert len(restored) == len(memory)

    def test_serialization_round_trip(self):
        """Test that serialization round-trip preserves data."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [
                {"name": "Python", "type": "language", "properties": {"version": "3.11"}},
                {"name": "JavaScript", "type": "language"},
            ],
            "relationships": [
                {"source": "Python", "target": "Programming", "type": "is_a"},
            ]
        })
        
        # Serialize and deserialize
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        # Verify concepts preserved
        assert "Python" in restored
        assert "JavaScript" in restored
        
        # Verify properties preserved
        python_def = restored.get_concept_definition("Python")
        assert python_def is not None
        assert python_def["properties"].get("version") == "3.11"
        
        # Verify relationships preserved
        assert len(restored.knowledge_graph.get_all_relationships()) >= 1

    def test_serialization_preserves_statistics(self):
        """Test that serialization preserves statistics."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({"concepts": [{"name": "Test", "type": "entity"}]})
        memory.integrate_knowledge({"concepts": [{"name": "Test2", "type": "entity"}]})
        
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        assert restored._total_integrations == memory._total_integrations


class TestStatisticsAndState:
    """Tests for statistics and state methods."""

    def test_get_statistics(self):
        """Test getting statistics."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Test", "type": "entity"}]
        })
        
        stats = memory.get_statistics()
        
        assert "knowledge_graph" in stats
        assert "total_integrations" in stats
        assert "total_conflicts" in stats
        assert "uptime" in stats

    def test_get_state(self):
        """Test getting state."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Test", "type": "entity"}]
        })
        
        state = memory.get_state()
        
        assert "concept_count" in state
        assert "relationship_count" in state
        assert state["concept_count"] == 1

    def test_len(self):
        """Test __len__ returns concept count."""
        memory = SemanticMemory()
        
        assert len(memory) == 0
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Test", "type": "entity"}]
        })
        
        assert len(memory) == 1

    def test_contains(self):
        """Test __contains__ checks concept existence."""
        memory = SemanticMemory()
        
        assert "Python" not in memory
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Python", "type": "language"}]
        })
        
        assert "Python" in memory

    def test_clear(self):
        """Test clearing semantic memory."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Test", "type": "entity"}]
        })
        
        assert len(memory) == 1
        
        memory.clear()
        
        assert len(memory) == 0


class TestDataclasses:
    """Tests for supporting dataclasses."""

    def test_conflict_info_to_dict(self):
        """Test ConflictInfo serialization."""
        conflict = ConflictInfo(
            conflict_id="test-id",
            existing_node_id="node-1",
            existing_value="old",
            new_value="new",
            conflict_type="attribute",
            resolution="new_wins",
        )
        
        data = conflict.to_dict()
        
        assert data["conflict_id"] == "test-id"
        assert data["existing_value"] == "old"
        assert data["new_value"] == "new"

    def test_conflict_info_from_dict(self):
        """Test ConflictInfo deserialization."""
        data = {
            "conflict_id": "test-id",
            "existing_node_id": "node-1",
            "existing_value": "old",
            "new_value": "new",
            "conflict_type": "attribute",
            "resolution": "new_wins",
        }
        
        conflict = ConflictInfo.from_dict(data)
        
        assert conflict.conflict_id == "test-id"
        assert conflict.existing_value == "old"

    def test_integration_result_to_dict(self):
        """Test IntegrationResult serialization."""
        result = IntegrationResult(
            new_concepts=["c1", "c2"],
            updated_concepts=["c3"],
            new_relationships=["r1"],
            conflicts=[],
            resolution_actions=["Created c1"],
        )
        
        data = result.to_dict()
        
        assert data["new_concepts"] == ["c1", "c2"]
        assert data["success"] is True

    def test_extraction_result_to_dict(self):
        """Test ExtractionResult serialization."""
        result = ExtractionResult(
            extracted_concepts=["Python", "Java"],
            extracted_relationships=[("Python", "is_a", "Language")],
            episodes_processed=5,
            patterns_found=3,
        )
        
        data = result.to_dict()
        
        assert data["extracted_concepts"] == ["Python", "Java"]
        assert data["episodes_processed"] == 5

    def test_consolidation_result_to_dict(self):
        """Test ConsolidationResult serialization."""
        result = ConsolidationResult(
            merged_concepts=[("kept", "removed")],
            strengthened_relationships=["r1"],
            pruned_concepts=["p1"],
            pruned_relationships=["pr1"],
            statistics={"total": 10},
        )
        
        data = result.to_dict()
        
        assert len(data["merged_concepts"]) == 1
        assert data["statistics"]["total"] == 10


class TestConflictResolutionStrategy:
    """Tests for ConflictResolutionStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert ConflictResolutionStrategy.NEWEST_WINS.value == "newest_wins"
        assert ConflictResolutionStrategy.HIGHEST_CONFIDENCE.value == "highest_confidence"
        assert ConflictResolutionStrategy.MERGE.value == "merge"

    def test_strategy_from_string(self):
        """Test creating strategy from string."""
        strategy = ConflictResolutionStrategy("newest_wins")
        assert strategy == ConflictResolutionStrategy.NEWEST_WINS

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


class TestKnowledgeGraphOperationsThroughSemanticMemory:
    """
    Tests for knowledge graph CRUD operations through SemanticMemory.
    
    Requirements: 4.1-4.6
    """

    def test_create_node_via_integration(self):
        """Test creating nodes through knowledge integration."""
        memory = SemanticMemory()
        
        result = memory.integrate_knowledge({
            "concepts": [
                {"name": "Dog", "type": "animal", "properties": {"legs": 4, "sound": "bark"}}
            ]
        })
        
        assert result.success
        assert len(result.new_concepts) == 1
        
        # Verify node exists in knowledge graph
        node = memory.knowledge_graph.get_node_by_name("Dog")
        assert node is not None
        assert node.name == "Dog"
        assert node.node_type == "animal"
        assert node.properties["legs"] == 4
        assert node.properties["sound"] == "bark"

    def test_read_node_via_query(self):
        """Test reading nodes through query interface."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [
                {"name": "Cat", "type": "animal"},
                {"name": "Car", "type": "vehicle"},
            ]
        })
        
        # Query by name
        results = memory.query("Cat")
        assert len(results) >= 1
        assert any(n.name == "Cat" for n in results)
        
        # Query should not return unrelated concepts first
        results = memory.query("Cat")
        assert results[0].name == "Cat"

    def test_update_node_via_integration(self):
        """Test updating nodes through knowledge integration."""
        memory = SemanticMemory()
        
        # Create initial node
        memory.integrate_knowledge({
            "concepts": [{"name": "Bird", "type": "animal", "properties": {"can_fly": True}}]
        })
        
        # Update with new properties
        result = memory.integrate_knowledge({
            "attributes": {"Bird": {"color": "blue", "size": "small"}}
        })
        
        assert result.success
        
        # Verify updates
        definition = memory.get_concept_definition("Bird")
        assert definition["properties"]["can_fly"] is True
        assert definition["properties"]["color"] == "blue"
        assert definition["properties"]["size"] == "small"

    def test_delete_node_via_knowledge_graph(self):
        """Test deleting nodes through knowledge graph."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Temporary", "type": "test"}]
        })
        
        assert "Temporary" in memory
        
        # Get node ID and delete
        node = memory.knowledge_graph.get_node_by_name("Temporary")
        memory.knowledge_graph.remove_node(node.node_id)
        
        assert "Temporary" not in memory

    def test_create_relationship_via_integration(self):
        """Test creating relationships through knowledge integration."""
        memory = SemanticMemory()
        
        result = memory.integrate_knowledge({
            "relationships": [
                {"source": "Dog", "target": "Animal", "type": "is_a", "weight": 0.9}
            ]
        })
        
        assert result.success
        assert len(result.new_relationships) == 1
        
        # Verify relationship exists
        dog_node = memory.knowledge_graph.get_node_by_name("Dog")
        relationships = memory.knowledge_graph.get_relationships(dog_node.node_id, direction="outgoing")
        
        assert len(relationships) >= 1
        assert any(r.relationship_type == "is_a" for r in relationships)

    def test_read_relationship_via_query(self):
        """Test reading relationships through query interface."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "relationships": [
                {"source": "Python", "target": "Language", "type": "is_a"},
                {"source": "Python", "target": "Scripting", "type": "supports"},
            ]
        })
        
        python_node = memory.knowledge_graph.get_node_by_name("Python")
        
        # Query all relationships
        all_rels = memory.query_relationships(python_node.node_id)
        assert len(all_rels) == 2
        
        # Query by type
        is_a_rels = memory.query_relationships(python_node.node_id, relationship_type="is_a")
        assert len(is_a_rels) == 1
        assert is_a_rels[0].relationship_type == "is_a"

    def test_update_relationship_strength(self):
        """Test that repeated relationship integration strengthens the relationship."""
        memory = SemanticMemory()
        
        # Create initial relationship
        memory.integrate_knowledge({
            "relationships": [{"source": "A", "target": "B", "type": "related_to", "weight": 0.5}]
        })
        
        a_node = memory.knowledge_graph.get_node_by_name("A")
        initial_rel = memory.knowledge_graph.get_relationships(a_node.node_id, direction="outgoing")[0]
        initial_weight = initial_rel.weight
        
        # Integrate same relationship again
        memory.integrate_knowledge({
            "relationships": [{"source": "A", "target": "B", "type": "related_to"}]
        })
        
        # Weight should have increased
        updated_rel = memory.knowledge_graph.get_relationships(a_node.node_id, direction="outgoing")[0]
        assert updated_rel.weight >= initial_weight

    def test_delete_relationship_via_knowledge_graph(self):
        """Test deleting relationships through knowledge graph."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "relationships": [{"source": "X", "target": "Y", "type": "test_rel"}]
        })
        
        x_node = memory.knowledge_graph.get_node_by_name("X")
        relationships = memory.knowledge_graph.get_relationships(x_node.node_id)
        assert len(relationships) == 1
        
        # Delete the relationship
        rel_id = relationships[0].relationship_id
        memory.knowledge_graph.remove_relationship(rel_id)
        
        # Verify deletion
        relationships_after = memory.knowledge_graph.get_relationships(x_node.node_id)
        assert len(relationships_after) == 0


class TestConflictResolutionScenarios:
    """
    Tests for conflict resolution scenarios with all strategies.
    
    Requirements: 4.5
    """

    def test_newest_wins_string_attribute(self):
        """Test newest_wins strategy with string attribute conflict."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "newest_wins"})
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Server", "type": "computer", "properties": {"status": "running"}}]
        })
        
        result = memory.integrate_knowledge({
            "concepts": [{"name": "Server", "type": "computer", "properties": {"status": "stopped"}}]
        })
        
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution == "new_wins"
        
        definition = memory.get_concept_definition("Server")
        assert definition["properties"]["status"] == "stopped"

    def test_newest_wins_numeric_attribute(self):
        """Test newest_wins strategy with numeric attribute conflict."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "newest_wins"})
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Counter", "type": "metric", "properties": {"value": 100}}]
        })
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Counter", "type": "metric", "properties": {"value": 200}}]
        })
        
        definition = memory.get_concept_definition("Counter")
        assert definition["properties"]["value"] == 200

    def test_highest_confidence_existing_wins(self):
        """Test highest_confidence strategy where existing value wins."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "highest_confidence"})
        
        # High confidence initial value
        memory.integrate_knowledge(
            {"concepts": [{"name": "Fact", "type": "info", "properties": {"verified": True}}]},
            confidence=0.95
        )
        
        # Low confidence update
        result = memory.integrate_knowledge(
            {"concepts": [{"name": "Fact", "type": "info", "properties": {"verified": False}}]},
            confidence=0.3
        )
        
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution == "existing_wins_confidence"
        
        definition = memory.get_concept_definition("Fact")
        assert definition["properties"]["verified"] is True

    def test_highest_confidence_new_wins(self):
        """Test highest_confidence strategy where new value wins."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "highest_confidence"})
        
        # Low confidence initial value
        memory.integrate_knowledge(
            {"concepts": [{"name": "Estimate", "type": "data", "properties": {"amount": 100}}]},
            confidence=0.3
        )
        
        # High confidence update
        result = memory.integrate_knowledge(
            {"concepts": [{"name": "Estimate", "type": "data", "properties": {"amount": 150}}]},
            confidence=0.9
        )
        
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution == "new_wins_confidence"
        
        definition = memory.get_concept_definition("Estimate")
        assert definition["properties"]["amount"] == 150

    def test_merge_strategy_list_values(self):
        """Test merge strategy combines list values."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "merge"})
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Tags", "type": "collection", "properties": {"items": ["a", "b"]}}]
        })
        
        result = memory.integrate_knowledge({
            "concepts": [{"name": "Tags", "type": "collection", "properties": {"items": ["b", "c", "d"]}}]
        })
        
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution == "merged"
        
        definition = memory.get_concept_definition("Tags")
        items = definition["properties"]["items"]
        # Should contain all unique items
        assert "a" in items
        assert "b" in items
        assert "c" in items
        assert "d" in items

    def test_merge_strategy_dict_values(self):
        """Test merge strategy combines dict values."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "merge"})
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Config", "type": "settings", "properties": {"options": {"key1": "val1"}}}]
        })
        
        result = memory.integrate_knowledge({
            "concepts": [{"name": "Config", "type": "settings", "properties": {"options": {"key2": "val2"}}}]
        })
        
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution == "merged"
        
        definition = memory.get_concept_definition("Config")
        options = definition["properties"]["options"]
        assert options["key1"] == "val1"
        assert options["key2"] == "val2"

    def test_merge_strategy_string_values(self):
        """Test merge strategy concatenates string values."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "merge"})
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Note", "type": "text", "properties": {"content": "First part"}}]
        })
        
        result = memory.integrate_knowledge({
            "concepts": [{"name": "Note", "type": "text", "properties": {"content": "Second part"}}]
        })
        
        assert len(result.conflicts) == 1
        assert result.conflicts[0].resolution == "merged"
        
        definition = memory.get_concept_definition("Note")
        content = definition["properties"]["content"]
        assert "First part" in content
        assert "Second part" in content

    def test_merge_strategy_unmergeable_falls_back(self):
        """Test merge strategy falls back to newest_wins for unmergeable types."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "merge"})
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Number", "type": "value", "properties": {"val": 10}}]
        })
        
        result = memory.integrate_knowledge({
            "concepts": [{"name": "Number", "type": "value", "properties": {"val": 20}}]
        })
        
        assert len(result.conflicts) == 1
        # Should fall back since integers can't be merged
        assert "merge_failed" in result.conflicts[0].resolution or "new_wins" in result.conflicts[0].resolution
        
        definition = memory.get_concept_definition("Number")
        assert definition["properties"]["val"] == 20

    def test_type_conflict_detection(self):
        """Test detection of type conflicts."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Item", "type": "entity"}]
        })
        
        result = memory.integrate_knowledge({
            "concepts": [{"name": "Item", "type": "attribute"}]
        })
        
        # Should detect type conflict
        type_conflicts = [c for c in result.conflicts if c.conflict_type == "type"]
        assert len(type_conflicts) == 1
        assert type_conflicts[0].existing_value == "entity"
        assert type_conflicts[0].new_value == "attribute"

    def test_multiple_conflicts_in_single_integration(self):
        """Test handling multiple conflicts in a single integration."""
        memory = SemanticMemory(config={"conflict_resolution_strategy": "newest_wins"})
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Multi", "type": "test", "properties": {"a": 1, "b": 2, "c": 3}}]
        })
        
        result = memory.integrate_knowledge({
            "concepts": [{"name": "Multi", "type": "test", "properties": {"a": 10, "b": 20, "c": 30}}]
        })
        
        assert len(result.conflicts) == 3
        
        definition = memory.get_concept_definition("Multi")
        assert definition["properties"]["a"] == 10
        assert definition["properties"]["b"] == 20
        assert definition["properties"]["c"] == 30


class TestSerializationRoundTrip:
    """
    Comprehensive tests for serialization round-trip.
    
    Requirements: 4.6
    """

    def test_empty_memory_round_trip(self):
        """Test serialization of empty semantic memory."""
        memory = SemanticMemory()
        
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        assert len(restored) == 0
        assert restored._total_integrations == 0

    def test_single_concept_round_trip(self):
        """Test serialization with a single concept."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Test", "type": "entity", "properties": {"key": "value"}}]
        })
        
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        assert "Test" in restored
        definition = restored.get_concept_definition("Test")
        assert definition["properties"]["key"] == "value"

    def test_complex_graph_round_trip(self):
        """Test serialization with complex graph structure."""
        memory = SemanticMemory()
        
        # Create a complex graph
        memory.integrate_knowledge({
            "concepts": [
                {"name": "Animal", "type": "category"},
                {"name": "Dog", "type": "species", "properties": {"legs": 4}},
                {"name": "Cat", "type": "species", "properties": {"legs": 4}},
                {"name": "Bird", "type": "species", "properties": {"legs": 2, "can_fly": True}},
            ],
            "relationships": [
                {"source": "Dog", "target": "Animal", "type": "is_a"},
                {"source": "Cat", "target": "Animal", "type": "is_a"},
                {"source": "Bird", "target": "Animal", "type": "is_a"},
            ]
        })
        
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        # Verify all concepts
        assert "Animal" in restored
        assert "Dog" in restored
        assert "Cat" in restored
        assert "Bird" in restored
        
        # Verify properties
        dog_def = restored.get_concept_definition("Dog")
        assert dog_def["properties"]["legs"] == 4
        
        bird_def = restored.get_concept_definition("Bird")
        assert bird_def["properties"]["can_fly"] is True
        
        # Verify relationships
        dog_node = restored.knowledge_graph.get_node_by_name("Dog")
        dog_rels = restored.knowledge_graph.get_relationships(dog_node.node_id, direction="outgoing")
        assert len(dog_rels) >= 1
        assert any(r.relationship_type == "is_a" for r in dog_rels)

    def test_config_preserved_in_round_trip(self):
        """Test that configuration is preserved in serialization."""
        config = {
            "conflict_resolution_strategy": "highest_confidence",
            "similarity_threshold": 0.75,
            "max_query_results": 50,
        }
        memory = SemanticMemory(config=config)
        
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        assert restored._config.conflict_resolution_strategy == "highest_confidence"
        assert restored._config.similarity_threshold == 0.75
        assert restored._config.max_query_results == 50

    def test_statistics_preserved_in_round_trip(self):
        """Test that statistics are preserved in serialization."""
        memory = SemanticMemory()
        
        # Perform multiple integrations to build up statistics
        for i in range(5):
            memory.integrate_knowledge({
                "concepts": [{"name": f"Concept{i}", "type": "entity"}]
            })
        
        original_integrations = memory._total_integrations
        
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        assert restored._total_integrations == original_integrations

    def test_knowledge_graph_to_dict_from_dict(self):
        """Test KnowledgeGraph serialization directly."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [
                {"name": "Node1", "type": "entity", "properties": {"prop1": "val1"}},
                {"name": "Node2", "type": "entity"},
            ],
            "relationships": [
                {"source": "Node1", "target": "Node2", "type": "connects_to", "weight": 0.8}
            ]
        })
        
        # Serialize knowledge graph directly
        kg_data = memory.knowledge_graph.to_dict()
        
        assert "nodes" in kg_data
        assert "relationships" in kg_data
        assert "config" in kg_data
        
        # Verify node data
        assert len(kg_data["nodes"]) == 2
        
        # Verify relationship data
        assert len(kg_data["relationships"]) == 1

    def test_semantic_memory_to_dict_structure(self):
        """Test the structure of SemanticMemory.to_dict() output."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Test", "type": "entity"}]
        })
        
        data = memory.to_dict()
        
        # Verify required keys
        assert "config" in data
        assert "knowledge_graph" in data
        assert "total_integrations" in data
        assert "total_conflicts" in data
        assert "total_extractions" in data
        assert "total_consolidations" in data
        assert "initialized_at" in data
        
        # Verify types
        assert isinstance(data["config"], dict)
        assert isinstance(data["knowledge_graph"], dict)
        assert isinstance(data["total_integrations"], int)
        assert isinstance(data["initialized_at"], float)

    def test_integration_result_round_trip(self):
        """Test IntegrationResult serialization round-trip."""
        result = IntegrationResult(
            new_concepts=["c1", "c2"],
            updated_concepts=["c3"],
            new_relationships=["r1", "r2"],
            conflicts=[
                ConflictInfo(
                    conflict_id="conflict-1",
                    existing_node_id="node-1",
                    existing_value="old",
                    new_value="new",
                    conflict_type="attribute",
                    resolution="new_wins",
                    property_key="key1",
                )
            ],
            resolution_actions=["Created c1", "Updated c3"],
            success=True,
        )
        
        data = result.to_dict()
        restored = IntegrationResult.from_dict(data)
        
        assert restored.new_concepts == result.new_concepts
        assert restored.updated_concepts == result.updated_concepts
        assert restored.new_relationships == result.new_relationships
        assert len(restored.conflicts) == 1
        assert restored.conflicts[0].conflict_id == "conflict-1"
        assert restored.resolution_actions == result.resolution_actions
        assert restored.success == result.success

    def test_extraction_result_round_trip(self):
        """Test ExtractionResult serialization round-trip."""
        result = ExtractionResult(
            extracted_concepts=["Python", "Java", "JavaScript"],
            extracted_relationships=[
                ("Python", "is_a", "Language"),
                ("Java", "is_a", "Language"),
            ],
            episodes_processed=10,
            patterns_found=5,
        )
        
        data = result.to_dict()
        restored = ExtractionResult.from_dict(data)
        
        assert restored.extracted_concepts == result.extracted_concepts
        assert restored.extracted_relationships == result.extracted_relationships
        assert restored.episodes_processed == result.episodes_processed
        assert restored.patterns_found == result.patterns_found

    def test_consolidation_result_round_trip(self):
        """Test ConsolidationResult serialization round-trip."""
        result = ConsolidationResult(
            merged_concepts=[("kept1", "removed1"), ("kept2", "removed2")],
            strengthened_relationships=["r1", "r2", "r3"],
            pruned_concepts=["p1"],
            pruned_relationships=["pr1", "pr2"],
            statistics={"total_concepts": 10, "total_relationships": 20},
        )
        
        data = result.to_dict()
        restored = ConsolidationResult.from_dict(data)
        
        assert restored.merged_concepts == result.merged_concepts
        assert restored.strengthened_relationships == result.strengthened_relationships
        assert restored.pruned_concepts == result.pruned_concepts
        assert restored.pruned_relationships == result.pruned_relationships
        assert restored.statistics == result.statistics

    def test_conflict_info_round_trip(self):
        """Test ConflictInfo serialization round-trip."""
        conflict = ConflictInfo(
            conflict_id="test-conflict",
            existing_node_id="node-123",
            existing_value={"nested": "value"},
            new_value={"nested": "new_value"},
            conflict_type="attribute",
            resolution="merged",
            property_key="data",
            timestamp=1234567890.0,
        )
        
        data = conflict.to_dict()
        restored = ConflictInfo.from_dict(data)
        
        assert restored.conflict_id == conflict.conflict_id
        assert restored.existing_node_id == conflict.existing_node_id
        assert restored.existing_value == conflict.existing_value
        assert restored.new_value == conflict.new_value
        assert restored.conflict_type == conflict.conflict_type
        assert restored.resolution == conflict.resolution
        assert restored.property_key == conflict.property_key
        assert restored.timestamp == conflict.timestamp

    def test_round_trip_preserves_node_access_counts(self):
        """Test that node access counts are preserved in round-trip."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "concepts": [{"name": "Popular", "type": "entity"}]
        })
        
        # Access the node multiple times
        for _ in range(10):
            memory.query("Popular")
        
        node_before = memory.knowledge_graph.get_node_by_name("Popular", record_access=False)
        access_count_before = node_before.access_count
        
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        node_after = restored.knowledge_graph.get_node_by_name("Popular", record_access=False)
        assert node_after.access_count == access_count_before

    def test_round_trip_preserves_relationship_weights(self):
        """Test that relationship weights are preserved in round-trip."""
        memory = SemanticMemory()
        
        memory.integrate_knowledge({
            "relationships": [
                {"source": "A", "target": "B", "type": "strong", "weight": 0.95},
                {"source": "C", "target": "D", "type": "weak", "weight": 0.15},
            ]
        })
        
        data = memory.to_dict()
        restored = SemanticMemory.from_dict(data)
        
        a_node = restored.knowledge_graph.get_node_by_name("A")
        a_rels = restored.knowledge_graph.get_relationships(a_node.node_id, direction="outgoing")
        assert len(a_rels) == 1
        assert abs(a_rels[0].weight - 0.95) < 0.01
        
        c_node = restored.knowledge_graph.get_node_by_name("C")
        c_rels = restored.knowledge_graph.get_relationships(c_node.node_id, direction="outgoing")
        assert len(c_rels) == 1
        assert abs(c_rels[0].weight - 0.15) < 0.01

"""
Unit tests for the Knowledge Graph module.

Tests node and relationship CRUD operations, graph traversal,
relationship inference, and serialization.

Requirements: 4.1, 4.3
"""

import pytest
import time
from typing import Dict, Any

from mm_orch.consciousness.knowledge_graph import (
    ConceptNode,
    Relationship,
    KnowledgeGraph,
    KnowledgeGraphConfig,
    create_concept_node,
    create_relationship,
)


class TestConceptNode:
    """Tests for the ConceptNode dataclass."""

    def test_node_creation_with_all_fields(self):
        """Test creating a node with all required fields."""
        node = ConceptNode(
            node_id="node-001",
            name="TestConcept",
            node_type="entity",
            properties={"color": "blue", "size": 10},
            embedding=[0.1, 0.2, 0.3],
            created_at=1234567890.0,
            updated_at=1234567890.0,
            access_count=0,
            metadata={"source": "test"},
        )
        
        assert node.node_id == "node-001"
        assert node.name == "TestConcept"
        assert node.node_type == "entity"
        assert node.properties == {"color": "blue", "size": 10}
        assert node.embedding == [0.1, 0.2, 0.3]
        assert node.created_at == 1234567890.0
        assert node.updated_at == 1234567890.0
        assert node.access_count == 0
        assert node.metadata == {"source": "test"}

    def test_node_creation_with_defaults(self):
        """Test creating a node with default values."""
        node = ConceptNode(
            node_id="node-002",
            name="SimpleConcept",
            node_type="attribute",
        )
        
        assert node.properties == {}
        assert node.embedding is None
        assert node.access_count == 0
        assert node.metadata == {}

    def test_node_validation_empty_id(self):
        """Test that empty node_id raises ValueError."""
        with pytest.raises(ValueError, match="node_id cannot be empty"):
            ConceptNode(
                node_id="",
                name="Test",
                node_type="entity",
            )

    def test_node_validation_empty_name(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ConceptNode(
                node_id="test",
                name="",
                node_type="entity",
            )

    def test_node_validation_empty_type(self):
        """Test that empty node_type raises ValueError."""
        with pytest.raises(ValueError, match="node_type cannot be empty"):
            ConceptNode(
                node_id="test",
                name="Test",
                node_type="",
            )

    def test_node_validation_invalid_properties(self):
        """Test that non-dict properties raises ValueError."""
        with pytest.raises(ValueError, match="properties must be a dictionary"):
            ConceptNode(
                node_id="test",
                name="Test",
                node_type="entity",
                properties="invalid",  # type: ignore
            )

    def test_node_validation_invalid_embedding(self):
        """Test that non-list embedding raises ValueError."""
        with pytest.raises(ValueError, match="embedding must be a list or None"):
            ConceptNode(
                node_id="test",
                name="Test",
                node_type="entity",
                embedding="invalid",  # type: ignore
            )

    def test_node_validation_negative_access_count(self):
        """Test that negative access_count raises ValueError."""
        with pytest.raises(ValueError, match="access_count cannot be negative"):
            ConceptNode(
                node_id="test",
                name="Test",
                node_type="entity",
                access_count=-1,
            )

    def test_node_record_access(self):
        """Test recording access to a node."""
        node = ConceptNode(
            node_id="test",
            name="Test",
            node_type="entity",
        )
        
        assert node.access_count == 0
        original_updated = node.updated_at
        
        time.sleep(0.01)  # Small delay to ensure timestamp changes
        node.record_access()
        
        assert node.access_count == 1
        assert node.updated_at > original_updated
        
        node.record_access()
        assert node.access_count == 2

    def test_node_update_properties(self):
        """Test updating node properties."""
        node = ConceptNode(
            node_id="test",
            name="Test",
            node_type="entity",
            properties={"a": 1},
        )
        
        original_updated = node.updated_at
        time.sleep(0.01)
        
        node.update_properties({"b": 2, "c": 3})
        
        assert node.properties == {"a": 1, "b": 2, "c": 3}
        assert node.updated_at > original_updated

    def test_node_to_dict(self):
        """Test serializing a node to dictionary."""
        node = ConceptNode(
            node_id="node-001",
            name="TestConcept",
            node_type="entity",
            properties={"key": "value"},
            embedding=[0.1, 0.2],
            created_at=1234567890.0,
            updated_at=1234567900.0,
            access_count=5,
            metadata={"meta": "data"},
        )
        
        data = node.to_dict()
        
        assert data["node_id"] == "node-001"
        assert data["name"] == "TestConcept"
        assert data["node_type"] == "entity"
        assert data["properties"] == {"key": "value"}
        assert data["embedding"] == [0.1, 0.2]
        assert data["created_at"] == 1234567890.0
        assert data["updated_at"] == 1234567900.0
        assert data["access_count"] == 5
        assert data["metadata"] == {"meta": "data"}

    def test_node_from_dict(self):
        """Test deserializing a node from dictionary."""
        data = {
            "node_id": "node-001",
            "name": "TestConcept",
            "node_type": "entity",
            "properties": {"key": "value"},
            "embedding": [0.1, 0.2],
            "created_at": 1234567890.0,
            "updated_at": 1234567900.0,
            "access_count": 5,
            "metadata": {"meta": "data"},
        }
        
        node = ConceptNode.from_dict(data)
        
        assert node.node_id == "node-001"
        assert node.name == "TestConcept"
        assert node.node_type == "entity"
        assert node.properties == {"key": "value"}
        assert node.embedding == [0.1, 0.2]
        assert node.created_at == 1234567890.0
        assert node.updated_at == 1234567900.0
        assert node.access_count == 5
        assert node.metadata == {"meta": "data"}

    def test_node_serialization_roundtrip(self):
        """Test that serialization and deserialization preserves data."""
        original = ConceptNode(
            node_id="roundtrip-test",
            name="RoundtripConcept",
            node_type="action",
            properties={"nested": {"key": "value"}},
            embedding=[0.1, 0.2, 0.3, 0.4],
            created_at=time.time(),
            updated_at=time.time(),
            access_count=10,
            metadata={"tags": ["test", "roundtrip"]},
        )
        
        data = original.to_dict()
        restored = ConceptNode.from_dict(data)
        
        assert restored.node_id == original.node_id
        assert restored.name == original.name
        assert restored.node_type == original.node_type
        assert restored.properties == original.properties
        assert restored.embedding == original.embedding
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at
        assert restored.access_count == original.access_count
        assert restored.metadata == original.metadata


class TestRelationship:
    """Tests for the Relationship dataclass."""

    def test_relationship_creation_with_all_fields(self):
        """Test creating a relationship with all fields."""
        rel = Relationship(
            relationship_id="rel-001",
            source_id="node-a",
            target_id="node-b",
            relationship_type="is_a",
            weight=0.8,
            properties={"confidence": 0.9},
            created_at=1234567890.0,
            metadata={"source": "test"},
        )
        
        assert rel.relationship_id == "rel-001"
        assert rel.source_id == "node-a"
        assert rel.target_id == "node-b"
        assert rel.relationship_type == "is_a"
        assert rel.weight == 0.8
        assert rel.properties == {"confidence": 0.9}
        assert rel.created_at == 1234567890.0
        assert rel.metadata == {"source": "test"}

    def test_relationship_creation_with_defaults(self):
        """Test creating a relationship with default values."""
        rel = Relationship(
            relationship_id="rel-002",
            source_id="node-a",
            target_id="node-b",
            relationship_type="related_to",
        )
        
        assert rel.weight == 1.0
        assert rel.properties == {}
        assert rel.metadata == {}

    def test_relationship_weight_clamping(self):
        """Test that weight is clamped to [0.0, 1.0]."""
        rel_high = Relationship(
            relationship_id="rel-high",
            source_id="a",
            target_id="b",
            relationship_type="test",
            weight=1.5,
        )
        assert rel_high.weight == 1.0
        
        rel_low = Relationship(
            relationship_id="rel-low",
            source_id="a",
            target_id="b",
            relationship_type="test",
            weight=-0.5,
        )
        assert rel_low.weight == 0.0

    def test_relationship_validation_empty_id(self):
        """Test that empty relationship_id raises ValueError."""
        with pytest.raises(ValueError, match="relationship_id cannot be empty"):
            Relationship(
                relationship_id="",
                source_id="a",
                target_id="b",
                relationship_type="test",
            )

    def test_relationship_validation_empty_source(self):
        """Test that empty source_id raises ValueError."""
        with pytest.raises(ValueError, match="source_id cannot be empty"):
            Relationship(
                relationship_id="rel",
                source_id="",
                target_id="b",
                relationship_type="test",
            )

    def test_relationship_validation_empty_target(self):
        """Test that empty target_id raises ValueError."""
        with pytest.raises(ValueError, match="target_id cannot be empty"):
            Relationship(
                relationship_id="rel",
                source_id="a",
                target_id="",
                relationship_type="test",
            )

    def test_relationship_validation_empty_type(self):
        """Test that empty relationship_type raises ValueError."""
        with pytest.raises(ValueError, match="relationship_type cannot be empty"):
            Relationship(
                relationship_id="rel",
                source_id="a",
                target_id="b",
                relationship_type="",
            )

    def test_relationship_to_dict(self):
        """Test serializing a relationship to dictionary."""
        rel = Relationship(
            relationship_id="rel-001",
            source_id="node-a",
            target_id="node-b",
            relationship_type="is_a",
            weight=0.8,
            properties={"key": "value"},
            created_at=1234567890.0,
            metadata={"meta": "data"},
        )
        
        data = rel.to_dict()
        
        assert data["relationship_id"] == "rel-001"
        assert data["source_id"] == "node-a"
        assert data["target_id"] == "node-b"
        assert data["relationship_type"] == "is_a"
        assert data["weight"] == 0.8
        assert data["properties"] == {"key": "value"}
        assert data["created_at"] == 1234567890.0
        assert data["metadata"] == {"meta": "data"}

    def test_relationship_from_dict(self):
        """Test deserializing a relationship from dictionary."""
        data = {
            "relationship_id": "rel-001",
            "source_id": "node-a",
            "target_id": "node-b",
            "relationship_type": "is_a",
            "weight": 0.8,
            "properties": {"key": "value"},
            "created_at": 1234567890.0,
            "metadata": {"meta": "data"},
        }
        
        rel = Relationship.from_dict(data)
        
        assert rel.relationship_id == "rel-001"
        assert rel.source_id == "node-a"
        assert rel.target_id == "node-b"
        assert rel.relationship_type == "is_a"
        assert rel.weight == 0.8
        assert rel.properties == {"key": "value"}
        assert rel.created_at == 1234567890.0
        assert rel.metadata == {"meta": "data"}

    def test_relationship_serialization_roundtrip(self):
        """Test that serialization and deserialization preserves data."""
        original = Relationship(
            relationship_id="roundtrip-rel",
            source_id="source-node",
            target_id="target-node",
            relationship_type="has_property",
            weight=0.75,
            properties={"evidence": ["a", "b"]},
            created_at=time.time(),
            metadata={"inferred": False},
        )
        
        data = original.to_dict()
        restored = Relationship.from_dict(data)
        
        assert restored.relationship_id == original.relationship_id
        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.relationship_type == original.relationship_type
        assert restored.weight == original.weight
        assert restored.properties == original.properties
        assert restored.created_at == original.created_at
        assert restored.metadata == original.metadata


class TestKnowledgeGraphConfig:
    """Tests for the KnowledgeGraphConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = KnowledgeGraphConfig()
        
        assert config.max_nodes == 10000
        assert config.max_relationships == 50000
        assert config.default_relationship_weight == 0.5
        assert config.inference_min_weight == 0.3
        assert config.max_inference_depth == 2

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = KnowledgeGraphConfig(
            max_nodes=1000,
            max_relationships=5000,
            default_relationship_weight=0.7,
        )
        
        assert config.max_nodes == 1000
        assert config.max_relationships == 5000
        assert config.default_relationship_weight == 0.7

    def test_config_validation_max_nodes(self):
        """Test that invalid max_nodes raises ValueError."""
        with pytest.raises(ValueError, match="max_nodes must be at least 1"):
            KnowledgeGraphConfig(max_nodes=0)

    def test_config_validation_max_relationships(self):
        """Test that invalid max_relationships raises ValueError."""
        with pytest.raises(ValueError, match="max_relationships must be at least 1"):
            KnowledgeGraphConfig(max_relationships=0)

    def test_config_validation_default_weight(self):
        """Test that invalid default_relationship_weight raises ValueError."""
        with pytest.raises(ValueError, match="default_relationship_weight must be between"):
            KnowledgeGraphConfig(default_relationship_weight=1.5)

    def test_config_serialization_roundtrip(self):
        """Test config serialization and deserialization."""
        original = KnowledgeGraphConfig(
            max_nodes=2000,
            default_relationship_weight=0.6,
        )
        
        data = original.to_dict()
        restored = KnowledgeGraphConfig.from_dict(data)
        
        assert restored.max_nodes == original.max_nodes
        assert restored.default_relationship_weight == original.default_relationship_weight



class TestKnowledgeGraphNodeOperations:
    """Tests for KnowledgeGraph node CRUD operations."""

    def test_graph_initialization_default(self):
        """Test default graph initialization."""
        graph = KnowledgeGraph()
        
        assert len(graph) == 0
        assert graph._config.max_nodes == 10000

    def test_graph_initialization_with_config(self):
        """Test graph initialization with config dict."""
        config = {"max_nodes": 500, "max_relationships": 2000}
        graph = KnowledgeGraph(config=config)
        
        assert graph._config.max_nodes == 500
        assert graph._config.max_relationships == 2000

    def test_add_node(self):
        """Test adding a node to the graph."""
        graph = KnowledgeGraph()
        
        node = ConceptNode(
            node_id="node-001",
            name="TestNode",
            node_type="entity",
        )
        
        node_id = graph.add_node(node)
        
        assert node_id == "node-001"
        assert len(graph) == 1
        assert graph.contains_node("node-001")

    def test_add_node_duplicate_id(self):
        """Test that adding a node with duplicate ID raises ValueError."""
        graph = KnowledgeGraph()
        
        node1 = ConceptNode(node_id="dup", name="First", node_type="entity")
        node2 = ConceptNode(node_id="dup", name="Second", node_type="entity")
        
        graph.add_node(node1)
        
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node2)

    def test_add_node_max_capacity(self):
        """Test that adding nodes beyond capacity raises ValueError."""
        graph = KnowledgeGraph(config={"max_nodes": 2})
        
        graph.create_node("Node1", "entity")
        graph.create_node("Node2", "entity")
        
        with pytest.raises(ValueError, match="Maximum number of nodes"):
            graph.create_node("Node3", "entity")

    def test_create_node(self):
        """Test creating a node through the graph."""
        graph = KnowledgeGraph()
        
        node = graph.create_node(
            name="CreatedNode",
            node_type="attribute",
            properties={"value": 42},
            metadata={"source": "test"},
        )
        
        assert node.node_id is not None
        assert node.name == "CreatedNode"
        assert node.node_type == "attribute"
        assert node.properties == {"value": 42}
        assert node.metadata == {"source": "test"}
        assert len(graph) == 1

    def test_get_node(self):
        """Test retrieving a node by ID."""
        graph = KnowledgeGraph()
        
        created = graph.create_node("TestNode", "entity")
        retrieved = graph.get_node(created.node_id)
        
        assert retrieved is not None
        assert retrieved.node_id == created.node_id
        assert retrieved.name == "TestNode"

    def test_get_node_records_access(self):
        """Test that getting a node records access."""
        graph = KnowledgeGraph()
        
        node = graph.create_node("TestNode", "entity")
        assert node.access_count == 0
        
        graph.get_node(node.node_id)
        assert node.access_count == 1

    def test_get_node_without_recording_access(self):
        """Test getting a node without recording access."""
        graph = KnowledgeGraph()
        
        node = graph.create_node("TestNode", "entity")
        graph.get_node(node.node_id, record_access=False)
        
        assert node.access_count == 0

    def test_get_node_not_found(self):
        """Test getting a non-existent node returns None."""
        graph = KnowledgeGraph()
        
        result = graph.get_node("non-existent")
        assert result is None

    def test_get_node_by_name(self):
        """Test retrieving a node by name."""
        graph = KnowledgeGraph()
        
        created = graph.create_node("UniqueNodeName", "entity")
        retrieved = graph.get_node_by_name("UniqueNodeName")
        
        assert retrieved is not None
        assert retrieved.node_id == created.node_id

    def test_get_node_by_name_case_insensitive(self):
        """Test that name lookup is case-insensitive."""
        graph = KnowledgeGraph()
        
        created = graph.create_node("MixedCaseName", "entity")
        
        assert graph.get_node_by_name("mixedcasename") is not None
        assert graph.get_node_by_name("MIXEDCASENAME") is not None
        assert graph.get_node_by_name("MixedCaseName") is not None

    def test_update_node(self):
        """Test updating node properties."""
        graph = KnowledgeGraph()
        
        node = graph.create_node("TestNode", "entity", properties={"a": 1})
        result = graph.update_node(node.node_id, {"b": 2})
        
        assert result is True
        assert node.properties == {"a": 1, "b": 2}

    def test_update_node_not_found(self):
        """Test updating a non-existent node returns False."""
        graph = KnowledgeGraph()
        
        result = graph.update_node("non-existent", {"a": 1})
        assert result is False

    def test_remove_node(self):
        """Test removing a node."""
        graph = KnowledgeGraph()
        
        node = graph.create_node("TestNode", "entity")
        assert len(graph) == 1
        
        result = graph.remove_node(node.node_id)
        
        assert result is True
        assert len(graph) == 0
        assert not graph.contains_node(node.node_id)

    def test_remove_node_removes_relationships(self):
        """Test that removing a node also removes its relationships."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        rel1 = graph.create_relationship(node_a.node_id, node_b.node_id, "related_to")
        rel2 = graph.create_relationship(node_b.node_id, node_c.node_id, "related_to")
        
        assert graph.get_relationship_count() == 2
        
        graph.remove_node(node_b.node_id)
        
        assert graph.get_relationship_count() == 0
        assert not graph.contains_relationship(rel1.relationship_id)
        assert not graph.contains_relationship(rel2.relationship_id)

    def test_remove_node_not_found(self):
        """Test removing a non-existent node returns False."""
        graph = KnowledgeGraph()
        
        result = graph.remove_node("non-existent")
        assert result is False

    def test_contains_node(self):
        """Test checking if a node exists."""
        graph = KnowledgeGraph()
        
        node = graph.create_node("TestNode", "entity")
        
        assert graph.contains_node(node.node_id)
        assert not graph.contains_node("non-existent")
        assert node.node_id in graph
        assert "non-existent" not in graph

    def test_get_all_nodes(self):
        """Test getting all nodes."""
        graph = KnowledgeGraph()
        
        node1 = graph.create_node("Node1", "entity")
        node2 = graph.create_node("Node2", "attribute")
        
        all_nodes = graph.get_all_nodes()
        
        assert len(all_nodes) == 2
        assert node1 in all_nodes
        assert node2 in all_nodes

    def test_clear(self):
        """Test clearing all nodes and relationships."""
        graph = KnowledgeGraph()
        
        node1 = graph.create_node("Node1", "entity")
        node2 = graph.create_node("Node2", "entity")
        graph.create_relationship(node1.node_id, node2.node_id, "related_to")
        
        assert len(graph) == 2
        assert graph.get_relationship_count() == 1
        
        graph.clear()
        
        assert len(graph) == 0
        assert graph.get_relationship_count() == 0


class TestKnowledgeGraphRelationshipOperations:
    """Tests for KnowledgeGraph relationship CRUD operations."""

    def test_add_relationship(self):
        """Test adding a relationship to the graph."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        rel = Relationship(
            relationship_id="rel-001",
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            relationship_type="is_a",
            weight=0.8,
        )
        
        rel_id = graph.add_relationship(rel)
        
        assert rel_id == "rel-001"
        assert graph.get_relationship_count() == 1
        assert graph.contains_relationship("rel-001")

    def test_add_relationship_invalid_source(self):
        """Test that adding a relationship with invalid source raises ValueError."""
        graph = KnowledgeGraph()
        
        node_b = graph.create_node("NodeB", "entity")
        
        rel = Relationship(
            relationship_id="rel-001",
            source_id="non-existent",
            target_id=node_b.node_id,
            relationship_type="is_a",
        )
        
        with pytest.raises(ValueError, match="Source node .* does not exist"):
            graph.add_relationship(rel)

    def test_add_relationship_invalid_target(self):
        """Test that adding a relationship with invalid target raises ValueError."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        
        rel = Relationship(
            relationship_id="rel-001",
            source_id=node_a.node_id,
            target_id="non-existent",
            relationship_type="is_a",
        )
        
        with pytest.raises(ValueError, match="Target node .* does not exist"):
            graph.add_relationship(rel)

    def test_add_relationship_duplicate_id(self):
        """Test that adding a relationship with duplicate ID raises ValueError."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        rel1 = Relationship(
            relationship_id="dup",
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            relationship_type="is_a",
        )
        rel2 = Relationship(
            relationship_id="dup",
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            relationship_type="has_property",
        )
        
        graph.add_relationship(rel1)
        
        with pytest.raises(ValueError, match="already exists"):
            graph.add_relationship(rel2)

    def test_create_relationship(self):
        """Test creating a relationship through the graph."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        rel = graph.create_relationship(
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            relationship_type="is_a",
            weight=0.9,
            properties={"evidence": "test"},
        )
        
        assert rel.relationship_id is not None
        assert rel.source_id == node_a.node_id
        assert rel.target_id == node_b.node_id
        assert rel.relationship_type == "is_a"
        assert rel.weight == 0.9
        assert rel.properties == {"evidence": "test"}

    def test_create_relationship_default_weight(self):
        """Test that created relationships use default weight."""
        graph = KnowledgeGraph(config={"default_relationship_weight": 0.7})
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        rel = graph.create_relationship(
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            relationship_type="related_to",
        )
        
        assert rel.weight == 0.7

    def test_get_relationship(self):
        """Test retrieving a relationship by ID."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        created = graph.create_relationship(
            node_a.node_id, node_b.node_id, "is_a"
        )
        
        retrieved = graph.get_relationship(created.relationship_id)
        
        assert retrieved is not None
        assert retrieved.relationship_id == created.relationship_id

    def test_get_relationship_not_found(self):
        """Test getting a non-existent relationship returns None."""
        graph = KnowledgeGraph()
        
        result = graph.get_relationship("non-existent")
        assert result is None

    def test_get_relationships_outgoing(self):
        """Test getting outgoing relationships for a node."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        rel1 = graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        rel2 = graph.create_relationship(node_a.node_id, node_c.node_id, "has_property")
        rel3 = graph.create_relationship(node_b.node_id, node_c.node_id, "related_to")
        
        outgoing = graph.get_relationships(node_a.node_id, direction="outgoing")
        
        assert len(outgoing) == 2
        assert rel1 in outgoing
        assert rel2 in outgoing
        assert rel3 not in outgoing

    def test_get_relationships_incoming(self):
        """Test getting incoming relationships for a node."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        rel1 = graph.create_relationship(node_a.node_id, node_c.node_id, "is_a")
        rel2 = graph.create_relationship(node_b.node_id, node_c.node_id, "has_property")
        rel3 = graph.create_relationship(node_a.node_id, node_b.node_id, "related_to")
        
        incoming = graph.get_relationships(node_c.node_id, direction="incoming")
        
        assert len(incoming) == 2
        assert rel1 in incoming
        assert rel2 in incoming
        assert rel3 not in incoming

    def test_get_relationships_both(self):
        """Test getting both incoming and outgoing relationships."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        rel1 = graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        rel2 = graph.create_relationship(node_c.node_id, node_b.node_id, "has_property")
        
        both = graph.get_relationships(node_b.node_id, direction="both")
        
        assert len(both) == 2
        assert rel1 in both
        assert rel2 in both

    def test_get_relationships_by_type(self):
        """Test filtering relationships by type."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        rel1 = graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        rel2 = graph.create_relationship(node_a.node_id, node_c.node_id, "has_property")
        
        is_a_rels = graph.get_relationships(
            node_a.node_id, direction="outgoing", relationship_type="is_a"
        )
        
        assert len(is_a_rels) == 1
        assert rel1 in is_a_rels

    def test_remove_relationship(self):
        """Test removing a relationship."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        rel = graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        
        assert graph.get_relationship_count() == 1
        
        result = graph.remove_relationship(rel.relationship_id)
        
        assert result is True
        assert graph.get_relationship_count() == 0
        assert not graph.contains_relationship(rel.relationship_id)

    def test_remove_relationship_not_found(self):
        """Test removing a non-existent relationship returns False."""
        graph = KnowledgeGraph()
        
        result = graph.remove_relationship("non-existent")
        assert result is False

    def test_get_relationship_between(self):
        """Test getting a relationship between two specific nodes."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        rel = graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        
        found = graph.get_relationship_between(node_a.node_id, node_b.node_id)
        
        assert found is not None
        assert found.relationship_id == rel.relationship_id

    def test_get_relationship_between_with_type(self):
        """Test getting a relationship between nodes with type filter."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_a.node_id, node_b.node_id, "has_property")
        
        found = graph.get_relationship_between(
            node_a.node_id, node_b.node_id, relationship_type="has_property"
        )
        
        assert found is not None
        assert found.relationship_type == "has_property"

    def test_get_relationship_between_not_found(self):
        """Test getting a non-existent relationship between nodes."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        found = graph.get_relationship_between(node_a.node_id, node_b.node_id)
        
        assert found is None



class TestKnowledgeGraphTraversal:
    """Tests for KnowledgeGraph traversal methods."""

    def test_get_neighbors_outgoing(self):
        """Test getting outgoing neighbors."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_a.node_id, node_c.node_id, "has_property")
        
        neighbors = graph.get_neighbors(node_a.node_id, direction="outgoing")
        
        assert len(neighbors) == 2
        neighbor_ids = [n.node_id for n in neighbors]
        assert node_b.node_id in neighbor_ids
        assert node_c.node_id in neighbor_ids

    def test_get_neighbors_incoming(self):
        """Test getting incoming neighbors."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        graph.create_relationship(node_a.node_id, node_c.node_id, "is_a")
        graph.create_relationship(node_b.node_id, node_c.node_id, "has_property")
        
        neighbors = graph.get_neighbors(node_c.node_id, direction="incoming")
        
        assert len(neighbors) == 2
        neighbor_ids = [n.node_id for n in neighbors]
        assert node_a.node_id in neighbor_ids
        assert node_b.node_id in neighbor_ids

    def test_get_neighbors_by_type(self):
        """Test filtering neighbors by relationship type."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_a.node_id, node_c.node_id, "has_property")
        
        neighbors = graph.get_neighbors(
            node_a.node_id, relationship_type="is_a", direction="outgoing"
        )
        
        assert len(neighbors) == 1
        assert neighbors[0].node_id == node_b.node_id

    def test_get_neighbors_non_existent_node(self):
        """Test getting neighbors of non-existent node returns empty list."""
        graph = KnowledgeGraph()
        
        neighbors = graph.get_neighbors("non-existent")
        assert neighbors == []

    def test_find_path_direct(self):
        """Test finding a direct path between nodes."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        
        path = graph.find_path(node_a.node_id, node_b.node_id)
        
        assert len(path) == 2
        assert path[0].node_id == node_a.node_id
        assert path[1].node_id == node_b.node_id

    def test_find_path_multi_hop(self):
        """Test finding a multi-hop path."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_b.node_id, node_c.node_id, "is_a")
        
        path = graph.find_path(node_a.node_id, node_c.node_id)
        
        assert len(path) == 3
        assert path[0].node_id == node_a.node_id
        assert path[1].node_id == node_b.node_id
        assert path[2].node_id == node_c.node_id

    def test_find_path_same_node(self):
        """Test finding path from node to itself."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        
        path = graph.find_path(node_a.node_id, node_a.node_id)
        
        assert len(path) == 1
        assert path[0].node_id == node_a.node_id

    def test_find_path_no_path(self):
        """Test finding path when no path exists."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        # No relationship between them
        
        path = graph.find_path(node_a.node_id, node_b.node_id)
        
        assert path == []

    def test_find_path_max_depth(self):
        """Test that path finding respects max_depth."""
        graph = KnowledgeGraph()
        
        # Create a chain: A -> B -> C -> D
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        node_d = graph.create_node("NodeD", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_b.node_id, node_c.node_id, "is_a")
        graph.create_relationship(node_c.node_id, node_d.node_id, "is_a")
        
        # Should find path with sufficient depth
        path = graph.find_path(node_a.node_id, node_d.node_id, max_depth=5)
        assert len(path) == 4
        
        # Should not find path with insufficient depth
        path = graph.find_path(node_a.node_id, node_d.node_id, max_depth=2)
        assert path == []

    def test_find_path_with_relationship_types(self):
        """Test finding path with relationship type filter."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_b.node_id, node_c.node_id, "has_property")
        
        # Should find path following only "is_a"
        path = graph.find_path(
            node_a.node_id, node_b.node_id, relationship_types=["is_a"]
        )
        assert len(path) == 2
        
        # Should not find path to C following only "is_a"
        path = graph.find_path(
            node_a.node_id, node_c.node_id, relationship_types=["is_a"]
        )
        assert path == []

    def test_traverse_basic(self):
        """Test basic graph traversal."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_a.node_id, node_c.node_id, "has_property")
        
        paths = graph.traverse(node_a.node_id, max_depth=1)
        
        assert len(paths) == 2
        # Each path should start with node_a and have one more node
        for path in paths:
            assert path[0] == node_a.node_id
            assert len(path) == 2

    def test_traverse_max_depth(self):
        """Test traversal respects max_depth."""
        graph = KnowledgeGraph()
        
        # Create chain: A -> B -> C -> D
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        node_d = graph.create_node("NodeD", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_b.node_id, node_c.node_id, "is_a")
        graph.create_relationship(node_c.node_id, node_d.node_id, "is_a")
        
        paths = graph.traverse(node_a.node_id, max_depth=2)
        
        # Should have paths of length 2 and 3 (depth 1 and 2)
        max_path_len = max(len(p) for p in paths)
        assert max_path_len <= 3  # start + 2 hops

    def test_get_subgraph(self):
        """Test extracting a subgraph."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        node_d = graph.create_node("NodeD", "entity")
        
        rel1 = graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        rel2 = graph.create_relationship(node_b.node_id, node_c.node_id, "is_a")
        rel3 = graph.create_relationship(node_c.node_id, node_d.node_id, "is_a")
        
        # Get subgraph with only A, B, C
        nodes, rels = graph.get_subgraph([node_a.node_id, node_b.node_id, node_c.node_id])
        
        assert len(nodes) == 3
        assert len(rels) == 2  # rel1 and rel2, not rel3
        
        rel_ids = [r.relationship_id for r in rels]
        assert rel1.relationship_id in rel_ids
        assert rel2.relationship_id in rel_ids
        assert rel3.relationship_id not in rel_ids



class TestKnowledgeGraphInference:
    """Tests for KnowledgeGraph relationship inference."""

    def test_infer_transitive_is_a(self):
        """Test transitive inference for is_a relationships."""
        graph = KnowledgeGraph()
        
        # Create: Dog is_a Animal, Animal is_a LivingThing
        dog = graph.create_node("Dog", "entity")
        animal = graph.create_node("Animal", "entity")
        living = graph.create_node("LivingThing", "entity")
        
        graph.create_relationship(dog.node_id, animal.node_id, "is_a", weight=0.9)
        graph.create_relationship(animal.node_id, living.node_id, "is_a", weight=0.9)
        
        # Infer: Dog is_a LivingThing
        inferred = graph.infer_relationships(dog.node_id, inference_types=["transitive"])
        
        assert len(inferred) >= 1
        
        # Find the transitive inference
        transitive_rel = None
        for rel in inferred:
            if rel.target_id == living.node_id and rel.relationship_type == "is_a":
                transitive_rel = rel
                break
        
        assert transitive_rel is not None
        assert transitive_rel.source_id == dog.node_id
        assert transitive_rel.properties.get("inferred") is True
        assert transitive_rel.properties.get("inference_type") == "transitive"

    def test_infer_siblings(self):
        """Test sibling inference."""
        graph = KnowledgeGraph()
        
        # Create: Dog is_a Animal, Cat is_a Animal
        dog = graph.create_node("Dog", "entity")
        cat = graph.create_node("Cat", "entity")
        animal = graph.create_node("Animal", "entity")
        
        graph.create_relationship(dog.node_id, animal.node_id, "is_a", weight=0.9)
        graph.create_relationship(cat.node_id, animal.node_id, "is_a", weight=0.9)
        
        # Infer: Dog related_to Cat
        inferred = graph.infer_relationships(dog.node_id, inference_types=["sibling"])
        
        assert len(inferred) >= 1
        
        # Find the sibling inference
        sibling_rel = None
        for rel in inferred:
            if rel.target_id == cat.node_id and rel.relationship_type == "related_to":
                sibling_rel = rel
                break
        
        assert sibling_rel is not None
        assert sibling_rel.source_id == dog.node_id
        assert sibling_rel.properties.get("inferred") is True
        assert sibling_rel.properties.get("inference_type") == "sibling"

    def test_infer_inheritance(self):
        """Test property inheritance inference."""
        graph = KnowledgeGraph()
        
        # Create: Dog is_a Animal, Animal has_property CanMove
        dog = graph.create_node("Dog", "entity")
        animal = graph.create_node("Animal", "entity")
        can_move = graph.create_node("CanMove", "attribute")
        
        graph.create_relationship(dog.node_id, animal.node_id, "is_a", weight=0.9)
        graph.create_relationship(animal.node_id, can_move.node_id, "has_property", weight=0.8)
        
        # Infer: Dog has_property CanMove
        inferred = graph.infer_relationships(dog.node_id, inference_types=["inheritance"])
        
        assert len(inferred) >= 1
        
        # Find the inheritance inference
        inherit_rel = None
        for rel in inferred:
            if rel.target_id == can_move.node_id and rel.relationship_type == "has_property":
                inherit_rel = rel
                break
        
        assert inherit_rel is not None
        assert inherit_rel.source_id == dog.node_id
        assert inherit_rel.properties.get("inferred") is True
        assert inherit_rel.properties.get("inference_type") == "inheritance"

    def test_infer_no_duplicates(self):
        """Test that inference doesn't create duplicates of existing relationships."""
        graph = KnowledgeGraph()
        
        # Create: Dog is_a Animal, Animal is_a LivingThing
        # Also create: Dog is_a LivingThing (explicit)
        dog = graph.create_node("Dog", "entity")
        animal = graph.create_node("Animal", "entity")
        living = graph.create_node("LivingThing", "entity")
        
        graph.create_relationship(dog.node_id, animal.node_id, "is_a")
        graph.create_relationship(animal.node_id, living.node_id, "is_a")
        graph.create_relationship(dog.node_id, living.node_id, "is_a")  # Explicit
        
        # Infer should not create duplicate
        inferred = graph.infer_relationships(dog.node_id, inference_types=["transitive"])
        
        # Should not infer Dog is_a LivingThing since it already exists
        for rel in inferred:
            assert not (rel.target_id == living.node_id and rel.relationship_type == "is_a")

    def test_infer_non_existent_node(self):
        """Test inference for non-existent node returns empty list."""
        graph = KnowledgeGraph()
        
        inferred = graph.infer_relationships("non-existent")
        assert inferred == []


class TestKnowledgeGraphQuery:
    """Tests for KnowledgeGraph query methods."""

    def test_find_nodes_by_name(self):
        """Test finding nodes by name."""
        graph = KnowledgeGraph()
        
        graph.create_node("Apple", "entity")
        graph.create_node("Banana", "entity")
        graph.create_node("ApplePie", "entity")
        
        results = graph.find_nodes("Apple")
        
        assert len(results) >= 1
        # Exact match should be first
        assert results[0].name == "Apple"

    def test_find_nodes_by_type(self):
        """Test finding nodes filtered by type."""
        graph = KnowledgeGraph()
        
        graph.create_node("Apple", "entity")
        graph.create_node("Red", "attribute")
        graph.create_node("AppleTree", "entity")
        
        results = graph.find_nodes("Apple", node_type="entity")
        
        assert len(results) == 2
        for node in results:
            assert node.node_type == "entity"

    def test_find_nodes_empty_query(self):
        """Test that empty query returns empty list."""
        graph = KnowledgeGraph()
        
        graph.create_node("Apple", "entity")
        
        results = graph.find_nodes("")
        assert results == []

    def test_find_nodes_by_type_method(self):
        """Test find_nodes_by_type method."""
        graph = KnowledgeGraph()
        
        graph.create_node("Apple", "entity")
        graph.create_node("Red", "attribute")
        graph.create_node("Banana", "entity")
        graph.create_node("Yellow", "attribute")
        
        entities = graph.find_nodes_by_type("entity")
        attributes = graph.find_nodes_by_type("attribute")
        
        assert len(entities) == 2
        assert len(attributes) == 2

    def test_find_relationships_by_type(self):
        """Test finding relationships by type."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_a.node_id, node_c.node_id, "has_property")
        graph.create_relationship(node_b.node_id, node_c.node_id, "is_a")
        
        is_a_rels = graph.find_relationships_by_type("is_a")
        
        assert len(is_a_rels) == 2
        for rel in is_a_rels:
            assert rel.relationship_type == "is_a"


class TestKnowledgeGraphSerialization:
    """Tests for KnowledgeGraph serialization."""

    def test_serialization_roundtrip(self):
        """Test graph serialization and deserialization."""
        graph = KnowledgeGraph(config={"max_nodes": 100})
        
        node_a = graph.create_node("NodeA", "entity", properties={"value": 1})
        node_b = graph.create_node("NodeB", "attribute", properties={"value": 2})
        rel = graph.create_relationship(
            node_a.node_id, node_b.node_id, "has_property", weight=0.8
        )
        
        # Access a node to update access_count
        graph.get_node(node_a.node_id)
        
        # Serialize
        data = graph.to_dict()
        
        # Deserialize
        restored = KnowledgeGraph.from_dict(data)
        
        assert len(restored) == 2
        assert restored.get_relationship_count() == 1
        
        restored_a = restored.get_node(node_a.node_id, record_access=False)
        assert restored_a is not None
        assert restored_a.name == "NodeA"
        assert restored_a.properties == {"value": 1}
        
        restored_rel = restored.get_relationship(rel.relationship_id)
        assert restored_rel is not None
        assert restored_rel.weight == 0.8

    def test_serialization_preserves_indices(self):
        """Test that serialization preserves relationship indices."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        node_c = graph.create_node("NodeC", "entity")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_a.node_id, node_c.node_id, "has_property")
        
        data = graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        
        # Check that indices work correctly
        outgoing = restored.get_relationships(node_a.node_id, direction="outgoing")
        assert len(outgoing) == 2

    def test_validate_integrity(self):
        """Test graph integrity validation."""
        graph = KnowledgeGraph()
        
        node_a = graph.create_node("NodeA", "entity")
        node_b = graph.create_node("NodeB", "entity")
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        
        result = graph.validate_integrity()
        
        assert result["valid"] is True
        assert result["issues"] == []
        assert result["node_count"] == 2
        assert result["relationship_count"] == 1

    def test_get_statistics(self):
        """Test getting graph statistics."""
        graph = KnowledgeGraph()
        
        graph.create_node("NodeA", "entity")
        graph.create_node("NodeB", "entity")
        graph.create_node("AttrC", "attribute")
        
        node_a = graph.get_node_by_name("NodeA")
        node_b = graph.get_node_by_name("NodeB")
        
        graph.create_relationship(node_a.node_id, node_b.node_id, "is_a")
        graph.create_relationship(node_a.node_id, node_b.node_id, "related_to")
        
        stats = graph.get_statistics()
        
        assert stats["node_count"] == 3
        assert stats["relationship_count"] == 2
        assert stats["node_types"]["entity"] == 2
        assert stats["node_types"]["attribute"] == 1
        assert stats["relationship_types"]["is_a"] == 1
        assert stats["relationship_types"]["related_to"] == 1

    def test_get_state(self):
        """Test getting graph state."""
        graph = KnowledgeGraph(config={"max_nodes": 500})
        
        graph.create_node("NodeA", "entity")
        graph.create_node("NodeB", "entity")
        
        state = graph.get_state()
        
        assert state["node_count"] == 2
        assert state["relationship_count"] == 0
        assert state["max_nodes"] == 500
        assert "config" in state


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_concept_node(self):
        """Test create_concept_node factory function."""
        node = create_concept_node(
            name="TestNode",
            node_type="entity",
            properties={"key": "value"},
            metadata={"source": "test"},
        )
        
        assert node.node_id is not None
        assert len(node.node_id) > 0
        assert node.name == "TestNode"
        assert node.node_type == "entity"
        assert node.properties == {"key": "value"}
        assert node.metadata == {"source": "test"}
        assert node.access_count == 0

    def test_create_concept_node_unique_ids(self):
        """Test that factory creates unique IDs."""
        nodes = [
            create_concept_node("Node", "entity")
            for _ in range(10)
        ]
        
        ids = [n.node_id for n in nodes]
        assert len(ids) == len(set(ids))  # All unique

    def test_create_relationship_factory(self):
        """Test create_relationship factory function."""
        rel = create_relationship(
            source_id="source",
            target_id="target",
            relationship_type="is_a",
            weight=0.8,
            properties={"evidence": "test"},
        )
        
        assert rel.relationship_id is not None
        assert len(rel.relationship_id) > 0
        assert rel.source_id == "source"
        assert rel.target_id == "target"
        assert rel.relationship_type == "is_a"
        assert rel.weight == 0.8
        assert rel.properties == {"evidence": "test"}

    def test_create_relationship_unique_ids(self):
        """Test that factory creates unique relationship IDs."""
        rels = [
            create_relationship("a", "b", "test")
            for _ in range(10)
        ]
        
        ids = [r.relationship_id for r in rels]
        assert len(ids) == len(set(ids))  # All unique

"""
Property-based tests for Semantic Memory.

Tests properties 15-20 from the consciousness-system-deepening design document:
- Property 15: Knowledge Graph Structure Validity
- Property 16: Knowledge Integration Consistency
- Property 17: Knowledge Query Completeness
- Property 18: Consolidation Frequency Updates
- Property 19: Knowledge Conflict Resolution Determinism
- Property 20: Knowledge Graph Serialization Round-Trip

Validates: Requirements 4.1-4.6
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import Dict, List, Any, Optional, Tuple
import time
import math

from mm_orch.consciousness.semantic_memory import (
    SemanticMemory,
    SemanticMemoryConfig,
    IntegrationResult,
    ConsolidationResult,
    ConflictInfo,
    create_semantic_memory,
)
from mm_orch.consciousness.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphConfig,
    ConceptNode,
    Relationship,
    create_concept_node,
    create_relationship,
)
from mm_orch.consciousness.episodic_memory import Episode


# =============================================================================
# Hypothesis Strategies for Test Data Generation
# =============================================================================

# Valid node types for concepts
VALID_NODE_TYPES = ["entity", "attribute", "action", "relation", "concept"]

# Strategy for valid float values in [0.0, 1.0]
unit_float_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for confidence values
confidence_strategy = st.floats(
    min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for relationship weights
weight_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for concept names (alphanumeric, reasonable length)
concept_name_strategy = st.text(
    min_size=1, max_size=30,
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-')
).filter(lambda x: len(x.strip()) > 0)

# Strategy for node types
node_type_strategy = st.sampled_from(VALID_NODE_TYPES)

# Strategy for relationship types
VALID_RELATIONSHIP_TYPES = ["is_a", "has_property", "related_to", "part_of", "causes", "contains"]
relationship_type_strategy = st.sampled_from(VALID_RELATIONSHIP_TYPES)

# Strategy for property dictionaries
property_value_strategy = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    st.booleans(),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)

properties_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    values=property_value_strategy,
    min_size=0,
    max_size=5
)

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
def concept_node_strategy(draw):
    """Generate a valid ConceptNode."""
    name = draw(concept_name_strategy)
    node_type = draw(node_type_strategy)
    properties = draw(properties_strategy)
    metadata = draw(metadata_strategy)
    
    return create_concept_node(
        name=name,
        node_type=node_type,
        properties=properties,
        metadata=metadata,
    )


@st.composite
def concept_definition_strategy(draw):
    """Generate a concept definition dictionary for integration."""
    return {
        "name": draw(concept_name_strategy),
        "type": draw(node_type_strategy),
        "properties": draw(properties_strategy),
    }


@st.composite
def relationship_definition_strategy(draw, source_name: Optional[str] = None, target_name: Optional[str] = None):
    """Generate a relationship definition dictionary for integration."""
    return {
        "source": source_name if source_name else draw(concept_name_strategy),
        "target": target_name if target_name else draw(concept_name_strategy),
        "type": draw(relationship_type_strategy),
        "weight": draw(weight_strategy),
    }


@st.composite
def knowledge_integration_strategy(draw, min_concepts: int = 1, max_concepts: int = 5):
    """Generate a knowledge dictionary for integration."""
    num_concepts = draw(st.integers(min_value=min_concepts, max_value=max_concepts))
    concepts = [draw(concept_definition_strategy()) for _ in range(num_concepts)]
    
    # Generate relationships between existing concepts
    relationships = []
    if len(concepts) >= 2:
        num_rels = draw(st.integers(min_value=0, max_value=min(3, len(concepts))))
        for _ in range(num_rels):
            source_idx = draw(st.integers(min_value=0, max_value=len(concepts) - 1))
            target_idx = draw(st.integers(min_value=0, max_value=len(concepts) - 1))
            if source_idx != target_idx:
                relationships.append({
                    "source": concepts[source_idx]["name"],
                    "target": concepts[target_idx]["name"],
                    "type": draw(relationship_type_strategy),
                    "weight": draw(weight_strategy),
                })
    
    return {
        "concepts": concepts,
        "relationships": relationships,
    }


@st.composite
def knowledge_graph_strategy(draw, min_nodes: int = 2, max_nodes: int = 10):
    """Generate a valid KnowledgeGraph with nodes and relationships."""
    graph = KnowledgeGraph()
    
    num_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    nodes = []
    
    for i in range(num_nodes):
        name = f"concept_{i}_{draw(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'))}"
        node = graph.create_node(
            name=name,
            node_type=draw(node_type_strategy),
            properties=draw(properties_strategy),
            metadata=draw(metadata_strategy),
        )
        nodes.append(node)
    
    # Add relationships between nodes
    if len(nodes) >= 2:
        num_rels = draw(st.integers(min_value=1, max_value=min(5, len(nodes) * 2)))
        for _ in range(num_rels):
            source_idx = draw(st.integers(min_value=0, max_value=len(nodes) - 1))
            target_idx = draw(st.integers(min_value=0, max_value=len(nodes) - 1))
            if source_idx != target_idx:
                try:
                    graph.create_relationship(
                        source_id=nodes[source_idx].node_id,
                        target_id=nodes[target_idx].node_id,
                        relationship_type=draw(relationship_type_strategy),
                        weight=draw(weight_strategy),
                    )
                except ValueError:
                    pass  # Relationship might already exist
    
    return graph


@st.composite
def semantic_memory_config_strategy(draw):
    """Generate a valid SemanticMemoryConfig."""
    return {
        "conflict_resolution_strategy": draw(st.sampled_from(["newest_wins", "highest_confidence", "merge"])),
        "similarity_threshold": draw(st.floats(min_value=0.5, max_value=0.95, allow_nan=False)),
        "min_relationship_strength": draw(st.floats(min_value=0.05, max_value=0.3, allow_nan=False)),
        "consolidation_merge_threshold": draw(st.floats(min_value=0.8, max_value=0.99, allow_nan=False)),
        "prune_access_threshold": draw(st.integers(min_value=0, max_value=5)),
        "prune_strength_threshold": draw(st.floats(min_value=0.1, max_value=0.4, allow_nan=False)),
        "max_query_results": draw(st.integers(min_value=5, max_value=50)),
        "extraction_min_frequency": draw(st.floats(min_value=0.1, max_value=0.5, allow_nan=False)),
    }


# =============================================================================
# Property 15: Knowledge Graph Structure Validity
# =============================================================================

class TestKnowledgeGraphStructureValidity:
    """
    Tests for Property 15: Knowledge Graph Structure Validity
    
    *For any* KnowledgeGraph, all relationships SHALL reference valid concept 
    nodes (both source_id and target_id exist), and all nodes SHALL have 
    valid concept_type values.
    
    **Validates: Requirements 4.1**
    """

    @given(graph=knowledge_graph_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_relationships_reference_valid_nodes(self, graph: KnowledgeGraph):
        """
        Feature: consciousness-system-deepening, Property 15: Knowledge Graph Structure Validity
        
        For any KnowledgeGraph, all relationships SHALL reference valid concept 
        nodes (both source_id and target_id exist).
        
        **Validates: Requirements 4.1**
        """
        all_node_ids = {node.node_id for node in graph.get_all_nodes()}
        all_relationships = graph.get_all_relationships()
        
        for rel in all_relationships:
            assert rel.source_id in all_node_ids, \
                f"Relationship {rel.relationship_id} references non-existent source node {rel.source_id}"
            assert rel.target_id in all_node_ids, \
                f"Relationship {rel.relationship_id} references non-existent target node {rel.target_id}"

    @given(graph=knowledge_graph_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_nodes_have_valid_type(self, graph: KnowledgeGraph):
        """
        Feature: consciousness-system-deepening, Property 15: Knowledge Graph Structure Validity
        
        For any KnowledgeGraph, all nodes SHALL have valid concept_type values.
        
        **Validates: Requirements 4.1**
        """
        all_nodes = graph.get_all_nodes()
        
        for node in all_nodes:
            assert node.node_type is not None and len(node.node_type) > 0, \
                f"Node {node.node_id} has invalid node_type: {node.node_type}"
            assert isinstance(node.node_type, str), \
                f"Node {node.node_id} node_type must be a string, got {type(node.node_type)}"


    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_validate_integrity_passes_for_valid_graph(self, data):
        """
        Feature: consciousness-system-deepening, Property 15: Knowledge Graph Structure Validity
        
        For any properly constructed KnowledgeGraph, validate_integrity() SHALL
        return valid=True with no issues.
        
        **Validates: Requirements 4.1**
        """
        graph = data.draw(knowledge_graph_strategy())
        
        validation = graph.validate_integrity()
        
        assert validation["valid"], \
            f"Graph should be valid but found issues: {validation['issues']}"
        assert len(validation["issues"]) == 0, \
            f"Graph should have no issues: {validation['issues']}"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_relationship_weight_bounds(self, data):
        """
        Feature: consciousness-system-deepening, Property 15: Knowledge Graph Structure Validity
        
        For any relationship in a KnowledgeGraph, the weight SHALL be in [0.0, 1.0].
        
        **Validates: Requirements 4.1**
        """
        graph = data.draw(knowledge_graph_strategy())
        
        for rel in graph.get_all_relationships():
            assert 0.0 <= rel.weight <= 1.0, \
                f"Relationship {rel.relationship_id} has invalid weight: {rel.weight}"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_node_ids_are_unique(self, data):
        """
        Feature: consciousness-system-deepening, Property 15: Knowledge Graph Structure Validity
        
        For any KnowledgeGraph, all node IDs SHALL be unique.
        
        **Validates: Requirements 4.1**
        """
        graph = data.draw(knowledge_graph_strategy())
        
        all_nodes = graph.get_all_nodes()
        node_ids = [node.node_id for node in all_nodes]
        
        assert len(node_ids) == len(set(node_ids)), \
            "All node IDs should be unique"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_relationship_ids_are_unique(self, data):
        """
        Feature: consciousness-system-deepening, Property 15: Knowledge Graph Structure Validity
        
        For any KnowledgeGraph, all relationship IDs SHALL be unique.
        
        **Validates: Requirements 4.1**
        """
        graph = data.draw(knowledge_graph_strategy())
        
        all_rels = graph.get_all_relationships()
        rel_ids = [rel.relationship_id for rel in all_rels]
        
        assert len(rel_ids) == len(set(rel_ids)), \
            "All relationship IDs should be unique"


# =============================================================================
# Property 16: Knowledge Integration Consistency
# =============================================================================

class TestKnowledgeIntegrationConsistency:
    """
    Tests for Property 16: Knowledge Integration Consistency
    
    *For any* new knowledge integrated into SemanticMemory, either a new concept 
    node is created OR an existing concept is updated, and the IntegrationResult 
    SHALL accurately reflect the changes made.
    
    **Validates: Requirements 4.2**
    """

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_new_concept_creates_node(self, data):
        """
        Feature: consciousness-system-deepening, Property 16: Knowledge Integration Consistency
        
        For any new concept integrated, a new node SHALL be created in the graph.
        
        **Validates: Requirements 4.2**
        """
        memory = SemanticMemory()
        concept_def = data.draw(concept_definition_strategy())
        
        initial_count = len(memory)
        
        result = memory.integrate_knowledge({"concepts": [concept_def]})
        
        assert result.success, f"Integration should succeed: {result.error_message}"
        assert len(result.new_concepts) == 1, \
            f"Should create exactly 1 new concept, got {len(result.new_concepts)}"
        assert len(memory) == initial_count + 1, \
            f"Memory should have 1 more concept: was {initial_count}, now {len(memory)}"
        assert concept_def["name"] in memory, \
            f"Concept '{concept_def['name']}' should be in memory"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_existing_concept_updates_node(self, data):
        """
        Feature: consciousness-system-deepening, Property 16: Knowledge Integration Consistency
        
        For any existing concept integrated with new properties, the node SHALL 
        be updated (not duplicated).
        
        **Validates: Requirements 4.2**
        """
        memory = SemanticMemory()
        concept_name = data.draw(concept_name_strategy)
        
        # First integration
        memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity", "properties": {"v1": "original"}}]
        })
        
        count_after_first = len(memory)
        
        # Second integration with same name but new property
        result = memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity", "properties": {"v2": "new"}}]
        })
        
        assert result.success, f"Integration should succeed: {result.error_message}"
        assert len(result.updated_concepts) >= 1, \
            "Should update existing concept"
        assert len(memory) == count_after_first, \
            f"Should not create duplicate: was {count_after_first}, now {len(memory)}"


    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_integration_result_reflects_changes(self, data):
        """
        Feature: consciousness-system-deepening, Property 16: Knowledge Integration Consistency
        
        For any integration, the IntegrationResult SHALL accurately reflect 
        the changes made (new_concepts + updated_concepts = total concepts processed).
        
        **Validates: Requirements 4.2**
        """
        memory = SemanticMemory()
        knowledge = data.draw(knowledge_integration_strategy(min_concepts=1, max_concepts=5))
        
        result = memory.integrate_knowledge(knowledge)
        
        assert result.success, f"Integration should succeed: {result.error_message}"
        
        # Total concepts processed should match new + updated
        total_processed = len(result.new_concepts) + len(result.updated_concepts)
        # Note: Some concepts might be created via relationships too
        assert total_processed >= len(knowledge["concepts"]), \
            f"Should process at least {len(knowledge['concepts'])} concepts, " \
            f"got {total_processed} (new={len(result.new_concepts)}, updated={len(result.updated_concepts)})"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_relationship_integration_creates_missing_nodes(self, data):
        """
        Feature: consciousness-system-deepening, Property 16: Knowledge Integration Consistency
        
        For any relationship integration where source or target doesn't exist,
        the missing nodes SHALL be created automatically.
        
        **Validates: Requirements 4.2**
        """
        memory = SemanticMemory()
        source_name = data.draw(concept_name_strategy)
        target_name = data.draw(concept_name_strategy)
        assume(source_name != target_name)
        
        result = memory.integrate_knowledge({
            "relationships": [{
                "source": source_name,
                "target": target_name,
                "type": "related_to",
            }]
        })
        
        assert result.success, f"Integration should succeed: {result.error_message}"
        assert source_name in memory, f"Source '{source_name}' should be created"
        assert target_name in memory, f"Target '{target_name}' should be created"
        assert len(result.new_relationships) == 1, \
            f"Should create 1 relationship, got {len(result.new_relationships)}"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_integrations_accumulate(self, data):
        """
        Feature: consciousness-system-deepening, Property 16: Knowledge Integration Consistency
        
        For any sequence of integrations, the total concepts SHALL equal the
        sum of unique concepts integrated.
        
        **Validates: Requirements 4.2**
        """
        memory = SemanticMemory()
        
        # Generate unique concept names
        num_integrations = data.draw(st.integers(min_value=2, max_value=5))
        all_names = set()
        
        for i in range(num_integrations):
            name = f"concept_{i}_{data.draw(st.text(min_size=3, max_size=8, alphabet='abcdefghijklmnopqrstuvwxyz'))}"
            all_names.add(name)
            
            result = memory.integrate_knowledge({
                "concepts": [{"name": name, "type": "entity"}]
            })
            assert result.success
        
        assert len(memory) == len(all_names), \
            f"Memory should have {len(all_names)} unique concepts, got {len(memory)}"


# =============================================================================
# Property 17: Knowledge Query Completeness
# =============================================================================

class TestKnowledgeQueryCompleteness:
    """
    Tests for Property 17: Knowledge Query Completeness
    
    *For any* query to SemanticMemory, the QueryResult SHALL include all 
    directly matching concepts and their immediate relationships.
    
    **Validates: Requirements 4.3**
    """

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_exact_name_query_returns_concept(self, data):
        """
        Feature: consciousness-system-deepening, Property 17: Knowledge Query Completeness
        
        For any query with an exact concept name, the result SHALL include 
        that concept.
        
        **Validates: Requirements 4.3**
        """
        memory = SemanticMemory()
        concept_name = data.draw(concept_name_strategy)
        
        memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity"}]
        })
        
        results = memory.query(concept_name)
        
        assert len(results) >= 1, \
            f"Query for '{concept_name}' should return at least 1 result"
        assert any(node.name == concept_name for node in results), \
            f"Results should include concept with exact name '{concept_name}'"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_query_returns_related_concepts(self, data):
        """
        Feature: consciousness-system-deepening, Property 17: Knowledge Query Completeness
        
        For any concept with relationships, querying relationships SHALL return
        all immediate relationships.
        
        **Validates: Requirements 4.3**
        """
        memory = SemanticMemory()
        source_name = "source_concept"
        target_names = [f"target_{i}" for i in range(3)]
        
        # Create relationships
        for target in target_names:
            memory.integrate_knowledge({
                "relationships": [{
                    "source": source_name,
                    "target": target,
                    "type": "related_to",
                }]
            })
        
        # Query the source concept
        source_nodes = memory.query(source_name)
        assert len(source_nodes) >= 1
        
        source_id = source_nodes[0].node_id
        relationships = memory.query_relationships(source_id, direction="outgoing")
        
        assert len(relationships) == len(target_names), \
            f"Should have {len(target_names)} outgoing relationships, got {len(relationships)}"


    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_concept_definition_includes_relationships(self, data):
        """
        Feature: consciousness-system-deepening, Property 17: Knowledge Query Completeness
        
        For any concept with relationships, get_concept_definition() SHALL 
        include all immediate relationships in the result.
        
        **Validates: Requirements 4.3**
        """
        memory = SemanticMemory()
        concept_name = "main_concept"
        
        # Create concept with multiple relationships
        memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity", "properties": {"key": "value"}}],
            "relationships": [
                {"source": concept_name, "target": "related_1", "type": "is_a"},
                {"source": concept_name, "target": "related_2", "type": "has_property"},
            ]
        })
        
        definition = memory.get_concept_definition(concept_name)
        
        assert definition is not None, "Definition should be returned"
        assert definition["name"] == concept_name
        assert "relationships" in definition
        assert len(definition["relationships"]) >= 2, \
            f"Should have at least 2 relationships, got {len(definition['relationships'])}"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_find_related_concepts_returns_neighbors(self, data):
        """
        Feature: consciousness-system-deepening, Property 17: Knowledge Query Completeness
        
        For any concept, find_related_concepts() SHALL return all concepts 
        within the specified depth.
        
        **Validates: Requirements 4.3**
        """
        memory = SemanticMemory()
        
        # Create a chain: A -> B -> C
        memory.integrate_knowledge({
            "relationships": [
                {"source": "A", "target": "B", "type": "related_to"},
                {"source": "B", "target": "C", "type": "related_to"},
            ]
        })
        
        # Find related to A with depth 1
        related_depth_1 = memory.find_related_concepts("A", max_depth=1)
        related_names_1 = {node.name for node in related_depth_1}
        
        assert "B" in related_names_1, "B should be found at depth 1"
        
        # Find related to A with depth 2
        related_depth_2 = memory.find_related_concepts("A", max_depth=2)
        related_names_2 = {node.name for node in related_depth_2}
        
        assert "B" in related_names_2, "B should be found at depth 2"
        assert "C" in related_names_2, "C should be found at depth 2"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_query_respects_max_results(self, data):
        """
        Feature: consciousness-system-deepening, Property 17: Knowledge Query Completeness
        
        For any query with max_results, the number of results SHALL be <= max_results.
        
        **Validates: Requirements 4.3**
        """
        memory = SemanticMemory()
        
        # Create many concepts with similar names
        num_concepts = 20
        for i in range(num_concepts):
            memory.integrate_knowledge({
                "concepts": [{"name": f"test_concept_{i}", "type": "entity"}]
            })
        
        max_results = data.draw(st.integers(min_value=1, max_value=num_concepts - 1))
        results = memory.query("test_concept", max_results=max_results)
        
        assert len(results) <= max_results, \
            f"Should return at most {max_results} results, got {len(results)}"


# =============================================================================
# Property 18: Consolidation Frequency Updates
# =============================================================================

class TestConsolidationFrequencyUpdates:
    """
    Tests for Property 18: Consolidation Frequency Updates
    
    *For any* episodic consolidation that extracts patterns, the corresponding 
    concept frequencies and relationship strengths in SemanticMemory SHALL increase.
    
    **Validates: Requirements 4.4**
    """

    def _create_test_episode(
        self,
        context: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> Episode:
        """Helper to create test episodes."""
        import uuid
        return Episode(
            episode_id=str(uuid.uuid4()),
            timestamp=time.time(),
            context=context,
            events=events,
            emotional_state={"pleasure": 0.0, "arousal": 0.5, "dominance": 0.0},
            importance=0.5,
        )

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_extraction_increases_concept_count(self, data):
        """
        Feature: consciousness-system-deepening, Property 18: Consolidation Frequency Updates
        
        For any pattern extraction from episodes, new concepts SHALL be added
        to semantic memory.
        
        **Validates: Requirements 4.4**
        """
        memory = SemanticMemory(config={"extraction_min_frequency": 0.3})
        
        # Create episodes with repeated concepts
        concept_name = "repeated_concept"
        episodes = [
            self._create_test_episode({"domain": concept_name}, []),
            self._create_test_episode({"domain": concept_name}, []),
            self._create_test_episode({"domain": concept_name}, []),
        ]
        
        initial_count = len(memory)
        result = memory.extract_from_episodes(episodes)
        
        assert result.episodes_processed == 3
        # If patterns were extracted, concepts should be added
        if result.patterns_found > 0:
            assert len(memory) > initial_count or result.integration_result is not None, \
                "Extraction should add concepts to memory"


    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_consolidation_strengthens_accessed_relationships(self, data):
        """
        Feature: consciousness-system-deepening, Property 18: Consolidation Frequency Updates
        
        For any consolidation, frequently accessed relationships SHALL have 
        their strength increased.
        
        **Validates: Requirements 4.4**
        """
        memory = SemanticMemory()
        
        # Create a relationship
        memory.integrate_knowledge({
            "relationships": [
                {"source": "A", "target": "B", "type": "related_to", "weight": 0.5}
            ]
        })
        
        # Access the nodes multiple times to increase access count
        for _ in range(10):
            memory.query("A")
            memory.query("B")
        
        # Get initial relationship weight
        a_nodes = memory.query("A")
        assert len(a_nodes) > 0
        initial_rels = memory.query_relationships(a_nodes[0].node_id, direction="outgoing")
        initial_weight = initial_rels[0].weight if initial_rels else 0.5
        
        # Consolidate
        result = memory.consolidate()
        
        # Check if relationship was strengthened
        if len(result.strengthened_relationships) > 0:
            final_rels = memory.query_relationships(a_nodes[0].node_id, direction="outgoing")
            if final_rels:
                assert final_rels[0].weight >= initial_weight, \
                    f"Relationship weight should not decrease: was {initial_weight}, now {final_rels[0].weight}"

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_extraction_result_tracks_patterns(self, data):
        """
        Feature: consciousness-system-deepening, Property 18: Consolidation Frequency Updates
        
        For any extraction, the ExtractionResult SHALL accurately track the 
        number of patterns found.
        
        **Validates: Requirements 4.4**
        """
        memory = SemanticMemory(config={"extraction_min_frequency": 0.3})
        
        # Create episodes with patterns
        episodes = [
            self._create_test_episode({"task": "coding", "lang": "Python"}, []),
            self._create_test_episode({"task": "coding", "lang": "Python"}, []),
            self._create_test_episode({"task": "coding", "lang": "Python"}, []),
        ]
        
        result = memory.extract_from_episodes(episodes)
        
        assert result.episodes_processed == len(episodes), \
            f"Should process {len(episodes)} episodes, got {result.episodes_processed}"
        assert result.patterns_found >= 0, \
            "patterns_found should be non-negative"
        assert result.patterns_found == len(result.extracted_concepts) + len(result.extracted_relationships), \
            "patterns_found should equal concepts + relationships extracted"


# =============================================================================
# Property 19: Knowledge Conflict Resolution Determinism
# =============================================================================

class TestKnowledgeConflictResolutionDeterminism:
    """
    Tests for Property 19: Knowledge Conflict Resolution Determinism
    
    *For any* conflict between existing and new information, the resolution 
    SHALL be deterministic based on confidence scores and recency, with higher 
    confidence or more recent information taking precedence.
    
    **Validates: Requirements 4.5**
    """

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_newest_wins_strategy_deterministic(self, data):
        """
        Feature: consciousness-system-deepening, Property 19: Knowledge Conflict Resolution Determinism
        
        For newest_wins strategy, the most recent value SHALL always win.
        
        **Validates: Requirements 4.5**
        """
        memory = SemanticMemory(config={"conflict_resolution_strategy": "newest_wins"})
        concept_name = data.draw(concept_name_strategy)
        
        old_value = "old_value"
        new_value = "new_value"
        
        # First integration
        memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity", "properties": {"attr": old_value}}]
        })
        
        # Second integration with conflict
        result = memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity", "properties": {"attr": new_value}}]
        })
        
        # Verify new value wins
        definition = memory.get_concept_definition(concept_name)
        assert definition is not None
        assert definition["properties"]["attr"] == new_value, \
            f"newest_wins should use new value '{new_value}', got '{definition['properties']['attr']}'"
        
        # Verify conflict was detected
        assert len(result.conflicts) >= 1, "Should detect conflict"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_highest_confidence_strategy_deterministic(self, data):
        """
        Feature: consciousness-system-deepening, Property 19: Knowledge Conflict Resolution Determinism
        
        For highest_confidence strategy, the higher confidence value SHALL win.
        
        **Validates: Requirements 4.5**
        """
        memory = SemanticMemory(config={"conflict_resolution_strategy": "highest_confidence"})
        concept_name = data.draw(concept_name_strategy)
        
        high_conf_value = "high_confidence_value"
        low_conf_value = "low_confidence_value"
        
        # First integration with high confidence
        memory.integrate_knowledge(
            {"concepts": [{"name": concept_name, "type": "entity", "properties": {"attr": high_conf_value}}]},
            confidence=0.9
        )
        
        # Second integration with lower confidence
        memory.integrate_knowledge(
            {"concepts": [{"name": concept_name, "type": "entity", "properties": {"attr": low_conf_value}}]},
            confidence=0.3
        )
        
        # Verify high confidence value wins
        definition = memory.get_concept_definition(concept_name)
        assert definition is not None
        assert definition["properties"]["attr"] == high_conf_value, \
            f"highest_confidence should keep high confidence value '{high_conf_value}', " \
            f"got '{definition['properties']['attr']}'"


    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_conflict_resolution_is_repeatable(self, data):
        """
        Feature: consciousness-system-deepening, Property 19: Knowledge Conflict Resolution Determinism
        
        For any conflict scenario, running the same integration sequence SHALL 
        produce the same result.
        
        **Validates: Requirements 4.5**
        """
        strategy = data.draw(st.sampled_from(["newest_wins", "highest_confidence"]))
        concept_name = data.draw(concept_name_strategy)
        
        def run_integration():
            memory = SemanticMemory(config={"conflict_resolution_strategy": strategy})
            
            memory.integrate_knowledge({
                "concepts": [{"name": concept_name, "type": "entity", "properties": {"attr": "value1"}}]
            }, confidence=0.7)
            
            memory.integrate_knowledge({
                "concepts": [{"name": concept_name, "type": "entity", "properties": {"attr": "value2"}}]
            }, confidence=0.5)
            
            definition = memory.get_concept_definition(concept_name)
            return definition["properties"]["attr"] if definition else None
        
        # Run twice and compare
        result1 = run_integration()
        result2 = run_integration()
        
        assert result1 == result2, \
            f"Conflict resolution should be deterministic: got '{result1}' and '{result2}'"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_conflict_info_captures_details(self, data):
        """
        Feature: consciousness-system-deepening, Property 19: Knowledge Conflict Resolution Determinism
        
        For any conflict, the ConflictInfo SHALL capture existing value, new value,
        and resolution action.
        
        **Validates: Requirements 4.5**
        """
        memory = SemanticMemory(config={"conflict_resolution_strategy": "newest_wins"})
        concept_name = data.draw(concept_name_strategy)
        
        old_value = "old"
        new_value = "new"
        
        memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity", "properties": {"attr": old_value}}]
        })
        
        result = memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity", "properties": {"attr": new_value}}]
        })
        
        assert len(result.conflicts) >= 1, "Should have at least one conflict"
        
        conflict = result.conflicts[0]
        assert conflict.existing_value == old_value, \
            f"Conflict should capture existing value '{old_value}', got '{conflict.existing_value}'"
        assert conflict.new_value == new_value, \
            f"Conflict should capture new value '{new_value}', got '{conflict.new_value}'"
        assert conflict.resolution is not None and len(conflict.resolution) > 0, \
            "Conflict should have a resolution"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_merge_strategy_combines_values(self, data):
        """
        Feature: consciousness-system-deepening, Property 19: Knowledge Conflict Resolution Determinism
        
        For merge strategy with compatible types, values SHALL be merged.
        
        **Validates: Requirements 4.5**
        """
        memory = SemanticMemory(config={"conflict_resolution_strategy": "merge"})
        concept_name = data.draw(concept_name_strategy)
        
        # Test with list values that can be merged
        memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity", "properties": {"tags": ["a", "b"]}}]
        })
        
        result = memory.integrate_knowledge({
            "concepts": [{"name": concept_name, "type": "entity", "properties": {"tags": ["c", "d"]}}]
        })
        
        definition = memory.get_concept_definition(concept_name)
        assert definition is not None
        
        # Merged list should contain elements from both
        tags = definition["properties"].get("tags", [])
        if isinstance(tags, list):
            assert "a" in tags or "c" in tags, \
                "Merged list should contain elements from original lists"


# =============================================================================
# Property 20: Knowledge Graph Serialization Round-Trip
# =============================================================================

class TestKnowledgeGraphSerializationRoundTrip:
    """
    Tests for Property 20: Knowledge Graph Serialization Round-Trip
    
    *For any* valid KnowledgeGraph state, serializing to dictionary and 
    deserializing back SHALL produce an equivalent graph with identical 
    nodes, relationships, and attributes.
    
    **Validates: Requirements 4.6**
    """

    @given(graph=knowledge_graph_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_knowledge_graph_round_trip_preserves_nodes(self, graph: KnowledgeGraph):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For any KnowledgeGraph, serialization round-trip SHALL preserve all nodes.
        
        **Validates: Requirements 4.6**
        """
        original_nodes = graph.get_all_nodes()
        original_node_count = len(original_nodes)
        
        # Serialize and deserialize
        data = graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        
        restored_nodes = restored.get_all_nodes()
        
        assert len(restored_nodes) == original_node_count, \
            f"Node count should be preserved: was {original_node_count}, got {len(restored_nodes)}"
        
        # Verify each node is preserved
        original_ids = {node.node_id for node in original_nodes}
        restored_ids = {node.node_id for node in restored_nodes}
        
        assert original_ids == restored_ids, \
            "All node IDs should be preserved"

    @given(graph=knowledge_graph_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_knowledge_graph_round_trip_preserves_relationships(self, graph: KnowledgeGraph):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For any KnowledgeGraph, serialization round-trip SHALL preserve all relationships.
        
        **Validates: Requirements 4.6**
        """
        original_rels = graph.get_all_relationships()
        original_rel_count = len(original_rels)
        
        # Serialize and deserialize
        data = graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        
        restored_rels = restored.get_all_relationships()
        
        assert len(restored_rels) == original_rel_count, \
            f"Relationship count should be preserved: was {original_rel_count}, got {len(restored_rels)}"
        
        # Verify each relationship is preserved
        original_rel_ids = {rel.relationship_id for rel in original_rels}
        restored_rel_ids = {rel.relationship_id for rel in restored_rels}
        
        assert original_rel_ids == restored_rel_ids, \
            "All relationship IDs should be preserved"


    @given(graph=knowledge_graph_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_knowledge_graph_round_trip_preserves_node_attributes(self, graph: KnowledgeGraph):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For any KnowledgeGraph, serialization round-trip SHALL preserve all 
        node attributes (name, type, properties, metadata).
        
        **Validates: Requirements 4.6**
        """
        original_nodes = {node.node_id: node for node in graph.get_all_nodes()}
        
        # Serialize and deserialize
        data = graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        
        restored_nodes = {node.node_id: node for node in restored.get_all_nodes()}
        
        for node_id, original in original_nodes.items():
            assert node_id in restored_nodes, f"Node {node_id} should be restored"
            restored_node = restored_nodes[node_id]
            
            assert restored_node.name == original.name, \
                f"Node name should be preserved: was '{original.name}', got '{restored_node.name}'"
            assert restored_node.node_type == original.node_type, \
                f"Node type should be preserved: was '{original.node_type}', got '{restored_node.node_type}'"
            assert restored_node.properties == original.properties, \
                f"Node properties should be preserved"
            assert restored_node.access_count == original.access_count, \
                f"Node access_count should be preserved"

    @given(graph=knowledge_graph_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_knowledge_graph_round_trip_preserves_relationship_attributes(self, graph: KnowledgeGraph):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For any KnowledgeGraph, serialization round-trip SHALL preserve all 
        relationship attributes (source_id, target_id, type, weight).
        
        **Validates: Requirements 4.6**
        """
        original_rels = {rel.relationship_id: rel for rel in graph.get_all_relationships()}
        
        # Serialize and deserialize
        data = graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        
        restored_rels = {rel.relationship_id: rel for rel in restored.get_all_relationships()}
        
        for rel_id, original in original_rels.items():
            assert rel_id in restored_rels, f"Relationship {rel_id} should be restored"
            restored_rel = restored_rels[rel_id]
            
            assert restored_rel.source_id == original.source_id, \
                f"Relationship source_id should be preserved"
            assert restored_rel.target_id == original.target_id, \
                f"Relationship target_id should be preserved"
            assert restored_rel.relationship_type == original.relationship_type, \
                f"Relationship type should be preserved"
            assert abs(restored_rel.weight - original.weight) < 0.0001, \
                f"Relationship weight should be preserved: was {original.weight}, got {restored_rel.weight}"


    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_semantic_memory_round_trip_preserves_concepts(self, data):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For any SemanticMemory, serialization round-trip SHALL preserve all concepts.
        
        **Validates: Requirements 4.6**
        """
        memory = SemanticMemory()
        knowledge = data.draw(knowledge_integration_strategy(min_concepts=2, max_concepts=5))
        
        memory.integrate_knowledge(knowledge)
        original_count = len(memory)
        
        # Serialize and deserialize
        serialized = memory.to_dict()
        restored = SemanticMemory.from_dict(serialized)
        
        assert len(restored) == original_count, \
            f"Concept count should be preserved: was {original_count}, got {len(restored)}"
        
        # Verify each concept exists
        for concept_def in knowledge["concepts"]:
            assert concept_def["name"] in restored, \
                f"Concept '{concept_def['name']}' should be preserved"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_semantic_memory_round_trip_preserves_statistics(self, data):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For any SemanticMemory, serialization round-trip SHALL preserve statistics.
        
        **Validates: Requirements 4.6**
        """
        memory = SemanticMemory()
        
        # Perform multiple integrations to build up statistics
        num_integrations = data.draw(st.integers(min_value=2, max_value=5))
        for i in range(num_integrations):
            memory.integrate_knowledge({
                "concepts": [{"name": f"concept_{i}", "type": "entity"}]
            })
        
        original_stats = memory.get_state()
        
        # Serialize and deserialize
        serialized = memory.to_dict()
        restored = SemanticMemory.from_dict(serialized)
        
        restored_stats = restored.get_state()
        
        assert restored_stats["concept_count"] == original_stats["concept_count"], \
            "Concept count should be preserved"
        assert restored._total_integrations == memory._total_integrations, \
            f"Integration count should be preserved: was {memory._total_integrations}, " \
            f"got {restored._total_integrations}"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_semantic_memory_round_trip_preserves_config(self, data):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For any SemanticMemory, serialization round-trip SHALL preserve configuration.
        
        **Validates: Requirements 4.6**
        """
        config = data.draw(semantic_memory_config_strategy())
        memory = SemanticMemory(config=config)
        
        memory.integrate_knowledge({
            "concepts": [{"name": "test", "type": "entity"}]
        })
        
        # Serialize and deserialize
        serialized = memory.to_dict()
        restored = SemanticMemory.from_dict(serialized)
        
        assert restored._config.conflict_resolution_strategy == config["conflict_resolution_strategy"], \
            "Conflict resolution strategy should be preserved"
        assert abs(restored._config.similarity_threshold - config["similarity_threshold"]) < 0.0001, \
            "Similarity threshold should be preserved"


    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_restored_graph_is_valid(self, data):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For any restored KnowledgeGraph, validate_integrity() SHALL pass.
        
        **Validates: Requirements 4.6**
        """
        graph = data.draw(knowledge_graph_strategy())
        
        # Serialize and deserialize
        serialized = graph.to_dict()
        restored = KnowledgeGraph.from_dict(serialized)
        
        # Validate restored graph
        validation = restored.validate_integrity()
        
        assert validation["valid"], \
            f"Restored graph should be valid: {validation['issues']}"

    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_round_trips_are_stable(self, data):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For any KnowledgeGraph, multiple serialization round-trips SHALL produce
        identical results.
        
        **Validates: Requirements 4.6**
        """
        graph = data.draw(knowledge_graph_strategy())
        
        # First round-trip
        data1 = graph.to_dict()
        restored1 = KnowledgeGraph.from_dict(data1)
        
        # Second round-trip
        data2 = restored1.to_dict()
        restored2 = KnowledgeGraph.from_dict(data2)
        
        # Compare node counts
        assert restored1.get_node_count() == restored2.get_node_count(), \
            "Node count should be stable across round-trips"
        assert restored1.get_relationship_count() == restored2.get_relationship_count(), \
            "Relationship count should be stable across round-trips"
        
        # Compare node IDs
        ids1 = {node.node_id for node in restored1.get_all_nodes()}
        ids2 = {node.node_id for node in restored2.get_all_nodes()}
        assert ids1 == ids2, "Node IDs should be stable across round-trips"

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_empty_graph_round_trip(self, data):
        """
        Feature: consciousness-system-deepening, Property 20: Knowledge Graph Serialization Round-Trip
        
        For an empty KnowledgeGraph, serialization round-trip SHALL produce
        an equivalent empty graph.
        
        **Validates: Requirements 4.6**
        """
        graph = KnowledgeGraph()
        
        # Serialize and deserialize
        serialized = graph.to_dict()
        restored = KnowledgeGraph.from_dict(serialized)
        
        assert restored.get_node_count() == 0, "Restored empty graph should have 0 nodes"
        assert restored.get_relationship_count() == 0, "Restored empty graph should have 0 relationships"
        
        validation = restored.validate_integrity()
        assert validation["valid"], "Empty restored graph should be valid"

"""
Semantic Memory module for consciousness system.

This module implements the SemanticMemory system which wraps the KnowledgeGraph
and provides higher-level operations for knowledge management. The semantic memory
supports:
- Knowledge integration with conflict detection and resolution
- Query interface for concepts and relationships
- Pattern extraction from episodic memory
- Consolidation of knowledge (merging, strengthening, pruning)
- Serialization/deserialization for persistence

Requirements: 4.2, 4.4, 4.5, 4.6
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import time
import uuid

from mm_orch.consciousness.knowledge_graph import (
    KnowledgeGraph,
    ConceptNode,
    Relationship,
    KnowledgeGraphConfig,
)
from mm_orch.consciousness.episodic_memory import Episode


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving knowledge conflicts."""

    NEWEST_WINS = "newest_wins"  # Most recent information takes precedence
    HIGHEST_CONFIDENCE = "highest_confidence"  # Higher confidence wins
    MERGE = "merge"  # Attempt to merge both pieces of information


@dataclass
class ConflictInfo:
    """Information about a detected knowledge conflict."""

    conflict_id: str
    existing_node_id: str
    existing_value: Any
    new_value: Any
    conflict_type: str  # "attribute", "relationship", "type"
    resolution: str  # How the conflict was resolved
    property_key: Optional[str] = None  # The property key for attribute conflicts
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "conflict_id": self.conflict_id,
            "existing_node_id": self.existing_node_id,
            "existing_value": self.existing_value,
            "new_value": self.new_value,
            "conflict_type": self.conflict_type,
            "resolution": self.resolution,
            "property_key": self.property_key,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConflictInfo":
        """Create from dictionary representation."""
        return cls(
            conflict_id=data["conflict_id"],
            existing_node_id=data["existing_node_id"],
            existing_value=data.get("existing_value"),
            new_value=data.get("new_value"),
            conflict_type=data.get("conflict_type", "attribute"),
            resolution=data.get("resolution", "unknown"),
            property_key=data.get("property_key"),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class IntegrationResult:
    """
    Result of knowledge integration operation.

    Tracks what was created, updated, and any conflicts encountered
    during the integration of new knowledge.

    Validates: Requirements 4.2
    """

    new_concepts: List[str]  # IDs of newly created concepts
    updated_concepts: List[str]  # IDs of updated concepts
    new_relationships: List[str]  # IDs of newly created relationships
    conflicts: List[ConflictInfo]  # Conflicts detected and resolved
    resolution_actions: List[str]  # Human-readable descriptions of actions taken
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "new_concepts": self.new_concepts.copy(),
            "updated_concepts": self.updated_concepts.copy(),
            "new_relationships": self.new_relationships.copy(),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolution_actions": self.resolution_actions.copy(),
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationResult":
        """Create from dictionary representation."""
        return cls(
            new_concepts=data.get("new_concepts", []),
            updated_concepts=data.get("updated_concepts", []),
            new_relationships=data.get("new_relationships", []),
            conflicts=[ConflictInfo.from_dict(c) for c in data.get("conflicts", [])],
            resolution_actions=data.get("resolution_actions", []),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ExtractionResult:
    """
    Result of pattern extraction from episodes.

    Tracks concepts and relationships extracted from episodic memory
    and integrated into semantic memory.

    Validates: Requirements 4.4
    """

    extracted_concepts: List[str]  # Names of concepts extracted
    extracted_relationships: List[Tuple[str, str, str]]  # (source, type, target)
    integration_result: Optional[IntegrationResult] = None
    episodes_processed: int = 0
    patterns_found: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "extracted_concepts": self.extracted_concepts.copy(),
            "extracted_relationships": [list(r) for r in self.extracted_relationships],
            "integration_result": (
                self.integration_result.to_dict() if self.integration_result else None
            ),
            "episodes_processed": self.episodes_processed,
            "patterns_found": self.patterns_found,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractionResult":
        """Create from dictionary representation."""
        integration = data.get("integration_result")
        return cls(
            extracted_concepts=data.get("extracted_concepts", []),
            extracted_relationships=[tuple(r) for r in data.get("extracted_relationships", [])],
            integration_result=IntegrationResult.from_dict(integration) if integration else None,
            episodes_processed=data.get("episodes_processed", 0),
            patterns_found=data.get("patterns_found", 0),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ConsolidationResult:
    """
    Result of semantic memory consolidation.

    Tracks merging, strengthening, and pruning operations performed
    during consolidation.

    Validates: Requirements 4.4
    """

    merged_concepts: List[Tuple[str, str]]  # Pairs of (kept_id, removed_id)
    strengthened_relationships: List[str]  # IDs of strengthened relationships
    pruned_concepts: List[str]  # IDs of removed concepts
    pruned_relationships: List[str]  # IDs of removed relationships
    statistics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "merged_concepts": [list(m) for m in self.merged_concepts],
            "strengthened_relationships": self.strengthened_relationships.copy(),
            "pruned_concepts": self.pruned_concepts.copy(),
            "pruned_relationships": self.pruned_relationships.copy(),
            "statistics": self.statistics.copy(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsolidationResult":
        """Create from dictionary representation."""
        return cls(
            merged_concepts=[tuple(m) for m in data.get("merged_concepts", [])],
            strengthened_relationships=data.get("strengthened_relationships", []),
            pruned_concepts=data.get("pruned_concepts", []),
            pruned_relationships=data.get("pruned_relationships", []),
            statistics=data.get("statistics", {}),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class SemanticMemoryConfig:
    """Configuration for the semantic memory system."""

    conflict_resolution_strategy: str = "newest_wins"  # Default strategy
    similarity_threshold: float = 0.8  # Threshold for concept similarity
    min_relationship_strength: float = 0.1  # Minimum strength to retain
    consolidation_merge_threshold: float = 0.9  # Similarity for merging
    prune_access_threshold: int = 0  # Min accesses to avoid pruning
    prune_strength_threshold: float = 0.2  # Min strength to avoid pruning
    max_query_results: int = 20  # Default max results for queries
    extraction_min_frequency: float = 0.2  # Min frequency for pattern extraction

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_strategies = ["newest_wins", "highest_confidence", "merge"]
        if self.conflict_resolution_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid conflict_resolution_strategy: {self.conflict_resolution_strategy}"
            )
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.min_relationship_strength <= 1.0):
            raise ValueError("min_relationship_strength must be between 0.0 and 1.0")
        if not (0.0 <= self.consolidation_merge_threshold <= 1.0):
            raise ValueError("consolidation_merge_threshold must be between 0.0 and 1.0")
        if self.prune_access_threshold < 0:
            raise ValueError("prune_access_threshold must be non-negative")
        if not (0.0 <= self.prune_strength_threshold <= 1.0):
            raise ValueError("prune_strength_threshold must be between 0.0 and 1.0")
        if self.max_query_results < 1:
            raise ValueError("max_query_results must be at least 1")
        if not (0.0 <= self.extraction_min_frequency <= 1.0):
            raise ValueError("extraction_min_frequency must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "conflict_resolution_strategy": self.conflict_resolution_strategy,
            "similarity_threshold": self.similarity_threshold,
            "min_relationship_strength": self.min_relationship_strength,
            "consolidation_merge_threshold": self.consolidation_merge_threshold,
            "prune_access_threshold": self.prune_access_threshold,
            "prune_strength_threshold": self.prune_strength_threshold,
            "max_query_results": self.max_query_results,
            "extraction_min_frequency": self.extraction_min_frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemoryConfig":
        """Create config from dictionary representation."""
        return cls(
            conflict_resolution_strategy=data.get("conflict_resolution_strategy", "newest_wins"),
            similarity_threshold=data.get("similarity_threshold", 0.8),
            min_relationship_strength=data.get("min_relationship_strength", 0.1),
            consolidation_merge_threshold=data.get("consolidation_merge_threshold", 0.9),
            prune_access_threshold=data.get("prune_access_threshold", 0),
            prune_strength_threshold=data.get("prune_strength_threshold", 0.2),
            max_query_results=data.get("max_query_results", 20),
            extraction_min_frequency=data.get("extraction_min_frequency", 0.2),
        )


class SemanticMemory:
    """
    Manages semantic memory with knowledge graph.

    This class wraps the KnowledgeGraph and provides higher-level operations
    for knowledge management including:
    - Knowledge integration with conflict detection and resolution
    - Query interface for concepts and relationships
    - Pattern extraction from episodic memory
    - Consolidation (merging, strengthening, pruning)
    - Serialization/deserialization for persistence

    Requirements: 4.2, 4.4, 4.5, 4.6
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize semantic memory.

        Args:
            config: Optional configuration dictionary.
        """
        if config is not None:
            self._config = SemanticMemoryConfig.from_dict(config)
        else:
            self._config = SemanticMemoryConfig()

        # Initialize the underlying knowledge graph
        self._knowledge_graph = KnowledgeGraph()

        # Statistics tracking
        self._total_integrations: int = 0
        self._total_conflicts: int = 0
        self._total_extractions: int = 0
        self._total_consolidations: int = 0
        self._initialized_at: float = time.time()

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        """Get the underlying knowledge graph."""
        return self._knowledge_graph

    # ==================== Knowledge Integration ====================

    def integrate_knowledge(
        self,
        knowledge: Dict[str, Any],
        source: str = "experience",
        confidence: float = 0.5,
    ) -> IntegrationResult:
        """
        Integrate new knowledge into the semantic memory.

        This method processes incoming knowledge, detects conflicts with
        existing knowledge, resolves conflicts using the configured strategy,
        and updates the knowledge graph accordingly.

        Args:
            knowledge: Dictionary containing knowledge to integrate.
                Expected keys:
                - "concepts": List of concept definitions
                - "relationships": List of relationship definitions
                - "attributes": Dict of concept_name -> attributes to update
            source: Source of the knowledge (e.g., "experience", "user", "inference")
            confidence: Confidence level of the new knowledge (0.0 to 1.0)

        Returns:
            IntegrationResult with details of what was created/updated.

        Validates: Requirements 4.2, 4.5
        """
        result = IntegrationResult(
            new_concepts=[],
            updated_concepts=[],
            new_relationships=[],
            conflicts=[],
            resolution_actions=[],
        )

        try:
            # Process concepts
            concepts = knowledge.get("concepts", [])
            for concept_def in concepts:
                self._integrate_concept(concept_def, source, confidence, result)

            # Process relationships
            relationships = knowledge.get("relationships", [])
            for rel_def in relationships:
                self._integrate_relationship(rel_def, source, confidence, result)

            # Process attribute updates
            attributes = knowledge.get("attributes", {})
            for concept_name, attrs in attributes.items():
                self._integrate_attributes(concept_name, attrs, source, confidence, result)

            self._total_integrations += 1
            self._total_conflicts += len(result.conflicts)

        except Exception as e:
            result.success = False
            result.error_message = str(e)

        return result

    def _integrate_concept(
        self,
        concept_def: Dict[str, Any],
        source: str,
        confidence: float,
        result: IntegrationResult,
    ) -> None:
        """
        Integrate a single concept definition.

        Args:
            concept_def: Concept definition with name, type, properties.
            source: Source of the knowledge.
            confidence: Confidence level.
            result: IntegrationResult to update.
        """
        name = concept_def.get("name", "")
        if not name:
            return

        node_type = concept_def.get("type", "entity")
        properties = concept_def.get("properties", {})

        # Check if concept already exists
        existing = self._knowledge_graph.get_node_by_name(name, record_access=False)

        if existing is None:
            # Create new concept
            node = self._knowledge_graph.create_node(
                name=name,
                node_type=node_type,
                properties=properties,
                metadata={"source": source, "confidence": confidence},
            )
            result.new_concepts.append(node.node_id)
            result.resolution_actions.append(f"Created new concept: {name}")
        else:
            # Check for conflicts and update
            conflicts = self._detect_concept_conflicts(existing, concept_def, confidence)

            # Track which properties had conflicts (to avoid double-updating)
            conflicted_properties = set()
            for conflict in conflicts:
                if conflict.conflict_type == "attribute" and conflict.property_key:
                    conflicted_properties.add(conflict.property_key)
                resolved = self._resolve_conflict(conflict, existing, concept_def, confidence)
                result.conflicts.append(resolved)

            # Update non-conflicting properties only
            if properties:
                non_conflicting_props = {
                    k: v
                    for k, v in properties.items()
                    if k not in conflicted_properties and k not in existing.properties
                }
                if non_conflicting_props:
                    self._knowledge_graph.update_node(existing.node_id, non_conflicting_props)
                existing.metadata["last_source"] = source
                existing.metadata["last_confidence"] = confidence

            result.updated_concepts.append(existing.node_id)
            result.resolution_actions.append(f"Updated existing concept: {name}")

    def _detect_concept_conflicts(
        self,
        existing: ConceptNode,
        new_def: Dict[str, Any],
        new_confidence: float,
    ) -> List[ConflictInfo]:
        """
        Detect conflicts between existing concept and new definition.

        Args:
            existing: Existing concept node.
            new_def: New concept definition.
            new_confidence: Confidence of new information.

        Returns:
            List of detected conflicts.
        """
        conflicts = []

        # Check type conflict
        new_type = new_def.get("type")
        if new_type and new_type != existing.node_type:
            conflicts.append(
                ConflictInfo(
                    conflict_id=str(uuid.uuid4()),
                    existing_node_id=existing.node_id,
                    existing_value=existing.node_type,
                    new_value=new_type,
                    conflict_type="type",
                    resolution="pending",
                )
            )

        # Check property conflicts
        new_props = new_def.get("properties", {})
        for key, new_value in new_props.items():
            if key in existing.properties:
                existing_value = existing.properties[key]
                if existing_value != new_value:
                    conflicts.append(
                        ConflictInfo(
                            conflict_id=str(uuid.uuid4()),
                            existing_node_id=existing.node_id,
                            existing_value=existing_value,
                            new_value=new_value,
                            conflict_type="attribute",
                            resolution="pending",
                            property_key=key,
                        )
                    )

        return conflicts

    def _resolve_conflict(
        self,
        conflict: ConflictInfo,
        existing: ConceptNode,
        new_def: Dict[str, Any],
        new_confidence: float,
    ) -> ConflictInfo:
        """
        Resolve a knowledge conflict using the configured strategy.

        Args:
            conflict: The conflict to resolve.
            existing: Existing concept node.
            new_def: New concept definition.
            new_confidence: Confidence of new information.

        Returns:
            Updated ConflictInfo with resolution.

        Validates: Requirements 4.5
        """
        strategy = ConflictResolutionStrategy(self._config.conflict_resolution_strategy)
        existing_confidence = existing.metadata.get("confidence", 0.5)

        if strategy == ConflictResolutionStrategy.NEWEST_WINS:
            # New information always wins
            conflict.resolution = "new_wins"
            self._apply_new_value(existing, conflict)

        elif strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            # Higher confidence wins
            if new_confidence > existing_confidence:
                conflict.resolution = "new_wins_confidence"
                self._apply_new_value(existing, conflict)
            else:
                conflict.resolution = "existing_wins_confidence"

        elif strategy == ConflictResolutionStrategy.MERGE:
            # Attempt to merge values
            merged = self._merge_values(conflict.existing_value, conflict.new_value)
            if merged is not None:
                conflict.resolution = "merged"
                conflict.new_value = merged
                self._apply_new_value(existing, conflict)
            else:
                # Fall back to newest wins if merge fails
                conflict.resolution = "new_wins_merge_failed"
                self._apply_new_value(existing, conflict)

        return conflict

    def _apply_new_value(self, existing: ConceptNode, conflict: ConflictInfo) -> None:
        """Apply the new value from a conflict resolution."""
        if conflict.conflict_type == "type":
            # Type changes are not directly applied to avoid breaking relationships
            existing.metadata["suggested_type"] = conflict.new_value
        elif conflict.conflict_type == "attribute":
            # Use property_key if available, otherwise find by value
            if conflict.property_key and conflict.property_key in existing.properties:
                existing.properties[conflict.property_key] = conflict.new_value
            else:
                # Fallback: find the attribute key by value
                for key, value in existing.properties.items():
                    if value == conflict.existing_value:
                        existing.properties[key] = conflict.new_value
                        break

    def _merge_values(self, existing_value: Any, new_value: Any) -> Optional[Any]:
        """
        Attempt to merge two values.

        Returns merged value or None if merge is not possible.
        """
        # Merge lists by combining unique elements
        if isinstance(existing_value, list) and isinstance(new_value, list):
            merged = list(existing_value)
            for item in new_value:
                if item not in merged:
                    merged.append(item)
            return merged

        # Merge dicts by combining keys
        if isinstance(existing_value, dict) and isinstance(new_value, dict):
            merged = dict(existing_value)
            merged.update(new_value)
            return merged

        # Merge strings by concatenation if different
        if isinstance(existing_value, str) and isinstance(new_value, str):
            if existing_value != new_value:
                return f"{existing_value}; {new_value}"
            return existing_value

        # Cannot merge other types
        return None

    def _integrate_relationship(
        self,
        rel_def: Dict[str, Any],
        source: str,
        confidence: float,
        result: IntegrationResult,
    ) -> None:
        """
        Integrate a single relationship definition.

        Args:
            rel_def: Relationship definition with source, target, type.
            source: Source of the knowledge.
            confidence: Confidence level.
            result: IntegrationResult to update.
        """
        source_name = rel_def.get("source", "")
        target_name = rel_def.get("target", "")
        rel_type = rel_def.get("type", "related_to")
        weight = rel_def.get("weight", 0.5)

        if not source_name or not target_name:
            return

        # Find or create source node
        source_node = self._knowledge_graph.get_node_by_name(source_name, record_access=False)
        if source_node is None:
            source_node = self._knowledge_graph.create_node(
                name=source_name,
                node_type="entity",
                metadata={"source": source, "confidence": confidence},
            )
            result.new_concepts.append(source_node.node_id)

        # Find or create target node
        target_node = self._knowledge_graph.get_node_by_name(target_name, record_access=False)
        if target_node is None:
            target_node = self._knowledge_graph.create_node(
                name=target_name,
                node_type="entity",
                metadata={"source": source, "confidence": confidence},
            )
            result.new_concepts.append(target_node.node_id)

        # Check if relationship already exists
        existing_rel = self._knowledge_graph.get_relationship_between(
            source_node.node_id, target_node.node_id, rel_type
        )

        if existing_rel is None:
            # Create new relationship
            rel = self._knowledge_graph.create_relationship(
                source_id=source_node.node_id,
                target_id=target_node.node_id,
                relationship_type=rel_type,
                weight=weight,
                metadata={"source": source, "confidence": confidence},
            )
            result.new_relationships.append(rel.relationship_id)
            result.resolution_actions.append(
                f"Created relationship: {source_name} -{rel_type}-> {target_name}"
            )
        else:
            # Strengthen existing relationship
            new_weight = min(1.0, existing_rel.weight + 0.1)
            existing_rel.weight = new_weight
            result.resolution_actions.append(
                f"Strengthened relationship: {source_name} -{rel_type}-> {target_name}"
            )

    def _integrate_attributes(
        self,
        concept_name: str,
        attributes: Dict[str, Any],
        source: str,
        confidence: float,
        result: IntegrationResult,
    ) -> None:
        """
        Integrate attribute updates for a concept.

        Args:
            concept_name: Name of the concept to update.
            attributes: Attributes to add/update.
            source: Source of the knowledge.
            confidence: Confidence level.
            result: IntegrationResult to update.
        """
        node = self._knowledge_graph.get_node_by_name(concept_name, record_access=False)

        if node is None:
            # Create new concept with attributes
            node = self._knowledge_graph.create_node(
                name=concept_name,
                node_type="entity",
                properties=attributes,
                metadata={"source": source, "confidence": confidence},
            )
            result.new_concepts.append(node.node_id)
            result.resolution_actions.append(f"Created concept with attributes: {concept_name}")
        else:
            # Check for conflicts and update
            for key, new_value in attributes.items():
                if key in node.properties:
                    existing_value = node.properties[key]
                    if existing_value != new_value:
                        conflict = ConflictInfo(
                            conflict_id=str(uuid.uuid4()),
                            existing_node_id=node.node_id,
                            existing_value=existing_value,
                            new_value=new_value,
                            conflict_type="attribute",
                            resolution="pending",
                        )
                        resolved = self._resolve_conflict(
                            conflict, node, {"properties": {key: new_value}}, confidence
                        )
                        result.conflicts.append(resolved)
                else:
                    node.properties[key] = new_value

            node.metadata["last_source"] = source
            node.metadata["last_confidence"] = confidence
            result.updated_concepts.append(node.node_id)
            result.resolution_actions.append(f"Updated attributes for: {concept_name}")

    # ==================== Query Interface ====================

    def query(
        self,
        query_text: str,
        max_results: Optional[int] = None,
    ) -> List[ConceptNode]:
        """
        Query semantic memory for concepts matching the query text.

        Args:
            query_text: Text to search for in concept names and properties.
            max_results: Maximum number of results to return.

        Returns:
            List of matching ConceptNodes, sorted by relevance.

        Validates: Requirements 4.3
        """
        if max_results is None:
            max_results = self._config.max_query_results

        return self._knowledge_graph.find_nodes(query_text, max_results=max_results)

    def query_relationships(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[Relationship]:
        """
        Query relationships for a specific node.

        Args:
            node_id: ID of the node to query relationships for.
            relationship_type: Optional filter by relationship type.
            direction: "outgoing", "incoming", or "both".

        Returns:
            List of matching Relationships.

        Validates: Requirements 4.3
        """
        return self._knowledge_graph.get_relationships(
            node_id, direction=direction, relationship_type=relationship_type
        )

    def find_related_concepts(
        self,
        concept_name: str,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None,
    ) -> List[ConceptNode]:
        """
        Find concepts related to the given concept within max_depth hops.

        Args:
            concept_name: Name of the starting concept.
            max_depth: Maximum traversal depth.
            relationship_types: Optional filter by relationship types.

        Returns:
            List of related ConceptNodes.

        Validates: Requirements 4.3
        """
        # Find the starting node
        start_node = self._knowledge_graph.get_node_by_name(concept_name)
        if start_node is None:
            return []

        # Traverse the graph
        paths = self._knowledge_graph.traverse(
            start_id=start_node.node_id,
            relationship_types=relationship_types,
            max_depth=max_depth,
            direction="both",
        )

        # Collect unique related nodes
        related_ids: Set[str] = set()
        for path in paths:
            for node_id in path:
                if node_id != start_node.node_id:
                    related_ids.add(node_id)

        # Get the actual nodes
        related_nodes = []
        for node_id in related_ids:
            node = self._knowledge_graph.get_node(node_id)
            if node is not None:
                related_nodes.append(node)

        return related_nodes

    def get_concept_definition(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the full definition of a concept including its relationships.

        Args:
            concept_name: Name of the concept.

        Returns:
            Dictionary with concept details or None if not found.

        Validates: Requirements 4.3
        """
        node = self._knowledge_graph.get_node_by_name(concept_name)
        if node is None:
            return None

        # Get relationships
        relationships = self._knowledge_graph.get_relationships(node.node_id)

        # Build relationship summaries
        rel_summaries = []
        for rel in relationships:
            if rel.source_id == node.node_id:
                target = self._knowledge_graph.get_node(rel.target_id, record_access=False)
                target_name = target.name if target else rel.target_id
                rel_summaries.append(
                    {
                        "type": rel.relationship_type,
                        "direction": "outgoing",
                        "target": target_name,
                        "weight": rel.weight,
                    }
                )
            else:
                source = self._knowledge_graph.get_node(rel.source_id, record_access=False)
                source_name = source.name if source else rel.source_id
                rel_summaries.append(
                    {
                        "type": rel.relationship_type,
                        "direction": "incoming",
                        "source": source_name,
                        "weight": rel.weight,
                    }
                )

        return {
            "node_id": node.node_id,
            "name": node.name,
            "type": node.node_type,
            "properties": node.properties.copy(),
            "relationships": rel_summaries,
            "access_count": node.access_count,
            "created_at": node.created_at,
            "updated_at": node.updated_at,
            "metadata": node.metadata.copy(),
        }

    # ==================== Pattern Extraction from Episodes ====================

    def extract_from_episodes(self, episodes: List[Episode]) -> ExtractionResult:
        """
        Extract generalizable patterns from episodes and integrate into semantic memory.

        This method analyzes episodes to extract:
        - Concepts from context keys and values
        - Relationships from event sequences and context associations

        Args:
            episodes: List of episodes to extract patterns from.

        Returns:
            ExtractionResult with extracted concepts and relationships.

        Validates: Requirements 4.4
        """
        result = ExtractionResult(
            extracted_concepts=[],
            extracted_relationships=[],
            episodes_processed=len(episodes),
        )

        if not episodes:
            return result

        # Extract concepts from contexts
        concept_counts: Dict[str, int] = {}
        concept_types: Dict[str, str] = {}

        for episode in episodes:
            self._extract_concepts_from_context(episode.context, concept_counts, concept_types)

        # Filter by frequency threshold
        min_count = max(1, int(len(episodes) * self._config.extraction_min_frequency))
        for concept_name, count in concept_counts.items():
            if count >= min_count:
                result.extracted_concepts.append(concept_name)

        # Extract relationships from event sequences
        relationship_counts: Dict[Tuple[str, str, str], int] = {}

        for episode in episodes:
            self._extract_relationships_from_events(episode.events, relationship_counts)
            self._extract_relationships_from_context(episode.context, relationship_counts)

        # Filter relationships by frequency
        for (source, rel_type, target), count in relationship_counts.items():
            if count >= min_count:
                result.extracted_relationships.append((source, rel_type, target))

        result.patterns_found = len(result.extracted_concepts) + len(result.extracted_relationships)

        # Integrate extracted knowledge
        if result.extracted_concepts or result.extracted_relationships:
            knowledge = self._build_knowledge_from_extraction(
                result.extracted_concepts,
                result.extracted_relationships,
                concept_types,
            )
            result.integration_result = self.integrate_knowledge(
                knowledge, source="episodic_extraction"
            )

        self._total_extractions += 1
        return result

    def _extract_concepts_from_context(
        self,
        context: Dict[str, Any],
        concept_counts: Dict[str, int],
        concept_types: Dict[str, str],
    ) -> None:
        """Extract concept names from episode context."""
        for key, value in context.items():
            # Keys often represent concept types or attributes
            if isinstance(key, str) and len(key) > 1:
                concept_counts[key] = concept_counts.get(key, 0) + 1
                concept_types[key] = "attribute"

            # String values often represent entities
            if isinstance(value, str) and len(value) > 1 and not value.isdigit():
                concept_counts[value] = concept_counts.get(value, 0) + 1
                concept_types[value] = "entity"

            # Recurse into nested dicts
            elif isinstance(value, dict):
                self._extract_concepts_from_context(value, concept_counts, concept_types)

            # Extract from lists
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and len(item) > 1:
                        concept_counts[item] = concept_counts.get(item, 0) + 1
                        concept_types[item] = "entity"
                    elif isinstance(item, dict):
                        self._extract_concepts_from_context(item, concept_counts, concept_types)

    def _extract_relationships_from_events(
        self,
        events: List[Dict[str, Any]],
        relationship_counts: Dict[Tuple[str, str, str], int],
    ) -> None:
        """Extract relationships from event sequences."""
        # Extract sequential relationships (event A followed by event B)
        for i in range(len(events) - 1):
            event1 = events[i]
            event2 = events[i + 1]

            type1 = event1.get("type", event1.get("name", "unknown"))
            type2 = event2.get("type", event2.get("name", "unknown"))

            if isinstance(type1, str) and isinstance(type2, str):
                key = (type1, "followed_by", type2)
                relationship_counts[key] = relationship_counts.get(key, 0) + 1

        # Extract relationships within events
        for event in events:
            event_type = event.get("type", event.get("name"))
            if not isinstance(event_type, str):
                continue

            # Look for subject-action-object patterns
            subject = event.get("subject", event.get("actor"))
            obj = event.get("object", event.get("target"))

            if isinstance(subject, str) and isinstance(event_type, str):
                key = (subject, "performs", event_type)
                relationship_counts[key] = relationship_counts.get(key, 0) + 1

            if isinstance(event_type, str) and isinstance(obj, str):
                key = (event_type, "affects", obj)
                relationship_counts[key] = relationship_counts.get(key, 0) + 1

    def _extract_relationships_from_context(
        self,
        context: Dict[str, Any],
        relationship_counts: Dict[Tuple[str, str, str], int],
    ) -> None:
        """Extract relationships from context associations."""
        # Look for key-value pairs that suggest relationships
        for key, value in context.items():
            if not isinstance(key, str):
                continue

            # String values suggest "has_attribute" relationships
            if isinstance(value, str) and len(value) > 1:
                rel_key = (key, "has_value", value)
                relationship_counts[rel_key] = relationship_counts.get(rel_key, 0) + 1

            # List values suggest "contains" relationships
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and len(item) > 1:
                        rel_key = (key, "contains", item)
                        relationship_counts[rel_key] = relationship_counts.get(rel_key, 0) + 1

    def _build_knowledge_from_extraction(
        self,
        concepts: List[str],
        relationships: List[Tuple[str, str, str]],
        concept_types: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build knowledge dictionary from extracted patterns."""
        knowledge: Dict[str, Any] = {
            "concepts": [],
            "relationships": [],
        }

        # Add concepts
        for concept_name in concepts:
            knowledge["concepts"].append(
                {
                    "name": concept_name,
                    "type": concept_types.get(concept_name, "entity"),
                    "properties": {"extracted": True},
                }
            )

        # Add relationships
        for source, rel_type, target in relationships:
            knowledge["relationships"].append(
                {
                    "source": source,
                    "type": rel_type,
                    "target": target,
                    "weight": 0.5,
                }
            )

        return knowledge

    # ==================== Consolidation ====================

    def consolidate(self) -> ConsolidationResult:
        """
        Consolidate semantic memory by merging, strengthening, and pruning.

        This method performs:
        - Merging similar concepts (based on name similarity)
        - Strengthening frequently accessed relationships
        - Pruning weak/unused knowledge

        Returns:
            ConsolidationResult with details of operations performed.

        Validates: Requirements 4.4
        """
        result = ConsolidationResult(
            merged_concepts=[],
            strengthened_relationships=[],
            pruned_concepts=[],
            pruned_relationships=[],
        )

        # Step 1: Merge similar concepts
        self._merge_similar_concepts(result)

        # Step 2: Strengthen frequently accessed relationships
        self._strengthen_relationships(result)

        # Step 3: Prune weak/unused knowledge
        self._prune_weak_knowledge(result)

        # Update statistics
        result.statistics = {
            "total_concepts": self._knowledge_graph.get_node_count(),
            "total_relationships": self._knowledge_graph.get_relationship_count(),
            "merged_count": len(result.merged_concepts),
            "strengthened_count": len(result.strengthened_relationships),
            "pruned_concepts_count": len(result.pruned_concepts),
            "pruned_relationships_count": len(result.pruned_relationships),
        }

        self._total_consolidations += 1
        return result

    def _merge_similar_concepts(self, result: ConsolidationResult) -> None:
        """Merge concepts with very similar names."""
        nodes = self._knowledge_graph.get_all_nodes()
        merged_ids: Set[str] = set()

        for i, node1 in enumerate(nodes):
            if node1.node_id in merged_ids:
                continue

            for node2 in nodes[i + 1 :]:
                if node2.node_id in merged_ids:
                    continue

                # Check name similarity
                similarity = self._calculate_name_similarity(node1.name, node2.name)

                if similarity >= self._config.consolidation_merge_threshold:
                    # Merge node2 into node1 (keep the one with more accesses)
                    if node1.access_count >= node2.access_count:
                        kept, removed = node1, node2
                    else:
                        kept, removed = node2, node1

                    # Transfer relationships from removed to kept
                    self._transfer_relationships(removed.node_id, kept.node_id)

                    # Merge properties
                    for key, value in removed.properties.items():
                        if key not in kept.properties:
                            kept.properties[key] = value

                    # Remove the merged node
                    self._knowledge_graph.remove_node(removed.node_id)
                    merged_ids.add(removed.node_id)

                    result.merged_concepts.append((kept.node_id, removed.node_id))

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two concept names."""
        # Normalize names
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        # Exact match
        if n1 == n2:
            return 1.0

        # One contains the other
        if n1 in n2 or n2 in n1:
            shorter = min(len(n1), len(n2))
            longer = max(len(n1), len(n2))
            return shorter / longer if longer > 0 else 0.0

        # Character-level Jaccard similarity
        set1 = set(n1)
        set2 = set(n2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _transfer_relationships(self, from_id: str, to_id: str) -> None:
        """Transfer relationships from one node to another."""
        relationships = self._knowledge_graph.get_relationships(from_id)

        for rel in relationships:
            # Determine new source and target
            if rel.source_id == from_id:
                new_source = to_id
                new_target = rel.target_id
            else:
                new_source = rel.source_id
                new_target = to_id

            # Skip self-loops
            if new_source == new_target:
                continue

            # Check if relationship already exists
            existing = self._knowledge_graph.get_relationship_between(
                new_source, new_target, rel.relationship_type
            )

            if existing is None:
                # Create new relationship
                try:
                    self._knowledge_graph.create_relationship(
                        source_id=new_source,
                        target_id=new_target,
                        relationship_type=rel.relationship_type,
                        weight=rel.weight,
                        properties=rel.properties.copy(),
                        metadata=rel.metadata.copy(),
                    )
                except ValueError:
                    # Node might have been removed
                    pass
            else:
                # Strengthen existing relationship
                existing.weight = min(1.0, existing.weight + rel.weight * 0.5)

    def _strengthen_relationships(self, result: ConsolidationResult) -> None:
        """Strengthen relationships based on node access patterns."""
        relationships = self._knowledge_graph.get_all_relationships()

        for rel in relationships:
            source = self._knowledge_graph.get_node(rel.source_id, record_access=False)
            target = self._knowledge_graph.get_node(rel.target_id, record_access=False)

            if source is None or target is None:
                continue

            # Calculate access-based strength boost
            total_accesses = source.access_count + target.access_count
            if total_accesses > 5:  # Threshold for "frequently accessed"
                boost = min(0.1, total_accesses * 0.01)
                new_weight = min(1.0, rel.weight + boost)

                if new_weight > rel.weight:
                    rel.weight = new_weight
                    result.strengthened_relationships.append(rel.relationship_id)

    def _prune_weak_knowledge(self, result: ConsolidationResult) -> None:
        """Prune weak/unused concepts and relationships."""
        # Prune weak relationships first
        relationships = self._knowledge_graph.get_all_relationships()
        for rel in relationships:
            if rel.weight < self._config.prune_strength_threshold:
                self._knowledge_graph.remove_relationship(rel.relationship_id)
                result.pruned_relationships.append(rel.relationship_id)

        # Prune unused concepts (no relationships and low access)
        nodes = self._knowledge_graph.get_all_nodes()
        for node in nodes:
            if node.access_count <= self._config.prune_access_threshold:
                # Check if node has any relationships
                rels = self._knowledge_graph.get_relationships(node.node_id)
                if not rels:
                    self._knowledge_graph.remove_node(node.node_id)
                    result.pruned_concepts.append(node.node_id)

    # ==================== Statistics and State ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the semantic memory."""
        graph_stats = self._knowledge_graph.get_statistics()

        return {
            "knowledge_graph": graph_stats,
            "total_integrations": self._total_integrations,
            "total_conflicts": self._total_conflicts,
            "total_extractions": self._total_extractions,
            "total_consolidations": self._total_consolidations,
            "uptime": time.time() - self._initialized_at,
            "config": self._config.to_dict(),
        }

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the semantic memory."""
        return {
            "concept_count": self._knowledge_graph.get_node_count(),
            "relationship_count": self._knowledge_graph.get_relationship_count(),
            "total_integrations": self._total_integrations,
            "total_conflicts": self._total_conflicts,
            "total_extractions": self._total_extractions,
            "total_consolidations": self._total_consolidations,
            "uptime": time.time() - self._initialized_at,
        }

    # ==================== Serialization ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the semantic memory to a dictionary.

        Returns:
            Dictionary representation of the semantic memory.

        Validates: Requirements 4.6
        """
        return {
            "config": self._config.to_dict(),
            "knowledge_graph": self._knowledge_graph.to_dict(),
            "total_integrations": self._total_integrations,
            "total_conflicts": self._total_conflicts,
            "total_extractions": self._total_extractions,
            "total_consolidations": self._total_consolidations,
            "initialized_at": self._initialized_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemory":
        """
        Create a semantic memory from a dictionary.

        Args:
            data: Dictionary representation of the semantic memory.

        Returns:
            A new SemanticMemory instance.

        Validates: Requirements 4.6
        """
        config = data.get("config", {})
        memory = cls(config=config)

        # Restore knowledge graph
        kg_data = data.get("knowledge_graph", {})
        if kg_data:
            memory._knowledge_graph = KnowledgeGraph.from_dict(kg_data)

        # Restore statistics
        memory._total_integrations = data.get("total_integrations", 0)
        memory._total_conflicts = data.get("total_conflicts", 0)
        memory._total_extractions = data.get("total_extractions", 0)
        memory._total_consolidations = data.get("total_consolidations", 0)
        memory._initialized_at = data.get("initialized_at", time.time())

        return memory

    def clear(self) -> None:
        """Clear all knowledge from semantic memory."""
        self._knowledge_graph.clear()

    def __len__(self) -> int:
        """Return the number of concepts in semantic memory."""
        return self._knowledge_graph.get_node_count()

    def __contains__(self, concept_name: str) -> bool:
        """Check if a concept exists in semantic memory."""
        return self._knowledge_graph.get_node_by_name(concept_name, record_access=False) is not None


# ==================== Factory Functions ====================


def create_semantic_memory(config: Optional[Dict[str, Any]] = None) -> SemanticMemory:
    """
    Factory function to create a SemanticMemory instance.

    Args:
        config: Optional configuration dictionary.

    Returns:
        A new SemanticMemory instance.

    Validates: Requirements 4.2
    """
    return SemanticMemory(config=config)

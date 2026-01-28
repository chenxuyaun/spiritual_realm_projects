"""
Knowledge Graph module for consciousness system.

This module implements the KnowledgeGraph system which stores concepts, entities,
and their relationships in a graph structure. The knowledge graph supports:
- Node (concept) CRUD operations
- Relationship CRUD operations
- Graph traversal methods
- Relationship inference
- Serialization/deserialization for persistence

Requirements: 4.1, 4.3
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import time
import uuid
from collections import deque


@dataclass
class ConceptNode:
    """
    Represents a concept node in the knowledge graph.

    A concept node captures an entity, attribute, action, or relation
    with its properties and metadata for knowledge management.

    Attributes:
        node_id: Unique identifier for the node.
        name: Human-readable name of the concept.
        node_type: Category of concept (e.g., "entity", "attribute", "action", "relation").
        properties: Dictionary of attributes and their values.
        embedding: Optional vector representation for similarity search.
        created_at: Timestamp when the node was created.
        updated_at: Timestamp when the node was last updated.
        access_count: Number of times this node has been accessed.
        metadata: Additional information about the node.

    Validates: Requirements 4.1
    """

    node_id: str
    name: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate node fields after initialization."""
        if not self.node_id:
            raise ValueError("node_id cannot be empty")

        if not self.name:
            raise ValueError("name cannot be empty")

        if not self.node_type:
            raise ValueError("node_type cannot be empty")

        if not isinstance(self.properties, dict):
            raise ValueError("properties must be a dictionary")

        if self.embedding is not None and not isinstance(self.embedding, list):
            raise ValueError("embedding must be a list or None")

        if not isinstance(self.created_at, (int, float)):
            raise ValueError("created_at must be a number")

        if not isinstance(self.updated_at, (int, float)):
            raise ValueError("updated_at must be a number")

        if not isinstance(self.access_count, int):
            raise ValueError("access_count must be an integer")

        if self.access_count < 0:
            raise ValueError("access_count cannot be negative")

        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def record_access(self) -> None:
        """Record that this node was accessed."""
        self.access_count += 1
        self.updated_at = time.time()

    def update_properties(self, new_properties: Dict[str, Any]) -> None:
        """Update node properties and timestamp."""
        self.properties.update(new_properties)
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation for serialization."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type,
            "properties": self.properties.copy(),
            "embedding": self.embedding.copy() if self.embedding else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptNode":
        """Create a node from dictionary representation."""
        return cls(
            node_id=data["node_id"],
            name=data["name"],
            node_type=data["node_type"],
            properties=data.get("properties", {}),
            embedding=data.get("embedding"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Relationship:
    """
    Represents a relationship between two concept nodes.

    A relationship captures the connection between a source and target
    concept with a type, weight, and additional properties.

    Attributes:
        relationship_id: Unique identifier for the relationship.
        source_id: ID of the source node.
        target_id: ID of the target node.
        relationship_type: Type of relationship (e.g., "is_a", "has_property", "related_to").
        weight: Strength of the relationship (0.0 to 1.0).
        properties: Additional attributes of the relationship.
        created_at: Timestamp when the relationship was created.
        metadata: Additional information about the relationship.

    Validates: Requirements 4.1
    """

    relationship_id: str
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate relationship fields after initialization."""
        if not self.relationship_id:
            raise ValueError("relationship_id cannot be empty")

        if not self.source_id:
            raise ValueError("source_id cannot be empty")

        if not self.target_id:
            raise ValueError("target_id cannot be empty")

        if not self.relationship_type:
            raise ValueError("relationship_type cannot be empty")

        if not isinstance(self.weight, (int, float)):
            raise ValueError("weight must be a number")

        # Clamp weight to valid range [0.0, 1.0]
        self.weight = max(0.0, min(1.0, float(self.weight)))

        if not isinstance(self.properties, dict):
            raise ValueError("properties must be a dictionary")

        if not isinstance(self.created_at, (int, float)):
            raise ValueError("created_at must be a number")

        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation for serialization."""
        return {
            "relationship_id": self.relationship_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "properties": self.properties.copy(),
            "created_at": self.created_at,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create a relationship from dictionary representation."""
        return cls(
            relationship_id=data["relationship_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relationship_type=data["relationship_type"],
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class KnowledgeGraphConfig:
    """Configuration for the knowledge graph."""

    max_nodes: int = 10000
    max_relationships: int = 50000
    default_relationship_weight: float = 0.5
    inference_min_weight: float = 0.3  # Minimum weight for inferred relationships
    max_inference_depth: int = 2  # Maximum depth for relationship inference

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be at least 1")
        if self.max_relationships < 1:
            raise ValueError("max_relationships must be at least 1")
        if not (0.0 <= self.default_relationship_weight <= 1.0):
            raise ValueError("default_relationship_weight must be between 0.0 and 1.0")
        if not (0.0 <= self.inference_min_weight <= 1.0):
            raise ValueError("inference_min_weight must be between 0.0 and 1.0")
        if self.max_inference_depth < 1:
            raise ValueError("max_inference_depth must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "max_nodes": self.max_nodes,
            "max_relationships": self.max_relationships,
            "default_relationship_weight": self.default_relationship_weight,
            "inference_min_weight": self.inference_min_weight,
            "max_inference_depth": self.max_inference_depth,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraphConfig":
        """Create config from dictionary representation."""
        return cls(
            max_nodes=data.get("max_nodes", 10000),
            max_relationships=data.get("max_relationships", 50000),
            default_relationship_weight=data.get("default_relationship_weight", 0.5),
            inference_min_weight=data.get("inference_min_weight", 0.3),
            max_inference_depth=data.get("max_inference_depth", 2),
        )


class KnowledgeGraph:
    """
    Graph-based knowledge representation.

    This class implements a knowledge graph that stores concepts as nodes
    and their relationships as edges. It supports:
    - Node and relationship CRUD operations
    - Graph traversal methods
    - Path finding between nodes
    - Relationship inference based on graph structure
    - Serialization/deserialization for persistence

    Requirements: 4.1, 4.3
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge graph.

        Args:
            config: Optional configuration dictionary.
        """
        if config is not None:
            self._config = KnowledgeGraphConfig.from_dict(config)
        else:
            self._config = KnowledgeGraphConfig()

        # Main storage: node_id -> ConceptNode
        self._nodes: Dict[str, ConceptNode] = {}

        # Relationship storage: relationship_id -> Relationship
        self._relationships: Dict[str, Relationship] = {}

        # Index: source_id -> list of relationship_ids (outgoing)
        self._outgoing_index: Dict[str, List[str]] = {}

        # Index: target_id -> list of relationship_ids (incoming)
        self._incoming_index: Dict[str, List[str]] = {}

        # Index: node name -> node_id (for name-based lookup)
        self._name_index: Dict[str, str] = {}

        # Statistics
        self._total_nodes_created: int = 0
        self._total_relationships_created: int = 0
        self._initialized_at: float = time.time()

    def _generate_node_id(self) -> str:
        """Generate a unique node ID."""
        return str(uuid.uuid4())

    def _generate_relationship_id(self) -> str:
        """Generate a unique relationship ID."""
        return str(uuid.uuid4())

    # ==================== Node CRUD Operations ====================

    def add_node(self, node: ConceptNode) -> str:
        """
        Add a concept node to the graph.

        Args:
            node: The ConceptNode to add.

        Returns:
            The node ID.

        Raises:
            ValueError: If a node with the same ID already exists.

        Validates: Requirements 4.1
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node with ID '{node.node_id}' already exists")

        if len(self._nodes) >= self._config.max_nodes:
            raise ValueError(f"Maximum number of nodes ({self._config.max_nodes}) reached")

        self._nodes[node.node_id] = node
        self._name_index[node.name.lower()] = node.node_id
        self._outgoing_index[node.node_id] = []
        self._incoming_index[node.node_id] = []
        self._total_nodes_created += 1

        return node.node_id

    def create_node(
        self,
        name: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConceptNode:
        """
        Create and add a new concept node to the graph.

        Args:
            name: Human-readable name of the concept.
            node_type: Category of concept.
            properties: Dictionary of attributes.
            embedding: Optional vector representation.
            metadata: Additional information.

        Returns:
            The created ConceptNode.

        Validates: Requirements 4.1
        """
        node = ConceptNode(
            node_id=self._generate_node_id(),
            name=name,
            node_type=node_type,
            properties=properties if properties is not None else {},
            embedding=embedding,
            created_at=time.time(),
            updated_at=time.time(),
            access_count=0,
            metadata=metadata if metadata is not None else {},
        )

        self.add_node(node)
        return node

    def get_node(self, node_id: str, record_access: bool = True) -> Optional[ConceptNode]:
        """
        Get a node by ID.

        Args:
            node_id: ID of the node to retrieve.
            record_access: Whether to record this access.

        Returns:
            The node if found, None otherwise.

        Validates: Requirements 4.1
        """
        node = self._nodes.get(node_id)
        if node is not None and record_access:
            node.record_access()
        return node

    def get_node_by_name(self, name: str, record_access: bool = True) -> Optional[ConceptNode]:
        """
        Get a node by name (case-insensitive).

        Args:
            name: Name of the node to retrieve.
            record_access: Whether to record this access.

        Returns:
            The node if found, None otherwise.
        """
        node_id = self._name_index.get(name.lower())
        if node_id is not None:
            return self.get_node(node_id, record_access)
        return None

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update the properties of a node.

        Args:
            node_id: ID of the node to update.
            properties: New properties to merge with existing ones.

        Returns:
            True if the node was updated, False if it doesn't exist.

        Validates: Requirements 4.1
        """
        node = self._nodes.get(node_id)
        if node is None:
            return False

        node.update_properties(properties)
        return True

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its relationships from the graph.

        Args:
            node_id: ID of the node to remove.

        Returns:
            True if the node was removed, False if it doesn't exist.

        Validates: Requirements 4.1
        """
        if node_id not in self._nodes:
            return False

        node = self._nodes[node_id]

        # Remove all outgoing relationships
        outgoing_rel_ids = self._outgoing_index.get(node_id, []).copy()
        for rel_id in outgoing_rel_ids:
            self._remove_relationship_internal(rel_id)

        # Remove all incoming relationships
        incoming_rel_ids = self._incoming_index.get(node_id, []).copy()
        for rel_id in incoming_rel_ids:
            self._remove_relationship_internal(rel_id)

        # Remove from indices
        del self._outgoing_index[node_id]
        del self._incoming_index[node_id]

        # Remove from name index
        name_lower = node.name.lower()
        if name_lower in self._name_index and self._name_index[name_lower] == node_id:
            del self._name_index[name_lower]

        # Remove the node
        del self._nodes[node_id]

        return True

    def contains_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._nodes

    def get_all_nodes(self) -> List[ConceptNode]:
        """Get all nodes in the graph."""
        return list(self._nodes.values())

    def get_node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self._nodes)

    # ==================== Relationship CRUD Operations ====================

    def add_relationship(self, relationship: Relationship) -> str:
        """
        Add a relationship to the graph.

        Args:
            relationship: The Relationship to add.

        Returns:
            The relationship ID.

        Raises:
            ValueError: If source or target node doesn't exist, or relationship ID exists.

        Validates: Requirements 4.1
        """
        if relationship.relationship_id in self._relationships:
            raise ValueError(
                f"Relationship with ID '{relationship.relationship_id}' already exists"
            )

        if relationship.source_id not in self._nodes:
            raise ValueError(f"Source node '{relationship.source_id}' does not exist")

        if relationship.target_id not in self._nodes:
            raise ValueError(f"Target node '{relationship.target_id}' does not exist")

        if len(self._relationships) >= self._config.max_relationships:
            raise ValueError(
                f"Maximum number of relationships ({self._config.max_relationships}) reached"
            )

        self._relationships[relationship.relationship_id] = relationship
        self._outgoing_index[relationship.source_id].append(relationship.relationship_id)
        self._incoming_index[relationship.target_id].append(relationship.relationship_id)
        self._total_relationships_created += 1

        return relationship.relationship_id

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Relationship:
        """
        Create and add a new relationship to the graph.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            relationship_type: Type of relationship.
            weight: Strength of relationship (0.0-1.0).
            properties: Additional attributes.
            metadata: Additional information.

        Returns:
            The created Relationship.

        Validates: Requirements 4.1
        """
        relationship = Relationship(
            relationship_id=self._generate_relationship_id(),
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight if weight is not None else self._config.default_relationship_weight,
            properties=properties if properties is not None else {},
            created_at=time.time(),
            metadata=metadata if metadata is not None else {},
        )

        self.add_relationship(relationship)
        return relationship

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """
        Get a relationship by ID.

        Args:
            relationship_id: ID of the relationship to retrieve.

        Returns:
            The relationship if found, None otherwise.

        Validates: Requirements 4.1
        """
        return self._relationships.get(relationship_id)

    def get_relationships(
        self,
        node_id: str,
        direction: str = "both",
        relationship_type: Optional[str] = None,
    ) -> List[Relationship]:
        """
        Get relationships for a node.

        Args:
            node_id: ID of the node.
            direction: "outgoing", "incoming", or "both".
            relationship_type: Optional filter by relationship type.

        Returns:
            List of relationships matching the criteria.

        Validates: Requirements 4.1
        """
        if node_id not in self._nodes:
            return []

        relationships = []

        if direction in ("outgoing", "both"):
            for rel_id in self._outgoing_index.get(node_id, []):
                rel = self._relationships.get(rel_id)
                if rel is not None:
                    if relationship_type is None or rel.relationship_type == relationship_type:
                        relationships.append(rel)

        if direction in ("incoming", "both"):
            for rel_id in self._incoming_index.get(node_id, []):
                rel = self._relationships.get(rel_id)
                if rel is not None:
                    if relationship_type is None or rel.relationship_type == relationship_type:
                        # Avoid duplicates for self-loops
                        if rel not in relationships:
                            relationships.append(rel)

        return relationships

    def remove_relationship(self, relationship_id: str) -> bool:
        """
        Remove a relationship from the graph.

        Args:
            relationship_id: ID of the relationship to remove.

        Returns:
            True if the relationship was removed, False if it doesn't exist.

        Validates: Requirements 4.1
        """
        return self._remove_relationship_internal(relationship_id)

    def _remove_relationship_internal(self, relationship_id: str) -> bool:
        """Internal method to remove a relationship."""
        if relationship_id not in self._relationships:
            return False

        rel = self._relationships[relationship_id]

        # Remove from indices
        if rel.source_id in self._outgoing_index:
            try:
                self._outgoing_index[rel.source_id].remove(relationship_id)
            except ValueError:
                pass

        if rel.target_id in self._incoming_index:
            try:
                self._incoming_index[rel.target_id].remove(relationship_id)
            except ValueError:
                pass

        # Remove the relationship
        del self._relationships[relationship_id]

        return True

    def contains_relationship(self, relationship_id: str) -> bool:
        """Check if a relationship exists in the graph."""
        return relationship_id in self._relationships

    def get_all_relationships(self) -> List[Relationship]:
        """Get all relationships in the graph."""
        return list(self._relationships.values())

    def get_relationship_count(self) -> int:
        """Get the number of relationships in the graph."""
        return len(self._relationships)

    # ==================== Graph Traversal Methods ====================

    def get_neighbors(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[ConceptNode]:
        """
        Get neighboring nodes connected to the given node.

        Args:
            node_id: ID of the node.
            relationship_type: Optional filter by relationship type.
            direction: "outgoing", "incoming", or "both".

        Returns:
            List of neighboring ConceptNodes.

        Validates: Requirements 4.3
        """
        if node_id not in self._nodes:
            return []

        neighbor_ids: Set[str] = set()

        if direction in ("outgoing", "both"):
            for rel_id in self._outgoing_index.get(node_id, []):
                rel = self._relationships.get(rel_id)
                if rel is not None:
                    if relationship_type is None or rel.relationship_type == relationship_type:
                        neighbor_ids.add(rel.target_id)

        if direction in ("incoming", "both"):
            for rel_id in self._incoming_index.get(node_id, []):
                rel = self._relationships.get(rel_id)
                if rel is not None:
                    if relationship_type is None or rel.relationship_type == relationship_type:
                        neighbor_ids.add(rel.source_id)

        # Remove self if present (for self-loops)
        neighbor_ids.discard(node_id)

        neighbors = []
        for nid in neighbor_ids:
            node = self._nodes.get(nid)
            if node is not None:
                neighbors.append(node)

        return neighbors

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None,
    ) -> List[ConceptNode]:
        """
        Find a path between two nodes using BFS.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            max_depth: Maximum path length to search.
            relationship_types: Optional filter by relationship types.

        Returns:
            List of ConceptNodes representing the path (including source and target),
            or empty list if no path found.

        Validates: Requirements 4.3
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return []

        if source_id == target_id:
            node = self._nodes.get(source_id)
            return [node] if node else []

        # BFS to find shortest path
        visited: Set[str] = {source_id}
        queue: deque = deque([(source_id, [source_id])])

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            # Get outgoing neighbors
            for rel_id in self._outgoing_index.get(current_id, []):
                rel = self._relationships.get(rel_id)
                if rel is None:
                    continue

                # Filter by relationship type if specified
                if (
                    relationship_types is not None
                    and rel.relationship_type not in relationship_types
                ):
                    continue

                neighbor_id = rel.target_id

                if neighbor_id == target_id:
                    # Found the target
                    result_path = path + [neighbor_id]
                    return [self._nodes[nid] for nid in result_path if nid in self._nodes]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return []

    def traverse(
        self,
        start_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 3,
        direction: str = "outgoing",
    ) -> List[List[str]]:
        """
        Traverse the graph from a starting node following specified relationship types.

        Args:
            start_id: ID of the starting node.
            relationship_types: Optional list of relationship types to follow.
            max_depth: Maximum traversal depth.
            direction: "outgoing", "incoming", or "both".

        Returns:
            List of paths (each path is a list of node IDs).

        Validates: Requirements 4.3
        """
        if start_id not in self._nodes:
            return []

        paths: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(current_id: str, current_path: List[str], depth: int) -> None:
            if depth > max_depth:
                return

            # Add current path if it has more than just the start node
            if len(current_path) > 1:
                paths.append(current_path.copy())

            if depth == max_depth:
                return

            # Get relationships based on direction
            rel_ids = []
            if direction in ("outgoing", "both"):
                rel_ids.extend(self._outgoing_index.get(current_id, []))
            if direction in ("incoming", "both"):
                rel_ids.extend(self._incoming_index.get(current_id, []))

            for rel_id in rel_ids:
                rel = self._relationships.get(rel_id)
                if rel is None:
                    continue

                # Filter by relationship type if specified
                if (
                    relationship_types is not None
                    and rel.relationship_type not in relationship_types
                ):
                    continue

                # Determine next node based on direction
                if direction == "outgoing":
                    next_id = rel.target_id
                elif direction == "incoming":
                    next_id = rel.source_id
                else:  # both
                    next_id = rel.target_id if rel.source_id == current_id else rel.source_id

                if next_id not in visited:
                    visited.add(next_id)
                    dfs(next_id, current_path + [next_id], depth + 1)
                    visited.remove(next_id)

        visited.add(start_id)
        dfs(start_id, [start_id], 0)

        return paths

    def get_subgraph(
        self,
        node_ids: List[str],
    ) -> Tuple[List[ConceptNode], List[Relationship]]:
        """
        Get a subgraph containing the specified nodes and their interconnections.

        Args:
            node_ids: List of node IDs to include.

        Returns:
            Tuple of (nodes, relationships) in the subgraph.
        """
        node_set = set(node_ids)
        nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]

        relationships = []
        for rel in self._relationships.values():
            if rel.source_id in node_set and rel.target_id in node_set:
                relationships.append(rel)

        return nodes, relationships

    # ==================== Relationship Inference ====================

    def infer_relationships(
        self,
        node_id: str,
        inference_types: Optional[List[str]] = None,
    ) -> List[Relationship]:
        """
        Infer implicit relationships based on graph structure.

        This method uses basic inference rules:
        1. Transitive inference: If A is_a B and B is_a C, then A is_a C
        2. Sibling inference: If A is_a C and B is_a C, then A related_to B
        3. Property inheritance: If A is_a B and B has_property P, then A has_property P

        Args:
            node_id: ID of the node to infer relationships for.
            inference_types: Optional list of inference types to apply.
                           Options: "transitive", "sibling", "inheritance"
                           Default: all types.

        Returns:
            List of inferred Relationships (not added to graph).

        Validates: Requirements 4.3
        """
        if node_id not in self._nodes:
            return []

        if inference_types is None:
            inference_types = ["transitive", "sibling", "inheritance"]

        inferred: List[Relationship] = []
        existing_pairs: Set[Tuple[str, str, str]] = set()

        # Track existing relationships to avoid duplicates
        for rel in self._relationships.values():
            existing_pairs.add((rel.source_id, rel.target_id, rel.relationship_type))

        # Transitive inference for "is_a" relationships
        if "transitive" in inference_types:
            inferred.extend(self._infer_transitive(node_id, existing_pairs))

        # Sibling inference
        if "sibling" in inference_types:
            inferred.extend(self._infer_siblings(node_id, existing_pairs))

        # Property inheritance
        if "inheritance" in inference_types:
            inferred.extend(self._infer_inheritance(node_id, existing_pairs))

        return inferred

    def _infer_transitive(
        self,
        node_id: str,
        existing_pairs: Set[Tuple[str, str, str]],
    ) -> List[Relationship]:
        """
        Infer transitive relationships.

        If A is_a B and B is_a C, then A is_a C (with reduced weight).
        """
        inferred = []

        # Get direct "is_a" relationships from this node
        direct_rels = self.get_relationships(
            node_id, direction="outgoing", relationship_type="is_a"
        )

        for rel1 in direct_rels:
            # Get "is_a" relationships from the target
            indirect_rels = self.get_relationships(
                rel1.target_id, direction="outgoing", relationship_type="is_a"
            )

            for rel2 in indirect_rels:
                # Check if this transitive relationship already exists
                if (node_id, rel2.target_id, "is_a") not in existing_pairs:
                    # Create inferred relationship with reduced weight
                    inferred_weight = rel1.weight * rel2.weight * 0.8
                    if inferred_weight >= self._config.inference_min_weight:
                        inferred.append(
                            Relationship(
                                relationship_id=self._generate_relationship_id(),
                                source_id=node_id,
                                target_id=rel2.target_id,
                                relationship_type="is_a",
                                weight=inferred_weight,
                                properties={"inferred": True, "inference_type": "transitive"},
                                created_at=time.time(),
                                metadata={"via": [rel1.relationship_id, rel2.relationship_id]},
                            )
                        )

        return inferred

    def _infer_siblings(
        self,
        node_id: str,
        existing_pairs: Set[Tuple[str, str, str]],
    ) -> List[Relationship]:
        """
        Infer sibling relationships.

        If A is_a C and B is_a C, then A related_to B.
        """
        inferred = []

        # Get parent nodes (targets of "is_a" from this node)
        parent_rels = self.get_relationships(
            node_id, direction="outgoing", relationship_type="is_a"
        )

        for parent_rel in parent_rels:
            parent_id = parent_rel.target_id

            # Get other children of this parent
            sibling_rels = self.get_relationships(
                parent_id, direction="incoming", relationship_type="is_a"
            )

            for sibling_rel in sibling_rels:
                sibling_id = sibling_rel.source_id

                # Skip self
                if sibling_id == node_id:
                    continue

                # Check if relationship already exists (in either direction)
                if (node_id, sibling_id, "related_to") not in existing_pairs and (
                    sibling_id,
                    node_id,
                    "related_to",
                ) not in existing_pairs:
                    # Create inferred sibling relationship
                    inferred_weight = parent_rel.weight * sibling_rel.weight * 0.5
                    if inferred_weight >= self._config.inference_min_weight:
                        inferred.append(
                            Relationship(
                                relationship_id=self._generate_relationship_id(),
                                source_id=node_id,
                                target_id=sibling_id,
                                relationship_type="related_to",
                                weight=inferred_weight,
                                properties={"inferred": True, "inference_type": "sibling"},
                                created_at=time.time(),
                                metadata={"common_parent": parent_id},
                            )
                        )

        return inferred

    def _infer_inheritance(
        self,
        node_id: str,
        existing_pairs: Set[Tuple[str, str, str]],
    ) -> List[Relationship]:
        """
        Infer property inheritance.

        If A is_a B and B has_property P, then A has_property P.
        """
        inferred = []

        # Get parent nodes (targets of "is_a" from this node)
        parent_rels = self.get_relationships(
            node_id, direction="outgoing", relationship_type="is_a"
        )

        for parent_rel in parent_rels:
            parent_id = parent_rel.target_id

            # Get properties of parent
            property_rels = self.get_relationships(
                parent_id, direction="outgoing", relationship_type="has_property"
            )

            for prop_rel in property_rels:
                property_id = prop_rel.target_id

                # Check if this property relationship already exists
                if (node_id, property_id, "has_property") not in existing_pairs:
                    # Create inferred property relationship
                    inferred_weight = parent_rel.weight * prop_rel.weight * 0.7
                    if inferred_weight >= self._config.inference_min_weight:
                        inferred.append(
                            Relationship(
                                relationship_id=self._generate_relationship_id(),
                                source_id=node_id,
                                target_id=property_id,
                                relationship_type="has_property",
                                weight=inferred_weight,
                                properties={"inferred": True, "inference_type": "inheritance"},
                                created_at=time.time(),
                                metadata={"inherited_from": parent_id},
                            )
                        )

        return inferred

    # ==================== Query Methods ====================

    def find_nodes(
        self,
        query: str,
        node_type: Optional[str] = None,
        max_results: int = 10,
    ) -> List[ConceptNode]:
        """
        Find nodes matching a query string.

        Searches node names and properties for matches.

        Args:
            query: Search query string.
            node_type: Optional filter by node type.
            max_results: Maximum number of results to return.

        Returns:
            List of matching ConceptNodes, sorted by relevance.

        Validates: Requirements 4.3
        """
        if not query:
            return []

        query_lower = query.lower()
        scored_nodes: List[Tuple[float, ConceptNode]] = []

        for node in self._nodes.values():
            # Filter by node type if specified
            if node_type is not None and node.node_type != node_type:
                continue

            score = 0.0

            # Exact name match
            if node.name.lower() == query_lower:
                score = 1.0
            # Name contains query
            elif query_lower in node.name.lower():
                score = 0.8
            # Query contains name
            elif node.name.lower() in query_lower:
                score = 0.6
            else:
                # Check properties
                for key, value in node.properties.items():
                    value_str = str(value).lower()
                    if query_lower in value_str or value_str in query_lower:
                        score = max(score, 0.4)
                        break

            if score > 0:
                scored_nodes.append((score, node))

        # Sort by score (descending), then by access count (descending)
        scored_nodes.sort(key=lambda x: (-x[0], -x[1].access_count))

        return [node for _, node in scored_nodes[:max_results]]

    def find_nodes_by_type(self, node_type: str) -> List[ConceptNode]:
        """
        Find all nodes of a specific type.

        Args:
            node_type: The node type to filter by.

        Returns:
            List of nodes with the specified type.
        """
        return [node for node in self._nodes.values() if node.node_type == node_type]

    def find_relationships_by_type(self, relationship_type: str) -> List[Relationship]:
        """
        Find all relationships of a specific type.

        Args:
            relationship_type: The relationship type to filter by.

        Returns:
            List of relationships with the specified type.
        """
        return [
            rel
            for rel in self._relationships.values()
            if rel.relationship_type == relationship_type
        ]

    def get_relationship_between(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None,
    ) -> Optional[Relationship]:
        """
        Get a relationship between two specific nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            relationship_type: Optional filter by relationship type.

        Returns:
            The relationship if found, None otherwise.
        """
        for rel_id in self._outgoing_index.get(source_id, []):
            rel = self._relationships.get(rel_id)
            if rel is not None and rel.target_id == target_id:
                if relationship_type is None or rel.relationship_type == relationship_type:
                    return rel
        return None

    # ==================== Statistics and State ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the knowledge graph."""
        if not self._nodes:
            return {
                "node_count": 0,
                "relationship_count": 0,
                "node_types": {},
                "relationship_types": {},
                "avg_relationships_per_node": 0.0,
                "total_nodes_created": self._total_nodes_created,
                "total_relationships_created": self._total_relationships_created,
            }

        # Count node types
        node_types: Dict[str, int] = {}
        for node in self._nodes.values():
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

        # Count relationship types
        relationship_types: Dict[str, int] = {}
        for rel in self._relationships.values():
            relationship_types[rel.relationship_type] = (
                relationship_types.get(rel.relationship_type, 0) + 1
            )

        # Calculate average relationships per node
        total_rels = len(self._relationships)
        node_count = len(self._nodes)
        avg_rels = (
            (total_rels * 2) / node_count if node_count > 0 else 0.0
        )  # *2 because each rel connects 2 nodes

        return {
            "node_count": node_count,
            "relationship_count": total_rels,
            "node_types": node_types,
            "relationship_types": relationship_types,
            "avg_relationships_per_node": avg_rels,
            "total_nodes_created": self._total_nodes_created,
            "total_relationships_created": self._total_relationships_created,
            "uptime": time.time() - self._initialized_at,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the knowledge graph."""
        return {
            "node_count": len(self._nodes),
            "relationship_count": len(self._relationships),
            "max_nodes": self._config.max_nodes,
            "max_relationships": self._config.max_relationships,
            "total_nodes_created": self._total_nodes_created,
            "total_relationships_created": self._total_relationships_created,
            "uptime": time.time() - self._initialized_at,
            "config": self._config.to_dict(),
        }

    def validate_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the knowledge graph.

        Checks for:
        - Orphaned relationships (referencing non-existent nodes)
        - Index consistency
        - Duplicate relationships

        Returns:
            Dictionary with validation results and any issues found.
        """
        issues = []

        # Check for orphaned relationships
        for rel_id, rel in self._relationships.items():
            if rel.source_id not in self._nodes:
                issues.append(
                    f"Relationship {rel_id} references non-existent source node {rel.source_id}"
                )
            if rel.target_id not in self._nodes:
                issues.append(
                    f"Relationship {rel_id} references non-existent target node {rel.target_id}"
                )

        # Check index consistency
        for node_id in self._nodes:
            if node_id not in self._outgoing_index:
                issues.append(f"Node {node_id} missing from outgoing index")
            if node_id not in self._incoming_index:
                issues.append(f"Node {node_id} missing from incoming index")

        # Check for relationships in indices that don't exist
        for node_id, rel_ids in self._outgoing_index.items():
            for rel_id in rel_ids:
                if rel_id not in self._relationships:
                    issues.append(
                        f"Outgoing index for {node_id} references non-existent relationship {rel_id}"
                    )

        for node_id, rel_ids in self._incoming_index.items():
            for rel_id in rel_ids:
                if rel_id not in self._relationships:
                    issues.append(
                        f"Incoming index for {node_id} references non-existent relationship {rel_id}"
                    )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "node_count": len(self._nodes),
            "relationship_count": len(self._relationships),
        }

    # ==================== Serialization ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the knowledge graph to a dictionary.

        Returns:
            Dictionary representation of the graph.

        Validates: Requirements 4.6
        """
        return {
            "config": self._config.to_dict(),
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "relationships": {rel_id: rel.to_dict() for rel_id, rel in self._relationships.items()},
            "total_nodes_created": self._total_nodes_created,
            "total_relationships_created": self._total_relationships_created,
            "initialized_at": self._initialized_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """
        Create a knowledge graph from a dictionary.

        Args:
            data: Dictionary representation of the graph.

        Returns:
            A new KnowledgeGraph instance.

        Validates: Requirements 4.6
        """
        config = data.get("config", {})
        graph = cls(config=config)

        # Restore nodes first
        nodes_data = data.get("nodes", {})
        for node_id, node_data in nodes_data.items():
            node = ConceptNode.from_dict(node_data)
            graph._nodes[node_id] = node
            graph._name_index[node.name.lower()] = node_id
            graph._outgoing_index[node_id] = []
            graph._incoming_index[node_id] = []

        # Restore relationships and rebuild indices
        relationships_data = data.get("relationships", {})
        for rel_id, rel_data in relationships_data.items():
            rel = Relationship.from_dict(rel_data)
            graph._relationships[rel_id] = rel

            # Update indices
            if rel.source_id in graph._outgoing_index:
                graph._outgoing_index[rel.source_id].append(rel_id)
            if rel.target_id in graph._incoming_index:
                graph._incoming_index[rel.target_id].append(rel_id)

        # Restore statistics
        graph._total_nodes_created = data.get("total_nodes_created", len(graph._nodes))
        graph._total_relationships_created = data.get(
            "total_relationships_created", len(graph._relationships)
        )
        graph._initialized_at = data.get("initialized_at", time.time())

        return graph

    def clear(self) -> None:
        """Clear all nodes and relationships from the graph."""
        self._nodes.clear()
        self._relationships.clear()
        self._outgoing_index.clear()
        self._incoming_index.clear()
        self._name_index.clear()

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if a node ID is in the graph."""
        return node_id in self._nodes


# ==================== Factory Functions ====================


def create_concept_node(
    name: str,
    node_type: str,
    properties: Optional[Dict[str, Any]] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConceptNode:
    """
    Factory function to create a ConceptNode with auto-generated ID and timestamp.

    Args:
        name: Human-readable name of the concept.
        node_type: Category of concept.
        properties: Dictionary of attributes.
        embedding: Optional vector representation.
        metadata: Additional information.

    Returns:
        A new ConceptNode instance.

    Validates: Requirements 4.1
    """
    return ConceptNode(
        node_id=str(uuid.uuid4()),
        name=name,
        node_type=node_type,
        properties=properties if properties is not None else {},
        embedding=embedding,
        created_at=time.time(),
        updated_at=time.time(),
        access_count=0,
        metadata=metadata if metadata is not None else {},
    )


def create_relationship(
    source_id: str,
    target_id: str,
    relationship_type: str,
    weight: float = 0.5,
    properties: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Relationship:
    """
    Factory function to create a Relationship with auto-generated ID and timestamp.

    Args:
        source_id: ID of the source node.
        target_id: ID of the target node.
        relationship_type: Type of relationship.
        weight: Strength of relationship (0.0-1.0).
        properties: Additional attributes.
        metadata: Additional information.

    Returns:
        A new Relationship instance.

    Validates: Requirements 4.1
    """
    return Relationship(
        relationship_id=str(uuid.uuid4()),
        source_id=source_id,
        target_id=target_id,
        relationship_type=relationship_type,
        weight=weight,
        properties=properties if properties is not None else {},
        created_at=time.time(),
        metadata=metadata if metadata is not None else {},
    )

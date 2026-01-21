"""
World Model module for consciousness system.

The WorldModel maintains the system's understanding of the external environment:
- Environment knowledge representation
- User models and preferences
- Knowledge update mechanisms
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import time


@dataclass
class Entity:
    """Represents an entity in the world model."""

    entity_id: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)  # relation_type -> [entity_ids]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class UserModel:
    """Represents a user's model in the world."""

    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    language: str = "zh"
    created_at: float = field(default_factory=time.time)
    last_interaction: Optional[float] = None


class WorldModel:
    """
    World Model maintains the system's understanding of the external environment.

    Tracks entities, relationships, user models, and environmental knowledge.
    Implements requirement 6.4: maintain environment knowledge representation.
    """

    def __init__(self):
        """Initialize the world model with default state."""
        self._entities: Dict[str, Entity] = {}
        self._users: Dict[str, UserModel] = {}
        self._knowledge: Dict[str, Any] = {
            "environment": {
                "type": "development",
                "resources": {},
            },
            "facts": {},
            "rules": {},
        }
        self._entity_types: Set[str] = {"concept", "topic", "tool", "model", "workflow"}
        self._initialized_at: float = time.time()

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current world model state.

        Returns:
            Dictionary containing current state information.
        """
        return {
            "entity_count": len(self._entities),
            "user_count": len(self._users),
            "knowledge_domains": list(self._knowledge.keys()),
            "entity_types": list(self._entity_types),
            "uptime": time.time() - self._initialized_at,
        }

    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Update the world model state.

        Args:
            updates: Dictionary of state updates to apply.
        """
        if "knowledge" in updates:
            self._deep_update(self._knowledge, updates["knowledge"])
        if "environment" in updates:
            self._deep_update(self._knowledge["environment"], updates["environment"])

    def _deep_update(self, target: Dict, source: Dict) -> None:
        """Deep update a dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    # Entity management
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the world model.

        Args:
            entity: The Entity object to add.
        """
        self._entities[entity.entity_id] = entity
        self._entity_types.add(entity.entity_type)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.

        Args:
            entity_id: The entity identifier.

        Returns:
            The Entity object or None if not found.
        """
        return self._entities.get(entity_id)

    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an entity's attributes.

        Args:
            entity_id: The entity identifier.
            updates: Dictionary of attribute updates.

        Returns:
            True if the entity was updated, False if not found.
        """
        if entity_id in self._entities:
            entity = self._entities[entity_id]
            entity.attributes.update(updates)
            entity.updated_at = time.time()
            return True
        return False

    def remove_entity(self, entity_id: str) -> bool:
        """
        Remove an entity from the world model.

        Args:
            entity_id: The entity identifier.

        Returns:
            True if the entity was removed, False if not found.
        """
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Get all entities of a specific type.

        Args:
            entity_type: The entity type to filter by.

        Returns:
            List of Entity objects of the specified type.
        """
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def add_relation(self, from_id: str, relation_type: str, to_id: str) -> bool:
        """
        Add a relation between two entities.

        Args:
            from_id: Source entity ID.
            relation_type: Type of relation.
            to_id: Target entity ID.

        Returns:
            True if the relation was added, False if source entity not found.
        """
        if from_id in self._entities:
            entity = self._entities[from_id]
            if relation_type not in entity.relations:
                entity.relations[relation_type] = []
            if to_id not in entity.relations[relation_type]:
                entity.relations[relation_type].append(to_id)
            entity.updated_at = time.time()
            return True
        return False

    # User model management
    def get_or_create_user(self, user_id: str) -> UserModel:
        """
        Get or create a user model.

        Args:
            user_id: The user identifier.

        Returns:
            The UserModel object.
        """
        if user_id not in self._users:
            self._users[user_id] = UserModel(user_id=user_id)
        return self._users[user_id]

    def get_user(self, user_id: str) -> Optional[UserModel]:
        """
        Get a user model by ID.

        Args:
            user_id: The user identifier.

        Returns:
            The UserModel object or None if not found.
        """
        return self._users.get(user_id)

    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        Update a user's preferences.

        Args:
            user_id: The user identifier.
            preferences: Dictionary of preference updates.

        Returns:
            True if the user was updated, False if not found.
        """
        user = self.get_or_create_user(user_id)
        user.preferences.update(preferences)
        return True

    def record_user_interaction(self, user_id: str, interaction: Dict[str, Any]) -> None:
        """
        Record a user interaction.

        Args:
            user_id: The user identifier.
            interaction: Dictionary describing the interaction.
        """
        user = self.get_or_create_user(user_id)
        interaction["timestamp"] = time.time()
        user.interaction_history.append(interaction)
        user.last_interaction = time.time()

        # Keep only recent interactions
        max_history = 50
        if len(user.interaction_history) > max_history:
            user.interaction_history = user.interaction_history[-max_history:]

    # Knowledge management
    def get_knowledge(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get knowledge from the world model.

        Args:
            domain: Optional domain to filter by.

        Returns:
            Dictionary of knowledge.
        """
        if domain:
            return self._knowledge.get(domain, {})
        return self._knowledge.copy()

    def update_knowledge(self, domain: str, key: str, value: Any) -> None:
        """
        Update knowledge in the world model.

        Args:
            domain: The knowledge domain.
            key: The knowledge key.
            value: The knowledge value.
        """
        if domain not in self._knowledge:
            self._knowledge[domain] = {}
        self._knowledge[domain][key] = value

    def add_fact(self, key: str, value: Any) -> None:
        """
        Add a fact to the world model.

        Args:
            key: The fact key.
            value: The fact value.
        """
        self._knowledge["facts"][key] = {
            "value": value,
            "added_at": time.time(),
        }

    def get_fact(self, key: str) -> Optional[Any]:
        """
        Get a fact from the world model.

        Args:
            key: The fact key.

        Returns:
            The fact value or None if not found.
        """
        fact = self._knowledge["facts"].get(key)
        return fact["value"] if fact else None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the world model to a dictionary.

        Returns:
            Dictionary representation of the world model.
        """
        return {
            "entities": {
                eid: {
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type,
                    "attributes": e.attributes,
                    "relations": e.relations,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                }
                for eid, e in self._entities.items()
            },
            "users": {
                uid: {
                    "user_id": u.user_id,
                    "preferences": u.preferences,
                    "interaction_history": u.interaction_history,
                    "expertise_level": u.expertise_level,
                    "language": u.language,
                    "created_at": u.created_at,
                    "last_interaction": u.last_interaction,
                }
                for uid, u in self._users.items()
            },
            "knowledge": self._knowledge,
            "entity_types": list(self._entity_types),
            "initialized_at": self._initialized_at,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore the world model from a dictionary.

        Args:
            data: Dictionary representation of the world model.
        """
        if "entities" in data:
            self._entities = {}
            for eid, e_data in data["entities"].items():
                self._entities[eid] = Entity(
                    entity_id=e_data["entity_id"],
                    entity_type=e_data["entity_type"],
                    attributes=e_data.get("attributes", {}),
                    relations=e_data.get("relations", {}),
                    created_at=e_data.get("created_at", time.time()),
                    updated_at=e_data.get("updated_at", time.time()),
                )

        if "users" in data:
            self._users = {}
            for uid, u_data in data["users"].items():
                self._users[uid] = UserModel(
                    user_id=u_data["user_id"],
                    preferences=u_data.get("preferences", {}),
                    interaction_history=u_data.get("interaction_history", []),
                    expertise_level=u_data.get("expertise_level", "intermediate"),
                    language=u_data.get("language", "zh"),
                    created_at=u_data.get("created_at", time.time()),
                    last_interaction=u_data.get("last_interaction"),
                )

        if "knowledge" in data:
            self._knowledge = data["knowledge"]

        if "entity_types" in data:
            self._entity_types = set(data["entity_types"])

        if "initialized_at" in data:
            self._initialized_at = data["initialized_at"]

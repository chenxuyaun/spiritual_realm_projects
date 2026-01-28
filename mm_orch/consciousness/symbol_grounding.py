"""
Symbol Grounding module for consciousness system.

This module implements the SymbolGroundingModule which connects abstract symbols
(words, identifiers) to perceptual experiences and meanings. The symbol grounding
system supports:
- Mapping between symbolic identifiers and their perceptual/experiential representations
- Grounding new symbols by associating them with episodic memories and semantic concepts
- Tracking grounding confidence based on consistency and frequency of associations
- Handling ambiguous groundings with multiple candidate groundings and probabilities
- Updating symbol groundings based on new experiences

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from mm_orch.consciousness.episodic_memory import EpisodicMemory
    from mm_orch.consciousness.semantic_memory import SemanticMemory


@dataclass
class GroundingCandidate:
    """
    A candidate grounding for a symbol.

    Represents one possible meaning/grounding for a symbol, linked to
    semantic concepts and episodic memories with associated probability.

    Attributes:
        concept_id: Link to semantic memory concept (may be None for ungrounded).
        episode_ids: List of supporting episode IDs from episodic memory.
        perceptual_features: Dictionary of perceptual features associated with this grounding.
        probability: Probability of this being the correct grounding (0.0 to 1.0).

    Validates: Requirements 5.1, 5.4
    """

    concept_id: Optional[str]
    episode_ids: List[str]
    perceptual_features: Dict[str, Any]
    probability: float

    def __post_init__(self) -> None:
        """Validate grounding candidate fields after initialization."""
        if not isinstance(self.episode_ids, list):
            raise ValueError("episode_ids must be a list")

        if not isinstance(self.perceptual_features, dict):
            raise ValueError("perceptual_features must be a dictionary")

        if not isinstance(self.probability, (int, float)):
            raise ValueError("probability must be a number")

        # Clamp probability to valid range [0.0, 1.0]
        self.probability = max(0.0, min(1.0, float(self.probability)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert grounding candidate to dictionary representation."""
        return {
            "concept_id": self.concept_id,
            "episode_ids": self.episode_ids.copy(),
            "perceptual_features": self.perceptual_features.copy(),
            "probability": self.probability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundingCandidate":
        """Create a grounding candidate from dictionary representation."""
        return cls(
            concept_id=data.get("concept_id"),
            episode_ids=data.get("episode_ids", []),
            perceptual_features=data.get("perceptual_features", {}),
            probability=data.get("probability", 0.0),
        )


@dataclass
class SymbolGrounding:
    """
    Represents a symbol's grounding to experiences.

    Contains all candidate groundings for a symbol along with confidence
    tracking and metadata for grounding management.

    Attributes:
        symbol: The symbolic identifier being grounded.
        groundings: List of candidate groundings with probabilities.
        confidence: Overall confidence in the symbol's grounding (0.0 to 1.0).
        grounding_count: Number of times this symbol has been grounded/updated.
        last_updated: Timestamp of the last grounding update.

    Validates: Requirements 5.1, 5.3, 5.4
    """

    symbol: str
    groundings: List[GroundingCandidate]
    confidence: float
    grounding_count: int
    last_updated: float

    def __post_init__(self) -> None:
        """Validate symbol grounding fields after initialization."""
        if not self.symbol:
            raise ValueError("symbol cannot be empty")

        if not isinstance(self.groundings, list):
            raise ValueError("groundings must be a list")

        if not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number")

        # Clamp confidence to valid range [0.0, 1.0]
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        if not isinstance(self.grounding_count, int):
            raise ValueError("grounding_count must be an integer")

        if self.grounding_count < 0:
            raise ValueError("grounding_count cannot be negative")

        if not isinstance(self.last_updated, (int, float)):
            raise ValueError("last_updated must be a number")

    def to_dict(self) -> Dict[str, Any]:
        """Convert symbol grounding to dictionary representation."""
        return {
            "symbol": self.symbol,
            "groundings": [g.to_dict() for g in self.groundings],
            "confidence": self.confidence,
            "grounding_count": self.grounding_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolGrounding":
        """Create a symbol grounding from dictionary representation."""
        return cls(
            symbol=data["symbol"],
            groundings=[GroundingCandidate.from_dict(g) for g in data.get("groundings", [])],
            confidence=data.get("confidence", 0.0),
            grounding_count=data.get("grounding_count", 0),
            last_updated=data.get("last_updated", time.time()),
        )


@dataclass
class SymbolGroundingConfig:
    """Configuration for the symbol grounding module."""

    min_confidence_threshold: float = 0.3  # Minimum confidence to consider grounded
    confidence_growth_rate: float = 0.1  # Confidence increase per consistent association
    confidence_decay_rate: float = 0.05  # Confidence decrease per inconsistent association
    max_candidates: int = 10  # Maximum number of grounding candidates per symbol
    context_similarity_threshold: float = 0.5  # Threshold for context matching
    probability_smoothing: float = 0.01  # Smoothing factor for probability normalization

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            raise ValueError("min_confidence_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence_growth_rate <= 1.0):
            raise ValueError("confidence_growth_rate must be between 0.0 and 1.0")
        if not (0.0 <= self.confidence_decay_rate <= 1.0):
            raise ValueError("confidence_decay_rate must be between 0.0 and 1.0")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if not (0.0 <= self.context_similarity_threshold <= 1.0):
            raise ValueError("context_similarity_threshold must be between 0.0 and 1.0")
        if self.probability_smoothing < 0:
            raise ValueError("probability_smoothing must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "min_confidence_threshold": self.min_confidence_threshold,
            "confidence_growth_rate": self.confidence_growth_rate,
            "confidence_decay_rate": self.confidence_decay_rate,
            "max_candidates": self.max_candidates,
            "context_similarity_threshold": self.context_similarity_threshold,
            "probability_smoothing": self.probability_smoothing,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolGroundingConfig":
        """Create config from dictionary representation."""
        return cls(
            min_confidence_threshold=data.get("min_confidence_threshold", 0.3),
            confidence_growth_rate=data.get("confidence_growth_rate", 0.1),
            confidence_decay_rate=data.get("confidence_decay_rate", 0.05),
            max_candidates=data.get("max_candidates", 10),
            context_similarity_threshold=data.get("context_similarity_threshold", 0.5),
            probability_smoothing=data.get("probability_smoothing", 0.01),
        )


class SymbolGroundingModule:
    """
    Manages symbol grounding to experiences and concepts.

    This class connects abstract symbols (words, identifiers) to perceptual
    experiences and meanings by integrating with the episodic and semantic
    memory systems. It supports:
    - Getting or creating groundings for symbols
    - Updating groundings based on new experiences
    - Resolving ambiguous groundings using context
    - Tracking grounding confidence
    - Identifying ungrounded symbols

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """

    def __init__(
        self,
        episodic_memory: "EpisodicMemory",
        semantic_memory: "SemanticMemory",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize with references to memory systems.

        Args:
            episodic_memory: Reference to the episodic memory system.
            semantic_memory: Reference to the semantic memory system.
            config: Optional configuration dictionary.
        """
        # Import here to avoid circular imports
        from mm_orch.consciousness.episodic_memory import EpisodicMemory
        from mm_orch.consciousness.semantic_memory import SemanticMemory

        if not isinstance(episodic_memory, EpisodicMemory):
            raise TypeError("episodic_memory must be an EpisodicMemory instance")
        if not isinstance(semantic_memory, SemanticMemory):
            raise TypeError("semantic_memory must be a SemanticMemory instance")

        self._episodic_memory = episodic_memory
        self._semantic_memory = semantic_memory

        if config is not None:
            self._config = SymbolGroundingConfig.from_dict(config)
        else:
            self._config = SymbolGroundingConfig()

        # Main storage: symbol -> SymbolGrounding
        self._groundings: Dict[str, SymbolGrounding] = {}

        # Statistics
        self._total_groundings_created: int = 0
        self._total_groundings_updated: int = 0
        self._total_ambiguity_resolutions: int = 0
        self._initialized_at: float = time.time()

    # ==================== Core Grounding Operations ====================

    def ground_symbol(self, symbol: str) -> SymbolGrounding:
        """
        Get or create grounding for a symbol.

        If the symbol already has a grounding, returns it. Otherwise,
        attempts to create a new grounding by querying both episodic
        and semantic memory for relevant associations.

        Args:
            symbol: The symbol to ground.

        Returns:
            SymbolGrounding with at least one grounding candidate.

        Validates: Requirements 5.1, 5.2
        Property 21: Symbol Grounding Existence
        Property 22: New Symbol Grounding Attempt
        """
        if not symbol:
            raise ValueError("symbol cannot be empty")

        # Normalize symbol for consistent lookup
        normalized_symbol = symbol.lower().strip()

        # Check if grounding already exists
        if normalized_symbol in self._groundings:
            return self._groundings[normalized_symbol]

        # Create new grounding by querying memory systems
        grounding = self._create_new_grounding(normalized_symbol)
        self._groundings[normalized_symbol] = grounding
        self._total_groundings_created += 1

        return grounding

    def _create_new_grounding(self, symbol: str) -> SymbolGrounding:
        """
        Create a new grounding for a symbol by querying memory systems.

        Queries both episodic and semantic memory to find relevant
        associations for the symbol.

        Args:
            symbol: The symbol to ground.

        Returns:
            New SymbolGrounding with candidates from memory systems.

        Validates: Requirements 5.2
        Property 22: New Symbol Grounding Attempt
        """
        candidates: List[GroundingCandidate] = []

        # Query semantic memory for concept matches
        semantic_candidates = self._query_semantic_memory(symbol)
        candidates.extend(semantic_candidates)

        # Query episodic memory for experience-based groundings
        episodic_candidates = self._query_episodic_memory(symbol)
        candidates.extend(episodic_candidates)

        # If no candidates found, create a default ungrounded candidate
        if not candidates:
            candidates.append(
                GroundingCandidate(
                    concept_id=None,
                    episode_ids=[],
                    perceptual_features={"symbol": symbol, "ungrounded": True},
                    probability=1.0,
                )
            )

        # Normalize probabilities to sum to 1.0
        candidates = self._normalize_probabilities(candidates)

        # Calculate initial confidence based on candidate quality
        confidence = self._calculate_initial_confidence(candidates)

        return SymbolGrounding(
            symbol=symbol,
            groundings=candidates,
            confidence=confidence,
            grounding_count=1,
            last_updated=time.time(),
        )

    def _query_semantic_memory(self, symbol: str) -> List[GroundingCandidate]:
        """
        Query semantic memory for concept matches.

        Args:
            symbol: The symbol to search for.

        Returns:
            List of grounding candidates from semantic memory.
        """
        candidates: List[GroundingCandidate] = []

        try:
            # Query semantic memory for matching concepts
            matching_concepts = self._semantic_memory.query(symbol, max_results=5)

            for concept in matching_concepts:
                # Calculate relevance score based on name similarity
                relevance = self._calculate_name_similarity(symbol, concept.name)

                if relevance > 0:
                    candidate = GroundingCandidate(
                        concept_id=concept.node_id,
                        episode_ids=[],
                        perceptual_features={
                            "concept_name": concept.name,
                            "concept_type": concept.node_type,
                            "properties": concept.properties.copy(),
                        },
                        probability=relevance,
                    )
                    candidates.append(candidate)
        except Exception:
            # If semantic memory query fails, continue without semantic candidates
            pass

        return candidates

    def _query_episodic_memory(self, symbol: str) -> List[GroundingCandidate]:
        """
        Query episodic memory for experience-based groundings.

        Args:
            symbol: The symbol to search for.

        Returns:
            List of grounding candidates from episodic memory.
        """
        candidates: List[GroundingCandidate] = []

        try:
            # Query episodic memory for episodes containing the symbol
            query_context = {"symbol": symbol, "query_text": symbol}
            matching_episodes = self._episodic_memory.retrieve_by_context_similarity(
                query_context, max_results=5
            )

            if matching_episodes:
                # Group episodes by common features to create candidates
                episode_ids = [ep.episode_id for ep in matching_episodes]

                # Extract common perceptual features from episodes
                perceptual_features = self._extract_common_features(matching_episodes)
                perceptual_features["source"] = "episodic_memory"
                perceptual_features["episode_count"] = len(episode_ids)

                # Calculate probability based on episode relevance
                avg_importance = sum(ep.importance for ep in matching_episodes) / len(
                    matching_episodes
                )

                candidate = GroundingCandidate(
                    concept_id=None,  # No semantic concept link yet
                    episode_ids=episode_ids,
                    perceptual_features=perceptual_features,
                    probability=avg_importance,
                )
                candidates.append(candidate)
        except Exception:
            # If episodic memory query fails, continue without episodic candidates
            pass

        return candidates

    def _extract_common_features(self, episodes: List) -> Dict[str, Any]:
        """
        Extract common perceptual features from a list of episodes.

        Args:
            episodes: List of Episode objects.

        Returns:
            Dictionary of common features.
        """
        if not episodes:
            return {}

        features: Dict[str, Any] = {}

        # Collect all context keys and their values
        key_values: Dict[str, List[Any]] = {}
        for episode in episodes:
            for key, value in episode.context.items():
                if key not in key_values:
                    key_values[key] = []
                key_values[key].append(value)

        # Find common values (appearing in majority of episodes)
        threshold = len(episodes) / 2
        for key, values in key_values.items():
            # Count value occurrences
            value_counts: Dict[Any, int] = {}
            for value in values:
                # Convert unhashable types to string for counting
                hashable_value = str(value) if isinstance(value, (dict, list)) else value
                value_counts[hashable_value] = value_counts.get(hashable_value, 0) + 1

            # Find most common value if it appears in majority
            for value, count in value_counts.items():
                if count >= threshold:
                    features[key] = value
                    break

        return features

    def _calculate_name_similarity(self, symbol: str, name: str) -> float:
        """
        Calculate similarity between a symbol and a concept name.

        Uses a simple character-based similarity metric.

        Args:
            symbol: The symbol being grounded.
            name: The concept name to compare.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not symbol or not name:
            return 0.0

        symbol_lower = symbol.lower()
        name_lower = name.lower()

        # Exact match
        if symbol_lower == name_lower:
            return 1.0

        # Substring match
        if symbol_lower in name_lower or name_lower in symbol_lower:
            shorter = min(len(symbol_lower), len(name_lower))
            longer = max(len(symbol_lower), len(name_lower))
            return shorter / longer

        # Character overlap (Jaccard similarity)
        symbol_chars = set(symbol_lower)
        name_chars = set(name_lower)
        intersection = len(symbol_chars & name_chars)
        union = len(symbol_chars | name_chars)

        if union == 0:
            return 0.0

        return intersection / union

    def _normalize_probabilities(
        self, candidates: List[GroundingCandidate]
    ) -> List[GroundingCandidate]:
        """
        Normalize candidate probabilities to sum to 1.0.

        Applies smoothing to ensure all candidates have non-zero probability.

        Args:
            candidates: List of grounding candidates.

        Returns:
            List of candidates with normalized probabilities.

        Validates: Requirements 5.4
        Property 24: Ambiguous Grounding Probability Distribution
        """
        if not candidates:
            return candidates

        # Apply smoothing
        smoothing = self._config.probability_smoothing
        for candidate in candidates:
            candidate.probability = max(candidate.probability, smoothing)

        # Calculate total probability
        total = sum(c.probability for c in candidates)

        # Normalize to sum to 1.0
        if total > 0:
            for candidate in candidates:
                candidate.probability = candidate.probability / total
        else:
            # Equal distribution if all zero
            equal_prob = 1.0 / len(candidates)
            for candidate in candidates:
                candidate.probability = equal_prob

        return candidates

    def _calculate_initial_confidence(self, candidates: List[GroundingCandidate]) -> float:
        """
        Calculate initial confidence based on candidate quality.

        Args:
            candidates: List of grounding candidates.

        Returns:
            Initial confidence score between 0.0 and 1.0.
        """
        if not candidates:
            return 0.0

        # Check if we have any grounded candidates (with concept_id or episode_ids)
        grounded_candidates = [
            c for c in candidates if c.concept_id is not None or len(c.episode_ids) > 0
        ]

        if not grounded_candidates:
            # No real groundings found
            return 0.1

        # Calculate confidence based on:
        # 1. Number of grounded candidates
        # 2. Highest probability among grounded candidates
        # 3. Evidence strength (episode count, concept properties)

        num_grounded = len(grounded_candidates)
        max_prob = max(c.probability for c in grounded_candidates)

        # More candidates with high probability = higher confidence
        confidence = min(0.3 + (num_grounded * 0.1) + (max_prob * 0.3), 0.8)

        return confidence

    # ==================== Grounding Updates ====================

    def update_grounding(
        self,
        symbol: str,
        context: Dict[str, Any],
        episode_id: Optional[str] = None,
    ) -> SymbolGrounding:
        """
        Update symbol grounding based on new experience.

        Updates the grounding for a symbol based on new contextual information
        and optionally links to a new episode. Adjusts confidence based on
        consistency with existing groundings.

        Args:
            symbol: The symbol to update grounding for.
            context: New contextual information about the symbol.
            episode_id: Optional ID of a new supporting episode.

        Returns:
            Updated SymbolGrounding.

        Validates: Requirements 5.3, 5.5
        Property 23: Grounding Confidence Evolution
        """
        if not symbol:
            raise ValueError("symbol cannot be empty")

        normalized_symbol = symbol.lower().strip()

        # Get or create grounding
        if normalized_symbol not in self._groundings:
            grounding = self.ground_symbol(symbol)
        else:
            grounding = self._groundings[normalized_symbol]

        # Find the most relevant candidate based on context
        best_candidate, similarity = self._find_best_matching_candidate(
            grounding.groundings, context
        )

        # Determine if this is a consistent or inconsistent association
        is_consistent = similarity >= self._config.context_similarity_threshold

        # Update confidence based on consistency
        if is_consistent:
            # Consistent association increases confidence
            grounding.confidence = min(
                1.0, grounding.confidence + self._config.confidence_growth_rate
            )

            # Update the best matching candidate
            if best_candidate is not None:
                # Add episode if provided
                if episode_id and episode_id not in best_candidate.episode_ids:
                    best_candidate.episode_ids.append(episode_id)

                # Update perceptual features with new context
                for key, value in context.items():
                    if key not in best_candidate.perceptual_features:
                        best_candidate.perceptual_features[key] = value

                # Increase probability of this candidate
                best_candidate.probability = min(1.0, best_candidate.probability * 1.1)
        else:
            # Inconsistent association decreases confidence
            grounding.confidence = max(
                0.0, grounding.confidence - self._config.confidence_decay_rate
            )

            # Create a new candidate for this context
            new_candidate = GroundingCandidate(
                concept_id=None,
                episode_ids=[episode_id] if episode_id else [],
                perceptual_features=context.copy(),
                probability=0.1,  # Start with low probability
            )

            # Add new candidate if we haven't reached max
            if len(grounding.groundings) < self._config.max_candidates:
                grounding.groundings.append(new_candidate)

        # Re-normalize probabilities
        grounding.groundings = self._normalize_probabilities(grounding.groundings)

        # Update metadata
        grounding.grounding_count += 1
        grounding.last_updated = time.time()

        self._total_groundings_updated += 1

        return grounding

    def _find_best_matching_candidate(
        self,
        candidates: List[GroundingCandidate],
        context: Dict[str, Any],
    ) -> tuple:
        """
        Find the candidate that best matches the given context.

        Args:
            candidates: List of grounding candidates.
            context: Context to match against.

        Returns:
            Tuple of (best_candidate, similarity_score).
        """
        if not candidates:
            return None, 0.0

        best_candidate = None
        best_similarity = 0.0

        for candidate in candidates:
            similarity = self._calculate_context_similarity(context, candidate.perceptual_features)

            # Weight by probability
            weighted_similarity = similarity * (0.5 + 0.5 * candidate.probability)

            if weighted_similarity > best_similarity:
                best_similarity = weighted_similarity
                best_candidate = candidate

        return best_candidate, best_similarity

    def _calculate_context_similarity(
        self,
        context1: Dict[str, Any],
        context2: Dict[str, Any],
    ) -> float:
        """
        Calculate similarity between two contexts.

        Args:
            context1: First context dictionary.
            context2: Second context dictionary.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not context1 or not context2:
            return 0.0

        keys1 = set(context1.keys())
        keys2 = set(context2.keys())

        # Find common keys
        common_keys = keys1 & keys2

        if not common_keys:
            return 0.0

        # Calculate weighted similarity
        total_score = 0.0
        max_possible_score = len(keys1)

        for key in common_keys:
            value1 = context1[key]
            value2 = context2[key]

            # Exact match
            if value1 == value2:
                total_score += 1.0
            # Type match with partial credit
            elif type(value1) == type(value2):
                if isinstance(value1, str) and isinstance(value2, str):
                    # String similarity
                    total_score += self._calculate_name_similarity(value1, value2)
                elif isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    # Numeric similarity (inverse of relative difference)
                    max_val = max(abs(value1), abs(value2), 1)
                    diff = abs(value1 - value2) / max_val
                    total_score += max(0, 1 - diff)
                else:
                    total_score += 0.3
            else:
                total_score += 0.1

        return total_score / max_possible_score if max_possible_score > 0 else 0.0

    # ==================== Ambiguity Resolution ====================

    def resolve_ambiguity(
        self,
        symbol: str,
        context: Dict[str, Any],
    ) -> GroundingCandidate:
        """
        Resolve ambiguous grounding using context.

        When a symbol has multiple candidate groundings, uses the provided
        context to select the most appropriate grounding.

        Args:
            symbol: The symbol to resolve.
            context: Context to use for disambiguation.

        Returns:
            The most appropriate GroundingCandidate for the context.

        Validates: Requirements 5.4
        """
        if not symbol:
            raise ValueError("symbol cannot be empty")

        normalized_symbol = symbol.lower().strip()

        # Get or create grounding
        if normalized_symbol not in self._groundings:
            grounding = self.ground_symbol(symbol)
        else:
            grounding = self._groundings[normalized_symbol]

        # Increment resolution count for all calls
        self._total_ambiguity_resolutions += 1

        # If only one candidate, return it
        if len(grounding.groundings) == 1:
            return grounding.groundings[0]

        # Find best matching candidate based on context
        best_candidate, _ = self._find_best_matching_candidate(grounding.groundings, context)

        # Return best candidate or first if no match found
        return best_candidate if best_candidate else grounding.groundings[0]

    # ==================== Confidence and Status ====================

    def get_grounding_confidence(self, symbol: str) -> float:
        """
        Get confidence score for a symbol's grounding.

        Args:
            symbol: The symbol to check.

        Returns:
            Confidence score between 0.0 and 1.0.
            Returns 0.0 if symbol has no grounding.

        Validates: Requirements 5.3
        """
        if not symbol:
            return 0.0

        normalized_symbol = symbol.lower().strip()

        if normalized_symbol not in self._groundings:
            return 0.0

        return self._groundings[normalized_symbol].confidence

    def get_ungrounded_symbols(self) -> List[str]:
        """
        Get symbols with low grounding confidence.

        Returns symbols whose confidence is below the configured
        minimum confidence threshold.

        Returns:
            List of symbol strings with low confidence.

        Validates: Requirements 5.3
        """
        ungrounded = []

        for symbol, grounding in self._groundings.items():
            if grounding.confidence < self._config.min_confidence_threshold:
                ungrounded.append(symbol)

        return ungrounded

    def is_grounded(self, symbol: str) -> bool:
        """
        Check if a symbol is considered grounded.

        A symbol is grounded if its confidence is at or above the
        minimum confidence threshold.

        Args:
            symbol: The symbol to check.

        Returns:
            True if the symbol is grounded, False otherwise.
        """
        return self.get_grounding_confidence(symbol) >= self._config.min_confidence_threshold

    def get_all_symbols(self) -> List[str]:
        """
        Get all symbols that have been grounded.

        Returns:
            List of all symbol strings.
        """
        return list(self._groundings.keys())

    def get_grounding(self, symbol: str) -> Optional[SymbolGrounding]:
        """
        Get the grounding for a symbol if it exists.

        Unlike ground_symbol(), this does not create a new grounding
        if one doesn't exist.

        Args:
            symbol: The symbol to look up.

        Returns:
            SymbolGrounding if found, None otherwise.
        """
        if not symbol:
            return None

        normalized_symbol = symbol.lower().strip()
        return self._groundings.get(normalized_symbol)

    # ==================== Statistics ====================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the symbol grounding module.

        Returns:
            Dictionary with statistics.
        """
        total_symbols = len(self._groundings)
        grounded_count = sum(
            1
            for g in self._groundings.values()
            if g.confidence >= self._config.min_confidence_threshold
        )
        ungrounded_count = total_symbols - grounded_count

        avg_confidence = 0.0
        avg_candidates = 0.0
        if total_symbols > 0:
            avg_confidence = sum(g.confidence for g in self._groundings.values()) / total_symbols
            avg_candidates = (
                sum(len(g.groundings) for g in self._groundings.values()) / total_symbols
            )

        return {
            "total_symbols": total_symbols,
            "grounded_count": grounded_count,
            "ungrounded_count": ungrounded_count,
            "average_confidence": avg_confidence,
            "average_candidates_per_symbol": avg_candidates,
            "total_groundings_created": self._total_groundings_created,
            "total_groundings_updated": self._total_groundings_updated,
            "total_ambiguity_resolutions": self._total_ambiguity_resolutions,
            "uptime_seconds": time.time() - self._initialized_at,
        }

    # ==================== Serialization ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the symbol grounding module state to dictionary representation.

        Returns:
            Dictionary representation of the module state.
        """
        return {
            "config": self._config.to_dict(),
            "groundings": {
                symbol: grounding.to_dict() for symbol, grounding in self._groundings.items()
            },
            "statistics": {
                "total_groundings_created": self._total_groundings_created,
                "total_groundings_updated": self._total_groundings_updated,
                "total_ambiguity_resolutions": self._total_ambiguity_resolutions,
                "initialized_at": self._initialized_at,
            },
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state from a dictionary representation.

        Args:
            state: Dictionary containing saved state.
        """
        # Load config if present
        if "config" in state:
            self._config = SymbolGroundingConfig.from_dict(state["config"])

        # Load groundings
        if "groundings" in state:
            self._groundings = {
                symbol: SymbolGrounding.from_dict(grounding_data)
                for symbol, grounding_data in state["groundings"].items()
            }

        # Load statistics
        if "statistics" in state:
            stats = state["statistics"]
            self._total_groundings_created = stats.get("total_groundings_created", 0)
            self._total_groundings_updated = stats.get("total_groundings_updated", 0)
            self._total_ambiguity_resolutions = stats.get("total_ambiguity_resolutions", 0)
            self._initialized_at = stats.get("initialized_at", time.time())

    def from_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore state from a dictionary representation.

        Alias for load_state() for consistency with other modules.

        Args:
            state: Dictionary containing saved state.
        """
        self.load_state(state)

    def clear(self) -> None:
        """
        Clear all groundings and reset statistics.
        """
        self._groundings.clear()
        self._total_groundings_created = 0
        self._total_groundings_updated = 0
        self._total_ambiguity_resolutions = 0
        self._initialized_at = time.time()

    # ==================== Advanced Operations ====================

    def link_to_concept(
        self,
        symbol: str,
        concept_id: str,
        probability_boost: float = 0.2,
    ) -> bool:
        """
        Link a symbol grounding to a semantic memory concept.

        Creates or updates a grounding candidate with the specified concept ID.

        Args:
            symbol: The symbol to link.
            concept_id: The concept ID from semantic memory.
            probability_boost: Amount to boost the candidate's probability.

        Returns:
            True if linking was successful, False otherwise.
        """
        if not symbol or not concept_id:
            return False

        normalized_symbol = symbol.lower().strip()

        # Get or create grounding
        if normalized_symbol not in self._groundings:
            grounding = self.ground_symbol(symbol)
        else:
            grounding = self._groundings[normalized_symbol]

        # Find existing candidate with this concept_id
        existing_candidate = None
        for candidate in grounding.groundings:
            if candidate.concept_id == concept_id:
                existing_candidate = candidate
                break

        if existing_candidate:
            # Boost existing candidate's probability
            existing_candidate.probability = min(
                1.0, existing_candidate.probability + probability_boost
            )
        else:
            # Create new candidate with concept link
            new_candidate = GroundingCandidate(
                concept_id=concept_id,
                episode_ids=[],
                perceptual_features={"linked_concept": concept_id},
                probability=probability_boost,
            )

            if len(grounding.groundings) < self._config.max_candidates:
                grounding.groundings.append(new_candidate)
            else:
                # Replace lowest probability candidate
                min_prob_idx = min(
                    range(len(grounding.groundings)),
                    key=lambda i: grounding.groundings[i].probability,
                )
                if grounding.groundings[min_prob_idx].probability < probability_boost:
                    grounding.groundings[min_prob_idx] = new_candidate

        # Re-normalize probabilities
        grounding.groundings = self._normalize_probabilities(grounding.groundings)

        # Increase confidence due to explicit linking
        grounding.confidence = min(1.0, grounding.confidence + 0.1)
        grounding.last_updated = time.time()

        return True

    def merge_groundings(self, symbol1: str, symbol2: str) -> Optional[SymbolGrounding]:
        """
        Merge groundings from two symbols into one.

        Combines the grounding candidates from both symbols into the first symbol's
        grounding, useful for synonyms or related terms.

        Args:
            symbol1: The primary symbol (will receive merged groundings).
            symbol2: The secondary symbol (groundings will be merged into symbol1).

        Returns:
            The merged SymbolGrounding, or None if either symbol doesn't exist.
        """
        normalized1 = symbol1.lower().strip() if symbol1 else ""
        normalized2 = symbol2.lower().strip() if symbol2 else ""

        if not normalized1 or not normalized2:
            return None

        if normalized1 not in self._groundings or normalized2 not in self._groundings:
            return None

        grounding1 = self._groundings[normalized1]
        grounding2 = self._groundings[normalized2]

        # Merge candidates from grounding2 into grounding1
        for candidate in grounding2.groundings:
            # Check if similar candidate already exists
            is_duplicate = False
            for existing in grounding1.groundings:
                if existing.concept_id == candidate.concept_id and existing.concept_id is not None:
                    # Merge episode IDs
                    for ep_id in candidate.episode_ids:
                        if ep_id not in existing.episode_ids:
                            existing.episode_ids.append(ep_id)
                    # Boost probability
                    existing.probability = min(
                        1.0, existing.probability + candidate.probability * 0.5
                    )
                    is_duplicate = True
                    break

            if not is_duplicate and len(grounding1.groundings) < self._config.max_candidates:
                grounding1.groundings.append(candidate)

        # Re-normalize probabilities
        grounding1.groundings = self._normalize_probabilities(grounding1.groundings)

        # Update confidence (average of both)
        grounding1.confidence = (grounding1.confidence + grounding2.confidence) / 2
        grounding1.grounding_count += grounding2.grounding_count
        grounding1.last_updated = time.time()

        return grounding1

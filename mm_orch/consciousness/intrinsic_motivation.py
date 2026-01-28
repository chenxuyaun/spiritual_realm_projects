"""
Intrinsic Motivation Engine module for consciousness system.

This module implements the IntrinsicMotivationEngine which manages intrinsic
motivation and curiosity-driven exploration. It calculates curiosity rewards
based on prediction error, tracks novelty and familiarity, and provides
exploration bonuses for action selection.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math
import time


@dataclass
class IntrinsicMotivationConfig:
    """Configuration for the intrinsic motivation engine."""

    base_curiosity_reward: float = 0.5
    prediction_error_weight: float = 0.7
    novelty_threshold: float = 0.5
    familiarity_decay_rate: float = 0.1
    curiosity_decay_rate: float = 0.15
    exploration_weight: float = 0.3
    max_familiarity_entries: int = 10000
    information_gain_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (0.0 <= self.base_curiosity_reward <= 1.0):
            raise ValueError("base_curiosity_reward must be between 0.0 and 1.0")
        if not (0.0 <= self.prediction_error_weight <= 1.0):
            raise ValueError("prediction_error_weight must be between 0.0 and 1.0")
        if not (0.0 <= self.novelty_threshold <= 1.0):
            raise ValueError("novelty_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.familiarity_decay_rate <= 1.0):
            raise ValueError("familiarity_decay_rate must be between 0.0 and 1.0")
        if not (0.0 <= self.curiosity_decay_rate <= 1.0):
            raise ValueError("curiosity_decay_rate must be between 0.0 and 1.0")
        if not (0.0 <= self.exploration_weight <= 1.0):
            raise ValueError("exploration_weight must be between 0.0 and 1.0")
        if self.max_familiarity_entries < 1:
            raise ValueError("max_familiarity_entries must be at least 1")
        if self.information_gain_scale <= 0:
            raise ValueError("information_gain_scale must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "base_curiosity_reward": self.base_curiosity_reward,
            "prediction_error_weight": self.prediction_error_weight,
            "novelty_threshold": self.novelty_threshold,
            "familiarity_decay_rate": self.familiarity_decay_rate,
            "curiosity_decay_rate": self.curiosity_decay_rate,
            "exploration_weight": self.exploration_weight,
            "max_familiarity_entries": self.max_familiarity_entries,
            "information_gain_scale": self.information_gain_scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntrinsicMotivationConfig":
        """Create config from dictionary representation."""
        return cls(
            base_curiosity_reward=data.get("base_curiosity_reward", 0.5),
            prediction_error_weight=data.get("prediction_error_weight", 0.7),
            novelty_threshold=data.get("novelty_threshold", 0.5),
            familiarity_decay_rate=data.get("familiarity_decay_rate", 0.1),
            curiosity_decay_rate=data.get("curiosity_decay_rate", 0.15),
            exploration_weight=data.get("exploration_weight", 0.3),
            max_familiarity_entries=data.get("max_familiarity_entries", 10000),
            information_gain_scale=data.get("information_gain_scale", 1.0),
        )


@dataclass
class FamiliarityEntry:
    """Tracks familiarity with a specific stimulus."""

    stimulus_hash: str
    encounter_count: int = 1
    familiarity_score: float = 0.0
    curiosity_level: float = 1.0
    first_encountered: float = field(default_factory=time.time)
    last_encountered: float = field(default_factory=time.time)
    cumulative_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "stimulus_hash": self.stimulus_hash,
            "encounter_count": self.encounter_count,
            "familiarity_score": self.familiarity_score,
            "curiosity_level": self.curiosity_level,
            "first_encountered": self.first_encountered,
            "last_encountered": self.last_encountered,
            "cumulative_reward": self.cumulative_reward,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FamiliarityEntry":
        """Create entry from dictionary representation."""
        return cls(
            stimulus_hash=data["stimulus_hash"],
            encounter_count=data.get("encounter_count", 1),
            familiarity_score=data.get("familiarity_score", 0.0),
            curiosity_level=data.get("curiosity_level", 1.0),
            first_encountered=data.get("first_encountered", time.time()),
            last_encountered=data.get("last_encountered", time.time()),
            cumulative_reward=data.get("cumulative_reward", 0.0),
        )


@dataclass
class ActionExplorationInfo:
    """Tracks exploration information for an action."""

    action: str
    attempt_count: int = 0
    contexts_seen: List[str] = field(default_factory=list)
    average_information_gain: float = 0.5
    last_attempted: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action": self.action,
            "attempt_count": self.attempt_count,
            "contexts_seen": self.contexts_seen.copy(),
            "average_information_gain": self.average_information_gain,
            "last_attempted": self.last_attempted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionExplorationInfo":
        """Create info from dictionary representation."""
        return cls(
            action=data["action"],
            attempt_count=data.get("attempt_count", 0),
            contexts_seen=data.get("contexts_seen", []),
            average_information_gain=data.get("average_information_gain", 0.5),
            last_attempted=data.get("last_attempted"),
        )


class IntrinsicMotivationEngine:
    """
    Manages intrinsic motivation and curiosity-driven exploration.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intrinsic motivation engine."""
        if config is not None:
            self._config = IntrinsicMotivationConfig.from_dict(config)
        else:
            self._config = IntrinsicMotivationConfig()
        self._familiarity_tracker: Dict[str, FamiliarityEntry] = {}
        self._action_exploration: Dict[str, ActionExplorationInfo] = {}
        self._total_curiosity_rewards: float = 0.0
        self._total_stimuli_encountered: int = 0
        self._initialized_at: float = time.time()

    def _hash_stimulus(self, stimulus: Any) -> str:
        """Generate a hash for a stimulus."""
        if isinstance(stimulus, str):
            content = stimulus
        elif isinstance(stimulus, dict):
            content = json.dumps(stimulus, sort_keys=True, default=str)
        else:
            content = str(stimulus)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Generate a hash for a context dictionary."""
        content = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _calculate_prediction_error(self, predicted_outcome: Any, actual_outcome: Any) -> float:
        """Calculate prediction error between predicted and actual outcomes."""
        if predicted_outcome is None and actual_outcome is None:
            return 0.0
        if predicted_outcome is None or actual_outcome is None:
            return 1.0
        if isinstance(predicted_outcome, (int, float)) and isinstance(actual_outcome, (int, float)):
            max_val = max(abs(predicted_outcome), abs(actual_outcome), 1.0)
            return min(1.0, abs(predicted_outcome - actual_outcome) / max_val)
        if isinstance(predicted_outcome, str) and isinstance(actual_outcome, str):
            if predicted_outcome == actual_outcome:
                return 0.0
            max_len = max(len(predicted_outcome), len(actual_outcome), 1)
            common = sum(1 for a, b in zip(predicted_outcome, actual_outcome) if a == b)
            return 1.0 - (common / max_len)
        if isinstance(predicted_outcome, dict) and isinstance(actual_outcome, dict):
            all_keys = set(predicted_outcome.keys()) | set(actual_outcome.keys())
            if not all_keys:
                return 0.0
            matches = sum(1 for k in all_keys if predicted_outcome.get(k) == actual_outcome.get(k))
            return 1.0 - (matches / len(all_keys))
        if isinstance(predicted_outcome, (list, tuple)) and isinstance(
            actual_outcome, (list, tuple)
        ):
            max_len = max(len(predicted_outcome), len(actual_outcome), 1)
            matches = sum(1 for a, b in zip(predicted_outcome, actual_outcome) if a == b)
            return 1.0 - (matches / max_len)
        return 0.0 if predicted_outcome == actual_outcome else 1.0

    def calculate_curiosity_reward(self, predicted_outcome: Any, actual_outcome: Any) -> float:
        """
        Calculate curiosity reward based on prediction error.
        Returns reward value (0.0 to 1.0) proportional to information gain.
        Validates: Requirements 2.1
        """
        prediction_error = self._calculate_prediction_error(predicted_outcome, actual_outcome)
        information_gain = math.log1p(prediction_error * self._config.information_gain_scale)
        max_info_gain = math.log1p(self._config.information_gain_scale)
        normalized_gain = information_gain / max_info_gain if max_info_gain > 0 else 0.0
        reward = (
            (1 - self._config.prediction_error_weight) * self._config.base_curiosity_reward
            + self._config.prediction_error_weight * normalized_gain
        )
        reward = min(1.0, max(0.0, reward))
        self._total_curiosity_rewards += reward
        return reward

    def get_novelty_score(self, stimulus: Any) -> float:
        """
        Get novelty score for a stimulus (0.0 familiar to 1.0 novel).
        Validates: Requirements 2.3
        """
        stimulus_hash = self._hash_stimulus(stimulus)
        if stimulus_hash not in self._familiarity_tracker:
            return 1.0
        entry = self._familiarity_tracker[stimulus_hash]
        novelty = (1.0 - entry.familiarity_score) * entry.curiosity_level
        return min(1.0, max(0.0, novelty))

    def update_familiarity(self, stimulus: Any) -> None:
        """
        Update familiarity tracking for a stimulus.
        Validates: Requirements 2.3
        """
        stimulus_hash = self._hash_stimulus(stimulus)
        current_time = time.time()
        self._total_stimuli_encountered += 1
        if stimulus_hash not in self._familiarity_tracker:
            self._familiarity_tracker[stimulus_hash] = FamiliarityEntry(
                stimulus_hash=stimulus_hash,
                encounter_count=1,
                familiarity_score=self._config.familiarity_decay_rate,
                curiosity_level=1.0 - self._config.familiarity_decay_rate,
                first_encountered=current_time,
                last_encountered=current_time,
            )
        else:
            entry = self._familiarity_tracker[stimulus_hash]
            entry.encounter_count += 1
            entry.last_encountered = current_time
            entry.familiarity_score = min(
                1.0,
                entry.familiarity_score
                + (1.0 - entry.familiarity_score) * self._config.familiarity_decay_rate,
            )
        self._prune_familiarity_tracker()

    def _prune_familiarity_tracker(self) -> None:
        """Remove least recently accessed entries if over limit."""
        if len(self._familiarity_tracker) <= self._config.max_familiarity_entries:
            return
        entries = sorted(self._familiarity_tracker.items(), key=lambda x: x[1].last_encountered)
        num_to_remove = len(entries) - self._config.max_familiarity_entries
        for i in range(num_to_remove):
            del self._familiarity_tracker[entries[i][0]]

    def get_exploration_bonus(self, action: str, context: Dict[str, Any]) -> float:
        """
        Get exploration bonus for an action in context.
        Validates: Requirements 2.4
        """
        context_hash = self._hash_context(context)
        if action not in self._action_exploration:
            return 1.0
        info = self._action_exploration[action]
        action_novelty = 1.0 / (1.0 + math.log1p(info.attempt_count))
        context_novelty = 1.0 if context_hash not in info.contexts_seen else 0.3
        info_gain_bonus = info.average_information_gain
        exploration_bonus = 0.4 * action_novelty + 0.3 * context_novelty + 0.3 * info_gain_bonus
        return min(1.0, max(0.0, exploration_bonus))

    def update_action_exploration(
        self, action: str, context: Dict[str, Any], information_gain: float
    ) -> None:
        """Update exploration tracking for an action."""
        context_hash = self._hash_context(context)
        current_time = time.time()
        if action not in self._action_exploration:
            self._action_exploration[action] = ActionExplorationInfo(
                action=action,
                attempt_count=1,
                contexts_seen=[context_hash],
                average_information_gain=information_gain,
                last_attempted=current_time,
            )
        else:
            info = self._action_exploration[action]
            info.attempt_count += 1
            info.last_attempted = current_time
            if context_hash not in info.contexts_seen:
                info.contexts_seen.append(context_hash)
                if len(info.contexts_seen) > 100:
                    info.contexts_seen = info.contexts_seen[-100:]
            n = info.attempt_count
            info.average_information_gain = (
                info.average_information_gain * (n - 1) + information_gain
            ) / n

    def decay_curiosity(self, stimulus: Any) -> None:
        """
        Apply curiosity decay for repeated stimulus.
        Validates: Requirements 2.5
        """
        stimulus_hash = self._hash_stimulus(stimulus)
        if stimulus_hash not in self._familiarity_tracker:
            return
        entry = self._familiarity_tracker[stimulus_hash]
        entry.curiosity_level = max(
            0.0, entry.curiosity_level * (1.0 - self._config.curiosity_decay_rate)
        )

    def get_intrinsic_reward(self, stimulus: Any) -> float:
        """
        Get the intrinsic reward for encountering a stimulus.
        Validates: Requirements 2.2
        """
        novelty = self.get_novelty_score(stimulus)
        if novelty > self._config.novelty_threshold:
            reward = self._config.base_curiosity_reward + (
                (1.0 - self._config.base_curiosity_reward)
                * (novelty - self._config.novelty_threshold)
                / (1.0 - self._config.novelty_threshold)
            )
        else:
            reward = self._config.base_curiosity_reward * (novelty / self._config.novelty_threshold)
        self.update_familiarity(stimulus)
        self.decay_curiosity(stimulus)
        stimulus_hash = self._hash_stimulus(stimulus)
        if stimulus_hash in self._familiarity_tracker:
            self._familiarity_tracker[stimulus_hash].cumulative_reward += reward
        return min(1.0, max(0.0, reward))

    def select_action_with_exploration(
        self,
        actions: List[str],
        context: Dict[str, Any],
        known_rewards: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float]:
        """
        Select an action balancing exploration and exploitation.
        Validates: Requirements 2.4
        """
        if not actions:
            raise ValueError("No actions provided for selection")
        if known_rewards is None:
            known_rewards = {}
        best_action = actions[0]
        best_score = float("-inf")
        for action in actions:
            exploration_bonus = self.get_exploration_bonus(action, context)
            known_reward = known_rewards.get(action, 0.5)
            score = (
                self._config.exploration_weight * exploration_bonus
                + (1.0 - self._config.exploration_weight) * known_reward
            )
            if score > best_score:
                best_score = score
                best_action = action
        return best_action, best_score

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the intrinsic motivation engine."""
        return {
            "total_stimuli_encountered": self._total_stimuli_encountered,
            "total_curiosity_rewards": self._total_curiosity_rewards,
            "tracked_stimuli_count": len(self._familiarity_tracker),
            "tracked_actions_count": len(self._action_exploration),
            "uptime": time.time() - self._initialized_at,
            "config": self._config.to_dict(),
        }

    def get_familiarity_entry(self, stimulus: Any) -> Optional[FamiliarityEntry]:
        """Get the familiarity entry for a stimulus."""
        stimulus_hash = self._hash_stimulus(stimulus)
        return self._familiarity_tracker.get(stimulus_hash)

    def get_action_exploration_info(self, action: str) -> Optional[ActionExplorationInfo]:
        """Get exploration info for an action."""
        return self._action_exploration.get(action)

    def reset_curiosity(self, stimulus: Any) -> None:
        """Reset curiosity for a specific stimulus."""
        stimulus_hash = self._hash_stimulus(stimulus)
        if stimulus_hash in self._familiarity_tracker:
            entry = self._familiarity_tracker[stimulus_hash]
            entry.curiosity_level = 1.0
            entry.familiarity_score = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the intrinsic motivation engine to a dictionary."""
        return {
            "config": self._config.to_dict(),
            "familiarity_tracker": {k: v.to_dict() for k, v in self._familiarity_tracker.items()},
            "action_exploration": {k: v.to_dict() for k, v in self._action_exploration.items()},
            "total_curiosity_rewards": self._total_curiosity_rewards,
            "total_stimuli_encountered": self._total_stimuli_encountered,
            "initialized_at": self._initialized_at,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore the intrinsic motivation engine from a dictionary."""
        if "config" in data:
            self._config = IntrinsicMotivationConfig.from_dict(data["config"])
        if "familiarity_tracker" in data:
            self._familiarity_tracker = {
                k: FamiliarityEntry.from_dict(v) for k, v in data["familiarity_tracker"].items()
            }
        if "action_exploration" in data:
            self._action_exploration = {
                k: ActionExplorationInfo.from_dict(v) for k, v in data["action_exploration"].items()
            }
        if "total_curiosity_rewards" in data:
            self._total_curiosity_rewards = data["total_curiosity_rewards"]
        if "total_stimuli_encountered" in data:
            self._total_stimuli_encountered = data["total_stimuli_encountered"]
        if "initialized_at" in data:
            self._initialized_at = data["initialized_at"]

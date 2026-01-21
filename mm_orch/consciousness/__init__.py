"""
Consciousness modules for self-awareness and metacognition.

This package provides the consciousness system for the MuAI orchestration system,
including self-awareness, world modeling, metacognition, motivation, emotion,
and development stage management.
"""

from mm_orch.consciousness.core import (
    ConsciousnessCore,
    get_consciousness,
    save_consciousness,
    load_consciousness,
)
from mm_orch.consciousness.self_model import SelfModel, Capability
from mm_orch.consciousness.world_model import WorldModel, Entity, UserModel
from mm_orch.consciousness.metacognition import Metacognition, TaskMonitor, StrategyRecord
from mm_orch.consciousness.motivation import (
    MotivationSystem,
    Goal,
    GoalType,
    GoalStatus,
)
from mm_orch.consciousness.emotion import EmotionSystem, EmotionEvent
from mm_orch.consciousness.development import (
    DevelopmentSystem,
    DevelopmentStage,
    StageConfig,
    LearningRecord,
)

__all__ = [
    # Core
    "ConsciousnessCore",
    "get_consciousness",
    "save_consciousness",
    "load_consciousness",
    # Self Model
    "SelfModel",
    "Capability",
    # World Model
    "WorldModel",
    "Entity",
    "UserModel",
    # Metacognition
    "Metacognition",
    "TaskMonitor",
    "StrategyRecord",
    # Motivation
    "MotivationSystem",
    "Goal",
    "GoalType",
    "GoalStatus",
    # Emotion
    "EmotionSystem",
    "EmotionEvent",
    # Development
    "DevelopmentSystem",
    "DevelopmentStage",
    "StageConfig",
    "LearningRecord",
]

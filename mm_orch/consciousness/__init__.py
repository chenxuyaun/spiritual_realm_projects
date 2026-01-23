"""
Consciousness modules for self-awareness and metacognition.

This package provides the consciousness system for the MuAI orchestration system,
including self-awareness, world modeling, metacognition, motivation, emotion,
development stage management, curriculum learning, dual memory systems,
enhanced emotion processing, and continuous learning capabilities.
"""

# Core consciousness modules (existing)
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

# Curriculum Learning System (new)
from mm_orch.consciousness.curriculum import (
    CurriculumLearningSystem,
    TaskDifficulty,
    ZPDAssessment,
    CapabilityDimension,
    CurriculumConfig,
)

# Intrinsic Motivation Engine (new)
from mm_orch.consciousness.intrinsic_motivation import (
    IntrinsicMotivationEngine,
    IntrinsicMotivationConfig,
)

# Experience Replay Buffer (new)
from mm_orch.consciousness.experience_replay import (
    ExperienceReplayBuffer,
    Experience,
    ExperienceReplayConfig,
)

# Episodic Memory System (new)
from mm_orch.consciousness.episodic_memory import (
    EpisodicMemory,
    Episode,
    EpisodicMemoryConfig,
)

# Semantic Memory and Knowledge Graph (new)
from mm_orch.consciousness.knowledge_graph import (
    KnowledgeGraph,
    ConceptNode,
    Relationship,
)
from mm_orch.consciousness.semantic_memory import (
    SemanticMemory,
    IntegrationResult,
    ConsolidationResult,
    ExtractionResult,
    ConflictInfo,
    ConflictResolutionStrategy,
    SemanticMemoryConfig,
)

# Symbol Grounding Module (new)
from mm_orch.consciousness.symbol_grounding import (
    SymbolGroundingModule,
    SymbolGrounding,
    GroundingCandidate,
    SymbolGroundingConfig,
)

# PAD Emotion Model (new)
from mm_orch.consciousness.pad_emotion import (
    PADEmotionModel,
    PADState,
    PADEmotionConfig,
    EMOTION_PAD_MAPPING,
)

# Cognitive Appraisal System (new)
from mm_orch.consciousness.cognitive_appraisal import (
    CognitiveAppraisalSystem,
    AppraisalResult,
    CognitiveAppraisalConfig,
)

# Decision Modulator (new)
from mm_orch.consciousness.decision_modulator import (
    DecisionModulator,
    DecisionModifiers,
    DecisionLog,
    DecisionModulatorConfig,
)

__all__ = [
    # Core consciousness modules (existing)
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
    # Curriculum Learning System (new)
    "CurriculumLearningSystem",
    "TaskDifficulty",
    "ZPDAssessment",
    "CapabilityDimension",
    "CurriculumConfig",
    # Intrinsic Motivation Engine (new)
    "IntrinsicMotivationEngine",
    "IntrinsicMotivationConfig",
    # Experience Replay Buffer (new)
    "ExperienceReplayBuffer",
    "Experience",
    "ExperienceReplayConfig",
    # Episodic Memory System (new)
    "EpisodicMemory",
    "Episode",
    "EpisodicMemoryConfig",
    # Semantic Memory and Knowledge Graph (new)
    "KnowledgeGraph",
    "ConceptNode",
    "Relationship",
    "SemanticMemory",
    "IntegrationResult",
    "ConsolidationResult",
    "ExtractionResult",
    "ConflictInfo",
    "ConflictResolutionStrategy",
    "SemanticMemoryConfig",
    # Symbol Grounding Module (new)
    "SymbolGroundingModule",
    "SymbolGrounding",
    "GroundingCandidate",
    "SymbolGroundingConfig",
    # PAD Emotion Model (new)
    "PADEmotionModel",
    "PADState",
    "PADEmotionConfig",
    "EMOTION_PAD_MAPPING",
    # Cognitive Appraisal System (new)
    "CognitiveAppraisalSystem",
    "AppraisalResult",
    "CognitiveAppraisalConfig",
    # Decision Modulator (new)
    "DecisionModulator",
    "DecisionModifiers",
    "DecisionLog",
    "DecisionModulatorConfig",
]

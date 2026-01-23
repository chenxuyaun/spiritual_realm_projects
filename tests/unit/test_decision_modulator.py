"""
Unit tests for DecisionModulator.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import pytest
import time

from mm_orch.consciousness.pad_emotion import PADEmotionModel, PADState
from mm_orch.consciousness.decision_modulator import (
    DecisionModulator,
    DecisionModifiers,
    DecisionLog,
    DecisionModulatorConfig,
)


class TestDecisionModifiers:
    """Test DecisionModifiers dataclass."""
    
    def test_valid_modifiers(self):
        """Test creating valid modifiers."""
        modifiers = DecisionModifiers(
            risk_tolerance=0.2,
            deliberation_time=1.5,
            exploration_bias=0.1,
            confidence_threshold=-0.1,
        )
        assert modifiers.risk_tolerance == 0.2
        assert modifiers.deliberation_time == 1.5
        assert modifiers.exploration_bias == 0.1
        assert modifiers.confidence_threshold == -0.1
    
    def test_invalid_risk_tolerance(self):
        """Test that invalid risk tolerance raises error."""
        with pytest.raises(ValueError, match="risk_tolerance"):
            DecisionModifiers(
                risk_tolerance=0.6,  # Out of range
                deliberation_time=1.0,
                exploration_bias=0.0,
                confidence_threshold=0.0,
            )
    
    def test_invalid_deliberation_time(self):
        """Test that invalid deliberation time raises error."""
        with pytest.raises(ValueError, match="deliberation_time"):
            DecisionModifiers(
                risk_tolerance=0.0,
                deliberation_time=2.5,  # Out of range
                exploration_bias=0.0,
                confidence_threshold=0.0,
            )
    
    def test_invalid_exploration_bias(self):
        """Test that invalid exploration bias raises error."""
        with pytest.raises(ValueError, match="exploration_bias"):
            DecisionModifiers(
                risk_tolerance=0.0,
                deliberation_time=1.0,
                exploration_bias=0.4,  # Out of range
                confidence_threshold=0.0,
            )
    
    def test_invalid_confidence_threshold(self):
        """Test that invalid confidence threshold raises error."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            DecisionModifiers(
                risk_tolerance=0.0,
                deliberation_time=1.0,
                exploration_bias=0.0,
                confidence_threshold=0.3,  # Out of range
            )
    
    def test_to_dict(self):
        """Test converting modifiers to dictionary."""
        modifiers = DecisionModifiers(
            risk_tolerance=0.2,
            deliberation_time=1.5,
            exploration_bias=0.1,
            confidence_threshold=-0.1,
        )
        data = modifiers.to_dict()
        assert data["risk_tolerance"] == 0.2
        assert data["deliberation_time"] == 1.5
        assert data["exploration_bias"] == 0.1
        assert data["confidence_threshold"] == -0.1
    
    def test_from_dict(self):
        """Test creating modifiers from dictionary."""
        data = {
            "risk_tolerance": 0.2,
            "deliberation_time": 1.5,
            "exploration_bias": 0.1,
            "confidence_threshold": -0.1,
        }
        modifiers = DecisionModifiers.from_dict(data)
        assert modifiers.risk_tolerance == 0.2
        assert modifiers.deliberation_time == 1.5
        assert modifiers.exploration_bias == 0.1
        assert modifiers.confidence_threshold == -0.1


class TestDecisionLog:
    """Test DecisionLog dataclass."""
    
    def test_create_log(self):
        """Test creating a decision log entry."""
        log = DecisionLog(
            timestamp=time.time(),
            decision="choose_strategy_A",
            emotional_state={"pleasure": 0.5, "arousal": 0.6, "dominance": 0.3},
            modifiers={"risk_tolerance": 0.2},
            outcome="success",
            metadata={"context": "test"},
        )
        assert log.decision == "choose_strategy_A"
        assert log.outcome == "success"
        assert log.metadata["context"] == "test"
    
    def test_log_to_dict(self):
        """Test converting log to dictionary."""
        log = DecisionLog(
            timestamp=123.456,
            decision="test_decision",
            emotional_state={"pleasure": 0.0},
            modifiers={"risk_tolerance": 0.0},
        )
        data = log.to_dict()
        assert data["timestamp"] == 123.456
        assert data["decision"] == "test_decision"
        assert data["emotional_state"] == {"pleasure": 0.0}
    
    def test_log_from_dict(self):
        """Test creating log from dictionary."""
        data = {
            "timestamp": 123.456,
            "decision": "test_decision",
            "emotional_state": {"pleasure": 0.0},
            "modifiers": {"risk_tolerance": 0.0},
            "outcome": "success",
            "metadata": {"key": "value"},
        }
        log = DecisionLog.from_dict(data)
        assert log.timestamp == 123.456
        assert log.decision == "test_decision"
        assert log.outcome == "success"
        assert log.metadata["key"] == "value"


class TestDecisionModulatorConfig:
    """Test DecisionModulatorConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DecisionModulatorConfig()
        assert config.dominance_risk_scale == 0.4
        assert config.base_risk_tolerance == 0.0
        assert config.arousal_deliberation_scale == 0.8
        assert config.base_deliberation_time == 1.0
        assert config.conservative_pleasure_threshold == -0.3
        assert config.max_log_size == 1000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DecisionModulatorConfig(
            dominance_risk_scale=0.5,
            conservative_pleasure_threshold=-0.4,
            max_log_size=500,
        )
        assert config.dominance_risk_scale == 0.5
        assert config.conservative_pleasure_threshold == -0.4
        assert config.max_log_size == 500
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = DecisionModulatorConfig()
        data = config.to_dict()
        assert "dominance_risk_scale" in data
        assert "conservative_pleasure_threshold" in data
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "dominance_risk_scale": 0.5,
            "conservative_pleasure_threshold": -0.4,
        }
        config = DecisionModulatorConfig.from_dict(data)
        assert config.dominance_risk_scale == 0.5
        assert config.conservative_pleasure_threshold == -0.4


class TestDecisionModulator:
    """Test DecisionModulator class."""
    
    def test_initialization(self):
        """Test modulator initialization."""
        pad_model = PADEmotionModel()
        modulator = DecisionModulator(pad_model)
        assert modulator._pad_model is pad_model
        assert modulator._total_decisions == 0
    
    def test_initialization_with_config(self):
        """Test modulator initialization with config."""
        pad_model = PADEmotionModel()
        config = {"dominance_risk_scale": 0.5}
        modulator = DecisionModulator(pad_model, config)
        assert modulator._config.dominance_risk_scale == 0.5
    
    def test_invalid_pad_model(self):
        """Test that invalid PAD model raises error."""
        with pytest.raises(TypeError, match="PADEmotionModel"):
            DecisionModulator("not a pad model")
    
    def test_get_modifiers(self):
        """Test getting decision modifiers."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(0.5, 0.6, 0.3))
        modulator = DecisionModulator(pad_model)
        
        modifiers = modulator.get_modifiers()
        assert isinstance(modifiers, DecisionModifiers)
        assert -0.5 <= modifiers.risk_tolerance <= 0.5
        assert 0.5 <= modifiers.deliberation_time <= 2.0
        assert -0.3 <= modifiers.exploration_bias <= 0.3
        assert -0.2 <= modifiers.confidence_threshold <= 0.2
    
    def test_adjust_risk_tolerance_high_dominance(self):
        """Test risk tolerance with high dominance."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(0.0, 0.5, 0.8))  # High dominance
        modulator = DecisionModulator(pad_model)
        
        risk = modulator.adjust_risk_tolerance(0.0)
        assert risk > 0.2, "High dominance should increase risk tolerance"
    
    def test_adjust_risk_tolerance_low_dominance(self):
        """Test risk tolerance with low dominance."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(0.0, 0.5, -0.8))  # Low dominance
        modulator = DecisionModulator(pad_model)
        
        risk = modulator.adjust_risk_tolerance(0.0)
        assert risk < -0.2, "Low dominance should decrease risk tolerance"
    
    def test_adjust_deliberation_high_arousal(self):
        """Test deliberation time with high arousal."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(0.0, 0.9, 0.0))  # High arousal
        modulator = DecisionModulator(pad_model)
        
        delib = modulator.adjust_deliberation(1.0)
        assert delib < 1.0, "High arousal should decrease deliberation time"
    
    def test_adjust_deliberation_low_arousal(self):
        """Test deliberation time with low arousal."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(0.0, 0.1, 0.0))  # Low arousal
        modulator = DecisionModulator(pad_model)
        
        delib = modulator.adjust_deliberation(1.0)
        assert delib > 1.0, "Low arousal should increase deliberation time"
    
    def test_adjust_strategy_confidence(self):
        """Test adjusting strategy confidence."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(0.5, 0.5, 0.5))
        modulator = DecisionModulator(pad_model)
        
        adjusted = modulator.adjust_strategy_confidence("strategy_A", 0.7)
        assert 0.0 <= adjusted <= 1.0, "Adjusted confidence should be in valid range"
    
    def test_adjust_strategy_confidence_clamping(self):
        """Test that confidence adjustment clamps to [0, 1]."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(-0.8, 0.5, -0.8))  # Very negative state
        modulator = DecisionModulator(pad_model)
        
        # Even with very low base confidence, should not go below 0
        adjusted = modulator.adjust_strategy_confidence("strategy_A", 0.05)
        assert adjusted >= 0.0
        
        # Even with very high base confidence, should not exceed 1
        pad_model.set_state(PADState(0.8, 0.5, 0.8))  # Very positive state
        adjusted = modulator.adjust_strategy_confidence("strategy_A", 0.95)
        assert adjusted <= 1.0
    
    def test_should_use_conservative_strategy_low_pleasure(self):
        """Test conservative strategy with low pleasure."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(-0.5, 0.5, 0.0))  # Low pleasure
        modulator = DecisionModulator(pad_model)
        
        assert modulator.should_use_conservative_strategy()
    
    def test_should_use_conservative_strategy_high_pleasure(self):
        """Test conservative strategy with high pleasure."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(0.5, 0.5, 0.0))  # High pleasure
        modulator = DecisionModulator(pad_model)
        
        assert not modulator.should_use_conservative_strategy()
    
    def test_log_decision(self):
        """Test logging a decision."""
        pad_model = PADEmotionModel()
        state = PADState(0.5, 0.6, 0.3)
        pad_model.set_state(state)
        modulator = DecisionModulator(pad_model)
        
        modulator.log_decision("test_decision", state, outcome="success")
        
        logs = modulator.get_decision_log()
        assert len(logs) == 1
        assert logs[0].decision == "test_decision"
        assert logs[0].outcome == "success"
        assert modulator._total_decisions == 1
    
    def test_log_decision_with_metadata(self):
        """Test logging a decision with metadata."""
        pad_model = PADEmotionModel()
        state = PADState(0.5, 0.6, 0.3)
        modulator = DecisionModulator(pad_model)
        
        metadata = {"context": "test", "priority": "high"}
        modulator.log_decision("test_decision", state, metadata=metadata)
        
        logs = modulator.get_decision_log()
        assert logs[0].metadata["context"] == "test"
        assert logs[0].metadata["priority"] == "high"
    
    def test_get_decision_log_limit(self):
        """Test getting decision log with limit."""
        pad_model = PADEmotionModel()
        state = PADState(0.0, 0.5, 0.0)
        modulator = DecisionModulator(pad_model)
        
        # Log multiple decisions
        for i in range(10):
            modulator.log_decision(f"decision_{i}", state)
        
        # Get last 5
        logs = modulator.get_decision_log(limit=5)
        assert len(logs) == 5
        assert logs[0].decision == "decision_5"
        assert logs[-1].decision == "decision_9"
    
    def test_get_decision_log_since(self):
        """Test getting decision log since timestamp."""
        pad_model = PADEmotionModel()
        state = PADState(0.0, 0.5, 0.0)
        modulator = DecisionModulator(pad_model)
        
        # Log some decisions
        modulator.log_decision("decision_1", state)
        time.sleep(0.01)
        cutoff = time.time()
        time.sleep(0.01)
        modulator.log_decision("decision_2", state)
        modulator.log_decision("decision_3", state)
        
        # Get logs since cutoff
        logs = modulator.get_decision_log(since=cutoff)
        assert len(logs) == 2
        assert all(log.timestamp >= cutoff for log in logs)
    
    def test_log_pruning(self):
        """Test that log is pruned when exceeding max size."""
        pad_model = PADEmotionModel()
        state = PADState(0.0, 0.5, 0.0)
        config = {"max_log_size": 10}
        modulator = DecisionModulator(pad_model, config)
        
        # Log more than max size
        for i in range(15):
            modulator.log_decision(f"decision_{i}", state)
        
        logs = modulator.get_decision_log()
        assert len(logs) == 10
        # Should keep the most recent ones
        assert logs[0].decision == "decision_5"
        assert logs[-1].decision == "decision_14"
    
    def test_get_statistics(self):
        """Test getting statistics."""
        pad_model = PADEmotionModel()
        pad_model.set_state(PADState(0.5, 0.6, 0.3))
        modulator = DecisionModulator(pad_model)
        
        # Log some decisions
        state = pad_model.get_state()
        modulator.log_decision("decision_1", state)
        modulator.log_decision("decision_2", state)
        
        stats = modulator.get_statistics()
        assert stats["total_decisions"] == 2
        assert stats["log_size"] == 2
        assert "current_emotional_state" in stats
        assert "current_modifiers" in stats
        assert "conservative_mode" in stats
    
    def test_analyze_decision_patterns(self):
        """Test analyzing decision patterns."""
        pad_model = PADEmotionModel()
        modulator = DecisionModulator(pad_model)
        
        # Log decisions with different emotional states
        modulator.log_decision("decision_1", PADState(0.5, 0.6, 0.3))
        modulator.log_decision("decision_2", PADState(-0.5, 0.4, -0.2))
        modulator.log_decision("decision_3", PADState(0.1, 0.5, 0.0))
        
        analysis = modulator.analyze_decision_patterns()
        assert analysis["total_decisions"] == 3
        assert "average_risk_tolerance" in analysis
        assert "average_deliberation_time" in analysis
        assert "conservative_decisions" in analysis
        assert "decisions_by_emotion" in analysis
    
    def test_analyze_decision_patterns_empty(self):
        """Test analyzing patterns with empty log."""
        pad_model = PADEmotionModel()
        modulator = DecisionModulator(pad_model)
        
        analysis = modulator.analyze_decision_patterns()
        assert analysis["total_decisions"] == 0
        assert analysis["average_risk_tolerance"] == 0.0
    
    def test_clear_log(self):
        """Test clearing the decision log."""
        pad_model = PADEmotionModel()
        state = PADState(0.0, 0.5, 0.0)
        modulator = DecisionModulator(pad_model)
        
        modulator.log_decision("decision_1", state)
        modulator.log_decision("decision_2", state)
        assert len(modulator.get_decision_log()) == 2
        
        modulator.clear_log()
        assert len(modulator.get_decision_log()) == 0
    
    def test_to_dict(self):
        """Test converting modulator to dictionary."""
        pad_model = PADEmotionModel()
        state = PADState(0.5, 0.6, 0.3)
        modulator = DecisionModulator(pad_model)
        modulator.log_decision("test_decision", state)
        
        data = modulator.to_dict()
        assert "config" in data
        assert "statistics" in data
        assert "decision_log" in data
        assert data["statistics"]["total_decisions"] == 1
    
    def test_load_state(self):
        """Test loading modulator state."""
        pad_model = PADEmotionModel()
        modulator = DecisionModulator(pad_model)
        
        state_data = {
            "config": {"max_log_size": 500},
            "statistics": {"total_decisions": 10, "initialized_at": 123.456},
            "decision_log": [
                {
                    "timestamp": 123.456,
                    "decision": "test",
                    "emotional_state": {"pleasure": 0.5, "arousal": 0.6, "dominance": 0.3},
                    "modifiers": {"risk_tolerance": 0.2},
                }
            ],
        }
        
        modulator.load_state(state_data)
        assert modulator._config.max_log_size == 500
        assert modulator._total_decisions == 10
        assert len(modulator._decision_log) == 1
        assert modulator._decision_log[0].decision == "test"
    
    def test_state_persistence_round_trip(self):
        """Test that state can be saved and restored."""
        pad_model = PADEmotionModel()
        state = PADState(0.5, 0.6, 0.3)
        modulator1 = DecisionModulator(pad_model)
        modulator1.log_decision("decision_1", state)
        modulator1.log_decision("decision_2", state)
        
        # Save state
        saved_state = modulator1.to_dict()
        
        # Create new modulator and load state
        pad_model2 = PADEmotionModel()
        modulator2 = DecisionModulator(pad_model2)
        modulator2.load_state(saved_state)
        
        # Verify state was restored
        assert modulator2._total_decisions == modulator1._total_decisions
        assert len(modulator2._decision_log) == len(modulator1._decision_log)
        assert modulator2._decision_log[0].decision == "decision_1"

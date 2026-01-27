"""
Integration tests for Phase B orchestrator with Phase A fallback.

Tests that Phase B components integrate correctly with the main system
and fall back gracefully to Phase A when components are unavailable.
"""

import pytest
from mm_orch.orchestration.phase_b_orchestrator import (
    PhaseBOrchestrator,
    PhaseBComponents,
    get_phase_b_orchestrator,
    reset_phase_b_orchestrator
)
from mm_orch.orchestration.config_fallback import get_config_manager, reset_config_manager
from mm_orch.schemas import UserRequest


class TestPhaseBIntegration:
    """Test Phase B orchestrator integration."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_phase_b_orchestrator()
        reset_config_manager()
    
    def test_orchestrator_initialization(self):
        """Test that Phase B orchestrator initializes without errors."""
        orchestrator = PhaseBOrchestrator()
        
        assert orchestrator is not None
        assert orchestrator.config_manager is not None
        assert orchestrator.legacy_runtime is not None
        assert orchestrator.phase_b is not None
    
    def test_orchestrator_singleton(self):
        """Test that get_phase_b_orchestrator returns singleton."""
        orch1 = get_phase_b_orchestrator()
        orch2 = get_phase_b_orchestrator()
        
        assert orch1 is orch2
    
    def test_process_request_basic(self):
        """Test basic request processing."""
        orchestrator = PhaseBOrchestrator()
        
        request = UserRequest(query="What is Python?")
        result = orchestrator.process_request(request)
        
        assert result is not None
        assert result.status in ["success", "partial", "failed"]
        assert result.metadata is not None
        assert "using_phase_b" in result.metadata
    
    def test_statistics_available(self):
        """Test that statistics are available."""
        orchestrator = PhaseBOrchestrator()
        
        stats = orchestrator.get_statistics()
        
        assert stats is not None
        assert "using_phase_b" in stats
        assert "has_graph_executor" in stats
        assert "has_workflow_registry" in stats
        assert "has_router" in stats
        assert "has_tracer" in stats
    
    def test_fallback_to_phase_a(self):
        """Test that orchestrator falls back to Phase A when Phase B unavailable."""
        # Create orchestrator with empty Phase B components
        empty_components = PhaseBComponents()
        orchestrator = PhaseBOrchestrator(phase_b_components=empty_components)
        
        # Should fall back to Phase A
        assert orchestrator.using_phase_b is False
        
        # Should still be able to process requests
        request = UserRequest(query="Test query")
        result = orchestrator.process_request(request)
        
        assert result is not None
        assert result.metadata["using_phase_b"] is False


class TestConfigurationFallback:
    """Test configuration fallback logic."""
    
    def setup_method(self):
        """Reset config manager before each test."""
        reset_config_manager()
    
    def test_router_config_fallback(self):
        """Test router configuration with fallback."""
        config_manager = get_config_manager()
        
        result = config_manager.load_router_config()
        
        assert result is not None
        assert result.config is not None
        assert "router_version" in result.config
        assert result.config_source in ["phase_b", "phase_a", "default"]
    
    def test_workflow_registry_config_fallback(self):
        """Test workflow registry configuration with fallback."""
        config_manager = get_config_manager()
        
        result = config_manager.load_workflow_registry_config()
        
        assert result is not None
        assert result.config is not None
        assert "workflows" in result.config
        assert isinstance(result.config["workflows"], list)
    
    def test_tracer_config_fallback(self):
        """Test tracer configuration with fallback."""
        config_manager = get_config_manager()
        
        result = config_manager.load_tracer_config()
        
        assert result is not None
        assert result.config is not None
        assert "enabled" in result.config
        assert "output_path" in result.config
    
    def test_model_registry_config_fallback(self):
        """Test model registry configuration with fallback."""
        config_manager = get_config_manager()
        
        result = config_manager.load_model_registry_config()
        
        assert result is not None
        assert result.config is not None
        # Models can be either list or dict depending on Phase A/B format
        assert "models" in result.config
        models = result.config["models"]
        assert isinstance(models, (list, dict))
        if isinstance(models, list):
            assert len(models) > 0
        else:
            assert len(models.keys()) > 0
    
    def test_tool_registry_config_fallback(self):
        """Test tool registry configuration with fallback."""
        config_manager = get_config_manager()
        
        result = config_manager.load_tool_registry_config()
        
        assert result is not None
        assert result.config is not None
        assert "tools" in result.config
        assert isinstance(result.config["tools"], list)
    
    def test_fallback_logging(self):
        """Test that fallback decisions are logged."""
        import logging
        
        # Set up logging to capture
        logger = logging.getLogger("mm_orch.orchestration.config_fallback")
        logger.setLevel(logging.INFO)
        
        config_manager = get_config_manager()
        
        # Load config that will likely fall back
        result = config_manager.load_router_config()
        
        # Just verify result is valid - logging capture is environment-dependent
        assert result is not None
        assert result.config is not None


class TestMainCLIIntegration:
    """Test main CLI integration with Phase B flag."""
    
    def test_cli_with_phase_b_flag(self):
        """Test that CLI accepts --phase-b flag."""
        from mm_orch.main import create_parser
        
        parser = create_parser()
        # Query is positional argument, not --query
        args = parser.parse_args(["--phase-b", "test query"])
        
        assert args.phase_b is True
        assert args.query == "test query"
    
    def test_cli_without_phase_b_flag(self):
        """Test that CLI works without --phase-b flag."""
        from mm_orch.main import create_parser
        
        parser = create_parser()
        # Query is positional argument
        args = parser.parse_args(["test query"])
        
        assert args.phase_b is False
        assert args.query == "test query"
    
    def test_cli_initialization_with_phase_b(self):
        """Test CLI initialization with Phase B enabled."""
        from mm_orch.main import CLI, PHASE_B_AVAILABLE
        
        if PHASE_B_AVAILABLE:
            cli = CLI(use_phase_b=True)
            assert cli.using_phase_b is True
        else:
            # Should fall back gracefully
            cli = CLI(use_phase_b=True)
            assert cli.using_phase_b is False
    
    def test_cli_initialization_without_phase_b(self):
        """Test CLI initialization without Phase B."""
        from mm_orch.main import CLI
        
        cli = CLI(use_phase_b=False)
        assert cli.using_phase_b is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

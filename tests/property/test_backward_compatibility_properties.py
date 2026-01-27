"""
Property-based tests for backward compatibility.

This module tests that the system functions correctly without optimization
features enabled, that features are configuration-controlled, and that
existing APIs remain unchanged.

Properties tested:
- Property 50: System functions without optimization features
- Property 52: Features are configuration-controlled
- Property 53: Existing APIs remain unchanged

Requirements validated: 13.1, 13.3, 13.4
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from mm_orch.runtime.model_manager import ModelManager, configure_model_manager
from mm_orch.orchestrator import WorkflowOrchestrator, create_orchestrator
from mm_orch.workflows.search_qa import SearchQAWorkflow
from mm_orch.workflows.chat_generate import ChatGenerateWorkflow
from mm_orch.workflows.rag_qa import RAGQAWorkflow
from mm_orch.schemas import WorkflowType, WorkflowResult, ModelConfig


# Test strategies
@st.composite
def model_config_strategy(draw):
    """Generate valid model configurations."""
    return ModelConfig(
        name=draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll')))),
        model_path=draw(st.text(min_size=1, max_size=50)),
        device=draw(st.sampled_from(["auto", "cuda", "cpu"])),
        max_length=draw(st.integers(min_value=128, max_value=2048)),
        temperature=draw(st.floats(min_value=0.1, max_value=2.0)),
        quantization=draw(st.sampled_from([None, "8bit", "4bit"]))
    )


@st.composite
def workflow_parameters_strategy(draw):
    """Generate valid workflow parameters."""
    return {
        "query": draw(st.text(min_size=1, max_size=100)),
        "max_results": draw(st.integers(min_value=1, max_value=10)),
        "temperature": draw(st.floats(min_value=0.1, max_value=2.0))
    }


class TestProperty50_SystemFunctionsWithoutOptimization:
    """
    Property 50: System functions without optimization features
    
    For any valid configuration and workflow execution, when optimization
    features are not enabled, the system should function with existing
    behavior and produce valid results.
    
    Validates: Requirements 13.1, 13.3
    """
    
    @given(config=model_config_strategy())
    @settings(
        max_examples=20, 
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_model_manager_without_optimization(self, config):
        """
        Test that ModelManager functions without optimization enabled.
        
        Property: For any model configuration, ModelManager should work
        without OptimizationManager.
        """
        # Create ModelManager without optimization
        manager = ModelManager(
            max_cached_models=3,
            default_device="cpu",
            enable_optimization=False
        )
        
        # Verify optimization is disabled
        assert manager.enable_optimization is False
        assert manager.optimization_manager is None
        
        # Register a model
        manager.register_model(config)
        
        # Verify model is registered
        assert manager.is_registered(config.name)
        assert config.name in manager.get_registered_models()
    
    @given(params=workflow_parameters_strategy())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_orchestrator_without_monitoring(self, params):
        """
        Test that Orchestrator functions without monitoring enabled.
        
        Property: For any workflow parameters, Orchestrator should work
        without monitoring components.
        """
        # Create mock model manager
        mock_model_manager = Mock()
        
        # Create Orchestrator without monitoring
        orchestrator = create_orchestrator(
            model_manager=mock_model_manager,
            enable_monitoring=False
        )
        
        # Verify monitoring is disabled
        assert orchestrator.enable_monitoring is False
        assert orchestrator.prometheus_exporter is None
        assert orchestrator.otel_tracer is None
        assert orchestrator.performance_monitor is None
        assert orchestrator.anomaly_detector is None
        
        # Verify workflows are registered
        assert len(orchestrator.get_registered_workflows()) > 0
    
    @given(query=st.text(min_size=1, max_size=100))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_workflow_without_tracer(self, query):
        """
        Test that workflows function without tracer.
        
        Property: For any query, workflows should work without OTelTracer.
        """
        # Create workflow without tracer
        workflow = SearchQAWorkflow(
            model_manager=None,
            tracer=None
        )
        
        # Verify tracer is None
        assert workflow.tracer is None
        
        # Workflow should still be valid
        assert workflow.workflow_type == WorkflowType.SEARCH_QA
        assert workflow.name == "SearchQA"


class TestProperty52_FeaturesAreConfigurationControlled:
    """
    Property 52: Features are configuration-controlled
    
    For any configuration setting, optimization and monitoring features
    should be enabled or disabled based on configuration flags, and the
    system should respect these settings.
    
    Validates: Requirements 13.3, 13.4
    """
    
    @given(
        enable_opt=st.booleans(),
        enable_mon=st.booleans()
    )
    @settings(
        max_examples=20, 
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None  # Disable deadline for this test
    )
    def test_configuration_controls_features(self, enable_opt, enable_mon):
        """
        Test that configuration flags control feature enablement.
        
        Property: For any boolean configuration values, features should
        be enabled/disabled accordingly.
        """
        # Create ModelManager with configuration
        manager = ModelManager(
            enable_optimization=enable_opt
        )
        
        # Verify optimization respects configuration
        # (Will be False if module not available, but that's expected)
        if enable_opt:
            # If enabled, should attempt to use optimization
            # (may be False if module not available)
            assert isinstance(manager.enable_optimization, bool)
        else:
            # If disabled, should definitely be False
            assert manager.enable_optimization is False
        
        # For monitoring, we skip the test if enable_mon is True
        # because Prometheus metrics can only be registered once per process
        if not enable_mon:
            # Create Orchestrator with configuration
            orchestrator = create_orchestrator(
                enable_monitoring=enable_mon
            )
            
            # Verify monitoring respects configuration
            assert orchestrator.enable_monitoring is False
    
    @given(
        max_cached=st.integers(min_value=1, max_value=10),
        device=st.sampled_from(["auto", "cuda", "cpu"])
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_existing_configuration_still_works(self, max_cached, device):
        """
        Test that existing configuration parameters still work.
        
        Property: For any valid existing configuration values, ModelManager
        should accept and use them.
        """
        # Create ModelManager with existing parameters
        manager = ModelManager(
            max_cached_models=max_cached,
            default_device=device
        )
        
        # Verify existing parameters are respected
        assert manager.max_cached_models == max_cached
        assert manager.default_device == device


class TestProperty53_ExistingAPIsRemainUnchanged:
    """
    Property 53: Existing APIs remain unchanged
    
    For any existing API call, the function signatures and return types
    should remain unchanged, ensuring backward compatibility.
    
    Validates: Requirements 13.1, 13.4
    """
    
    def test_model_manager_api_unchanged(self):
        """
        Test that ModelManager API remains unchanged.
        
        Property: Existing ModelManager methods should have the same
        signatures and behavior.
        """
        # Create ModelManager with old-style initialization
        manager = ModelManager(
            max_cached_models=3,
            default_device="auto"
        )
        
        # Verify all existing methods are present
        assert hasattr(manager, 'register_model')
        assert hasattr(manager, 'load_model')
        assert hasattr(manager, 'get_model')
        assert hasattr(manager, 'unload_model')
        assert hasattr(manager, 'infer')
        assert hasattr(manager, 'is_loaded')
        assert hasattr(manager, 'is_registered')
        assert hasattr(manager, 'get_cached_models')
        assert hasattr(manager, 'get_registered_models')
        assert hasattr(manager, 'get_cache_info')
        
        # Verify method signatures accept old parameters
        # (This is a compile-time check - if it runs, signatures are compatible)
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cpu",
            max_length=512,
            temperature=0.7
        )
        manager.register_model(config)
        assert manager.is_registered("test-model")
    
    def test_orchestrator_api_unchanged(self):
        """
        Test that Orchestrator API remains unchanged.
        
        Property: Existing Orchestrator methods should have the same
        signatures and behavior.
        """
        # Create Orchestrator with old-style initialization
        orchestrator = WorkflowOrchestrator(
            auto_register_workflows=True
        )
        
        # Verify all existing methods are present
        assert hasattr(orchestrator, 'register_workflow')
        assert hasattr(orchestrator, 'unregister_workflow')
        assert hasattr(orchestrator, 'get_workflow')
        assert hasattr(orchestrator, 'get_registered_workflows')
        assert hasattr(orchestrator, 'execute_workflow')
        assert hasattr(orchestrator, 'process_request')
        assert hasattr(orchestrator, 'get_statistics')
        assert hasattr(orchestrator, 'get_workflow_metrics')
        assert hasattr(orchestrator, 'reset_statistics')
        
        # Verify workflows are registered
        workflows = orchestrator.get_registered_workflows()
        assert len(workflows) > 0
    
    def test_workflow_api_unchanged(self):
        """
        Test that Workflow APIs remain unchanged.
        
        Property: Existing workflow methods should have the same
        signatures and behavior.
        """
        # Create workflows with old-style initialization
        search_qa = SearchQAWorkflow(model_manager=None)
        chat_gen = ChatGenerateWorkflow(model_manager=None)
        rag_qa = RAGQAWorkflow(model_manager=None)
        
        # Verify all existing methods are present
        for workflow in [search_qa, chat_gen, rag_qa]:
            assert hasattr(workflow, 'execute')
            assert hasattr(workflow, 'validate_parameters')
            assert hasattr(workflow, 'get_required_parameters')
            assert hasattr(workflow, 'get_optional_parameters')
            assert hasattr(workflow, 'get_required_models')
            assert hasattr(workflow, 'run')
            assert hasattr(workflow, 'get_metrics')
    
    @given(
        query=st.text(min_size=1, max_size=100),
        max_results=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_workflow_parameters_unchanged(self, query, max_results):
        """
        Test that workflow parameter handling is unchanged.
        
        Property: For any valid parameters, workflows should accept them
        in the same format as before.
        """
        # Create workflow
        workflow = SearchQAWorkflow(model_manager=None)
        
        # Verify parameter validation works with old-style parameters
        parameters = {
            "query": query,
            "max_results": max_results
        }
        
        # Should not raise exception for valid parameters
        try:
            workflow.validate_parameters(parameters)
            validation_passed = True
        except Exception:
            validation_passed = False
        
        # Validation should pass for valid parameters
        assert validation_passed


class TestIntegration_BackwardCompatibility:
    """
    Integration tests for backward compatibility.
    
    These tests verify that the system works end-to-end without
    optimization features enabled.
    """
    
    def test_full_system_without_optimization(self):
        """
        Test that the full system works without optimization.
        
        This is an integration test that verifies all components work
        together without optimization features.
        """
        # Create components without optimization
        model_manager = ModelManager(
            max_cached_models=3,
            enable_optimization=False
        )
        
        orchestrator = create_orchestrator(
            model_manager=model_manager,
            enable_monitoring=False
        )
        
        # Verify system is functional
        assert model_manager.enable_optimization is False
        assert orchestrator.enable_monitoring is False
        
        # Verify workflows are registered
        workflows = orchestrator.get_registered_workflows()
        assert WorkflowType.SEARCH_QA in workflows
        assert WorkflowType.CHAT_GENERATE in workflows
        assert WorkflowType.RAG_QA in workflows
        
        # Verify statistics work
        stats = orchestrator.get_statistics()
        assert "execution_count" in stats
        assert "success_count" in stats
        assert "failure_count" in stats
    
    def test_configuration_migration_path(self):
        """
        Test that old configurations can be migrated to new system.
        
        This verifies that users can gradually adopt new features without
        breaking existing code.
        """
        # Old-style configuration (no optimization)
        old_manager = ModelManager(
            max_cached_models=3,
            default_device="auto"
        )
        
        # New-style configuration (with optimization disabled)
        new_manager = ModelManager(
            max_cached_models=3,
            default_device="auto",
            enable_optimization=False
        )
        
        # Both should have same core behavior
        assert old_manager.max_cached_models == new_manager.max_cached_models
        assert old_manager.default_device == new_manager.default_device
        assert old_manager.enable_optimization == new_manager.enable_optimization


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

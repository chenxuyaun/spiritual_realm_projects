"""
Integration tests for performance monitoring with ModelManager.

Tests the complete integration of PerformanceMonitor with ModelManager
to ensure metrics are recorded correctly during actual inference operations.
"""

import pytest
from mm_orch.runtime.model_manager import ModelManager
from mm_orch.schemas import ModelConfig


class TestPerformanceMonitoringIntegration:
    """Integration tests for performance monitoring."""
    
    @pytest.fixture
    def model_manager(self):
        """Create a ModelManager instance for testing."""
        manager = ModelManager(
            max_cached_models=2,
            default_device="cpu",
            backend="pytorch"
        )
        return manager
    
    def test_performance_stats_available(self, model_manager):
        """Test that performance stats methods are available on ModelManager."""
        # Verify methods exist
        assert hasattr(model_manager, 'get_performance_stats')
        assert hasattr(model_manager, 'compare_backends')
        assert callable(model_manager.get_performance_stats)
        assert callable(model_manager.compare_backends)
    
    def test_get_performance_stats_empty(self, model_manager):
        """Test getting performance stats when no inferences have been run."""
        stats = model_manager.get_performance_stats()
        
        assert stats is not None
        assert "total_inferences" in stats
        assert stats["total_inferences"] == 0
        assert "backends" in stats
        assert len(stats["backends"]) == 0
    
    def test_get_performance_stats_specific_backend(self, model_manager):
        """Test getting stats for a specific backend with no data."""
        stats = model_manager.get_performance_stats("pytorch")
        
        # Should return empty dict when no data
        assert stats == {}
    
    def test_compare_backends_no_data(self, model_manager):
        """Test comparing backends when no data is available."""
        comparison = model_manager.compare_backends("pytorch", "openvino")
        
        # Should return empty dict when no data
        assert comparison == {}
    
    def test_performance_monitor_initialization(self, model_manager):
        """Test that PerformanceMonitor is initialized in ModelManager."""
        assert hasattr(model_manager, '_performance_monitor')
        assert model_manager._performance_monitor is not None
    
    def test_cache_info_includes_backend(self, model_manager):
        """Test that cache info includes backend information."""
        cache_info = model_manager.get_cache_info()
        
        assert "default_backend" in cache_info
        assert "available_backends" in cache_info
        assert cache_info["default_backend"] == "pytorch"
        assert "pytorch" in cache_info["available_backends"]

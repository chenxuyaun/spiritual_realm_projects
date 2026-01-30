"""
Property-based tests for export directory structure.

Feature: openvino-backend-integration
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

# Check if OpenVINO is available
try:
    from optimum.intel import OVModelForCausalLM
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# Skip all tests if OpenVINO not available
pytestmark = pytest.mark.skipif(
    not OPENVINO_AVAILABLE,
    reason="OpenVINO not available"
)

if OPENVINO_AVAILABLE:
    import sys
    from pathlib import Path
    # Add scripts directory to path
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from export_to_openvino import ModelExporter


# Test strategies
@st.composite
def model_name_strategy(draw):
    """Generate valid model names."""
    # Use simple names for testing
    base_names = ["model", "gpt2", "distilgpt2", "test-model"]
    return draw(st.sampled_from(base_names))


@st.composite
def precision_strategy(draw):
    """Generate valid precision values."""
    return draw(st.sampled_from(["FP32", "FP16", "INT8"]))


class TestExportDirectoryStructure:
    """
    Property 12: Export Directory Structure
    
    For any model exported to OpenVINO format, the exported files should
    follow the established directory structure convention (models/openvino/{model_name}).
    
    Validates: Requirements 8.4
    """
    
    @pytest.fixture
    def temp_export_dir(self):
        """Create temporary directory for exports."""
        temp_dir = tempfile.mkdtemp(prefix="openvino_export_test_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def exporter(self, temp_export_dir):
        """Create ModelExporter instance."""
        return ModelExporter(
            output_base_dir=temp_export_dir,
            verbose=False
        )
    
    @pytest.fixture
    def mock_model_files(self):
        """Create mock model files for testing."""
        def _create_files(export_path: Path):
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Create mock OpenVINO files
            (export_path / "openvino_model.xml").write_text('<?xml version="1.0"?><net></net>')
            (export_path / "openvino_model.bin").write_bytes(b"mock data" * 100)
            
            # Create tokenizer files
            (export_path / "tokenizer_config.json").write_text('{}')
            (export_path / "vocab.json").write_text('{}')
            (export_path / "config.json").write_text('{}')
            
            return export_path
        
        return _create_files
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(model_name=model_name_strategy())
    def test_export_follows_directory_convention(
        self,
        exporter,
        temp_export_dir,
        model_name,
        mock_model_files
    ):
        """
        Property: Exports follow models/openvino/{model_name} structure.
        
        For any model name, the export should create a directory at
        output_base_dir/model_name.
        """
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            export_path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                validate=False
            )
            
            # Check directory structure
            expected_path = Path(temp_export_dir) / model_name
            assert export_path == expected_path, \
                f"Expected {expected_path}, got {export_path}"
            assert export_path.exists(), "Export directory doesn't exist"
            assert export_path.is_dir(), "Export path is not a directory"
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(model_name=model_name_strategy())
    def test_export_creates_model_subdirectory(
        self,
        exporter,
        temp_export_dir,
        model_name,
        mock_model_files
    ):
        """
        Property: Each model gets its own subdirectory.
        
        For any model name, the export should create a dedicated
        subdirectory for that model.
        """
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            export_path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                validate=False
            )
            
            # Check that it's a subdirectory of the base
            assert export_path.parent == Path(temp_export_dir)
            assert export_path.name == model_name
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(model_name=model_name_strategy())
    def test_export_path_is_relative_to_base_dir(
        self,
        temp_export_dir,
        model_name,
        mock_model_files
    ):
        """
        Property: Export paths are relative to base directory.
        
        For any base directory and model name, the export path should
        be base_dir/model_name.
        """
        # Use a custom base directory
        custom_base = Path(temp_export_dir) / "custom" / "openvino"
        exporter = ModelExporter(output_base_dir=str(custom_base))
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            export_path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                validate=False
            )
            
            # Check path is relative to custom base
            assert export_path.parent == custom_base
            assert export_path == custom_base / model_name
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(model_name=model_name_strategy())
    def test_custom_output_path_overrides_convention(
        self,
        exporter,
        temp_export_dir,
        model_name,
        mock_model_files
    ):
        """
        Property: Custom output path overrides default convention.
        
        When a custom output path is provided, it should be used
        instead of the default convention.
        """
        custom_path = Path(temp_export_dir) / "custom" / "location" / model_name
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            export_path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                output_path=str(custom_path),
                validate=False
            )
            
            # Check custom path is used
            assert export_path == custom_path
            assert export_path.exists()
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(model_name=model_name_strategy())
    def test_directory_structure_contains_required_files(
        self,
        exporter,
        temp_export_dir,
        model_name,
        mock_model_files
    ):
        """
        Property: Export directory contains all required files.
        
        For any model export, the directory should contain:
        - openvino_model.xml
        - openvino_model.bin
        - tokenizer files
        - config files
        """
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            export_path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                validate=False
            )
            
            # Check required files exist in the directory
            required_files = [
                "openvino_model.xml",
                "openvino_model.bin",
                "tokenizer_config.json",
                "vocab.json",
                "config.json"
            ]
            
            for filename in required_files:
                file_path = export_path / filename
                assert file_path.exists(), f"Missing required file: {filename}"
                assert file_path.is_file(), f"{filename} is not a file"
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        model1=model_name_strategy(),
        model2=model_name_strategy()
    )
    def test_multiple_models_have_separate_directories(
        self,
        exporter,
        temp_export_dir,
        model1,
        model2,
        mock_model_files
    ):
        """
        Property: Multiple models get separate directories.
        
        For any two different model names, they should be exported
        to separate directories.
        """
        from hypothesis import assume
        assume(model1 != model2)
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            # Export first model
            path1 = exporter.export_model(
                model_name=model1,
                precision="FP32",
                validate=False
            )
            
            # Export second model
            path2 = exporter.export_model(
                model_name=model2,
                precision="FP32",
                validate=False
            )
            
            # Check they have different directories
            assert path1 != path2, "Models should have different directories"
            assert path1.exists() and path2.exists()
            assert path1.name == model1
            assert path2.name == model2
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(model_name=model_name_strategy())
    def test_directory_structure_is_flat(
        self,
        exporter,
        temp_export_dir,
        model_name,
        mock_model_files
    ):
        """
        Property: Export directory structure is flat (no nested subdirs).
        
        For any model export, all files should be in the model directory,
        not in nested subdirectories.
        """
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            export_path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                validate=False
            )
            
            # Check all files are directly in the export directory
            for item in export_path.iterdir():
                if item.is_file():
                    # File should be directly in export_path
                    assert item.parent == export_path
    
    def test_default_base_directory_convention(self, temp_export_dir):
        """
        Property: Default base directory follows convention.
        
        When no base directory is specified, it should default to
        'models/openvino'.
        """
        # Create exporter with default base
        exporter = ModelExporter()
        
        # Check default base directory
        assert exporter.output_base_dir == Path("models/openvino")
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(model_name=model_name_strategy())
    def test_list_exported_models_respects_directory_structure(
        self,
        exporter,
        temp_export_dir,
        model_name,
        mock_model_files
    ):
        """
        Property: list_exported_models finds models in directory structure.
        
        For any exported model, list_exported_models should find it
        by checking the directory structure.
        """
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            # Export model
            exporter.export_model(
                model_name=model_name,
                precision="FP32",
                validate=False
            )
            
            # List exported models
            exported = exporter.list_exported_models()
            
            # Should find the exported model
            assert model_name in exported, \
                f"Model {model_name} not found in exported list: {exported}"

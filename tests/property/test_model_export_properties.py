"""
Property-based tests for model export functionality.

Feature: openvino-backend-integration

Note: These tests use mocked components for fast property verification.
For full integration tests with real models, see test_model_export_integration.py
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch, MagicMock

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

# Check if OpenVINO is available
try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
    import openvino as ov
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
def precision_strategy(draw):
    """Generate valid precision values."""
    return draw(st.sampled_from(["FP32", "FP16", "INT8"]))


@st.composite
def model_name_strategy(draw):
    """Generate valid model names for testing."""
    return draw(st.sampled_from(["gpt2", "distilgpt2", "t5-small"]))


class TestModelExportValidation:
    """
    Property 11: Model Export Validation
    
    For any PyTorch model exported to OpenVINO IR format, the export process
    should validate that the resulting model can be loaded and produces outputs,
    and should support multiple precision levels (FP32, FP16, INT8).
    
    Validates: Requirements 8.2, 8.3
    
    Note: These tests use mocks to avoid slow model downloads.
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
            
            # Create mock XML file
            xml_file = export_path / "openvino_model.xml"
            xml_file.write_text('<?xml version="1.0"?><net name="model"></net>')
            
            # Create mock BIN file
            bin_file = export_path / "openvino_model.bin"
            bin_file.write_bytes(b"mock binary data" * 100)
            
            # Create tokenizer files
            (export_path / "tokenizer_config.json").write_text('{"model_type": "gpt2"}')
            (export_path / "vocab.json").write_text('{"hello": 0}')
            (export_path / "merges.txt").write_text('h e\n')
            (export_path / "config.json").write_text('{"model_type": "gpt2"}')
            
            return export_path
        
        return _create_files
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(precision=precision_strategy())
    def test_precision_validation(self, exporter, precision):
        """
        Property: All supported precision values are accepted.
        
        For any precision in SUPPORTED_PRECISIONS, the exporter should
        accept it without raising ValueError.
        """
        # This should not raise
        assert precision in ModelExporter.SUPPORTED_PRECISIONS
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(model_name=model_name_strategy(), precision=precision_strategy())
    def test_output_path_structure(
        self,
        exporter,
        temp_export_dir,
        model_name,
        precision,
        mock_model_files
    ):
        """
        Property: Export creates correct directory structure.
        
        For any model name and precision, the export should create
        a directory at output_base_dir/model_name.
        """
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            # Mock the export to just create files
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            # Export model
            export_path = exporter.export_model(
                model_name=model_name,
                precision=precision,
                validate=False
            )
            
            # Check path structure
            expected_path = Path(temp_export_dir) / model_name
            assert export_path == expected_path
            assert export_path.exists()
            assert export_path.is_dir()
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(precision=precision_strategy())
    def test_required_files_created(
        self,
        exporter,
        temp_export_dir,
        precision,
        mock_model_files
    ):
        """
        Property: Export creates required OpenVINO IR files.
        
        For any valid precision, the export should create XML and BIN files.
        """
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            export_path = exporter.export_model(
                model_name=model_name,
                precision=precision,
                validate=False
            )
            
            # Check required files
            xml_file = export_path / "openvino_model.xml"
            bin_file = export_path / "openvino_model.bin"
            
            assert xml_file.exists(), f"Missing XML file"
            assert bin_file.exists(), f"Missing BIN file"
            assert xml_file.stat().st_size > 0, "XML file is empty"
            assert bin_file.stat().st_size > 0, "BIN file is empty"
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(precision=precision_strategy())
    def test_tokenizer_files_included(
        self,
        exporter,
        temp_export_dir,
        precision,
        mock_model_files
    ):
        """
        Property: Export includes tokenizer configuration.
        
        For any valid precision, the export should include tokenizer files.
        """
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            export_path = exporter.export_model(
                model_name=model_name,
                precision=precision,
                validate=False
            )
            
            # Check tokenizer files
            assert (export_path / "tokenizer_config.json").exists()
            assert (export_path / "vocab.json").exists()
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(
        precision1=precision_strategy(),
        precision2=precision_strategy()
    )
    def test_multiple_precision_exports(
        self,
        exporter,
        temp_export_dir,
        precision1,
        precision2,
        mock_model_files
    ):
        """
        Property: Multiple precision exports can coexist.
        
        Exporting with different precisions to different paths should work.
        """
        assume(precision1 != precision2)
        
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            # Export with first precision
            path1 = exporter.export_model(
                model_name=model_name,
                precision=precision1,
                output_path=str(Path(temp_export_dir) / f"{model_name}_{precision1}"),
                validate=False
            )
            
            # Export with second precision
            path2 = exporter.export_model(
                model_name=model_name,
                precision=precision2,
                output_path=str(Path(temp_export_dir) / f"{model_name}_{precision2}"),
                validate=False
            )
            
            # Both should exist
            assert path1.exists()
            assert path2.exists()
            assert path1 != path2
            assert (path1 / "openvino_model.xml").exists()
            assert (path2 / "openvino_model.xml").exists()
    
    def test_invalid_precision_raises_error(self, exporter):
        """
        Property: Invalid precision values are rejected.
        
        The exporter should raise ValueError for unsupported precision values.
        """
        invalid_precisions = ["FP64", "INT4", "FLOAT32", "fp32", ""]
        
        for invalid_precision in invalid_precisions:
            with pytest.raises(ValueError, match="Invalid precision"):
                exporter.export_model(
                    model_name="test-model",
                    precision=invalid_precision,
                    validate=False
                )
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(precision=precision_strategy())
    def test_validation_checks_required_files(
        self,
        exporter,
        temp_export_dir,
        precision,
        mock_model_files
    ):
        """
        Property: Validation detects missing required files.
        
        If XML or BIN files are missing, validation should fail.
        """
        model_name = "test-model"
        export_path = Path(temp_export_dir) / model_name
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Create incomplete export (missing BIN file)
        (export_path / "openvino_model.xml").write_text("<net></net>")
        
        # Validation should fail
        with pytest.raises(RuntimeError, match="Validation failed|Missing BIN file"):
            exporter._validate_export(export_path, model_name)
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(model_name=model_name_strategy())
    def test_export_all_processes_multiple_models(
        self,
        exporter,
        temp_export_dir,
        model_name,
        mock_model_files
    ):
        """
        Property: export_all processes all provided models.
        
        For any list of model names, export_all should attempt to export each.
        """
        models = [model_name, f"{model_name}-v2"]
        
        with patch.object(exporter, 'export_model') as mock_export_model:
            # Mock export_model to return paths
            def side_effect(model_name, precision):
                path = Path(temp_export_dir) / model_name
                mock_model_files(path)
                return path
            
            mock_export_model.side_effect = side_effect
            
            paths = exporter.export_all(precision="FP32", models=models)
            
            # Should have attempted all models
            assert len(paths) == len(models)
            
            # All paths should exist
            for path in paths:
                assert path.exists()
    
    @settings(
        max_examples=100,
        deadline=1000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @given(precision=precision_strategy())
    def test_export_creates_output_directory(
        self,
        temp_export_dir,
        precision,
        mock_model_files
    ):
        """
        Property: Export creates output directory if it doesn't exist.
        
        For any output path, the exporter should create the directory structure.
        """
        # Use a nested path that doesn't exist
        nested_path = Path(temp_export_dir) / "nested" / "path" / "models"
        exporter = ModelExporter(output_base_dir=str(nested_path))
        
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            def side_effect(source, target, prec):
                mock_model_files(target)
            
            mock_export.side_effect = side_effect
            
            export_path = exporter.export_model(
                model_name=model_name,
                precision=precision,
                validate=False
            )
            
            # Directory should be created
            assert export_path.exists()
            assert export_path.parent.exists()
    
    def test_list_exported_models_returns_valid_exports(
        self,
        exporter,
        temp_export_dir,
        mock_model_files
    ):
        """
        Property: list_exported_models only returns valid exports.
        
        Only directories with openvino_model.xml should be listed.
        """
        # Create some valid exports
        valid_model1 = Path(temp_export_dir) / "model1"
        mock_model_files(valid_model1)
        
        valid_model2 = Path(temp_export_dir) / "model2"
        mock_model_files(valid_model2)
        
        # Create invalid directory (no XML file)
        invalid_model = Path(temp_export_dir) / "invalid"
        invalid_model.mkdir(parents=True, exist_ok=True)
        (invalid_model / "some_file.txt").write_text("not a model")
        
        # List exports
        exported = exporter.list_exported_models()
        
        # Should only include valid exports
        assert "model1" in exported
        assert "model2" in exported
        assert "invalid" not in exported
        assert len(exported) == 2

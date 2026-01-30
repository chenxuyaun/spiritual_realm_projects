"""
Unit tests for model export utility.

Tests specific examples and edge cases for the export_to_openvino script.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Check if OpenVINO is available
try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
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


class TestModelExporter:
    """Unit tests for ModelExporter class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix="export_test_")
        yield temp_dir
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def exporter(self, temp_dir):
        """Create ModelExporter instance."""
        return ModelExporter(output_base_dir=temp_dir, verbose=False)
    
    @pytest.fixture
    def mock_model_files(self):
        """Helper to create mock model files."""
        def _create(path: Path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "openvino_model.xml").write_text('<?xml version="1.0"?><net></net>')
            (path / "openvino_model.bin").write_bytes(b"data" * 100)
            (path / "tokenizer_config.json").write_text('{}')
            (path / "vocab.json").write_text('{}')
            (path / "config.json").write_text('{}')
            return path
        return _create
    
    def test_initialization_with_default_params(self):
        """Test ModelExporter initialization with defaults."""
        exporter = ModelExporter()
        
        assert exporter.output_base_dir == Path("models/openvino")
        assert exporter.SUPPORTED_PRECISIONS == ["FP32", "FP16", "INT8"]
    
    def test_initialization_with_custom_params(self, temp_dir):
        """Test ModelExporter initialization with custom parameters."""
        exporter = ModelExporter(output_base_dir=temp_dir, verbose=True)
        
        assert exporter.output_base_dir == Path(temp_dir)
    
    def test_export_with_fp32_precision(self, exporter, temp_dir, mock_model_files):
        """Test export with FP32 precision."""
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            mock_export.side_effect = lambda s, t, p: mock_model_files(t)
            
            path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                validate=False
            )
            
            assert path.exists()
            assert (path / "openvino_model.xml").exists()
            mock_export.assert_called_once()
            assert mock_export.call_args[0][2] == "FP32"
    
    def test_export_with_fp16_precision(self, exporter, temp_dir, mock_model_files):
        """Test export with FP16 precision."""
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            mock_export.side_effect = lambda s, t, p: mock_model_files(t)
            
            path = exporter.export_model(
                model_name=model_name,
                precision="FP16",
                validate=False
            )
            
            assert path.exists()
            mock_export.assert_called_once()
            assert mock_export.call_args[0][2] == "FP16"
    
    def test_export_with_int8_precision(self, exporter, temp_dir, mock_model_files):
        """Test export with INT8 precision (should warn about calibration)."""
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            mock_export.side_effect = lambda s, t, p: mock_model_files(t)
            
            path = exporter.export_model(
                model_name=model_name,
                precision="INT8",
                validate=False
            )
            
            assert path.exists()
            mock_export.assert_called_once()
    
    def test_export_with_invalid_precision_raises_error(self, exporter):
        """Test that invalid precision raises ValueError."""
        with pytest.raises(ValueError, match="Invalid precision"):
            exporter.export_model(
                model_name="test-model",
                precision="FP64",
                validate=False
            )
    
    def test_export_with_custom_output_path(self, exporter, temp_dir, mock_model_files):
        """Test export with custom output path."""
        model_name = "test-model"
        custom_path = Path(temp_dir) / "custom" / "path"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            mock_export.side_effect = lambda s, t, p: mock_model_files(t)
            
            path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                output_path=str(custom_path),
                validate=False
            )
            
            assert path == custom_path
            assert path.exists()
    
    def test_export_creates_directory_structure(self, exporter, temp_dir, mock_model_files):
        """Test that export creates necessary directory structure."""
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            mock_export.side_effect = lambda s, t, p: mock_model_files(t)
            
            path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                validate=False
            )
            
            assert path.parent.exists()
            assert path.exists()
            assert path.is_dir()
    
    def test_validation_detects_missing_xml_file(self, exporter, temp_dir):
        """Test that validation detects missing XML file."""
        model_name = "test-model"
        export_path = Path(temp_dir) / model_name
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Create only BIN file, missing XML
        (export_path / "openvino_model.bin").write_bytes(b"data")
        
        with pytest.raises(RuntimeError, match="Missing XML file|Validation failed"):
            exporter._validate_export(export_path, model_name)
    
    def test_validation_detects_missing_bin_file(self, exporter, temp_dir):
        """Test that validation detects missing BIN file."""
        model_name = "test-model"
        export_path = Path(temp_dir) / model_name
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Create only XML file, missing BIN
        (export_path / "openvino_model.xml").write_text("<net></net>")
        
        with pytest.raises(RuntimeError, match="Missing BIN file|Validation failed"):
            exporter._validate_export(export_path, model_name)
    
    def test_export_all_with_default_models(self, exporter, temp_dir, mock_model_files):
        """Test export_all with default model list."""
        with patch.object(exporter, 'export_model') as mock_export:
            mock_export.side_effect = lambda name, **kwargs: mock_model_files(
                Path(temp_dir) / name
            )
            
            paths = exporter.export_all(precision="FP32")
            
            # Should export default models
            assert len(paths) == len(ModelExporter.DEFAULT_MODELS)
            assert mock_export.call_count == len(ModelExporter.DEFAULT_MODELS)
    
    def test_export_all_with_custom_models(self, exporter, temp_dir, mock_model_files):
        """Test export_all with custom model list."""
        custom_models = ["model1", "model2", "model3"]
        
        with patch.object(exporter, 'export_model') as mock_export:
            mock_export.side_effect = lambda name, **kwargs: mock_model_files(
                Path(temp_dir) / name
            )
            
            paths = exporter.export_all(precision="FP32", models=custom_models)
            
            assert len(paths) == len(custom_models)
            assert mock_export.call_count == len(custom_models)
    
    def test_export_all_handles_failures_gracefully(self, exporter, temp_dir):
        """Test that export_all continues on failures."""
        models = ["model1", "model2", "model3"]
        
        with patch.object(exporter, 'export_model') as mock_export:
            # Make second model fail
            def side_effect(name, **kwargs):
                if name == "model2":
                    raise RuntimeError("Export failed")
                return Path(temp_dir) / name
            
            mock_export.side_effect = side_effect
            
            paths = exporter.export_all(precision="FP32", models=models)
            
            # Should have 2 successful exports
            assert len(paths) == 2
            assert mock_export.call_count == 3
    
    def test_list_exported_models_empty_directory(self, exporter):
        """Test list_exported_models with empty directory."""
        exported = exporter.list_exported_models()
        assert exported == []
    
    def test_list_exported_models_with_valid_exports(self, exporter, temp_dir, mock_model_files):
        """Test list_exported_models with valid exports."""
        # Create some valid exports
        mock_model_files(Path(temp_dir) / "model1")
        mock_model_files(Path(temp_dir) / "model2")
        
        exported = exporter.list_exported_models()
        
        assert len(exported) == 2
        assert "model1" in exported
        assert "model2" in exported
    
    def test_list_exported_models_ignores_invalid_directories(
        self,
        exporter,
        temp_dir,
        mock_model_files
    ):
        """Test that list_exported_models ignores invalid directories."""
        # Create valid export
        mock_model_files(Path(temp_dir) / "valid_model")
        
        # Create invalid directory (no XML file)
        invalid_dir = Path(temp_dir) / "invalid_model"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        (invalid_dir / "some_file.txt").write_text("not a model")
        
        exported = exporter.list_exported_models()
        
        assert len(exported) == 1
        assert "valid_model" in exported
        assert "invalid_model" not in exported
    
    def test_export_with_model_path_parameter(self, exporter, temp_dir, mock_model_files):
        """Test export with custom model_path parameter."""
        model_name = "test-model"
        custom_model_path = "/custom/path/to/model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            mock_export.side_effect = lambda s, t, p: mock_model_files(t)
            
            path = exporter.export_model(
                model_name=model_name,
                precision="FP32",
                model_path=custom_model_path,
                validate=False
            )
            
            # Should use custom model path as source
            assert mock_export.call_args[0][0] == custom_model_path
    
    def test_export_without_validation(self, exporter, temp_dir, mock_model_files):
        """Test export with validation disabled."""
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            with patch.object(exporter, '_validate_export') as mock_validate:
                mock_export.side_effect = lambda s, t, p: mock_model_files(t)
                
                exporter.export_model(
                    model_name=model_name,
                    precision="FP32",
                    validate=False
                )
                
                # Validation should not be called
                mock_validate.assert_not_called()
    
    def test_export_with_validation(self, exporter, temp_dir, mock_model_files):
        """Test export with validation enabled."""
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            with patch.object(exporter, '_validate_export') as mock_validate:
                mock_export.side_effect = lambda s, t, p: mock_model_files(t)
                
                exporter.export_model(
                    model_name=model_name,
                    precision="FP32",
                    validate=True
                )
                
                # Validation should be called
                mock_validate.assert_called_once()
    
    def test_export_failure_raises_runtime_error(self, exporter):
        """Test that export failures raise RuntimeError."""
        model_name = "test-model"
        
        with patch.object(exporter, '_export_with_optimum') as mock_export:
            mock_export.side_effect = Exception("Export failed")
            
            with pytest.raises(RuntimeError, match="Export failed"):
                exporter.export_model(
                    model_name=model_name,
                    precision="FP32",
                    validate=False
                )
    
    def test_supported_precisions_constant(self):
        """Test that SUPPORTED_PRECISIONS constant is correct."""
        assert ModelExporter.SUPPORTED_PRECISIONS == ["FP32", "FP16", "INT8"]
    
    def test_default_models_constant(self):
        """Test that DEFAULT_MODELS constant is defined."""
        assert hasattr(ModelExporter, 'DEFAULT_MODELS')
        assert isinstance(ModelExporter.DEFAULT_MODELS, list)
        assert len(ModelExporter.DEFAULT_MODELS) > 0

"""
Property-based tests for pre-load file validation.

Feature: openvino-backend-integration
Property 13: Pre-Load File Validation

This module tests that the OpenVINO backend properly validates model files
before attempting to load them, providing clear error messages when files
are missing or invalid.

Requirements tested:
    - 9.3: Check model XML and BIN files exist before loading
    - 9.3: Provide clear error messages for missing files
    - 9.3: Include troubleshooting suggestions in errors
"""

import os
import tempfile
import shutil
from hypothesis import given, strategies as st, settings, assume
import pytest

from mm_orch.runtime.openvino_backend import OpenVINOBackend
from mm_orch.runtime.backend_exceptions import FileValidationError


# Feature: openvino-backend-integration, Property 13: Pre-Load File Validation
# For any model load request, the system should validate that required model files
# exist before attempting to load them, and should provide clear error messages
# if files are missing.


@settings(max_examples=100, deadline=None)
@given(
    model_name=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=ord('a')),
        min_size=1,
        max_size=20
    ),
    missing_file=st.sampled_from(['directory', 'xml', 'bin', 'both', 'empty_xml', 'empty_bin'])
)
def test_file_validation_detects_missing_files(model_name, missing_file):
    """
    Property: For any model name and any type of missing/invalid file,
    the validation should detect the issue and raise FileValidationError
    with helpful troubleshooting information.
    
    This test verifies that:
    1. Missing directories are detected
    2. Missing XML files are detected
    3. Missing BIN files are detected
    4. Empty files are detected
    5. Error messages contain troubleshooting steps
    6. Error messages mention the export script
    """
    # Filter out invalid model names
    assume(model_name.strip() != '')
    assume(not model_name.startswith('.'))
    assume('/' not in model_name and '\\' not in model_name)
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'models', model_name)
        openvino_path = os.path.join(temp_dir, 'models', 'openvino', model_name)
        
        # Create backend with fallback disabled to test validation
        config = {'enable_fallback': False}
        backend = OpenVINOBackend('CPU', config)
        
        # Setup the scenario based on missing_file parameter
        if missing_file == 'directory':
            # Don't create the directory at all
            pass
        
        elif missing_file == 'xml':
            # Create directory and BIN file, but not XML
            os.makedirs(openvino_path, exist_ok=True)
            bin_file = os.path.join(openvino_path, 'openvino_model.bin')
            with open(bin_file, 'wb') as f:
                f.write(b'dummy weights data')
        
        elif missing_file == 'bin':
            # Create directory and XML file, but not BIN
            os.makedirs(openvino_path, exist_ok=True)
            xml_file = os.path.join(openvino_path, 'openvino_model.xml')
            with open(xml_file, 'w') as f:
                f.write('<net>dummy model</net>')
        
        elif missing_file == 'both':
            # Create directory but no files
            os.makedirs(openvino_path, exist_ok=True)
        
        elif missing_file == 'empty_xml':
            # Create both files but XML is empty
            os.makedirs(openvino_path, exist_ok=True)
            xml_file = os.path.join(openvino_path, 'openvino_model.xml')
            bin_file = os.path.join(openvino_path, 'openvino_model.bin')
            with open(xml_file, 'w') as f:
                pass  # Empty file
            with open(bin_file, 'wb') as f:
                f.write(b'dummy weights')
        
        elif missing_file == 'empty_bin':
            # Create both files but BIN is empty
            os.makedirs(openvino_path, exist_ok=True)
            xml_file = os.path.join(openvino_path, 'openvino_model.xml')
            bin_file = os.path.join(openvino_path, 'openvino_model.bin')
            with open(xml_file, 'w') as f:
                f.write('<net>dummy</net>')
            with open(bin_file, 'wb') as f:
                pass  # Empty file
        
        # Attempt validation - should raise FileValidationError
        with pytest.raises(FileValidationError) as exc_info:
            backend._validate_model_files(model_name, openvino_path)
        
        error_message = str(exc_info.value)
        
        # Verify error message contains helpful information
        # All error messages should mention the export script
        assert 'export_to_openvino.py' in error_message, \
            "Error message should mention the export script"
        
        # All error messages should contain troubleshooting steps
        assert 'Troubleshooting' in error_message or 'troubleshooting' in error_message, \
            "Error message should contain troubleshooting steps"
        
        # Verify specific error messages based on scenario
        if missing_file == 'directory':
            assert 'directory not found' in error_message.lower(), \
                "Should indicate directory is missing"
        
        elif missing_file == 'xml':
            assert 'xml' in error_message.lower(), \
                "Should indicate XML file is missing"
        
        elif missing_file == 'bin':
            assert 'bin' in error_message.lower() or 'weights' in error_message.lower(), \
                "Should indicate BIN/weights file is missing"
        
        elif missing_file in ['empty_xml', 'empty_bin']:
            assert 'empty' in error_message.lower() or 'zero bytes' in error_message.lower(), \
                "Should indicate file is empty"


@settings(max_examples=50, deadline=None)
@given(
    model_name=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=ord('a')),
        min_size=1,
        max_size=20
    ),
    xml_size=st.integers(min_value=10, max_value=1000),
    bin_size=st.integers(min_value=100, max_value=10000)
)
def test_file_validation_passes_with_valid_files(model_name, xml_size, bin_size):
    """
    Property: For any model name with valid (non-empty) XML and BIN files,
    the validation should pass without raising exceptions.
    
    This test verifies that:
    1. Valid files pass validation
    2. No false positives for valid configurations
    3. File size validation works correctly
    """
    # Filter out invalid model names
    assume(model_name.strip() != '')
    assume(not model_name.startswith('.'))
    assume('/' not in model_name and '\\' not in model_name)
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        openvino_path = os.path.join(temp_dir, 'models', 'openvino', model_name)
        os.makedirs(openvino_path, exist_ok=True)
        
        # Create valid XML file
        xml_file = os.path.join(openvino_path, 'openvino_model.xml')
        with open(xml_file, 'w') as f:
            f.write('<net>' + 'x' * xml_size + '</net>')
        
        # Create valid BIN file
        bin_file = os.path.join(openvino_path, 'openvino_model.bin')
        with open(bin_file, 'wb') as f:
            f.write(b'W' * bin_size)
        
        # Create backend
        config = {'enable_fallback': False}
        backend = OpenVINOBackend('CPU', config)
        
        # Validation should pass without raising exceptions
        try:
            backend._validate_model_files(model_name, openvino_path)
            # If we get here, validation passed (which is expected)
            assert True
        except FileValidationError as e:
            # Validation should not fail for valid files
            pytest.fail(f"Validation failed for valid files: {e}")


@settings(max_examples=50, deadline=None)
@given(
    model_name=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=ord('a')),
        min_size=1,
        max_size=20
    )
)
def test_error_messages_contain_model_name(model_name):
    """
    Property: For any model name, error messages should reference the specific
    model name to help users identify which model has the problem.
    
    This test verifies that:
    1. Error messages are model-specific
    2. Users can identify which model failed
    3. Export commands reference the correct model
    """
    # Filter out invalid model names
    assume(model_name.strip() != '')
    assume(not model_name.startswith('.'))
    assume('/' not in model_name and '\\' not in model_name)
    
    # Create temporary directory (but no model files)
    with tempfile.TemporaryDirectory() as temp_dir:
        openvino_path = os.path.join(temp_dir, 'models', 'openvino', model_name)
        
        # Create backend
        config = {'enable_fallback': False}
        backend = OpenVINOBackend('CPU', config)
        
        # Attempt validation - should raise FileValidationError
        with pytest.raises(FileValidationError) as exc_info:
            backend._validate_model_files(model_name, openvino_path)
        
        error_message = str(exc_info.value)
        
        # Error message should contain the model name
        # (either in the path or in the export command)
        assert model_name in error_message, \
            f"Error message should reference model name '{model_name}'"


def test_validation_error_hierarchy():
    """
    Test that FileValidationError is properly categorized in the exception hierarchy.
    
    This ensures that error handling code can catch validation errors specifically
    or catch all backend errors generically.
    """
    from mm_orch.runtime.backend_exceptions import (
        BackendError,
        ModelLoadError,
        FileValidationError
    )
    
    # FileValidationError should be a subclass of ModelLoadError
    assert issubclass(FileValidationError, ModelLoadError), \
        "FileValidationError should inherit from ModelLoadError"
    
    # ModelLoadError should be a subclass of BackendError
    assert issubclass(ModelLoadError, BackendError), \
        "ModelLoadError should inherit from BackendError"
    
    # Therefore FileValidationError should also be a BackendError
    assert issubclass(FileValidationError, BackendError), \
        "FileValidationError should inherit from BackendError"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

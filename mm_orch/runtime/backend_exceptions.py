"""
Custom exceptions for backend operations.

This module defines a hierarchy of exceptions for different types of backend errors,
making it easier to handle and categorize errors appropriately.
"""


class BackendError(Exception):
    """
    Base exception for all backend-related errors.
    
    All backend-specific exceptions inherit from this class, allowing
    for easy catching of any backend error.
    """
    pass


class ConfigurationError(BackendError):
    """
    Raised when backend configuration is invalid or malformed.
    
    Examples:
        - Invalid backend name
        - Invalid device name
        - Malformed configuration file
        - Missing required configuration fields
    """
    pass


class BackendInitializationError(BackendError):
    """
    Raised when backend initialization fails.
    
    Examples:
        - Backend library not installed
        - Backend initialization failure
        - Device allocation failure
        - Incompatible backend version
    """
    pass


class ModelLoadError(BackendError):
    """
    Raised when model loading fails.
    
    Examples:
        - Model files not found
        - Model format incompatible
        - Insufficient memory
        - Model corruption
    """
    pass


class InferenceError(BackendError):
    """
    Raised when inference execution fails.
    
    Examples:
        - Invalid input format
        - Model execution failure
        - Tokenization errors
        - Output decoding errors
    """
    pass


class DeviceError(BackendError):
    """
    Raised when device-related operations fail.
    
    Examples:
        - Device not available
        - Device allocation failure
        - Device compatibility issues
    """
    pass


class FileValidationError(ModelLoadError):
    """
    Raised when model file validation fails.
    
    Examples:
        - Required model files missing
        - Model files corrupted
        - Incomplete model export
    """
    pass

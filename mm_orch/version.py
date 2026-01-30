"""
Version information for MuAI Orchestration System.

This module provides version information that can be imported by other modules.
"""

__version__ = "1.0.0"
__version_info__ = (1, 0, 0)

# Release information
RELEASE_NAME = "First Stable Release"
RELEASE_DATE = "2026-01-30"
RELEASE_STATUS = "stable"  # alpha, beta, rc, stable

# Build information
BUILD_NUMBER = "20260130"
GIT_COMMIT = ""  # Will be filled by CI/CD

# Compatibility
MIN_PYTHON_VERSION = (3, 8)
RECOMMENDED_PYTHON_VERSION = (3, 9)

# Feature flags
FEATURES = {
    "multi_workflow_orchestration": True,
    "consciousness_modules": True,
    "real_model_integration": True,
    "advanced_optimization": True,
    "openvino_backend": True,
    "phase_b_architecture": True,
    "router_v3": True,
    "router_v3_mode_chat": True,  # NEW: Verified and working
    "structured_lesson_output": True,  # NEW: MVP implemented
}


def get_version() -> str:
    """
    Get the version string.
    
    Returns:
        Version string (e.g., "1.0.0-rc1")
    """
    return __version__


def get_version_info() -> tuple:
    """
    Get the version information tuple.
    
    Returns:
        Version tuple (e.g., (1, 0, 0, "rc1"))
    """
    return __version_info__


def get_full_version() -> str:
    """
    Get the full version string with build information.
    
    Returns:
        Full version string (e.g., "1.0.0-rc1+20260130")
    """
    version = __version__
    if BUILD_NUMBER:
        version += f"+{BUILD_NUMBER}"
    if GIT_COMMIT:
        version += f".{GIT_COMMIT[:7]}"
    return version


def is_feature_enabled(feature: str) -> bool:
    """
    Check if a feature is enabled.
    
    Args:
        feature: Feature name
        
    Returns:
        True if feature is enabled, False otherwise
    """
    return FEATURES.get(feature, False)


def get_release_info() -> dict:
    """
    Get release information.
    
    Returns:
        Dictionary with release information
    """
    return {
        "version": __version__,
        "version_info": __version_info__,
        "release_name": RELEASE_NAME,
        "release_date": RELEASE_DATE,
        "release_status": RELEASE_STATUS,
        "build_number": BUILD_NUMBER,
        "git_commit": GIT_COMMIT,
        "features": FEATURES,
    }


# Compatibility check
def check_python_version() -> bool:
    """
    Check if the current Python version is compatible.
    
    Returns:
        True if compatible, False otherwise
    """
    import sys
    
    current_version = sys.version_info[:2]
    return current_version >= MIN_PYTHON_VERSION


if __name__ == "__main__":
    # Print version information when run as script
    import json
    
    print("MuAI Orchestration System")
    print("=" * 50)
    print(f"Version: {get_version()}")
    print(f"Full Version: {get_full_version()}")
    print(f"Release Name: {RELEASE_NAME}")
    print(f"Release Date: {RELEASE_DATE}")
    print(f"Release Status: {RELEASE_STATUS}")
    print(f"Build Number: {BUILD_NUMBER}")
    print()
    print("Features:")
    print(json.dumps(FEATURES, indent=2))
    print()
    print(f"Python Version Check: {'✅ Compatible' if check_python_version() else '❌ Incompatible'}")

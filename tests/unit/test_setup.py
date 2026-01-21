"""
Basic setup verification test.
"""
import pytest


def test_python_version():
    """Verify Python version is 3.8+"""
    import sys
    assert sys.version_info >= (3, 8), "Python 3.8+ is required"


def test_imports():
    """Verify core package can be imported"""
    import mm_orch
    assert mm_orch.__version__ == "0.1.0"


def test_hypothesis_configuration():
    """Verify Hypothesis is configured correctly"""
    from hypothesis import settings
    profile = settings.get_profile("default")
    assert profile is not None

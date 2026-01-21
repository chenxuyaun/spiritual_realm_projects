"""
Pytest configuration and fixtures for the test suite.
"""
import pytest
from hypothesis import settings, Verbosity

# Configure Hypothesis for property-based testing
# Each property test will run at least 100 examples
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.register_profile("ci", max_examples=200, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=50, verbosity=Verbosity.normal)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

# Load the default profile
settings.load_profile("default")


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests to ensure isolation."""
    # This will be populated as we implement consciousness modules
    yield
    # Cleanup code here if needed

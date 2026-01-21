.PHONY: help install test lint format type-check clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-property - Run property-based tests only"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run flake8 linter"
	@echo "  make format       - Format code with black"
	@echo "  make type-check   - Run mypy type checker"
	@echo "  make clean        - Remove generated files"

install:
	pip install -r requirements.txt

test:
	pytest tests/

test-unit:
	pytest tests/unit/

test-property:
	pytest tests/property/

test-cov:
	pytest --cov=mm_orch --cov-report=html --cov-report=term tests/

lint:
	flake8 mm_orch/ tests/

format:
	black mm_orch/ tests/

type-check:
	mypy mm_orch/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".hypothesis" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage

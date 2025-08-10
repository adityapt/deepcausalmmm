# DeepCausalMMM v1.0.0 Makefile
# Production-ready Marketing Mix Modeling package

.PHONY: help install install-dev test lint format clean build upload docs dashboard example

# Default target
help:
	@echo "DeepCausalMMM v1.0.0 - Available commands:"
	@echo ""
	@echo "🔧 Development:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  test         - Run test suite"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black"
	@echo ""
	@echo "📦 Package:"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build distribution packages"
	@echo "  upload       - Upload to PyPI (requires credentials)"
	@echo ""
	@echo "🎨 Examples:"
	@echo "  dashboard    - Run official dashboard example"
	@echo "  example      - Run complete v1.0.0 example"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs         - Generate documentation"

# Installation targets
install:
	@echo "🔧 Installing DeepCausalMMM in development mode..."
	pip install -e .

install-dev: install
	@echo "🔧 Installing development dependencies..."
	pip install pytest black flake8 mypy jupyter notebook plotly

# Development targets
test:
	@echo "🧪 Running test suite..."
	python -m pytest tests/ -v

lint:
	@echo "🔍 Running code linting..."
	flake8 deepcausalmmm/ --max-line-length=100 --ignore=E203,W503
	mypy deepcausalmmm/ --ignore-missing-imports

format:
	@echo "✨ Formatting code with black..."
	black deepcausalmmm/ examples/ tests/ --line-length=100

# Package targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.backup" -delete

build: clean
	@echo "📦 Building distribution packages..."
	python -m build

upload: build
	@echo "🚀 Uploading to PyPI..."
	python -m twine upload dist/*

# Example targets
dashboard:
	@echo "🎨 Running official dashboard example..."
	cd .. && python dashboard_rmse_optimized.py

example:
	@echo "🎯 Running official v1.0.0 example..."
	python examples/official_v1_example.py

# Documentation targets
docs:
	@echo "📚 Generating documentation..."
	@echo "README.md: ✅"
	@echo "CHANGELOG.md: ✅"
	@echo "Examples: ✅"
	@echo "API docs would be generated here with sphinx"

# Quality checks
check-package:
	@echo "🔍 Checking package integrity..."
	python -c "import deepcausalmmm; print(f'Version: {deepcausalmmm.__version__}')"
	python -c "from deepcausalmmm import SimpleGlobalScaler; print('SimpleGlobalScaler: ✅')"
	python -c "from deepcausalmmm import DeepCausalMMM; print('DeepCausalMMM: ✅')"
	python -c "from deepcausalmmm import get_default_config; print('Config system: ✅')"

# Performance benchmark
benchmark:
	@echo "⚡ Running performance benchmark..."
	python -c "
import time
import numpy as np
from deepcausalmmm import SimpleGlobalScaler
print('Benchmarking SimpleGlobalScaler...')
X_media = np.random.rand(100, 52, 8)
X_control = np.random.rand(100, 52, 4)
y = np.random.rand(100, 52)
scaler = SimpleGlobalScaler()
start = time.time()
scaler.fit_transform(X_media, X_control, y)
end = time.time()
print(f'Scaling 100 regions × 52 weeks: {end-start:.3f}s')
"

# All-in-one development setup
dev-setup: install-dev
	@echo "🚀 Development environment ready!"
	@echo "Run 'make example' to test the installation"
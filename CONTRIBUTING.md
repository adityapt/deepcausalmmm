# Contributing to DeepCausalMMM

Thank you for your interest in contributing to DeepCausalMMM! This document provides guidelines for contributing to our advanced Media Mix Modeling package.

## Project Philosophy

### **Zero Hardcoding Principle**
- **All parameters must be configurable** via `config.py`
- **No magic numbers** in model code
- **Dataset agnostic** - works on any MMM dataset
- **Learnable parameters** preferred over fixed constants

### **Performance First**
- **Proven configurations** should not be changed without extensive testing
- **Benchmark against baseline** before and after changes
- **Maintain generalization** with proper train/holdout validation
- **Document performance impact** in pull requests

### **Code Quality Standards**
- **Type hints** for all function parameters and returns
- **Comprehensive docstrings** with examples
- **Error handling** with informative messages
- **Modular design** with clear separation of concerns

## Getting Started

### Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/adityapt/deepcausalmmm.git
cd deepcausalmmm
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**
```bash
pip install -e .
# Or install with development dependencies
pip install -e .[dev]
```

4. **Verify installation**
```bash
python -c "from deepcausalmmm import DeepCausalMMM, get_device; print('Package installed successfully!')"
```

5. **Run tests to ensure setup**
```bash
python -m pytest tests/ -v
```

## Project Structure

```
deepcausalmmm/                      # Project root
├── pyproject.toml                  # Package configuration and dependencies
├── README.md                       # Documentation
├── LICENSE                         # MIT License
├── CHANGELOG.md                    # Version history and changes
├── CONTRIBUTING.md                 # Development guidelines (this file)
├── CODE_OF_CONDUCT.md              # Code of conduct
├── CITATION.cff                    # Citation metadata for Zenodo/GitHub
├── Makefile                        # Build and development tasks
├── MANIFEST.in                     # Package manifest for distribution
│
├── deepcausalmmm/                  # Main package directory
│   ├── __init__.py                 # Package initialization and exports
│   ├── cli.py                      # Command-line interface
│   ├── exceptions.py               # Custom exception classes
│   │
│   ├── core/                       #  CRITICAL: Core model components
│   │   ├── __init__.py            # Core module initialization
│   │   ├── config.py              #  CRITICAL: All configurations
│   │   ├── unified_model.py       #  CRITICAL: Main model architecture
│   │   ├── trainer.py             # ModelTrainer class for training
│   │   ├── data.py                # UnifiedDataPipeline for data processing
│   │   ├── scaling.py             # Linear scaling (y/y_mean) for data normalization
│   │   ├── seasonality.py         # Seasonal decomposition utilities
│   │   ├── dag_model.py           # DAG learning and causal inference
│   │   ├── inference.py           # Model inference and prediction
│   │   ├── train_model.py         # Training functions and utilities
│   │   └── visualization.py       # Core visualization components
│   │
│   ├── postprocess/                # Analysis and post-processing
│   │   ├── __init__.py            # Postprocess module initialization
│   │   ├── analysis.py            # Statistical analysis utilities
│   │   ├── comprehensive_analysis.py  # Comprehensive analyzer
│   │   ├── response_curves.py     # Non-linear response curve fitting (Hill equations)
│   │   ├── optimization.py        # Budget optimization with response curves
│   │   ├── optimization_utils.py  # Optimization utility functions
│   │   └── dag_postprocess.py     # DAG post-processing and analysis
│   │
│   └── utils/                      # Utility functions
│       ├── __init__.py            # Utils module initialization
│       ├── device.py              # GPU/CPU device detection
│       └── data_generator.py      # Synthetic data generation (ConfigurableDataGenerator)
│
├── examples/                       # Example scripts and notebooks
│   ├── quickstart.ipynb           # Interactive Jupyter notebook for Google Colab
│   ├── dashboard_rmse_optimized.py # Comprehensive dashboard with 14+ visualizations
│   ├── example_response_curves.py  # Response curve fitting examples
│   └── example_budget_optimization.py  # Budget optimization workflow
│
├── tests/                          # Test suite
│   ├── __init__.py                # Test package initialization
│   ├── unit/                      # Unit tests
│   │   ├── __init__.py
│   │   ├── test_config.py         # Configuration tests
│   │   ├── test_model.py          # Model architecture tests
│   │   ├── test_scaling.py        # Data scaling tests
│   │   └── test_response_curves.py # Response curve fitting tests
│   └── integration/               # Integration tests
│       ├── __init__.py
│       └── test_end_to_end.py     # End-to-end integration tests
│
├── docs/                           # Documentation
│   ├── Makefile                   # Documentation build tasks
│   ├── make.bat                   # Windows documentation build
│   ├── requirements.txt           # Documentation dependencies
│   └── source/                    # Sphinx documentation source
│       ├── conf.py               # Sphinx configuration
│       ├── index.rst             # Documentation index
│       ├── installation.rst      # Installation guide
│       ├── quickstart.rst        # Quick start guide
│       ├── contributing.rst      # Contributing guide
│       └── api/                  # API documentation
│           ├── index.rst
│           ├── core.rst
│           ├── data.rst
│           ├── trainer.rst
│           ├── inference.rst
│           ├── analysis.rst
│           ├── response_curves.rst # Response curves API
│           ├── optimization.rst    # Budget optimization API
│           ├── utils.rst
│           └── exceptions.rst
│
└── JOSS/                           # Journal of Open Source Software submission
    ├── paper.md                   # JOSS paper manuscript
    ├── paper.bib                  # Bibliography
    ├── figure_dag_professional.png # DAG visualization figure
    └── figure_response_curve_simple.png # Response curve figure
```

## Development Guidelines

### **1. Configuration Management**

** DO:**
```python
# In unified_model.py
def __init__(self, config: Dict[str, Any]):
    self.hidden_dim = config['hidden_dim']
    self.dropout = config['dropout']
    self.gru_layers = config['gru_layers']
```

** DON'T:**
```python
# NEVER hardcode values
def __init__(self):
    self.hidden_dim = 320  # HARDCODED
    self.dropout = 0.08    # HARDCODED
```

### **2. Model Architecture Changes**

**Before making changes to core model:**
1. **Benchmark current performance** on test dataset
2. **Create feature branch** with descriptive name
3. **Implement with config parameters** - no hardcoding
4. **Test extensively** on multiple datasets
5. **Document performance impact**

**Critical Files (Require Extra Care):**
- `core/unified_model.py` - Main model architecture
- `core/config.py` - All configurations
- `core/trainer.py` - Training optimization

### **3. Adding New Features**

**Feature Development Process:**
1. **Create issue** describing the feature and motivation
2. **Design with configurability** in mind
3. **Add config parameters** with sensible defaults
4. **Implement with type hints** and docstrings
5. **Add comprehensive tests**
6. **Update documentation**
7. **Benchmark performance impact**

**Example: Adding New Loss Function**
```python
# 1. Add to config.py
'use_focal_loss': False,
'focal_alpha': 0.25,
'focal_gamma': 2.0,

# 2. Implement in trainer.py with config
def _calculate_loss(self, predictions, targets):
    if self.config.get('use_focal_loss', False):
        return self._focal_loss(predictions, targets)
    return self._huber_loss(predictions, targets)

# 3. Add tests
def test_focal_loss_calculation():
    # Test implementation
```

### **4. Data Processing**

**Scaling and Normalization Rules:**
- **Media variables**: SOV (Share-of-Voice) scaling
- **Control variables**: Z-score normalization  
- **Seasonality**: Min-Max scaling per region (0-1)
- **Target variable**: Linear scaling (y/y_mean per region) - **Changed in v1.0.19**

** DO:**
- Use `UnifiedDataPipeline` for all data processing
- Ensure consistent train/holdout transformations
- Document scaling choices with business justification

** DON'T:**
- Apply different scaling to train/holdout
- Use hardcoded scaling parameters
- Skip data validation steps

### **5. Testing Requirements**

**All contributions must include:**
1. **Unit tests** for new functions/classes
2. **Integration tests** for feature workflows
3. **Performance regression tests**
4. **Documentation tests** (docstring examples)

**Test Categories:**
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests  
python -m pytest tests/integration/

# Performance tests
python -m pytest tests/performance/

# All tests
python -m pytest tests/ -v --cov=deepcausalmmm
```

### **6. Documentation Standards**

**All functions must have comprehensive docstrings:**
```python
def calculate_contributions(
    self, 
    media_data: torch.Tensor, 
    coefficients: torch.Tensor,
    transform_back: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Calculate economic contributions for media channels.
    
    Args:
        media_data: Media spend data [regions, time, channels]
        coefficients: Learned coefficients [regions, time, channels]
        transform_back: Whether to apply inverse scaling transform
        
    Returns:
        Dict containing:
            - 'total_contributions': Total economic impact per channel
            - 'regional_contributions': Contributions by region
            - 'temporal_contributions': Contributions over time
            
    Example:
        >>> contributions = model.calculate_contributions(
        ...     media_data=processed_media,
        ...     coefficients=learned_coeffs,
        ...     transform_back=True
        ... )
        >>> print(f"Total impact: {contributions['total_contributions'].sum()}")
    """
```

## Performance Standards

### **Benchmark Requirements**

**Performance Standards (v1.0.19 Benchmarks):**
- **Training R²**: ≥ 0.93 (baseline achieved: 0.950)
- **Holdout R²**: ≥ 0.80 (baseline achieved: 0.842)
- **Performance Gap**: ≤ 15% relative to training (baseline: 10.8 percentage points)
- **Training Stability**: No coefficient explosion or divergence
- **Attribution Quality**: Components sum to 100% with <5% error
- **Business Logic**: Contributions should be realistic and interpretable (e.g., media 30-50%)

**Before Merging:**
1. **Run full benchmark** on standard dataset
2. **Compare against baseline** performance
3. **Document any performance changes**
4. **Get approval** for performance degradation >2%

### **Performance Testing**
```python
# Example performance test
def test_model_performance_regression():
    """Ensure new changes don't degrade performance."""
    config = get_config()
    
    # Load standard test dataset
    data = load_benchmark_data()
    
    # Train model
    results = train_and_evaluate(config, data)
    
    # Assert performance thresholds (v1.0.19 standards)
    assert results['holdout_r2'] >= 0.80  # Target: 0.80+
    assert results['performance_gap_pct'] <= 15.0  # Gap ≤ 15%
    assert results['attribution_additivity_error'] < 0.05  # <5% error
```

## Pull Request Process

### **1. Pre-submission Checklist**
- [ ] **Code follows zero-hardcoding principle**
- [ ] **All tests pass** (`python -m pytest tests/`)
- [ ] **Performance benchmarks meet thresholds**
- [ ] **Documentation updated** (README, docstrings)
- [ ] **Type hints added** to new functions
- [ ] **Config parameters added** for new features
- [ ] **No linting errors** (`flake8 deepcausalmmm/`)

### **2. PR Description Template**
```markdown
## Purpose
Brief description of what this PR accomplishes.

## Changes Made
- List of specific changes
- New features added
- Bug fixes included

## Performance Impact
- Benchmark results before/after
- Any performance changes documented
- Justification for performance changes

## Testing
- Tests added/modified
- Test coverage maintained
- Performance regression tests included

## Documentation
- README updated if needed
- Docstrings added/updated
- Examples provided for new features

## Breaking Changes
- Any breaking changes listed
- Migration guide provided if needed
```

### **3. Review Process**
1. **Automated checks** must pass (tests, linting, performance)
2. **Code review** by maintainers
3. **Performance review** if core components changed
4. **Documentation review** for clarity and completeness
5. **Final approval** and merge

## Bug Reports

### **Issue Template**
```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen.

## Actual Behavior  
What actually happens.

## Environment
- Python version:
- PyTorch version:
- Operating System:
- DeepCausalMMM version:

## Additional Context
- Error messages
- Stack traces
- Sample data (if applicable)
```

## Feature Requests

### **Enhancement Template**
```markdown
## Feature Description
Clear description of the proposed feature.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Implementation
High-level approach to implementing the feature.

## Performance Considerations
Any potential impact on model performance.

## Related Issues
Links to related issues or discussions.
```

## Code Style

### **Python Style Guidelines**
- **PEP 8** compliance with 100-character line limit
- **Type hints** for all function signatures
- **Descriptive variable names** (no abbreviations)
- **Consistent formatting** using `black` formatter

### **Naming Conventions**
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`  
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### **Import Organization**
```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Local imports
from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.utils.device import get_device
```

## Recognition

Contributors will be recognized in:
- **CHANGELOG.md** for each release
- **README.md** contributors section
- **GitHub releases** with contributor highlights

## Community

- **Be respectful** and constructive in discussions
- **Help others** learn and contribute
- **Share knowledge** through documentation and examples
- **Celebrate successes** and learn from failures

---

Thank you for contributing to DeepCausalMMM! Together we're building the future of Marketing Mix Modeling.

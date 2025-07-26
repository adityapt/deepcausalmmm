# Contributing to DeepCausalMMM

Thank you for your interest in contributing to DeepCausalMMM! This document outlines how you can help improve this deep learning + causal inference framework for Marketing Mix Modeling.

## 🚀 Quick Start

### Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/adityapt/deepcausalmmm.git
cd deepcausalmmm
```

2. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black isort flake8 mypy jupyter matplotlib seaborn
```

3. **Verify Installation**
```bash
python -c "import deepcausalmmm; print('Installation successful!')"
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=deepcausalmmm tests/

# Run specific test categories
pytest tests/test_model.py -v
pytest tests/test_causal.py -v
```

## 📊 Working with Marketing Data

When developing features, use realistic marketing data patterns:

```python
# Example test data structure
marketing_data = {
    'date': pd.date_range('2020-01-01', periods=104, freq='W'),
    'revenue': np.random.lognormal(10, 0.3, 104),
    'tv_spend': np.random.lognormal(8, 0.5, 104),
    'digital_spend': np.random.lognormal(7.5, 0.4, 104),
    'price': np.random.normal(50, 2, 104),
    'region': np.random.choice(['A', 'B', 'C'], 104)
}
```

## 🎯 Areas for Contribution

### High Priority
- **Model Architecture**: Improvements to GRU-based time series modeling
- **Causal Structure**: Enhancements to Bayesian network integration
- **Performance**: Optimization of training speed and memory usage
- **Documentation**: Examples, tutorials, and API documentation

### Medium Priority  
- **Transformations**: New adstock and saturation functions
- **Validation**: Cross-validation and model selection methods
- **Visualization**: Better plotting and analysis tools
- **CLI**: Enhanced command-line interface features

### Research Areas
- **Hierarchical Models**: Multi-region/multi-product modeling
- **Uncertainty Quantification**: Better confidence intervals
- **Causal Discovery**: Automated structure learning

## 🔬 Development Guidelines

### Code Style
```bash
# Format code
black deepcausalmmm/ tests/ examples/
isort deepcausalmmm/ tests/ examples/

# Lint code
flake8 deepcausalmmm/ tests/

# Type checking
mypy deepcausalmmm/
```

### Model Development
- **PyTorch Conventions**: Follow PyTorch best practices for model architecture
- **Reproducibility**: Set random seeds in tests and examples
- **GPU Support**: Ensure code works on both CPU and GPU
- **Memory Efficiency**: Consider memory usage for large datasets

### Testing Marketing Models
- **Synthetic Data**: Create realistic synthetic marketing data for tests
- **Edge Cases**: Test with small datasets, missing data, extreme values
- **Statistical Properties**: Verify model outputs have expected statistical properties
- **Convergence**: Test that training converges properly

## 📈 Performance Considerations

### Benchmarking
When making performance changes, benchmark against:
- **Training Speed**: Time to train on 1000 samples
- **Memory Usage**: Peak memory during training
- **Prediction Speed**: Inference time for batch predictions
- **Model Quality**: R², MAPE, and other accuracy metrics

### Example Benchmark
```python
import time
import torch
from deepcausalmmm import GRUCausalMMM

# Benchmark training time
start_time = time.time()
model = GRUCausalMMM(...)
model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f}s")
```

## 🧬 Causal Inference Guidelines

### Causal Structure
- **Domain Knowledge**: Incorporate marketing domain knowledge in causal graphs
- **Identifiability**: Ensure causal effects are identifiable
- **Validation**: Test causal claims against known ground truth when possible
- **Assumptions**: Clearly document causal assumptions

### Marketing-Specific Considerations
- **Adstock Effects**: Model carryover effects properly
- **Saturation**: Handle diminishing returns correctly
- **Seasonality**: Account for seasonal patterns
- **Media Interactions**: Consider channel interactions

## 📚 Documentation

### Code Documentation
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Add type hints to all functions
- **Examples**: Include usage examples in docstrings

### Example Docstring
```python
def calculate_adstock(media_spend: np.ndarray, 
                     adstock_rate: float = 0.5) -> np.ndarray:
    """Apply adstock transformation to media spending data.
    
    Args:
        media_spend: Array of media spending values over time
        adstock_rate: Adstock decay rate between 0 and 1
        
    Returns:
        Array of adstocked media values
        
    Example:
        >>> spend = np.array([100, 200, 150, 300])
        >>> adstocked = calculate_adstock(spend, adstock_rate=0.3)
        >>> print(adstocked)
        [100.0, 230.0, 219.0, 365.7]
    """
```

### Tutorials
When adding new features, consider creating:
- **Jupyter Notebooks**: Interactive examples
- **CLI Examples**: Command-line usage
- **Real Data Examples**: Using actual marketing datasets

## 🐛 Bug Reports

When reporting bugs, include:
- **Environment**: Python version, PyTorch version, OS
- **Data**: Sample data that reproduces the issue (anonymized)
- **Expected vs Actual**: What you expected vs what happened
- **Code**: Minimal reproducible example
- **Error Messages**: Full traceback

### Bug Report Template
```markdown
## Bug Description
Brief description of the issue

## Environment
- Python version: 
- PyTorch version:
- DeepCausalMMM version:
- OS:

## Reproduction Code
```python
# Minimal code to reproduce the issue
```

## Expected Behavior
What should happen

## Actual Behavior
What actually happens (include error messages)
```

## 🚀 Feature Requests

For new features, please describe:
- **Use Case**: What marketing problem does this solve?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Impact**: Who would benefit from this feature?

## 📊 Data Privacy

### Marketing Data Sensitivity
- **No Real Data**: Never commit actual marketing spend or revenue data
- **Synthetic Examples**: Use synthetic data for examples and tests
- **Anonymization**: If using real data patterns, anonymize completely
- **Documentation**: Mark any sensitive parameters clearly

## ⚡ Performance Testing

### Marketing Model Benchmarks
Test your changes against these benchmarks:

```python
# Standard benchmark dataset sizes
SMALL_DATASET = (52, 5)    # 1 year, 5 channels
MEDIUM_DATASET = (104, 10) # 2 years, 10 channels  
LARGE_DATASET = (208, 15)  # 4 years, 15 channels

# Expected performance (approximate)
# Training time should be < 30s for SMALL_DATASET
# Memory usage should be < 2GB for LARGE_DATASET
# R² should be > 0.8 on synthetic data with known structure
```

## 🤝 Community

### Getting Help
- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For private inquiries (puttaparthy.aditya@gmail.com)

### Code of Conduct
Be respectful, inclusive, and constructive in all interactions. We welcome contributors from all backgrounds and experience levels.

## 🔄 Release Process

### Version Numbering
- **Major**: Breaking API changes (1.0.0 → 2.0.0)
- **Minor**: New features, backward compatible (1.0.0 → 1.1.0)  
- **Patch**: Bug fixes (1.0.0 → 1.0.1)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in setup files
- [ ] Performance benchmarks run
- [ ] Example notebooks tested

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Happy Contributing!** 🎉

Your contributions help advance the state of Marketing Mix Modeling and make better marketing decisions possible for everyone. 

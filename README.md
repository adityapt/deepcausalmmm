# DeepCausalMMM

[![PyPI version](https://badge.fury.io/py/deepcausalmmm.svg)](https://badge.fury.io/py/deepcausalmmm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Deep Learning + Bayesian Networks based causal Marketing Mix Modeling**

A novel approach combining GRU-based time-series modeling with Bayesian networks for causal effect estimation in marketing mix modeling, going beyond classical adstock regressions or purely econometric approaches.

## 🚀 Features

- **GRU-based Time Series Modeling**: Captures complex temporal dependencies in marketing data
- **Bayesian Network Causal Structure**: Incorporates domain knowledge and causal relationships
- **Adstock & Saturation Transformations**: Models media carryover effects and diminishing returns
- **Multi-region Support**: Handles geographic variations in marketing effectiveness
- **Comprehensive Analytics**: Feature importance, contribution analysis, and ROAS calculations
- **Forecasting Capabilities**: Generate predictions with uncertainty quantification
- **Production Ready**: CLI interface, configuration management, and extensive testing

## 🎯 What Makes This Different

Unlike existing causal-ML libraries (EconML, CausalML) that focus on heterogeneous treatment effects, DeepCausalMMM specifically targets **marketing mix modeling** with:

- **Recurrent Neural Networks** for time-varying coefficients
- **Bayesian Network integration** for causal structure learning
- **Marketing-specific transformations** (adstock, Hill saturation)
- **End-to-end MMM pipeline** from data preprocessing to actionable insights

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install deepcausalmmm
```

### From Source

```bash
git clone https://github.com/yourusername/deepcausalmmm.git
cd deepcausalmmm
pip install -e .
```

### Dependencies

- Python 3.8+
- PyTorch >= 1.12.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.5.0
- pgmpy >= 0.1.20 (optional, for Bayesian networks)

## 🚀 Quick Start

### 1. Basic Usage

```python
import pandas as pd
from deepcausalmmm import GRUCausalMMM, prepare_data_for_training, train_model

# Load your marketing data
df = pd.read_csv('marketing_data.csv')

# Configure the model
config = {
    'marketing_vars': ['tv_spend', 'digital_spend', 'radio_spend'],
    'control_vars': ['price', 'promotion'],
    'dependent_var': 'revenue',
    'epochs': 1000,
    'hidden_size': 64
}

# Prepare data
data_dict = prepare_data_for_training(df, config)

# Initialize and train model
model = GRUCausalMMM(
    A_prior=data_dict['media_adjacency'],
    n_media=len(config['marketing_vars']),
    ctrl_dim=len(config['control_vars']),
    hidden=config['hidden_size']
)

results = train_model(model, data_dict, config)
print(f"R² Score: {results['test_metrics']['r2']:.4f}")
```

### 2. Command Line Interface

```bash
# Train a model
deepcausalmmm train data.csv --output-dir results/

# Make predictions
deepcausalmmm predict model.pth test_data.csv --output predictions.csv

# Generate forecasts
deepcausalmmm forecast model.pth data.csv --horizon 12 --output forecasts.csv

# Analyze results
deepcausalmmm analyze model.pth data.csv --output-dir analysis/
```

### 3. Complete Example

```python
from deepcausalmmm import (
    GRUCausalMMM, 
    prepare_data_for_training, 
    train_model_with_validation,
    get_feature_importance,
    forecast
)

# Load and prepare data
data_dict = prepare_data_for_training(df, config)

# Train model with validation
model = GRUCausalMMM(...)
results = train_model_with_validation(model, data_dict, config)

# Analyze feature importance
importance = get_feature_importance(model, data_dict['X_m'], data_dict['X_c'], data_dict['R'])

# Generate forecasts
forecasts = forecast(model, data_dict['X_m'], data_dict['X_c'], data_dict['R'], horizon=12)
```

## 📊 Data Format

Your marketing data should include:

| Column | Description | Required |
|--------|-------------|----------|
| `date` | Date/time column | Yes |
| `revenue` | Target variable | Yes |
| `tv_spend`, `digital_spend`, etc. | Media spending variables | Yes |
| `price`, `promotion`, etc. | Control variables | Optional |
| `region` | Geographic region | Optional |

Example:
```csv
date,region,tv_spend,digital_spend,radio_spend,price,promotion,revenue
2020-01-01,Region_A,1000,800,500,10.5,0,15000
2020-01-08,Region_A,1200,900,600,10.2,1,16500
...
```

## 🔧 Configuration

The package uses a flexible configuration system:

```python
from deepcausalmmm import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config.update({
    # Model architecture
    'hidden_size': 64,
    'learning_rate': 1e-3,
    'epochs': 10000,
    
    # Data preprocessing
    'burn_in_weeks': 4,
    'test_size': 0.2,
    
    # Feature engineering
    'apply_adstock': True,
    'apply_saturation': True,
    
    # Training
    'early_stopping_patience': 50,
    'gradient_clipping': 1.0,
})
```

## 📈 Model Architecture

The DeepCausalMMM model combines several key components:

1. **CausalEncoder**: Bayesian Network-based graph encoder for media variables
2. **Adstock Transformation**: Models media carryover effects over time
3. **Hill Saturation**: Captures diminishing returns in media effectiveness
4. **GRU Network**: Generates time-varying coefficients for media variables only
5. **Multi-region Support**: Handles geographic variations

```
Input Data → CausalEncoder → Adstock → Hill → GRU → Predictions
                ↓              ↓        ↓      ↓
            Belief Vectors  Carryover  Saturation  Time-varying β(t)
```

## 📊 Outputs & Analytics

The model provides comprehensive outputs:

### 1. Model Performance
- R² Score, RMSE, MAPE
- Training history and convergence
- Validation metrics

### 2. Feature Importance
- Media channel effectiveness rankings
- Contribution analysis over time
- ROAS calculations

### 3. Causal Effects
- Intervention analysis
- Counterfactual predictions
- Uncertainty quantification

### 4. Forecasts
- Multi-period predictions
- Confidence intervals
- Scenario analysis

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=deepcausalmmm tests/
```

## 📚 Documentation

- [API Reference](https://deepcausalmmm.readthedocs.io/)
- [Examples](examples/)
- [Configuration Guide](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on PyTorch for deep learning capabilities
- Inspired by advances in causal inference and marketing science
- Thanks to the open-source community for foundational libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/adityapt/deepcausalmmm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/adityapt/deepcausalmmm/discussions)
- **Email**: puttaparthy.aditya@gmail.com

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

---

**DeepCausalMMM** - Bringing deep learning and causal inference to marketing mix modeling 🚀

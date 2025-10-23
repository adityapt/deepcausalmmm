# DeepCausalMMM 🚀

**Advanced Marketing Mix Modeling with Causal Inference and Deep Learning**

[![Documentation](https://readthedocs.org/projects/deepcausalmmm/badge/?version=latest)](https://deepcausalmmm.readthedocs.io/en/latest/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityapt/deepcausalmmm/blob/main/examples/quickstart.ipynb)
[![PyPI version](https://badge.fury.io/py/deepcausalmmm.svg)](https://badge.fury.io/py/deepcausalmmm)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17274024.svg)](https://doi.org/10.5281/zenodo.17274024)
[![MMM](https://img.shields.io/badge/Marketing%20Mix-Modeling-brightgreen)](https://en.wikipedia.org/wiki/Marketing_mix_modeling)
[![Deep Learning](https://img.shields.io/badge/Deep-Learning-blue)](https://pytorch.org)
[![Causal DAG](https://img.shields.io/badge/Causal-DAG-purple)](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
[![GRU](https://img.shields.io/badge/Neural-GRU-orange)](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Key Features

### ✅ **No Hardcoding**
- **100% Learnable Parameters**: All model parameters learned from data
- **Config-Driven**: Every setting configurable via `config.py`
- **Dataset Agnostic**: Works on any MMM dataset without modifications

### 🧠 **Advanced Architecture**
- **GRU-Based Temporal Modeling**: Captures complex time-varying effects
- **DAG Learning**: Discovers causal relationships between channels
- **Learnable Coefficient Bounds**: Channel-specific, data-driven constraints
- **Data-Driven Seasonality**: Automatic seasonal decomposition per region

### 📊 **Robust Statistical Methods**
- **Huber Loss**: Robust to outliers and extreme values
- **Multiple Metrics**: RMSE, R², MAE, Trimmed RMSE, Log-space metrics
- **Advanced Regularization**: L1/L2, sparsity, coefficient-specific penalties
- **Gradient Clipping**: Parameter-specific clipping for stability

### 🔬 **Comprehensive Analysis**
- **14+ Interactive Visualizations**: Complete dashboard with insights
- **Response Curves**: Non-linear saturation analysis with Hill equations
- **DMA-Level Contributions**: True economic impact calculation
- **Channel Effectiveness**: Detailed performance analysis
- **DAG Visualization**: Interactive causal network graphs

## 🚀 Quick Start

### Installation

#### From PyPI (Recommended)
```bash
pip install deepcausalmmm
```

#### From GitHub (Development Version)
```bash
pip install git+https://github.com/adityapt/deepcausalmmm.git
```

#### Manual Installation
```bash
# Clone repository
git clone https://github.com/adityapt/deepcausalmmm.git
cd deepcausalmmm
pip install -e .
```

#### Dependencies Only
```bash
pip install torch pandas numpy plotly networkx statsmodels scikit-learn tqdm
```

### Basic Usage

```python
import pandas as pd
from deepcausalmmm import DeepCausalMMM, get_device
from deepcausalmmm.core import get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline

# Load your data
data = pd.read_csv('your_mmm_data.csv')

# Get optimized configuration
config = get_default_config()

# Check device availability
device = get_device()
print(f"Using device: {device}")

# Process data with unified pipeline
pipeline = UnifiedDataPipeline(config)
processed_data = pipeline.fit_transform(data)

# Train with ModelTrainer (recommended approach)
trainer = ModelTrainer(config)
model, results = trainer.train(processed_data)

# Generate comprehensive dashboard
python dashboard_rmse_optimized.py  # Run the main dashboard script
```

### One-Command Analysis

```bash
# Run from the project root directory
python dashboard_rmse_optimized.py
```

### Package Import Test

```python
# Verify installation works
from deepcausalmmm import DeepCausalMMM, get_device
from deepcausalmmm.core import get_default_config

print(" DeepCausalMMM package imported successfully!")
print(f"Device: {get_device()}")
```

## 📁 Project Structure

```
deepcausalmmm/                      # Project root
├── pyproject.toml                  # Package configuration and dependencies
├── README.md                       # This documentation
├── LICENSE                         # MIT License
├── CHANGELOG.md                    # Version history and changes
├── CONTRIBUTING.md                 # Development guidelines
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
│   ├── core/                       # Core model components
│   │   ├── __init__.py            # Core module initialization
│   │   ├── config.py              # Optimized configuration parameters
│   │   ├── unified_model.py       # Main DeepCausalMMM model architecture
│   │   ├── trainer.py             # ModelTrainer class for training
│   │   ├── data.py                # UnifiedDataPipeline for data processing
│   │   ├── scaling.py             # SimpleGlobalScaler for data normalization
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
│   └── example_response_curves.py  # Response curve fitting examples
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
│       ├── api/                  # API documentation
│       │   ├── index.rst
│       │   ├── core.rst
│       │   ├── data.rst
│       │   ├── trainer.rst
│       │   ├── inference.rst
│       │   ├── analysis.rst
│       │   ├── response_curves.rst # Response curves API
│       │   ├── utils.rst
│       │   └── exceptions.rst
│       ├── examples/             # Example documentation
│       │   └── index.rst
│       └── tutorials/            # Tutorial documentation
│           └── index.rst
│
└── JOSS/                           # Journal of Open Source Software submission
    ├── paper.md                   # JOSS paper manuscript
    ├── paper.bib                  # Bibliography
    ├── figure_dag_professional.png # DAG visualization figure
    └── figure_response_curve_simple.png # Response curve figure
```

## 🎨 Dashboard Features

The comprehensive dashboard includes:

1. **📈 Performance Metrics**: Training vs Holdout comparison
2. **📊 Actual vs Predicted**: Time series visualization
3. **🎯 Holdout Scatter**: Generalization assessment
4. **💰 Economic Contributions**: Total KPI per channel
5. **🥧 Contribution Breakdown**: Donut chart with percentages
6. **💧 Waterfall Analysis**: Decomposed contribution flow
7. **📺 Channel Effectiveness**: Coefficient distributions
8. **🔗 DAG Network**: Interactive causal relationships
9. **🔥 DAG Heatmap**: Adjacency matrix visualization
10. **📊 Stacked Contributions**: Time-based channel impact
11. **📈 Individual Channels**: Detailed channel analysis
12. **📏 Scaled Data**: Normalized time series
13. **🎛️ Control Variables**: External factor analysis
14. **📉 Response Curves**: Non-linear response curves (diminishing returns analysis) with Hill equations

## ⚙️ Configuration

Key configuration parameters:

```python
{
    # Model Architecture
    'hidden_dim': 320,           # Optimal hidden dimension
    'dropout': 0.08,             # Proven stable dropout
    'gru_layers': 1,             # Single layer for stability
    
    # Training Parameters  
    'n_epochs': 6500,            # Optimal convergence epochs
    'learning_rate': 0.009,      # Fine-tuned learning rate
    'temporal_regularization': 0.04,  # Proven regularization
    
    # Loss Function
    'use_huber_loss': True,      # Robust to outliers
    'huber_delta': 0.3,          # Optimal delta value
    
    # Data Processing
    'holdout_ratio': 0.08,       # Optimal train/test split
    'burn_in_weeks': 6,          # Stabilization period
}
```

## 🔬 Advanced Features

### Learnable Parameters
- **Media Coefficient Bounds**: `F.softplus(coeff_max_raw) * torch.sigmoid(media_coeffs_raw)`
- **Control Coefficients**: Unbounded with gradient clipping
- **Trend Damping**: `torch.exp(trend_damping_raw)` 
- **Baseline Components**: Non-negative via `F.softplus`
- **Seasonal Coefficient**: Learnable seasonal contribution

### Data Processing
- **SOV Scaling**: Share-of-voice normalization for media channels
- **Z-Score Normalization**: For control variables (weather, events, etc.)
- **Min-Max Seasonality**: Regional seasonal scaling (0-1) using `seasonal_decompose`
- **Consistent Transforms**: Same scaling applied to train/holdout splits
- **DMA-Level Processing**: True economic contributions calculated per region

### Regularization Strategy
- **Coefficient L2**: Channel-specific regularization
- **Sparsity Control**: GRU parameter sparsity
- **DAG Regularization**: Acyclicity constraints
- **Gradient Clipping**: Parameter-specific clipping

### Response Curves
- **Hill Saturation Modeling**: Non-linear response curves with Hill equations
- **Automatic Curve Fitting**: Fits S-shaped saturation curves to channel data
- **National-Level Aggregation**: Aggregates DMA-week data to national weekly level
- **Proportional Allocation**: Correctly scales log-space contributions to original scale
- **Interactive Visualizations**: Plotly-based interactive response curve plots
- **Performance Metrics**: R², slope, and saturation point for each channel

```python
from deepcausalmmm.postprocess import ResponseCurveFit

# Fit response curves to channel data
fitter = ResponseCurveFit(
    data=channel_data,
    x_col='impressions',
    y_col='contributions',
    model_level='national',
    date_col='week'
)

# Get fitted parameters
slope, saturation = fitter.fit_curve()
r2_score = fitter.calculate_r2_and_plot(save_path='response_curve.html')

print(f"Slope: {slope:.3f}, Saturation: {saturation:.3f}, R²: {r2_score:.3f}")
```

## 📊 Performance Benchmarks

*Performance benchmarks will be added with masked/anonymized data to demonstrate model capabilities while protecting proprietary information.*

## 🛠️ Development

### Requirements
- Python 3.8+
- PyTorch 1.13+
- pandas 1.5+
- numpy 1.21+
- plotly 5.11+
- statsmodels 0.13+
- scikit-learn 1.1+

### Testing
```bash
python -m pytest tests/
```

### Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🎉 Success Stories

> "Achieved 93% holdout R² with only 3.6% performance gap - exceptional generalization!"

> "Zero hardcoding approach makes it work perfectly on our different datasets without any modifications"

> "The comprehensive dashboard with 14+ interactive visualizations including response curves provides insights we never had before"

> "DMA-level contributions and DAG learning revealed true causal relationships between our marketing channels"

## 🤝 Support

- **Documentation**: Comprehensive README with examples
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Performance**: All configurations battle-tested and production-ready
- **Zero Hardcoding**: Fully generalizable across different datasets and industries

## 📚 Documentation

- **📖 Full Documentation**: [deepcausalmmm.readthedocs.io](https://deepcausalmmm.readthedocs.io/)
- **🚀 Quick Start Guide**: [Installation & Usage](https://deepcausalmmm.readthedocs.io/en/latest/quickstart.html)
- **📋 API Reference**: [Complete API Documentation](https://deepcausalmmm.readthedocs.io/en/latest/api/)
- **📝 Tutorials**: [Step-by-step Guides](https://deepcausalmmm.readthedocs.io/en/latest/tutorials/)
- **💡 Examples**: [Practical Use Cases](https://deepcausalmmm.readthedocs.io/en/latest/examples/)

## 📖 Citation

If you use DeepCausalMMM in your research, please cite:

```bibtex
@article{tirumala2025deepcausalmmm,
  title={DeepCausalMMM: A Deep Learning Framework for Marketing Mix Modeling with Causal Inference},
  author={Puttaparthi Tirumala, Aditya},
  journal={arXiv preprint arXiv:2510.13087},
  year={2025}
}
```

Or click the **"Cite this repository"** button on GitHub for other citation formats (APA, Chicago, MLA).

## 🔗 Quick Links

- **Main Dashboard**: `dashboard_rmse_optimized.py` - Complete analysis pipeline
- **Core Model**: `deepcausalmmm/core/unified_model.py` - DeepCausalMMM architecture
- **Configuration**: `deepcausalmmm/core/config.py` - All tunable parameters
- **Data Pipeline**: `deepcausalmmm/core/data.py` - Data processing and scaling

---

**DeepCausalMMM** - Where Deep Learning meets Causal Inference for Superior Marketing Mix Modeling 🚀

**arXiv preprint** - https://www.arxiv.org/abs/2510.13087

*Built with ❤️ for the MMM community*

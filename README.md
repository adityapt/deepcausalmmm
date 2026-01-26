# DeepCausalMMM

**Advanced Marketing Mix Modeling with Causal Inference and Deep Learning**

[![Documentation](https://readthedocs.org/projects/deepcausalmmm/badge/?version=latest)](https://deepcausalmmm.readthedocs.io/en/latest/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityapt/deepcausalmmm/blob/main/examples/quickstart.ipynb)
[![PyPI version](https://badge.fury.io/py/deepcausalmmm.svg)](https://badge.fury.io/py/deepcausalmmm)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16934842.svg)](https://doi.org/10.5281/zenodo.16934842)
[![MMM](https://img.shields.io/badge/Marketing%20Mix-Modeling-brightgreen)](https://en.wikipedia.org/wiki/Marketing_mix_modeling)
[![Deep Learning](https://img.shields.io/badge/Deep-Learning-blue)](https://pytorch.org)
[![Causal DAG](https://img.shields.io/badge/Causal-DAG-purple)](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
[![GRU](https://img.shields.io/badge/Neural-GRU-orange)](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Key Features

### Advanced Architecture
- **Config-Driven**: Every setting configurable via `config.py`
- **GRU-Based Temporal Modeling**: Captures complex time-varying effects
- **DAG Learning**: Discovers causal relationships between channels
- **Learnable Coefficient Bounds**: Channel-specific, data-driven constraints
- **Data-Driven Seasonality**: Automatic seasonal decomposition per region

### Robust Statistical Methods
- **Huber Loss**: Robust to outliers and extreme values
- **Multiple Metrics**: RMSE, R², MAE, Trimmed RMSE, Log-space metrics
- **Advanced Regularization**: L1/L2, sparsity, coefficient-specific penalties
- **Gradient Clipping**: Parameter-specific clipping for stability

### Comprehensive Analysis
- **14+ Interactive Visualizations**: Complete dashboard with insights
- **Response Curves**: Non-linear saturation analysis with Hill equations
- **Budget Optimization**: Constrained optimization for optimal channel allocation
- **DMA-Level Contributions**: True economic impact calculation
- **Channel Effectiveness**: Detailed performance analysis
- **DAG Visualization**: Interactive causal network graphs

## Quick Start

### Installation

#### From PyPI (Recommended)
```bash
pip install deepcausalmmm
```

#### From GitHub (Development Version - v1.0.19)
```bash
# Install the latest development version with all fixes
pip install git+https://github.com/adityapt/deepcausalmmm.git@feature/remove-log-scaling
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

---

## IMPORTANT: Version Compatibility

**The code examples in this README are for version 1.0.19+ (unreleased)**

If you installed from PyPI (`pip install deepcausalmmm`), you have **v1.0.18** which has a **completely different API**. The examples below will NOT work for v1.0.18.

### For Current PyPI Users (v1.0.18):
```bash
# Option 1: Install the development version (recommended for testing)
pip uninstall deepcausalmmm
pip install git+https://github.com/adityapt/deepcausalmmm.git@feature/remove-log-scaling

# Option 2: Clone and install locally
git clone https://github.com/adityapt/deepcausalmmm.git
cd deepcausalmmm
git checkout feature/remove-log-scaling
pip install -e .

# Option 3: Wait for v1.0.19 release (coming soon)
```

### API Changes in v1.0.19:
- **New**: `pipeline` parameter in `ModelTrainer.train()`
- **Fixed**: Proper data scaling without leakage
- **Fixed**: Correct attribution calculation
- **New**: `ConfigurableDataGenerator.generate_mmm_dataset()` method
- **Removed**: `y_full_for_baseline` parameter (data leakage fix)

---

### Basic Usage

```python
import numpy as np
from deepcausalmmm import get_device
from deepcausalmmm.core import get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline
from deepcausalmmm.utils.data_generator import ConfigurableDataGenerator

# Generate synthetic data for testing
# You can replace this with your own data in the same format
n_regions = 50        # Number of DMAs/regions
n_weeks = 104         # 2 years of weekly data
n_media = 13          # Number of media channels
n_control = 3         # Number of control variables

generator = ConfigurableDataGenerator()
X_media, X_control, y = generator.generate_mmm_dataset(
    n_regions=n_regions,
    n_weeks=n_weeks,
    n_media_channels=n_media,
    n_control_channels=n_control
)

# Expected data format:
# X_media: [n_regions, n_weeks, n_media_channels] - Media inputs
# X_control: [n_regions, n_weeks, n_control_variables] - Control variables
# y: [n_regions, n_weeks] - Target variable (KPI units)

# Get configuration
config = get_default_config()
config['n_epochs'] = 200  # Adjust as needed

# Check device availability
device = get_device()
print(f"Using device: {device}")

# Initialize pipeline
pipeline = UnifiedDataPipeline(config)

# Split data temporally (train/holdout)
train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)

# Process training data
train_tensors = pipeline.fit_and_transform_training(train_data)
holdout_tensors = pipeline.transform_holdout(holdout_data)

# Create trainer and model
trainer = ModelTrainer(config)
model = trainer.create_model(
    n_media=train_tensors['X_media'].shape[2],
    n_control=train_tensors['X_control'].shape[2],
    n_regions=train_tensors['X_media'].shape[0]
)
trainer.create_optimizer_and_scheduler()

# Train model
results = trainer.train(
    train_tensors['X_media'], train_tensors['X_control'],
    train_tensors['R'], train_tensors['y'],
    holdout_tensors['X_media'], holdout_tensors['X_control'],
    holdout_tensors['R'], holdout_tensors['y'],
    pipeline=pipeline,
    verbose=True
)

# View results
print(f"Training R²: {results['final_train_r2']:.3f}")
print(f"Holdout R²: {results['final_holdout_r2']:.3f}")
```

### Running Examples (Requires Cloning Repository)

The `examples/` folder is only available if you clone the repository (not included in pip install):

```bash
# Clone the repository first
git clone https://github.com/adityapt/deepcausalmmm.git
cd deepcausalmmm

# Install the package
pip install -e .

# Run the comprehensive dashboard (uses real-world anonymized data)
python examples/dashboard_rmse_optimized.py

# Or run other examples
python examples/example_budget_optimization.py
python examples/example_response_curves.py
```

### Package Import Test

```python
# Verify installation works
from deepcausalmmm import DeepCausalMMM, get_device
from deepcausalmmm.core import get_default_config

print("DeepCausalMMM package imported successfully!")
print(f"Device: {get_device()}")
```

## Project Structure

```
deepcausalmmm/                      # Project root
├── pyproject.toml                  # Package configuration and dependencies
├── README.md                       # This documentation
├── LICENSE                         # MIT License
├── CHANGELOG.md                    # Version history and changes
├── RELEASE_NOTES_1.0.19.md         # Latest release notes
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
│   ├── example_budget_optimization.py  # Budget optimization workflow
│   └── data/                      # Example data directory
│       └── MMM Data.csv           # Anonymized real-world MMM dataset
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
│       │   ├── optimization.rst    # Budget optimization API
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

## Dashboard Features

The comprehensive dashboard includes:

1. **Performance Metrics**: Training vs Holdout comparison
2. **Actual vs Predicted**: Time series visualization
3. **Holdout Scatter**: Generalization assessment
4. **Economic Contributions**: Total KPI per channel
5. **Contribution Breakdown**: Donut chart with percentages
6. **Waterfall Analysis**: Decomposed contribution flow
7. **Channel Effectiveness**: Coefficient distributions
8. **DAG Network**: Interactive causal relationships
9. **DAG Heatmap**: Adjacency matrix visualization
10. **Stacked Contributions**: Time-based channel impact
11. **Individual Channels**: Detailed channel analysis
12. **Scaled Data**: Normalized time series
13. **Control Variables**: External factor analysis
14. **Response Curves**: Non-linear response curves (diminishing returns analysis) with Hill equations

## Configuration

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

## Advanced Features

### Learnable Parameters
- **Media Coefficient Bounds**: `F.softplus(coeff_max_raw) * torch.sigmoid(media_coeffs_raw)`
- **Control Coefficients**: Unbounded with gradient clipping
- **Trend Damping**: `torch.exp(trend_damping_raw)` 
- **Baseline Components**: Non-negative via `F.softplus`
- **Seasonal Coefficient**: Learnable seasonal contribution

### Data Processing
- **Linear Scaling**: Target scaled by regional mean (y/y_mean) for balanced training
- **SOV Scaling**: Share-of-voice normalization for media channels
- **Z-Score Normalization**: For control variables (weather, events, etc.)
- **Min-Max Seasonality**: Regional seasonal scaling (0-1) using `seasonal_decompose`
- **Consistent Transforms**: Same scaling applied to train/holdout splits
- **DMA-Level Processing**: True economic contributions calculated per region
- **Attribution Priors**: Media contribution regularization (40% target) with dynamic loss scaling
- **Data-Driven Hill Initialization**: Hill parameters initialized from channel-specific SOV percentiles

### Regularization Strategy
- **Coefficient L2**: Channel-specific regularization
- **Sparsity Control**: GRU parameter sparsity
- **DAG Regularization**: Acyclicity constraints
- **Gradient Clipping**: Parameter-specific clipping

### Response Curves
- **Hill Saturation Modeling**: Non-linear response curves with Hill equations
- **Data-Driven Initialization**: Hill `g` parameter initialized from channel-specific SOV 60th percentile
- **Automatic Curve Fitting**: Fits S-shaped saturation curves to channel data
- **National-Level Aggregation**: Aggregates DMA-week data to national weekly level
- **Linear Scaling**: Direct scaling with prediction_scale × y_mean for accurate attribution
- **Interactive Visualizations**: Plotly-based interactive response curve plots
- **Performance Metrics**: R², slope, and saturation point for each channel

```python
import pandas as pd
import numpy as np
from deepcausalmmm.postprocess import ResponseCurveFit

# After training your model, prepare channel data for response curve fitting
# The data should have 'week_monday', 'spend', 'impressions', and 'predicted' columns

# Example: Create channel data from your model results
# Replace this with actual data extraction from your trained model
n_weeks = 104
channel_data = pd.DataFrame({
    'week_monday': pd.date_range('2024-01-01', periods=n_weeks, freq='W'),
    'spend': np.random.uniform(10000, 50000, n_weeks),       # Replace with actual spend
    'impressions': np.random.uniform(100000, 500000, n_weeks),  # Replace with actual impressions
    'predicted': np.random.uniform(1000, 5000, n_weeks)      # Replace with model predictions
})

# Fit response curves to channel data
fitter = ResponseCurveFit(
    data=channel_data,
    model_level='Overall',
    date_col='week_monday'
)

# Fit the curve and generate visualization
fitter.fit(
    x_label='Impressions',
    y_label='Predicted KPI Units',
    title='Channel Response Curve',
    save_figure=True,
    output_path='response_curve.html'
)

print(f"Slope: {fitter.slope:.3f}, Saturation: {fitter.saturation:.0f}, R²: {fitter.r_2:.3f}")
```

### Budget Optimization
- **Constrained Optimization**: Find optimal budget allocation across channels
- **Multiple Methods**: SLSQP (default), trust-constr, differential evolution, hybrid
- **Hill Equation Integration**: Uses fitted response curves for saturation modeling
- **Channel Constraints**: Set min/max spend limits based on business requirements
- **Scenario Comparison**: Compare current vs optimal allocations
- **ROI Maximization**: Maximize predicted response subject to budget and constraints

```python
import pandas as pd
from deepcausalmmm import optimize_budget_from_curves

# After training your model and fitting response curves...
# Create a DataFrame with fitted curve parameters for each channel
# Replace this with actual fitted parameters from your response curve fitting

fitted_curves_df = pd.DataFrame({
    'channel': ['TV', 'Search', 'Social', 'Display', 'Radio'],
    'top': [2.5, 3.0, 2.2, 1.8, 2.0],              # Hill parameter (slope at inflection)
    'bottom': [0.0, 0.0, 0.0, 0.0, 0.0],           # Minimum response
    'saturation': [500000, 300000, 200000, 150000, 400000],  # Saturation point (impressions)
    'slope': [0.002, 0.003, 0.004, 0.002, 0.001]   # Initial slope
})

# Optimize budget allocation
result = optimize_budget_from_curves(
    budget=1_000_000,
    curve_params=fitted_curves_df,
    num_weeks=52,
    constraints={
        'TV': {'lower': 100000, 'upper': 600000},
        'Search': {'lower': 150000, 'upper': 500000},
        'Social': {'lower': 50000, 'upper': 300000}
    },
    method='SLSQP'
)

# View results
if result.success:
    print(f"Optimal Allocation: {result.allocation}")
    print(f"Predicted Response: {result.predicted_response:,.0f}")
    print(result.by_channel)
```

**Example Output:**
```
Optimal Allocation: {'TV': 100000, 'Search': 420000, 'Social': 300000, ...}
Predicted Response: 627,788

Detailed Metrics:
  channel  total_spend  weekly_spend  roi  spend_pct  response_pct  saturation_pct
   Search      420,000      8,076.92  0.56      42.0%        37.8%           323%
   Social      300,000      5,769.23  0.73      30.0%        34.8%           288%
       TV      100,000      1,923.08  0.13      10.0%         2.1%            64%
```

See `examples/example_budget_optimization.py` for complete workflow and tips.

## Performance Benchmarks

**Real-World Validation** (190 regions, 109 weeks, 13 channels, 7 controls):

- **Training R²**: 0.956 | **Holdout R²**: 0.839
- **Performance Gap**: 12.2% (indicating strong generalization)
- **Generalization Gap**: 10.8% (excellent out-of-sample performance)
- **Temporal Split**: 92.7% training (101 weeks) / 7.3% holdout (8 weeks)

**Attribution Breakdown** (with 40% media prior regularization):
- **Media**: 38.6% (close to 40% target)
- **Baseline**: 35.4%
- **Seasonality**: 25.7%
- **Controls**: 0.2%

**Key Achievements**:
- Components sum to 100% with perfect additivity (0.000% error)
- Realistic attribution through prior-based regularization
- Data-driven Hill parameters prevent similar attribution across channels

## Development

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

## License

MIT License - see [LICENSE](LICENSE) file.

## Success Stories

> "Achieved 84% holdout R² with 10.8% performance gap - strong generalization on real-world data with 190 regions!"

> "Attribution priors with dynamic loss scaling solved the attribution explosion problem - media now at realistic 38.6%"

> "Data-driven Hill initialization enables channel-specific saturation curves"

> "The comprehensive dashboard with 14+ interactive visualizations including response curves provides insights we never had before"

> "DMA-level contributions and DAG learning revealed true causal relationships between our marketing channels"

## Support

- **Documentation**: Comprehensive README with examples
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Performance**: All configurations battle-tested and production-ready
- **Zero Hardcoding**: Fully generalizable across different datasets and industries

## Documentation

- **Full Documentation**: [deepcausalmmm.readthedocs.io](https://deepcausalmmm.readthedocs.io/)
- **Quick Start Guide**: [Installation & Usage](https://deepcausalmmm.readthedocs.io/en/latest/quickstart.html)
- **API Reference**: [Complete API Documentation](https://deepcausalmmm.readthedocs.io/en/latest/api/)
- **Tutorials**: [Step-by-step Guides](https://deepcausalmmm.readthedocs.io/en/latest/tutorials/)
- **Examples**: [Practical Use Cases](https://deepcausalmmm.readthedocs.io/en/latest/examples/)

## Roadmap

### Version 1.0.20 (Q2 2025)
- **NOTEARS DAG Learning**: Full implementation of the NOTEARS (DAGs with NO TEARS) continuous optimization method for discovering arbitrary DAG structures
- **Enhanced Causal Discovery**: Move beyond upper triangular constraints to learn more flexible causal relationships between marketing channels

## Citation

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

## Quick Links

- **Main Dashboard**: `dashboard_rmse_optimized.py` - Complete analysis pipeline
- **Budget Optimization**: `examples/example_budget_optimization.py` - End-to-end optimization workflow
- **Core Model**: `deepcausalmmm/core/unified_model.py` - DeepCausalMMM architecture
- **Configuration**: `deepcausalmmm/core/config.py` - All tunable parameters
- **Data Pipeline**: `deepcausalmmm/core/data.py` - Data processing and scaling

---

**DeepCausalMMM** - Where Deep Learning meets Causal Inference for Superior Marketing Mix Modeling

**arXiv preprint** - https://www.arxiv.org/abs/2510.13087

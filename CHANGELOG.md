# Changelog

All notable changes to DeepCausalMMM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.18] - 2025-10-22

### Professional Code Standards & Documentation

### Changed
- **Logging System**: Replaced all print statements with proper logging throughout core library
- **Package Logger**: Added centralized logging configuration in `__init__.py`
- **Emoji Removal**: Removed all emojis from core library, postprocessing, CHANGELOG, and CODE_OF_CONDUCT
- **JOSS Paper**: Updated with correct Zenodo DOI, test suite description, and requirement clarifications
- **arXiv Submission**: Updated to match JOSS paper changes for consistency

### Core Library Updates
- **`deepcausalmmm/__init__.py`**: Added `_setup_logging()` function for package-wide logger
- **`core/train_model.py`**: Replaced 70 print statements with logging (info/warning/debug)
- **`core/data.py`**: Replaced 46 print statements with logging
- **`core/unified_model.py`**: Replaced 17 print statements with logging
- **`core/seasonality.py`**: Replaced 7 print statements with logging
- **`core/trainer.py`**: Replaced 3 print statements with logging
- **`core/visualization.py`**: Replaced 1 print statement with logging

### Postprocessing Updates
- **`postprocess/analysis.py`**: Replaced 4 print statements with logging
- **`postprocess/response_curves.py`**: Replaced 4 print statements with logging (info/error)
- **`postprocess/comprehensive_analysis.py`**: Replaced 65 print statements with logging, removed 8 emojis from plot titles

### Documentation Updates
- **`CHANGELOG.md`**: Removed all emojis for professional presentation
- **`CODE_OF_CONDUCT.md`**: Removed all emojis
- **`JOSS/paper.md`**: Updated test suite description, Zenodo DOI to concept DOI (10.5281/zenodo.16934842)
- **`JOSS/arxiv_submission/paper_arxiv.tex`**: Synced with JOSS paper updates

### Technical Details
- **Logger Name**: `'deepcausalmmm'` (accessible via `logging.getLogger('deepcausalmmm')`)
- **Default Level**: INFO
- **Format**: `'%(levelname)s - %(message)s'`
- **Output**: stdout
- **User Control**: Users can adjust logging level via standard Python logging configuration

### Backward Compatibility
- CLI and example scripts retain print statements (user-facing output)
- All existing APIs unchanged
- Logging can be controlled by users without code changes

## [1.0.17] - 2025-10-05

### Response Curves & Saturation Analysis

### Added
- **Response Curve Module**: Complete non-linear response curve fitting with Hill equations
- **`postprocess/response_curves.py`**: `ResponseCurveFit` class for saturation analysis
- **National-Level Aggregation**: Automatic aggregation from DMA-week to national weekly data
- **Proportional Allocation**: Correct scaling of log-space contributions to original scale
- **Interactive Visualizations**: Plotly-based interactive response curve plots with hover details
- **Performance Metrics**: R², slope, and saturation point calculation for each channel
- **Dashboard Integration**: Response curves section added to comprehensive dashboard

### Enhanced
- **Hill Parameter Constraints**: Enforced slope `a >= 2.0` for proper S-shaped curves
- **Hill Initialization**: Improved `hill_a` initialization to `2.5` for natural learning above floor
- **Inverse Transform**: Enhanced `inverse_transform_contributions` with proportional allocation method
- **Waterfall Chart**: Fixed total calculation to use actual predictions instead of component sum
- **All Dashboard Plots**: Updated to use proportionally allocated contributions for consistency

### API
```python
from deepcausalmmm.postprocess import ResponseCurveFit

# Fit response curves
fitter = ResponseCurveFit(
    data=channel_data,
    x_col='impressions',
    y_col='contributions',
    model_level='national',
    date_col='week'
)

slope, saturation = fitter.fit_curve()
r2_score = fitter.calculate_r2_and_plot(save_path='response_curve.html')
```

### Documentation
- **README.md**: Added Response Curves section with usage examples
- **API Documentation**: Added response curves module documentation
- **Dashboard Features**: Updated to include response curve visualization (14+ charts)

### Fixed
- **Log-space Scaling**: Corrected contribution scaling using proportional allocation
- **Waterfall Totals**: Fixed incorrect total calculation in waterfall charts
- **Burn-in Handling**: Properly trimmed burn-in padding from all components
- **Shape Mismatches**: Resolved tensor shape inconsistencies in inverse transforms

### Technical Details
- **Hill Equation**: `y = x^a / (x^a + g^a)` where `a` is slope and `g` is half-saturation
- **Slope Constraint**: `a = torch.clamp(F.softplus(hill_a), 2.0, 5.0)` for S-shaped curves
- **Proportional Allocation**: `component_orig = (component_log / total_log) × y_pred_orig`
- **Aggregation**: Groups by week and sums impressions/contributions nationally

### Backward Compatibility
- **`ResponseCurveFitter`**: Maintained as alias for backward compatibility
- **Legacy Methods**: `Hill`, `get_param`, `regression`, `fit_model` still available
- **Legacy Parameters**: `Modellevel`, `Datecol` supported with deprecation path

## [1.0.0] - 2025-07-15

###  PRODUCTION RELEASE - Good Performance

### Added
- ** Complete DeepCausalMMM Package**: Production-ready MMM with causal inference
- ** Advanced Model Architecture**: GRU-based temporal modeling with DAG learning
- ** Zero Hardcoding Philosophy**: All parameters learnable and configurable
- ** Comprehensive Analysis Suite**: 13 interactive visualizations and insights
- ** Unified Data Pipeline**: Consistent data processing with proper scaling
- ** Robust Statistical Methods**: Huber loss, multiple metrics, advanced regularization

### Core Components
- **`core/unified_model.py`**: Main DeepCausalMMM model with learnable parameters
- **`core/trainer.py`**: ModelTrainer with advanced optimization strategies
- **`core/config.py`**: Comprehensive configuration system (no hardcoding)
- **`core/data.py`**: UnifiedDataPipeline for consistent data processing
- **`core/scaling.py`**: SimpleGlobalScaler with proper transformations
- **`core/seasonality.py`**: Data-driven seasonal decomposition

### Analysis & Visualization
- **`postprocess/comprehensive_analyzer.py`**: Complete analysis engine
- **`postprocess/inference.py`**: Model inference and prediction utilities
- **`postprocess/visualization.py`**: Interactive dashboard creation
- **Dashboard Features**: 13 comprehensive visualizations including DAG networks, waterfall charts, economic contributions

### Key Features
- ** Learnable Coefficient Bounds**: Channel-specific, data-driven constraints
- ** Data-Driven Seasonality**: Automatic seasonal decomposition per region
- ** Non-Negative Constraints**: Baseline and seasonality always positive
- ** DAG Learning**: Discovers causal relationships between channels
- ** Robust Loss Functions**: Huber loss for outlier resistance
- ** Advanced Regularization**: L1/L2, sparsity, coefficient-specific penalties
- ** Gradient Clipping**: Parameter-specific clipping for training stability

### Performance Optimizations
- **Optimal Configuration**: 6500 epochs, 0.009 LR, 0.04 temporal regularization
- **Smart Early Stopping**: Prevents overfitting while maximizing performance
- **Burn-in Stabilization**: 6-week warm-start for GRU stability
- **Holdout Strategy**: 8% holdout ratio for optimal train/test balance

### Data Processing
- **SOV Scaling**: Share-of-voice normalization for media variables
- **Z-Score Normalization**: For control variables
- **Min-Max Seasonality**: Regional seasonal scaling (0-1 range)
- **Log1p Transformation**: For target variable with proper inverse transforms

### Documentation
- ** Comprehensive README**: Complete usage guide with examples
- ** Contributing Guidelines**: Development standards and processes
- ** Changelog**: Detailed version history
- ** Configuration Guide**: All parameters documented with optimal values

### Testing & Quality
- **Unit Tests**: Comprehensive test coverage for all components
- **Integration Tests**: End-to-end workflow validation
- **Performance Regression Tests**: Ensures consistent benchmarks
- **Code Quality**: Type hints, docstrings, linting compliance

## [0.9.0] - 2024-01-10 - BETA RELEASE

### Added
- **Initial Model Architecture**: Basic GRU-based MMM implementation
- **Basic Data Processing**: Simple scaling and normalization
- **Core Training Loop**: ModelTrainer with basic optimization
- **Simple Visualizations**: Basic plotting functionality

### Performance
- **Training R²**: ~0.85
- **Holdout R²**: ~0.65
- **Performance Gap**: ~20%

### Issues Resolved in v1.0.0
- **Hardcoding Issues**: Eliminated all hardcoded parameters
- **Poor Generalization**: Improved from 20% to 3.6% performance gap
- **Limited Analysis**: Expanded from 3 to 13 comprehensive visualizations
- **Data Inconsistency**: Implemented unified data pipeline
- **Training Instability**: Added advanced regularization and gradient clipping

## [0.8.0] - 2024-01-05 - ALPHA RELEASE

### Added
- **Proof of Concept**: Basic MMM implementation
- **Simple GRU Model**: Single-layer GRU for temporal modeling
- **Basic Training**: Simple MSE loss with basic optimization
- **Minimal Visualization**: Single performance plot

### Known Issues (Fixed in Later Versions)
- **Hardcoded Parameters**: Many values hardcoded in model
- **Poor Performance**: High RMSE, low R²
- **No Seasonality**: Missing seasonal components
- **Limited Analysis**: No comprehensive insights
- **Data Leakage**: Inconsistent train/holdout processing

## Development Milestones

###  Key Achievements Across Versions

| Version | Training R² | Holdout R² | Performance Gap | Key Innovation |
|---------|-------------|------------|-----------------|----------------|
| v0.8.0  | 0.70       | 0.45       | 35%            | Basic GRU Model |
| v0.9.0  | 0.85       | 0.65       | 20%            | Improved Training |
| **v1.0.0** | **0.965**  | **0.930**  | **3.6%**      | **Zero Hardcoding + Advanced Architecture** |

###  Technical Evolution

**v0.8.0 → v0.9.0:**
- Improved model architecture
- Better regularization
- Enhanced data processing

**v0.9.0 → v1.0.0:**
- **Complete architectural overhaul**
- **Zero hardcoding philosophy**
- **Advanced regularization strategies**
- **Data-driven seasonality**
- **Comprehensive analysis suite**
- **Production-ready performance**

###  Performance Improvements

**RMSE Reduction Journey:**
- v0.8.0: ~800k visits (110% error)
- v0.9.0: ~600k visits (80% error)  
- **v1.0.0: 325k visits (38.7% error)** 

**Generalization Improvements:**
- v0.8.0: 35% performance gap
- v0.9.0: 20% performance gap
- **v1.0.0: 3.6% performance gap** 

###  Feature Evolution

**Visualization Expansion:**
- v0.8.0: 1 basic plot
- v0.9.0: 5 standard plots
- **v1.0.0: 13 comprehensive interactive visualizations** 

**Analysis Depth:**
- v0.8.0: Basic performance metrics
- v0.9.0: Channel-level insights
- **v1.0.0: Complete business intelligence suite** 

## Future Roadmap

### [1.1.0] - Planned Features
- **Multi-Objective Optimization**: Simultaneous RMSE and business constraint optimization
- **Automated Hyperparameter Tuning**: Bayesian optimization for config parameters
- **Real-Time Inference**: Streaming prediction capabilities
- **Advanced Causal Discovery**: Enhanced DAG learning algorithms

### [1.2.0] - Advanced Features  
- **Ensemble Methods**: Multiple model combination strategies
- **Uncertainty Quantification**: Confidence intervals for predictions
- **Transfer Learning**: Pre-trained models for quick deployment
- **Cloud Integration**: AWS/GCP deployment utilities

### [2.0.0] - Next Generation
- **Transformer Architecture**: Attention-based temporal modeling
- **Multi-Modal Learning**: Integration of external data sources
- **Federated Learning**: Distributed training across datasets
- **AutoML Integration**: Automated model selection and tuning

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and how to contribute to future releases.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.



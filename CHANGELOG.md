# Changelog

All notable changes to DeepCausalMMM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.21] - 2026-05-23

### Added
- **NOTEARS DAG learning** (Zheng et al., 2018): opt-in continuous structure learning via `config['dag_mode'] = 'notears'`. Replaces the fixed upper-triangular mask with the smooth acyclicity penalty **h(W) = tr(exp(W ⊙ W)) − d**, optimised under an augmented Lagrangian **0.5·ρ·h(W)² + α·h(W)** plus L1 sparsity on the adjacency. Default **`'dag_mode': 'triangular'`** is unchanged for existing workflows.
- **`DeepCausalMMM.h_acyclicity()`**, **`notears_update_duals(factor=...)`**, and **`threshold_dag(eps)`** for training-time dual updates and post-training pruned adjacency inspection.
- **NOTEARS config keys** in **`get_default_config()`**: `notears_lambda1`, `notears_rho_init`, `notears_alpha_init`, `notears_rho_max`, `notears_dual_update_every`, `notears_threshold`, `notears_warmup_epochs`, `notears_dual_factor`, `dag_temperature`, `notears_group_l1`.
- **Huber-first warmup**: `notears_warmup_epochs` trains with prediction loss only, then enables the NOTEARS penalty via a `notears_active` gate so the fit stabilises before acyclicity pressure.
- **Column-group L1** (`notears_group_l1`): encourages each channel to depend on a focused parent set rather than uniform weak edges from all channels.
- **Temperature-scaled DAG edges** (`dag_temperature < 1`): sharpens sigmoid adjacency toward near-{0,1} weights for clearer structure.
- **Trainer logging**: `[NOTEARS]` messages for warmup start/end and periodic dual updates (h, ρ, α).
- **`tests/unit/test_notears.py`**: NOTEARS smoke tests (forward/backward, acyclicity, dual updates, warmup gate).

### Fixed
- **`examples/dashboard_rmse_optimized.py`**: DAG network/CSV use **`model.threshold_dag(eps=notears_threshold)`**; heatmap uses masked temperature-scaled adjacency (aligned with forward pass, not raw `sigmoid(adj_logits)`).
- **`get_viz_params()`** / heatmap viz defaults: **`correlation_threshold`** fallback **0.05** (was 0.65).
- **`ModelTrainer`**: fallback defaults for **`notears_lambda1`** and **`notears_dual_factor`** match **`config.py`**.

### Changed
- **`dag_interaction()`**: load-bearing parent blend per channel — `x_j ← (1 − mix_j)·x_j + mix_j·Σ_i adj[i,j]·x_i` with **per-channel** `mix_j` (replacing a single global scalar and additive `x + w·(x @ adj)`), so gradients target informative parents instead of a uniform adjacency floor.
- **`interaction_weight`**: one learnable mix scalar per media channel (initialised near 20% DAG blend).
- **NOTEARS adjacency init**: aligned with triangular-mode scale so edges contribute to predictions from epoch 0; acyclicity is driven down during warmup + outer-loop updates rather than by near-zero initial W.
- **`ModelTrainer`**: passes `dag_temperature` and `notears_group_l1` into the model; dual updates use configurable `notears_dual_factor` (default gentler than the initial implementation).
- **`examples/dashboard_rmse_optimized.py` DAG network**: global top-N strongest edges (default 15), weight-normalised arrow sizing, edge strength labels, and optional **`dag_adjacency.csv`** export beside the plot.
- **Default DAG viz threshold**: `visualization.correlation_threshold` lowered to **0.05** (NOTEARS edge weights typically sit around 0.10–0.20); added **`visualization.dag_top_n_edges`**.

### Documentation
- **`README.md`**: NOTEARS feature description, enablement example, roadmap, and v1.0.21 API notes.
- **`docs/source/quickstart.rst`**: NOTEARS configuration subsection.
- **`docs/source/tutorials/dag_notears.rst`**: full Sphinx guide (config keys, training, inspection).
- **`docs/source/index.rst`**, **`tutorials/index.rst`**, **`api/core.rst`**: cross-links and NOTEARS coverage.
- **`JOSS/paper.md`**: Summary, State of the Field, and Software Design aligned with opt-in NOTEARS.
- **`RELEASE_NOTES_1.0.21.md`**: release summary for v1.0.21.
- **`DeepCausalMMM`** / **`get_default_config()`** docstrings updated for autodoc.

## [1.0.20] - 2026-04-18

### Fixed
- **`examples/dashboard_rmse_optimized.py`**: Region-wise missing-value fill uses `groupby(...).transform(lambda series: series.ffill().bfill())` instead of deprecated / removed grouped `fillna(method='ffill'|'bfill')`, compatible with pandas 2.2+ and 3.x (including environments where `SeriesGroupBy.fillna` is unavailable).
- **`DeepCausalMMM.forward()` contract vs callers**: **`InferenceManager.predict()`**, **`UnifiedDataPipeline.predict_and_postprocess`**, and **`train_model`** paths now unpack **`(predictions, media_coeffs, media_contributions, outputs)`** and take **control contributions** from **`outputs['control_contributions']`**, matching the current model implementation (fixes wrong media/control tensors when `n_media ≠ n_control`, broken 3-value unpack, and **`return_contributions=False`** returning a tuple instead of predictions).
- **`JOSS/paper.bib`**: Wrapped corporate author names in double braces (`{{...}}`) so BibTeX stops reordering them into initials-first form (`Meridian2024`, `PyMCMarketing2024`, `RobynGitHub`); also protected product names (`{Meridian}`, `{PyMC-Marketing}`, `{Robyn}`) in titles. Addresses JOSS review feedback on malformed citations.
- **`postprocess/analysis.py`**: **`ModelAnalyzer.analyze_predictions`** now calls **`InferenceManager.predict`** with the correct signature (NumPy inputs, optional **`return_media_coefficients=True`** for plots); **`_generate_plots`** uses **`media_contributions`**, **`burn_in_weeks`** from the underlying model, and no longer mis-passes region tensors as keyword arguments.
- **`examples/example_response_curves.py`**: Forward unpack matches **`(predictions, media_coeffs, media_contributions, outputs)`** with **control** from **`outputs['control_contributions']`**; seasonal slice uses **`outputs['seasonal_contribution']`** (singular key).
- **`examples/dashboard_rmse_optimized.py`**: Holdout **`model(...)`** unpack uses accurate names / discards for unused middle returns (behavior unchanged; clarity only).
- **`ModelTrainer` final metrics**: Train/holdout reporting trims **`burn_in_weeks`** (aligned with config / pipeline padding) in **scaled** space before **`inverse_transform_target`** and RMSE/R², so scores exclude padded stabilization weeks; holdout scaled-space metrics use the same trim. **`examples/dashboard_rmse_optimized.py`**: holdout scatter matches that evaluation; printed holdout MAE uses **`holdout_mae_orig`**.

### Added
- **`tests/integration/test_dashboard_rmse_optimized.py`**: Regression test that loads `load_real_mmm_data()` on `examples/data/MMM Data.csv` so the dashboard data path stays covered in CI.
- **`tests/unit/test_inference.py`**: Regression coverage for **`InferenceManager.predict()`** with **`return_contributions`** true/false; integration/unit forward-pass tests assert coefficient vs contribution shapes and **`outputs`** consistency; optional **`return_media_coefficients`** coverage.
- **`InferenceManager.predict(..., return_media_coefficients=...)`**: Optional time-varying media coefficients in the result dict (single forward pass) for analyzers and tooling.

### Documentation
- **`installation.rst`** / **`index.rst`**: Aligned install guidance with **`pyproject.toml`** and README (Python **3.9+**, full runtime dependency list including **scipy**, **networkx**, **tqdm**, note that **numpy** is capped below **2.0** in metadata); **PyPI** first, then GitHub; removed nonexistent **`[dev]`** / **`[visualization]`** / **`[docs]`** extras; documented real **`[test]`** extra and doc build requirements file.
- **`contributing.rst`**: Replaced **`pip install -e .[dev]`** with **`pip install -e .`** and **`pip install -e .[test]`** to match optional dependencies.
- **`README.md`** / **`CONTRIBUTING.md`**: Development requirements aligned with **`pyproject.toml`**; **“Dependencies only”** pip snippet uses the same **version specifiers** as **`[project] dependencies`**; **`CONTRIBUTING.md`** no longer references nonexistent **`[dev]`** extra; **Benchmarks** temporal split matches default **12%** holdout (~96 / ~13 weeks on 109 observed weeks).
- **Sphinx**: Added missing **`docs/source/api/cli.rst`** and **`visualization.rst`** (``deepcausalmmm.cli``, ``deepcausalmmm.core.visualization``); added **`docs/source/examples/retail_mmm.rst`** and **`multi_region.rst`** so API and Examples toctrees resolve; **`docs/requirements.txt`** includes **scipy** for autodoc imports used by optimization/response-curve modules.
- **JOSS (`paper.md`)**: **Software Design → Implementation Details**: versioning bullet states **semantic versioning** and **documented breaking changes** (notably **v1.0.19** vs **v1.0.18 and earlier**), with pointers to README/CHANGELOG—replacing a blanket “backward compatibility guarantees” phrase that conflicted with published release notes; optional minor/patch milestone wording was removed for brevity.
- **`deepcausalmmm.core.unified_model.DeepCausalMMM.forward`**: Docstring and examples updated to match the actual return tuple **`(predictions, media_coefficients, media_contributions, outputs)`** and real **`outputs`** keys (`contributions`, `control_contributions`, `seasonal_contribution`, etc.).
- **JOSS (`paper.md`)**: Comparative **Table 1** on `examples/data/MMM Data.csv` (same split as `pymc_aligned_dcm_config.json`) versus PyMC-Marketing, Meridian, and a national weekly Ridge baseline (Robyn-style inputs; not Meta’s full Robyn unless `robynpy` is used); corrected train/holdout week description (~96 / ~13 observed weeks at 12% holdout); **Research Impact Statement** reframed (niche, reproducible comparison, near-term significance, honest limits on early uptake); **Reproducibility** references `examples/mmm_three_way_benchmark.ipynb` for Table 1.
- **README.md**: **Development history** note—substantial design/prototyping predates the public GitHub history; bursty commits reflect integration, docs, tests, and packaging.
- **`CITATION.cff`**: Title aligned with the JOSS `paper.md` title (*"DeepCausalMMM: A Deep Learning Framework for Marketing Mix Modeling with Causal Structure Learning"*); version bumped to `1.0.20`; `date-released` updated.

## [1.0.19] - 2026-01-25

### Linear Scaling & Attribution Architecture

### Breaking Changes
- **Scaling Method**: Changed from log1p transformation to linear scaling (y/y_mean per region) for additive attribution
- **Attribution Calculation**: Components now sum exactly to 100% in original scale
- **Model Output**: All predictions and contributions in KPI units (not log-space)

### Added
- **Linear Scaling Architecture**: y/y_mean per region scaling enabling perfect additivity
- **Attribution Prior Regularization**: Configurable media contribution targets (e.g., 30-50%)
- **Dynamic Loss Scaling**: Automatic scaling of regularization losses to match prediction loss magnitude
- **Data-Driven Hill Initialization**: Channel-specific Hill parameter (g) initialization from SOV percentiles
- **Seasonal Regularization**: Optional regularization to prevent seasonal suppression
- **Integrated Attribution**: Components (baseline, seasonal, media, controls) sum to 100% with negligible error

### Enhanced
- **`core/unified_model.py`**: Redesigned forward pass for linear scaling and additive attribution
- **`core/scaling.py`**: Complete rewrite for linear scaling (y/y_mean) and proper inverse transforms
- **`core/trainer.py`**: Added data-driven Hill initialization call, dynamic loss scaling for regularization
- **`examples/dashboard_rmse_optimized.py`**: Fixed 3 critical bugs in waterfall calculations
- **Waterfall Chart**: Fixed double-counting seasonality, double inverse transformation, wrong prediction_scale
- **Response Curves**: Unified visualization with all channels in single plot, curves start from zero

### Fixed
- **Data Leakage Bug**: `y_mean_per_region` now calculated from training data only (not holdout)
- **Seasonal Initialization**: Corrected tensor slicing to prevent seasonal suppression
- **Double-Counting Seasonality**: Separated baseline and seasonal contributions for attribution
- **Waterfall Inflation**: Fixed 3 bugs causing 4-10x inflation in contribution totals
- **Hill Parameter Similarity**: All channels now learn distinct saturation curves via data-driven initialization
- **Example Scripts**: Fixed `example_response_curves.py` for current data format and API

### Changed
- **Terminology**: Replaced "visits" with "KPI units" throughout documentation
- **Attribution Display**: Waterfall and donut charts now show correct totals in original scale
- **Response Curves**: Simplified to single unified plot (removed scatter, parameter table)
- **Visualization Thresholds**: Lowered DAG network threshold to 0.30 for better visibility
- **Configuration**: Added `media_contribution_prior`, `attribution_reg_weight`, `seasonal_prior`, `seasonal_reg_weight`

### Performance
- **Training R²**: 0.950 (target 0.93 achieved)
- **Holdout R²**: 0.842 (target 0.8 achieved with moderate regularization)
- **Performance Gap**: 10.8% (strong generalization on temporal holdout)
- **Attribution Accuracy**: Media 38.7%, Baseline 40.3%, Seasonal 25.6%, Controls 0.9%
- **Additivity Error**: <0.1% (components sum exactly to 100%)
- **Training Stability**: Improved with dynamic loss scaling and balanced regularization

### Documentation Updates
- **README.md**: Updated with linear scaling, attribution priors, data-driven Hill initialization, "KPI units" terminology
- **JOSS/paper.md**: Added Software Design sections for linear scaling, attribution priors, Hill initialization
- **docs/source/quickstart.rst**: Already current with pipeline parameter and zero-leakage best practices
- **examples/quickstart.ipynb**: Fixed to use current API, removed emojis
- **examples/example_response_curves.py**: Fixed data loading for current CSV structure, updated API calls

### API Changes
```python
# Linear scaling is now default (no user code changes needed)
# Components automatically sum to 100% in original scale

# Configure attribution priors (optional)
config['media_contribution_prior'] = 0.40  # Target 40% media attribution
config['attribution_reg_weight'] = 0.5     # Regularization strength (0.5 = balanced)

# Seasonal regularization (optional)
config['seasonal_prior'] = 0.20           # Target 20% seasonal contribution
config['seasonal_reg_weight'] = 0.2       # Regularization strength
```

### Technical Details
- **Scaling Method**: `y_scaled = y / y_mean_per_region` (replaces log1p)
- **Inverse Transform**: `y_orig = y_scaled * prediction_scale * y_mean_per_region`
- **Attribution Loss**: `(media_contribution - prior * total_prediction)²` with dynamic scaling
- **Hill Initialization**: `g_init = SOV_75th_percentile` per channel (not uniform)
- **Seasonality**: Min-max [0, 1] scaling with optional regularization to prevent suppression

### Backward Compatibility
- **API**: No breaking changes to user-facing API
- **Model Loading**: Old log-space models not compatible (retrain required)
- **Data Format**: Same data format requirements
- **Configuration**: New config keys are optional (sensible defaults)

### Migration Guide
For users upgrading from v1.0.18:
1. **Retrain models**: Log-space models not compatible with linear scaling
2. **Update scripts**: Change "visits" to "KPI units" in custom visualizations
3. **Configure priors**: Set `media_contribution_prior` if you have business targets
4. **Review attribution**: Contributions now sum to 100% and may differ from log-space results

## [1.0.18] - 2025-10-22

### Budget Optimization & Documentation Cleanup

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



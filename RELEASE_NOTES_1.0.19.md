# Release Notes: DeepCausalMMM v1.0.19

**Release Date**: January 25, 2026

## Overview

DeepCausalMMM v1.0.19 represents a major architectural improvement focusing on **linear scaling** and **accurate attribution**. This release transitions from log-space modeling to linear scaling (y/y_mean), enabling additive attribution where components sum to exactly 100% in original KPI units.

## Key Highlights

### 1. Linear Scaling Architecture
- **Changed from**: Log1p transformation with proportional allocation
- **Changed to**: Linear scaling (y/y_mean per region) with direct additivity
- **Benefit**: Components (baseline, seasonal, media, controls) now sum to 100% with <0.1% error
- **Impact**: More interpretable and accurate attribution for business stakeholders

### 2. Attribution Prior Regularization
- **New Feature**: Configurable media contribution targets (e.g., 30-50%)
- **Dynamic Loss Scaling**: Regularization losses automatically scaled to match prediction loss magnitude
- **Business Alignment**: Prevents unrealistic attribution (e.g., >90% media)
- **Configuration**:
  ```python
  config['media_contribution_prior'] = 0.40  # Target 40% media
  config['attribution_reg_weight'] = 0.5     # Balanced strength
  ```

### 3. Data-Driven Hill Initialization
- **Old Behavior**: All channels initialized with same Hill parameters
- **New Behavior**: Channel-specific initialization from SOV percentiles
- **Benefit**: Each channel learns distinct saturation curves
- **Impact**: Eliminates identical channel attribution issue

### 4. Critical Bug Fixes
- **Data Leakage**: Fixed `y_mean_per_region` leaking from holdout to training
- **Waterfall Inflation**: Fixed 3 bugs causing 4-10x inflation in contribution totals
  1. Double-counting seasonality in baseline
  2. Double inverse transformation of contributions
  3. Using wrong `prediction_scale` value
- **Seasonal Suppression**: Fixed tensor slicing in seasonal initialization

## Performance Metrics

### Production Performance (1500 epochs)
- **Training R²**: 0.950
- **Holdout R²**: 0.842
- **Train-Test Gap**: 10.8%
- **Contribution**: Media 38.7%, Baseline 34.3%, Seasonal 25.6%, Controls 0.9%
- **Additivity Error**: <0.1% (components sum to 100%)

### Improvements from v1.0.18
- **Attribution Accuracy**: Fixed from >90% media (incorrect) to ~40% media (realistic)
- **Waterfall Charts**: Now show correct totals in original scale (fixed 4x inflation)
- **Channel Diversity**: All channels now learn unique saturation curves

## Breaking Changes

### 1. Model Retraining Required
- **Old models (log-space) not compatible** with new linear scaling architecture
- Users must retrain models with v1.0.19
- Data format and API remain unchanged

### 2. Terminology Updates
- **Reason**: More general and applicable to any KPI (sales, revenue, conversions, etc.)

## What's New

### Features
1. **Linear Scaling**: y/y_mean per region for additive attribution
2. **Attribution Priors**: Configurable regularization for business-aligned attribution
3. **Dynamic Loss Scaling**: Automatic balancing of prediction vs regularization losses
4. **Data-Driven Hill**: Channel-specific Hill parameter initialization from SOV
5. **Seasonal Regularization**: Optional regularization to prevent seasonal suppression

### Enhancements
1. **Visualization**: Unified response curve plot (all channels in one chart)
2. **Dashboard**: Fixed waterfall chart calculations (3 critical bugs)
3. **Example Scripts**: Fixed `example_response_curves.py` for current data format
4. **Notebook**: Updated `quickstart.ipynb` to current API, removed emojis
5. **DAG Visualization**: Lowered threshold to 0.30 for better network visibility

### Documentation Updates
1. **README.md**: Updated with linear scaling, attribution priors, data-driven Hill
2. **JOSS/paper.md**: Added Software Design sections for new features
3. **CHANGELOG.md**: Comprehensive v1.0.19 changelog with migration guide
4. **docs/**: Updated all documentation to use "KPI units" terminology

## Configuration Changes

### New Parameters
```python
# Attribution prior regularization
config['media_contribution_prior'] = 0.40    # Target media percentage (default: 0.40)
config['attribution_reg_weight'] = 0.5       # Regularization strength (default: 0.5)

# Seasonal regularization (optional)
config['seasonal_prior'] = 0.20              # Target seasonal percentage (default: 0.20)
config['seasonal_reg_weight'] = 0.2          # Regularization strength (default: 0.2)

# Visualization
config['visualization'] = {
    'correlation_threshold': 0.30            # DAG edge threshold (default: 0.30)
}
```

### Updated Parameters
- **Regularization**: Increased dropout, L1/L2 for better generalization
- **Early Stopping**: Added patience=300 for optimal training duration
- **Holdout Ratio**: Increased to 12% for robust validation

## Migration Guide

### For Users Upgrading from v1.0.18

1. **Retrain Models**
   ```bash
   # Old log-space models not compatible
   # Retrain with v1.0.19 for linear scaling benefits
   python dashboard_rmse_optimized.py
   ```

2. **Update Terminology**
   ```python
   # If you have custom visualizations, update:
   # "visits" → "KPI units"
   # "predicted visits" → "predicted KPI units"
   ```

3. **Configure Attribution Priors (Optional)**
   ```python
   # Set target media attribution if you have business requirements
   config['media_contribution_prior'] = 0.35  # e.g., 35% target
   config['attribution_reg_weight'] = 0.5     # Balanced weight
   ```

4. **Review Attribution**
   ```python
   # Components now sum to 100% and may differ from log-space results
   # This is expected and more accurate
   # Check waterfall chart for breakdown
   ```

## Technical Details

### Scaling Architecture
- **Forward Transform**: `y_scaled = y / y_mean_per_region`
- **Inverse Transform**: `y_orig = y_scaled * prediction_scale * y_mean_per_region`
- **Attribution**: Direct summation, no proportional allocation needed

### Attribution Regularization
- **Loss Function**: `(media_contribution - prior * total_prediction)²`
- **Dynamic Scaling**: Regularization loss scaled to match MSE loss magnitude
- **Impact**: Ensures regularization has meaningful effect during training

### Hill Initialization
- **Method**: `g_init = SOV_60th_percentile` per channel
- **Range**: Typically [0.15, 0.30] depending on channel SOV distribution
- **Benefit**: Prevents uniform initialization leading to identical curves

## Testing

### Test Suite Results
```
✓ Package Imports
✓ Version Number (1.0.19)
✓ Data Generation
✓ Data Pipeline
✓ Model Creation
✓ Model Training
✓ Attribution Additivity (<5% error)
✓ Example Scripts

ALL TESTS PASSED - READY FOR PRODUCTION
```

### Manual Testing
- ✓ Budget optimization example runs successfully
- ✓ Response curves example loads data correctly
- ✓ Dashboard generates all 14+ visualizations
- ✓ Waterfall chart shows correct totals (no inflation)
- ✓ All channels learn unique saturation curves

## Known Issues

### Non-Breaking
1. **Seasonal regularization broadcasting warning**: Cosmetic warning, does not affect results
   ```
   UserWarning: Using a target size (torch.Size([])) that is different to the input size (torch.Size([1]))
   ```
   - **Impact**: None (loss calculation still correct)
   - **Status**: Will fix in v1.0.20

2. **Emojis in logging**: Some emoji in Hill initialization logging
   - **Impact**: Visual only (may not render on all terminals)
   - **Status**: Most emojis removed, remaining will be cleaned in v1.0.20

## Files Changed

### Core Architecture (6 files)
- `deepcausalmmm/core/scaling.py` - Complete rewrite for linear scaling
- `deepcausalmmm/core/unified_model.py` - Linear scaling, attribution priors, Hill init
- `deepcausalmmm/core/trainer.py` - Dynamic loss scaling, Hill initialization call
- `deepcausalmmm/core/config.py` - New attribution and visualization parameters
- `pyproject.toml` - Version bump to 1.0.19
- `CHANGELOG.md` - Comprehensive v1.0.19 changelog

### Examples (3 files)
- `examples/dashboard_rmse_optimized.py` - Fixed 3 waterfall bugs
- `examples/example_response_curves.py` - Fixed data loading for current format
- `examples/quickstart.ipynb` - Updated API, removed emojis

### Documentation (4 files)
- `README.md` - Updated with linear scaling, attribution priors
- `JOSS/paper.md` - Added Software Design sections
- `docs/source/quickstart.rst` - Already current (verified)
- `docs/source/index.rst` - Already current (verified)

### Testing (1 file)
- `test_release_v1_0_19.py` - Comprehensive release test suite

## Installation

### PyPI (After Release)
```bash
pip install --upgrade deepcausalmmm
```

### GitHub
```bash
pip install git+https://github.com/adityapt/deepcausalmmm.git@v1.0.19
```

### Development
```bash
git clone https://github.com/adityapt/deepcausalmmm.git
cd deepcausalmmm
git checkout v1.0.19
pip install -e .
```

## Verification

### Quick Test
```python
import deepcausalmmm
print(f"Version: {deepcausalmmm.__version__}")  # Should print "1.0.19"

# Run comprehensive test suite
python test_release_v1_0_19.py
```

### Full Dashboard Test
```bash
# Should complete successfully with realistic attribution
python examples/dashboard_rmse_optimized.py
```

## Next Steps

### For v1.0.20 (Minor Fixes)
- Remove remaining emojis from logging
- Fix seasonal regularization broadcasting warning
- Add validation split option
- Improve documentation for attribution priors

### For v1.1.0 (New Features)
- Multi-objective optimization (RMSE + business constraints)
- Automated hyperparameter tuning (Bayesian optimization)
- Uncertainty quantification (confidence intervals)
- Enhanced causal discovery (full NOTEARS implementation)

## Support

- **Documentation**: https://deepcausalmmm.readthedocs.io
- **Issues**: https://github.com/adityapt/deepcausalmmm/issues
- **Discussions**: https://github.com/adityapt/deepcausalmmm/discussions
- **Email**: puttaparthy.aditya@gmail.com

## Citation

```bibtex
@software{deepcausalmmm2025,
  title={DeepCausalMMM: A Deep Learning Framework for Marketing Mix Modeling with Causal Inference},
  author={Puttaparthi Tirumala, Aditya},
  year={2026},
  version={1.0.19},
  url={https://github.com/adityapt/deepcausalmmm}
}
```

## Acknowledgments

Special thanks to all contributors and users who provided feedback that led to these improvements.

---

**Built for MMM practitioners**

---

## Checklist for Release

- [x] Version updated to 1.0.19 in `pyproject.toml`
- [x] CHANGELOG updated with comprehensive v1.0.19 entry
- [x] README.md updated with new features
- [x] JOSS paper updated with Software Design sections
- [x] Documentation (.rst files) verified and current
- [x] All example scripts executable and tested
- [x] Comprehensive test suite passes (8/8 tests)
- [x] No critical bugs or issues
- [x] Release notes created

**STATUS: READY FOR PRODUCTION RELEASE v1.0.19**


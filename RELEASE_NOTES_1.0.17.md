# Release Notes: DeepCausalMMM v1.0.17

**Release Date:** October 5, 2025

## 🎯 Major Feature: Response Curves & Saturation Analysis

This release introduces comprehensive non-linear response curve fitting capabilities, enabling marketers to understand saturation effects and optimize budget allocation across channels.

---

## ✨ New Features

### 📉 Response Curve Module (`postprocess/response_curves.py`)

Complete implementation of Hill equation-based saturation analysis:

- **`ResponseCurveFit` Class**: Fits S-shaped saturation curves to channel data
- **National-Level Aggregation**: Automatically aggregates DMA-week data to national weekly level
- **Interactive Visualizations**: Plotly-based plots with hover details and curve equations
- **Performance Metrics**: R², slope, and half-saturation point for each channel
- **Backward Compatibility**: Maintains `ResponseCurveFitter` alias and legacy method names

**Key Capabilities:**
```python
from deepcausalmmm.postprocess import ResponseCurveFit

fitter = ResponseCurveFit(
    data=channel_data,
    model_level='Overall',
    date_col='week_monday'
)

fitter.fit(
    x_label='Impressions',
    y_label='Contributions',
    title='Channel Response Curve',
    save_figure=True,
    output_path='response_curve.html'
)

print(f"Slope: {fitter.slope:.3f}")
print(f"Half-Saturation: {fitter.saturation:,.0f}")
print(f"R²: {fitter.r_2:.3f}")
```

### 📊 Dashboard Integration

Response curves now integrated into the comprehensive dashboard:

- **Summary Table**: R², slope, and saturation for all channels
- **Individual Plots**: Interactive response curve for each channel
- **Sorted by Quality**: Channels ranked by R² score
- **Direct Access**: Embedded iframes in master dashboard

---

## 🔧 Enhancements

### Hill Transformation Improvements

**Enforced Minimum Slope** (`unified_model.py`):
- Changed slope constraint from `[0.1, 2.0]` to `[2.0, 5.0]`
- Ensures proper S-shaped curves with clear diminishing returns
- Improved initialization: `hill_a = 2.5` (instead of 1.5)

**Formula:**
```python
a = torch.clamp(F.softplus(hill_a), 2.0, 5.0)
y = x^a / (x^a + g^a)
```

### Proportional Allocation Method

**Enhanced `inverse_transform_contributions`** (`scaling.py`):
- Correctly scales log-space contributions to original scale
- Uses proportional allocation: `component_orig = (component_log / total_log) × y_pred_orig`
- Handles baseline, media, control, and seasonal components
- Prevents double-counting and maintains consistency

**Benefits:**
- Accurate contribution attribution
- Proper handling of log-space modeling
- Consistent with total predictions

### Waterfall Chart Fixes

**Corrected Total Calculation** (`dashboard_rmse_optimized.py`):
- Changed from summing individual `expm1` components
- Now uses actual `predictions_orig.sum()`
- Eliminates incorrect scaling artifacts

**All Dashboard Plots Updated:**
- Contributions Stacked Area Chart
- Individual Contributions Lines
- Channel Effectiveness Analysis
- Contribution Percentages
- All now use proportionally allocated contributions

---

## 📚 Documentation

### README.md Updates

- Added Response Curves section with usage examples
- Updated feature count: 14+ interactive visualizations
- Included Hill equation explanation and best practices
- Added code examples for response curve fitting

### ReadTheDocs Documentation

**New API Documentation** (`docs/source/api/response_curves.rst`):
- Complete API reference for `ResponseCurveFit`
- Usage examples and best practices
- Parameter interpretation guide
- Troubleshooting common issues
- Mathematical background on Hill equation

**Updated Quickstart Guide** (`docs/source/quickstart.rst`):
- Added Response Curves Analysis section
- Included practical examples
- Explained business value and use cases

**Updated API Index** (`docs/source/api/index.rst`):
- Added response_curves to table of contents
- Organized under "Postprocessing and Analysis"

---

## 🧪 Testing

### New Test Suite (`tests/unit/test_response_curves.py`)

Comprehensive test coverage with 12 test cases:

**Unit Tests:**
- Initialization and configuration
- Hill equation calculation
- Curve fitting accuracy
- Prediction functionality
- Summary generation
- Backward compatibility

**Edge Cases:**
- Minimal data points
- Zero contributions
- Monotonic data
- Sparse data handling

**Integration Tests:**
- Full workflow from data to fitted curve
- Parameter validation
- Prediction accuracy

**Results:** ✅ All 28 tests pass (12 new + 16 existing)

---

## 🔄 Backward Compatibility

### Maintained Compatibility

**Alias Support:**
```python
# Both work identically
from deepcausalmmm.postprocess import ResponseCurveFit
from deepcausalmmm.postprocess import ResponseCurveFitter  # Alias
```

**Legacy Method Names:**
```python
fitter.Hill(x, *params)        # → fitter._hill_equation(x, *params)
fitter.get_param()             # → fitter.fit_curve()
fitter.regression()            # → fitter.calculate_r2_and_plot()
fitter.fit_model()             # → fitter.fit()
```

**Legacy Parameter Names:**
```python
fitter.Modellevel  # → fitter.model_level
fitter.Datecol     # → fitter.date_col
```

---

## 🐛 Bug Fixes

### Fixed Issues

1. **Log-space Scaling**: Corrected contribution scaling using proportional allocation
2. **Waterfall Totals**: Fixed incorrect total calculation (was summing `expm1` of components)
3. **Burn-in Handling**: Properly trimmed burn-in padding from all components
4. **Shape Mismatches**: Resolved tensor shape inconsistencies in inverse transforms
5. **Dashboard Plots**: Ensured all plots use consistently scaled contributions

---

## 📦 Package Updates

### Modified Files

**Core Module:**
- `deepcausalmmm/__init__.py`: Added ResponseCurveFit export
- `deepcausalmmm/postprocess/__init__.py`: Added ResponseCurveFit export
- `deepcausalmmm/postprocess/response_curves.py`: **NEW** - Complete response curve module
- `deepcausalmmm/core/unified_model.py`: Hill parameter constraints (lines 178, 580)
- `deepcausalmmm/core/scaling.py`: Proportional allocation method
- `deepcausalmmm/core/config.py`: Updated default epochs to 2500

**Dashboard:**
- `dashboard_rmse_optimized.py`: Integrated response curves section

**Tests:**
- `tests/unit/test_response_curves.py`: **NEW** - Comprehensive test suite

**Documentation:**
- `README.md`: Added response curves section
- `CHANGELOG.md`: Added v1.0.17 entry
- `docs/source/api/response_curves.rst`: **NEW** - Complete API docs
- `docs/source/api/index.rst`: Added response_curves
- `docs/source/quickstart.rst`: Added response curves section

**Configuration:**
- `pyproject.toml`: Version bump to 1.0.17

---

## 🎯 Technical Details

### Hill Equation

The response curve uses the Hill equation for saturation modeling:

```
y = x^a / (x^a + g^a)
```

Where:
- `x`: Input variable (impressions or spend)
- `y`: Output variable (contributions or response)
- `a`: Slope parameter (controls steepness of S-curve)
- `g`: Half-saturation point (x value where y = 0.5)

### Slope Constraint

**Enforced Range:** `a ∈ [2.0, 5.0]`

**Rationale:**
- `a >= 2.0`: Ensures proper S-shaped curve with clear saturation
- `a < 2.0`: Results in gentle, nearly linear curves
- `a >= 3.0`: Very strong saturation effects

### Proportional Allocation

**Formula:**
```python
component_orig = (component_log / total_log) × y_pred_orig
```

**Where:**
- `component_log`: Individual component in log-space
- `total_log`: Sum of all components in log-space
- `y_pred_orig`: Total prediction in original scale

**Ensures:**
- Sum of components equals total prediction
- Correct handling of log-space modeling
- No double-counting or scaling artifacts

---

## 📈 Performance

### Model Performance (Unchanged)

- **Training R²**: 0.965
- **Holdout R²**: 0.930
- **Performance Gap**: 3.6%
- **Training RMSE**: 254,911 visits (34.7%)
- **Holdout RMSE**: 324,584 visits (38.7%)

### Response Curve Fitting

**Typical Results:**
- **R² Range**: 0.60 - 0.95 (depending on channel)
- **Slope Range**: 2.0 - 4.5 (enforced minimum)
- **Saturation Points**: Vary by channel (data-driven)

---

## 🚀 Migration Guide

### Upgrading from v1.0.16

**Installation:**
```bash
pip install --upgrade deepcausalmmm
```

**No Breaking Changes:**
- All existing code continues to work
- New features are additive
- Backward compatibility maintained

**New Capabilities:**
```python
# Import response curves
from deepcausalmmm.postprocess import ResponseCurveFit

# Fit curves to your channel data
fitter = ResponseCurveFit(
    data=channel_data,
    model_level='Overall'
)

fitter.fit(
    x_label='Impressions',
    y_label='Contributions',
    save_figure=True,
    output_path='response_curve.html'
)
```

---

## 🎓 Use Cases

### Budget Optimization

Identify channels with:
- **Low saturation**: Room for increased investment
- **High saturation**: Diminishing returns, reallocate budget
- **Optimal spend**: Half-saturation point indicates sweet spot

### Channel Comparison

Compare across channels:
- **Efficiency**: Which channels saturate faster?
- **Capacity**: Which channels can handle more volume?
- **Returns**: Which channels show strongest S-curves?

### Strategic Planning

Inform decisions:
- **Investment levels**: Where to increase/decrease spend
- **Channel mix**: Optimal portfolio allocation
- **Forecasting**: Predict outcomes at different spend levels

---

## 🙏 Acknowledgments

Special thanks to the MMM community for feedback and testing that led to these improvements.

---

## 📞 Support

- **Documentation**: [deepcausalmmm.readthedocs.io](https://deepcausalmmm.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/adityapt/deepcausalmmm/issues)
- **Examples**: See `examples/` directory

---

**DeepCausalMMM v1.0.17** - Now with comprehensive response curve analysis! 🚀📉

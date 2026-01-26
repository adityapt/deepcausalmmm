Response Curves
===============

The response curves module provides non-linear saturation analysis using Hill equations to model diminishing returns in marketing channels.

Overview
--------

Response curves help you understand:

* **Saturation Points**: When additional spend/impressions yield diminishing returns
* **Optimal Allocation**: Which channels have room for increased investment
* **S-Shaped Relationships**: Non-linear effects of marketing activities
* **Channel Efficiency**: Compare saturation across different channels

Key Features
------------

* **Hill Equation Fitting**: Fits S-shaped saturation curves to channel data
* **Automatic Aggregation**: Aggregates DMA-week data to national weekly level
* **Direct Attribution**: Works with additive contributions from linear scaling (v1.0.19+)
* **Interactive Visualizations**: Plotly-based plots with hover details
* **Performance Metrics**: R², slope, and saturation point for each channel
* **Backward Compatibility**: Maintains support for legacy method names

ResponseCurveFit Class
----------------------

.. autoclass:: deepcausalmmm.postprocess.response_curves.ResponseCurveFit
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Basic Usage
-----------

Fitting Response Curves
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from deepcausalmmm.postprocess import ResponseCurveFit
    import pandas as pd

    # Prepare your data
    channel_data = pd.DataFrame({
        'week': [1, 2, 3, ...],
        'impressions': [10000, 15000, 20000, ...],
        'contributions': [500000, 650000, 750000, ...]
    })

    # Initialize fitter
    fitter = ResponseCurveFit(
        data=channel_data,
        x_col='impressions',
        y_col='contributions',
        model_level='national',
        date_col='week'
    )

    # Fit the curve
    slope, saturation = fitter.fit_curve()
    print(f"Slope (a): {slope:.3f}")
    print(f"Half-Saturation Point (g): {saturation:.0f}")

Calculating R² and Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Calculate R² and generate interactive plot
    r2_score = fitter.calculate_r2_and_plot(
        save_path='response_curve_channel.html'
    )
    print(f"R² Score: {r2_score:.3f}")

Complete Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from deepcausalmmm.postprocess import ResponseCurveFit
    import pandas as pd

    # Load channel data (impressions and contributions)
    df = pd.read_csv('channel_data.csv')

    # Initialize and fit
    fitter = ResponseCurveFit(
        data=df,
        x_col='impressions',
        y_col='contributions',
        model_level='national',
        date_col='week'
    )

    # Get fitted parameters
    slope, saturation = fitter.fit_curve()

    # Generate plot and get R²
    r2 = fitter.calculate_r2_and_plot(save_path='curve.html')

    # Interpret results
    print(f"Channel Saturation Analysis:")
    print(f"  Slope (a): {slope:.3f}")
    print(f"  Half-Saturation (g): {saturation:,.0f} impressions")
    print(f"  Fit Quality (R²): {r2:.3f}")
    
    if slope >= 2.0:
        print("  Strong S-shaped curve (diminishing returns)")
    else:
        print("  Gentle curve (less pronounced saturation)")
    
    if r2 >= 0.8:
        print("  Excellent fit")
    elif r2 >= 0.6:
        print("  Good fit")
    else:
        print("  Moderate fit - review data quality")

Advanced Usage
--------------

Batch Processing Multiple Channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from deepcausalmmm.postprocess import ResponseCurveFit

    # Assume you have a DataFrame with multiple channels
    all_channels_data = pd.read_csv('all_channels.csv')
    
    results = []
    for channel in all_channels_data['channel'].unique():
        # Filter data for this channel
        channel_df = all_channels_data[
            all_channels_data['channel'] == channel
        ].copy()
        
        # Fit response curve
        fitter = ResponseCurveFit(
            data=channel_df,
            x_col='impressions',
            y_col='contributions',
            model_level='national',
            date_col='week'
        )
        
        slope, saturation = fitter.fit_curve()
        r2 = fitter.calculate_r2_and_plot(
            save_path=f'curves/{channel}_response_curve.html'
        )
        
        results.append({
            'channel': channel,
            'slope': slope,
            'saturation': saturation,
            'r2': r2
        })
    
    # Create summary DataFrame
    summary = pd.DataFrame(results)
    summary = summary.sort_values('r2', ascending=False)
    print(summary)

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

**Slope (a) Parameter:**

* ``a >= 3.0``: Very strong S-curve, rapid saturation
* ``2.0 <= a < 3.0``: Strong S-curve, clear diminishing returns
* ``1.0 <= a < 2.0``: Gentle curve, gradual saturation
* ``a < 1.0``: Very gentle, almost linear

**Half-Saturation Point (g):**

* The impression/spend level where the channel reaches 50% of maximum effect
* Lower values indicate faster saturation
* Compare across channels to identify efficiency

**R² Score:**

* ``R² >= 0.8``: Excellent fit, high confidence
* ``0.6 <= R² < 0.8``: Good fit, reasonable confidence
* ``0.4 <= R² < 0.6``: Moderate fit, review data
* ``R² < 0.4``: Poor fit, investigate data quality or model assumptions

Hill Equation
-------------

The response curve uses the Hill equation:

.. math::

    y = \\frac{x^a}{x^a + g^a}

Where:

* ``x``: Input variable (impressions or spend)
* ``y``: Output variable (contributions or response)
* ``a``: Slope parameter (controls steepness of S-curve)
* ``g``: Half-saturation point (x value where y = 0.5)

Properties:

* **Monotonic**: Always increasing
* **Bounded**: Output between 0 and 1 (when normalized)
* **S-Shaped**: When ``a >= 2.0``
* **Half-Saturation**: ``y(g) = 0.5``

Technical Details
-----------------

Fitting Algorithm
~~~~~~~~~~~~~~~~~

The module uses ``scipy.optimize.curve_fit`` with:

* **Initial Guess**: ``a=1``, ``g=median(x)``
* **Bounds**: ``a ∈ [0.01, 100]``, ``g ∈ [0.01, max(x) × 10]``
* **Method**: Trust Region Reflective (default)
* **Max Iterations**: 10,000

Data Preprocessing
~~~~~~~~~~~~~~~~~~

1. **Aggregation**: Groups by date column and sums x and y
2. **Sorting**: Sorts by x values for consistent fitting
3. **Normalization**: Internally normalizes y for numerical stability
4. **Scaling**: Scales fitted curve back to original y scale

Backward Compatibility
----------------------

Legacy Method Names
~~~~~~~~~~~~~~~~~~~

For backward compatibility, the following legacy method names are supported:

.. code-block:: python

    fitter = ResponseCurveFit(data=df, x_col='x', y_col='y')
    
    # New API (recommended)
    result = fitter._hill_equation(x, a, g)
    slope, sat = fitter.fit_curve()
    r2 = fitter.calculate_r2_and_plot()
    
    # Legacy API (still works)
    result = fitter.Hill(x, a, g)
    slope, sat = fitter.get_param()
    r2 = fitter.regression()
    slope, sat = fitter.fit_model()  # Alias for fit_curve

Legacy Parameter Names
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # New API (recommended)
    fitter = ResponseCurveFit(
        data=df,
        x_col='impressions',
        y_col='contributions',
        model_level='national',
        date_col='week'
    )
    
    # Legacy API (still works)
    fitter = ResponseCurveFit(
        data=df,
        x_col='impressions',
        y_col='contributions',
        Modellevel='national',  # Old name
        Datecol='week'  # Old name
    )

ResponseCurveFitter Alias
~~~~~~~~~~~~~~~~~~~~~~~~~

The original class name ``ResponseCurveFitter`` is maintained as an alias:

.. code-block:: python

    from deepcausalmmm.postprocess import ResponseCurveFitter
    
    # This works identically to ResponseCurveFit
    fitter = ResponseCurveFitter(data=df, x_col='x', y_col='y')

Best Practices
--------------

Data Quality
~~~~~~~~~~~~

* **Sufficient Points**: Use at least 20-30 data points for reliable fitting
* **Range Coverage**: Ensure data covers a wide range of x values
* **Outlier Handling**: Remove or investigate extreme outliers before fitting
* **Monotonicity**: Response curves assume y generally increases with x

Aggregation Level
~~~~~~~~~~~~~~~~~

* **National Weekly**: Recommended for most analyses (reduces noise)
* **Regional**: Use when analyzing regional differences
* **DMA-Level**: Use with caution (high variance)

Model Validation
~~~~~~~~~~~~~~~~

* **R² Threshold**: Aim for R² >= 0.6 for reliable insights
* **Visual Inspection**: Always review the generated plots
* **Business Logic**: Ensure fitted parameters make business sense
* **Cross-Validation**: Test on holdout periods when possible

Common Issues
~~~~~~~~~~~~~

**Poor Fit (Low R²)**:

* Check for outliers or data quality issues
* Verify monotonic relationship between x and y
* Consider if Hill equation is appropriate for your data
* Try different aggregation levels

**Unrealistic Parameters**:

* Very high slope (a > 10): May indicate overfitting
* Very high saturation (g >> max(x)): Channel not reaching saturation
* Very low saturation (g << median(x)): Most data in saturated region

**Convergence Issues**:

* Increase max iterations
* Try different initial guesses
* Check for numerical issues (very large/small values)
* Normalize your data before fitting

Examples
--------

See the ``examples/`` directory for complete examples:

* ``example_response_curves.py``: Full workflow with DeepCausalMMM integration
* ``dashboard_rmse_optimized.py``: Dashboard with integrated response curves

See Also
--------

* :doc:`analysis`: General analysis utilities
* :doc:`core`: Core model components
* :doc:`../tutorials/index`: Step-by-step tutorials
* :doc:`../examples/index`: Practical examples

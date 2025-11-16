Budget Optimization
===================

The budget optimization module provides tools to optimize marketing budget allocation across channels using fitted response curves from your trained DeepCausalMMM model.

Core Classes
------------

.. autoclass:: deepcausalmmm.postprocess.optimization.BudgetOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: deepcausalmmm.postprocess.optimization.OptimizationResult
   :members:
   :undoc-members:
   :show-inheritance:

Optimization Functions
----------------------

.. autofunction:: deepcausalmmm.postprocess.optimization.optimize_budget_from_curves

Utility Functions
-----------------

.. autofunction:: deepcausalmmm.postprocess.optimization_utils.prepare_optimization_data

.. autofunction:: deepcausalmmm.postprocess.optimization_utils.fit_response_curves_batch

.. autofunction:: deepcausalmmm.postprocess.optimization_utils.create_optimizer_from_model_output

.. autofunction:: deepcausalmmm.postprocess.optimization_utils.compare_current_vs_optimal

.. autofunction:: deepcausalmmm.postprocess.optimization_utils.generate_optimization_report

Usage Example
-------------

After training your model and fitting response curves:

.. code-block:: python

    from deepcausalmmm import BudgetOptimizer, optimize_budget_from_curves
    
    # Option 1: Quick optimization from curves DataFrame
    result = optimize_budget_from_curves(
        budget=1_000_000,
        curve_params=fitted_curves_df,  # DataFrame with columns: channel, top, bottom, saturation, slope
        num_weeks=52,
        constraints={'TV': {'lower': 100000, 'upper': 600000}}
    )
    
    # Option 2: Using BudgetOptimizer directly
    optimizer = BudgetOptimizer(
        budget=1_000_000,
        channels=['TV', 'Search', 'Social'],
        response_curves={
            'TV': {'top': 500000, 'bottom': 0, 'saturation': 300000, 'slope': 1.5},
            'Search': {'top': 400000, 'bottom': 0, 'saturation': 250000, 'slope': 1.3},
            'Social': {'top': 300000, 'bottom': 0, 'saturation': 200000, 'slope': 1.2}
        },
        num_weeks=52,
        method='SLSQP'
    )
    
    # Set constraints
    optimizer.set_constraints({
        'TV': {'lower': 100000, 'upper': 600000},
        'Search': {'lower': 150000, 'upper': 500000}
    })
    
    # Run optimization
    result = optimizer.optimize()
    
    # View results
    print(f"Success: {result.success}")
    print(f"Predicted Response: {result.predicted_response:,.0f}")
    print(result.by_channel)

Optimization Methods
--------------------

The optimizer supports multiple optimization methods:

- **SLSQP** (default): Sequential Least Squares Programming - fast and stable
- **trust-constr**: Trust-region constrained optimization - robust for complex constraints
- **differential_evolution**: Global optimization - explores entire search space
- **hybrid**: Combines global search with local refinement

Response Curves
---------------

The optimizer uses Hill equation curves fitted from your model:

.. math::

    response = bottom + (top - bottom) \\frac{spend^{slope}}{saturation^{slope} + spend^{slope}}

Where:
    - **top**: Maximum response (saturation level)
    - **bottom**: Minimum response (typically 0)
    - **saturation**: Spend level at half-maximum response
    - **slope**: Steepness of the curve (elasticity)


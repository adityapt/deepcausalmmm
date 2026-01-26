End-to-End MMM Analysis
========================

This tutorial walks through a complete Marketing Mix Modeling workflow using DeepCausalMMM, from data generation to budget optimization.

Overview
--------

In this tutorial, you'll learn how to:

1. Generate synthetic MMM data
2. Configure and train a model
3. Analyze attribution results
4. Fit response curves
5. Optimize budget allocation

This entire workflow uses synthetic data, so you can run it immediately without any external data files.

Step 1: Setup and Data Generation
----------------------------------

First, let's import necessary libraries and generate synthetic MMM data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import torch
    from deepcausalmmm import get_device
    from deepcausalmmm.core import get_default_config
    from deepcausalmmm.core.trainer import ModelTrainer
    from deepcausalmmm.core.data import UnifiedDataPipeline
    from deepcausalmmm.utils.data_generator import ConfigurableDataGenerator
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize synthetic data generator
    generator = ConfigurableDataGenerator()
    
    # Generate MMM dataset
    # Returns: X_media [regions, weeks, channels], X_control, y [regions, weeks]
    X_media, X_control, y = generator.generate_mmm_dataset(
        n_regions=10,
        n_weeks=104,
        n_media_channels=5,
        n_control_channels=3
    )
    
    print(f"Generated data shapes:")
    print(f"  Media: {X_media.shape}")      # (10, 104, 5)
    print(f"  Control: {X_control.shape}")  # (10, 104, 3)
    print(f"  Target: {y.shape}")           # (10, 104)

Understanding the Data
~~~~~~~~~~~~~~~~~~~~~~

Our synthetic dataset represents:

- **10 regions** (e.g., different DMAs or geographic markets)
- **104 weeks** (~2 years of data)
- **5 media channels** (e.g., TV, Search, Social, Display, Radio)
- **3 control variables** (e.g., price, promotion, competitor spend)
- **Target variable** (e.g., sales, revenue, or conversions)

Step 2: Configure the Model
----------------------------

DeepCausalMMM is highly configurable. Let's start with the default configuration and customize key parameters:

.. code-block:: python

    # Get default configuration
    config = get_default_config()
    
    # Customize for our use case
    config['n_epochs'] = 500  # Reduce for faster training (default: 1500)
    config['learning_rate'] = 0.001
    
    # Attribution prior regularization (optional but recommended)
    config['media_contribution_prior'] = 0.40  # Target 40% media attribution
    config['attribution_reg_weight'] = 0.5     # Balanced regularization
    
    # Check device
    device = get_device()
    print(f"Training on: {device}")

Key Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **n_epochs**: Number of training iterations (500-1500)
- **media_contribution_prior**: Target percentage for media attribution (0.30-0.50)
- **attribution_reg_weight**: Regularization strength (0.0-1.0)
- **holdout_ratio**: Validation split percentage (default: 0.12)

Step 3: Prepare Data Pipeline
------------------------------

The UnifiedDataPipeline handles all data preprocessing:

.. code-block:: python

    # Initialize pipeline
    pipeline = UnifiedDataPipeline(config)
    
    # Split data temporally (training: 88%, holdout: 12%)
    train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
    
    # Transform training data (fit scalers)
    train_tensors = pipeline.fit_and_transform_training(train_data)
    
    # Transform holdout data (use fitted scalers)
    holdout_tensors = pipeline.transform_holdout(holdout_data)
    
    print(f"Training weeks: {train_tensors['y'].shape[1]}")
    print(f"Holdout weeks: {holdout_tensors['y'].shape[1]}")

What the Pipeline Does
~~~~~~~~~~~~~~~~~~~~~~~

1. **Temporal Split**: Splits data by time (last 12% for validation)
2. **Scaling**: 
   
   - Media: SOV (Share of Voice) scaling
   - Controls: Z-score normalization
   - Target: Linear scaling (y/y_mean per region)

3. **Tensor Conversion**: Converts to PyTorch tensors

Step 4: Create and Train Model
-------------------------------

Now let's create the model and train it:

.. code-block:: python

    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Create model
    model = trainer.create_model(
        n_media=train_tensors['X_media'].shape[2],      # 5 channels
        n_control=train_tensors['X_control'].shape[2],  # 3 controls
        n_regions=train_tensors['X_media'].shape[0]     # 10 regions
    )
    
    # Create optimizer and scheduler
    trainer.create_optimizer_and_scheduler()
    
    print("Model architecture created successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

Training the Model
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Train model (this will take a few minutes)
    results = trainer.train(
        train_tensors['X_media'], 
        train_tensors['X_control'],
        train_tensors['R'],  # Regional indicators
        train_tensors['y'],
        holdout_tensors['X_media'], 
        holdout_tensors['X_control'],
        holdout_tensors['R'],
        holdout_tensors['y'],
        pipeline=pipeline,
        verbose=True
    )
    
    # View results
    print("\nTraining Results:")
    print(f"  Training R²: {results['final_train_r2']:.3f}")
    print(f"  Holdout R²: {results['final_holdout_r2']:.3f}")
    print(f"  Training RMSE: {results['final_train_rmse']:,.0f}")
    print(f"  Holdout RMSE: {results['final_holdout_rmse']:,.0f}")

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

Performance on synthetic data varies depending on:

- **Number of epochs**: More epochs (1500-2500) improve convergence
- **Data complexity**: Random synthetic data is harder to fit than real data with consistent patterns
- **Regularization**: Attribution priors may constrain fit quality initially

For real-world MMM data with consistent marketing patterns, expect significantly better performance (see benchmark results in README.md)

Step 5: Analyze Attribution
----------------------------

Extract and analyze channel contributions:

.. code-block:: python

    # Get model predictions and contributions
    # Note: model.forward() returns a tuple: (y_pred, media_coeffs, media_contrib, outputs_dict)
    with torch.no_grad():
        y_pred, media_coeffs, media_contrib_direct, outputs = model(
            train_tensors['X_media'],
            train_tensors['X_control'],
            train_tensors['R']
        )
    
    # Extract contributions from outputs dictionary (already in original scale)
    media_contrib = outputs['contributions'].cpu().numpy()  # [regions, weeks, channels]
    baseline_contrib = outputs['baseline'].cpu().numpy()
    seasonal_contrib = outputs['seasonal_contribution'].cpu().numpy()
    control_contrib = outputs['control_contributions'].cpu().numpy()
    
    # Calculate total contributions
    total_media = media_contrib.sum()
    total_baseline = baseline_contrib.sum()
    total_seasonal = seasonal_contrib.sum()
    total_control = control_contrib.sum()
    total_predicted = y_pred.sum().cpu().numpy()  # Use y_pred from tuple return
    
    # Calculate percentages
    print("\nAttribution Breakdown:")
    print(f"  Media: {(total_media/total_predicted)*100:.1f}%")
    print(f"  Baseline: {(total_baseline/total_predicted)*100:.1f}%")
    print(f"  Seasonal: {(total_seasonal/total_predicted)*100:.1f}%")
    print(f"  Controls: {(total_control/total_predicted)*100:.1f}%")
    
    # Verify additivity (should be ~100%)
    total_pct = ((total_media + total_baseline + total_seasonal + total_control) / 
                 total_predicted) * 100
    print(f"\nAdditivity Check: {total_pct:.2f}% (should be ~100%)")

Channel-Level Attribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Sum contributions across regions and time for each channel
    channel_contributions = media_contrib.sum(axis=(0, 1))
    
    # Create channel names
    channel_names = [f'Channel_{i+1}' for i in range(len(channel_contributions))]
    
    # Display results
    print("\nChannel Contributions:")
    for name, contrib in zip(channel_names, channel_contributions):
        pct = (contrib / total_media) * 100
        print(f"  {name}: {contrib:,.0f} ({pct:.1f}% of media)")

Step 6: Fit Response Curves
----------------------------

Analyze saturation and diminishing returns for each channel:

.. code-block:: python

    from deepcausalmmm.postprocess import ResponseCurveFit
    
    # Prepare data for response curve fitting
    results_curves = []
    
    # Important: Use train_data (not X_media) to match contrib dimensions
    # And remove padding from contributions
    burn_in_weeks = config.get('burn_in_weeks', 6)
    media_contrib_no_padding = media_contrib[:, burn_in_weeks:, :]  # Remove padding
    
    for channel_idx in range(train_data['X_media'].shape[2]):
        # Aggregate data for this channel across regions
        channel_spend = train_data['X_media'][:, :, channel_idx].sum(axis=0)  # [weeks]
        channel_contrib = media_contrib_no_padding[:, :, channel_idx].sum(axis=0)  # [weeks]
        
        # ResponseCurveFit expects specific column names
        df = pd.DataFrame({
            'week_monday': pd.date_range('2023-01-01', periods=len(channel_spend), freq='W-MON'),
            'spend': channel_spend,
            'impressions': channel_spend,  # Use spend as impressions if not separate
            'predicted': channel_contrib
        })
        
        # Fit response curve
        fitter = ResponseCurveFit(
            data=df,
            bottom_param=False,  # Assume zero response at zero spend
            model_level='Overall',  # 'Overall' for aggregated, 'DMA' for region-level
            date_col='week_monday'
        )
        
        try:
            # Fit the curve
            fitted_df = fitter.fit()
            
            if fitted_df is not None and hasattr(fitter, 'slope'):
                results_curves.append({
                    'channel': f'Channel_{channel_idx+1}',
                    'slope_a': fitter.slope,
                    'half_saturation_g': fitter.saturation,
                    'top': fitter.top
                })
                
                print(f"\nChannel {channel_idx+1}:")
                print(f"  Slope (a): {fitter.slope:.2f}")
                print(f"  Half-saturation (g): {fitter.saturation:,.0f}")
                print(f"  Top (max response): {fitter.top:,.0f}")
            else:
                print(f"Channel {channel_idx+1}: Curve fitting failed - insufficient data variation")
            
        except Exception as e:
            print(f"Channel {channel_idx+1}: Curve fitting failed - {e}")
    
    # Create summary DataFrame
    if len(results_curves) > 0:
        curves_df = pd.DataFrame(results_curves)
        print("\nResponse Curve Summary:")
        print(curves_df.to_string(index=False))
    else:
        print("\nWarning: No response curves could be fitted. This may occur with:")
        print("  - Insufficient training epochs (try 1500-2500)")
        print("  - Random synthetic data without consistent patterns")
        print("  - Low channel spend variation")

Interpreting Response Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Slope Parameter (a)**:

- ``a >= 2.0``: Strong S-curve with clear saturation
- ``1.0 <= a < 2.0``: Moderate saturation
- ``a < 1.0``: Almost linear (limited saturation observed)

**Half-Saturation Point (g)**:

- Lower values: Channel saturates quickly
- Higher values: Channel has more room for growth

**R² Score**:

- ``>= 0.8``: Excellent fit
- ``0.6-0.8``: Good fit
- ``< 0.6``: Review data quality

Step 7: Budget Optimization
----------------------------

Use fitted response curves to optimize budget allocation:

.. code-block:: python

    from deepcausalmmm.postprocess import BudgetOptimizer
    
    # Initialize optimizer
    optimizer = BudgetOptimizer(
        response_curves_df=curves_df,
        method='SLSQP'
    )
    
    # Define total budget (e.g., $1M)
    total_budget = 1_000_000
    
    # Set channel-specific constraints (optional)
    # Note: Underscores in numbers (e.g., 100_000) are for readability (Python 3.6+)
    constraints = {
        'Channel_1': {'lower': 100_000, 'upper': 400_000},
        'Channel_2': {'lower': 150_000, 'upper': 500_000},
        'Channel_3': {'lower': 50_000, 'upper': 300_000},
        'Channel_4': {'lower': 100_000, 'upper': 400_000},
        'Channel_5': {'lower': 50_000, 'upper': 250_000},
    }
    
    # Optimize allocation
    optimal_allocation = optimizer.optimize_budget(
        total_budget=total_budget,
        constraints=constraints
    )
    
    # Display results
    print("\nOptimal Budget Allocation:")
    for channel, amount in optimal_allocation.items():
        pct = (amount / total_budget) * 100
        print(f"  {channel}: ${amount:,.0f} ({pct:.1f}%)")
    
    # Calculate expected ROI
    expected_roi = optimizer.calculate_expected_roi(optimal_allocation)
    print(f"\nExpected Total ROI: {expected_roi:.2f}x")

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Set Realistic Constraints**: Use historical min/max as guides
2. **Try Different Methods**: 'SLSQP' (fast) or 'trust-constr' (robust)
3. **Validate Results**: Compare optimal allocation with historical spend
4. **Consider Seasonality**: Run separate optimizations for different periods

Step 8: Save and Export Results
--------------------------------

Save your model and results for future use:

.. code-block:: python

    # Save trained model
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'results': results,
        'pipeline': pipeline
    }, 'mmm_model_trained.pth')
    
    # Export attribution results
    attribution_df = pd.DataFrame({
        'component': ['Media', 'Baseline', 'Seasonal', 'Controls'],
        'contribution': [total_media, total_baseline, total_seasonal, total_control],
        'percentage': [
            (total_media/total_predicted)*100,
            (total_baseline/total_predicted)*100,
            (total_seasonal/total_predicted)*100,
            (total_control/total_predicted)*100
        ]
    })
    attribution_df.to_csv('attribution_results.csv', index=False)
    
    # Export response curves
    curves_df.to_csv('response_curves.csv', index=False)
    
    print("\nResults saved successfully!")

Complete Script
---------------

Here's the complete workflow in one script:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import torch
    from deepcausalmmm import get_device
    from deepcausalmmm.core import get_default_config
    from deepcausalmmm.core.trainer import ModelTrainer
    from deepcausalmmm.core.data import UnifiedDataPipeline
    from deepcausalmmm.utils.data_generator import ConfigurableDataGenerator
    from deepcausalmmm.postprocess import ResponseCurveFit, BudgetOptimizer
    
    # Set seed
    np.random.seed(42)
    
    # 1. Generate data
    generator = ConfigurableDataGenerator(
        base_sales=50000, n_media_channels=5, n_control_vars=3
    )
    X_media, X_control, y = generator.generate_mmm_dataset(
        n_regions=10, n_weeks=104
    )
    
    # 2. Configure and prepare
    config = get_default_config()
    config['n_epochs'] = 500
    pipeline = UnifiedDataPipeline(config)
    train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
    train_tensors = pipeline.fit_and_transform_training(train_data)
    holdout_tensors = pipeline.transform_holdout(holdout_data)
    
    # 3. Train model
    trainer = ModelTrainer(config)
    model = trainer.create_model(
        n_media=5, n_control=3, n_regions=10
    )
    trainer.create_optimizer_and_scheduler()
    results = trainer.train(
        train_tensors['X_media'], train_tensors['X_control'],
        train_tensors['R'], train_tensors['y'],
        holdout_tensors['X_media'], holdout_tensors['X_control'],
        holdout_tensors['R'], holdout_tensors['y'],
        pipeline=pipeline, verbose=True
    )
    
    # 4. Analyze results
    print(f"Training R²: {results['final_train_r2']:.3f}")
    print(f"Holdout R²: {results['final_holdout_r2']:.3f}")
    
    # 5. (Optional) Fit response curves and optimize budget
    # See detailed steps above for response curve and optimization code

Next Steps
----------

Now that you've completed this tutorial, you can:

1. **Try Real Data**: Replace synthetic data with your actual MMM data
2. **Customize Configuration**: Experiment with different hyperparameters
3. **Advanced Visualization**: Run the full dashboard (``examples/dashboard_rmse_optimized.py``)
4. **Budget Optimization**: Use the complete optimization workflow (``examples/example_budget_optimization.py``)

Troubleshooting
---------------

**Low Holdout R²**:

- Increase training epochs
- Adjust regularization (L1/L2 weights)
- Check data quality

**Unrealistic Attribution**:

- Set attribution priors (``media_contribution_prior``)
- Increase ``attribution_reg_weight``
- Review channel correlation

**Training Instability**:

- Reduce learning rate
- Increase gradient clipping
- Check for data outliers

See Also
--------

* :doc:`../quickstart`: Quick start guide
* :doc:`../api/trainer`: ModelTrainer API reference
* :doc:`../api/response_curves`: Response curve fitting
* :doc:`../examples/index`: More examples


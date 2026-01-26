Quick Start Guide
=================

This guide will get you up and running with DeepCausalMMM in just a few minutes.

Basic Usage
-----------

Here's a complete example of training a DeepCausalMMM model:

.. code-block:: python

    import numpy as np
    from deepcausalmmm import get_device
    from deepcausalmmm.core import get_default_config
    from deepcausalmmm.core.trainer import ModelTrainer
    from deepcausalmmm.core.data import UnifiedDataPipeline

    # 1. Prepare your data (or load from CSV and reshape)
    # Data shape: [n_regions, n_weeks, n_channels]
    X_media = np.random.uniform(100, 5000, (10, 52, 5))    # Media spend/impressions
    X_control = np.random.uniform(0, 1, (10, 52, 3))       # Control variables
    y = np.random.uniform(1000, 10000, (10, 52))           # Target variable

    # 2. Get optimized configuration
    config = get_default_config()
    
    # 3. Check device availability
    device = get_device()
    print(f"Using device: {device}")

    # 4. Process data with unified pipeline
    pipeline = UnifiedDataPipeline(config)
    train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
    train_tensors = pipeline.fit_and_transform_training(train_data)
    holdout_tensors = pipeline.transform_holdout(holdout_data)

    # 5. Train with ModelTrainer (recommended approach)
    trainer = ModelTrainer(config)
    model = trainer.create_model(
        n_media=train_tensors['X_media'].shape[2],
        n_control=train_tensors['X_control'].shape[2],
        n_regions=train_tensors['X_media'].shape[0]
    )
    trainer.create_optimizer_and_scheduler()
    
    results = trainer.train(
        train_tensors['X_media'], train_tensors['X_control'],
        train_tensors['R'], train_tensors['y'],
        holdout_tensors['X_media'], holdout_tensors['X_control'],
        holdout_tensors['R'], holdout_tensors['y'],
        pipeline=pipeline,
        verbose=True
    )

    # 6. View results
    print(f"Training RMSE: {results['final_train_rmse']:.0f}")
    print(f"Training R²: {results['final_train_r2']:.3f}")
    print(f"Holdout RMSE: {results['final_holdout_rmse']:.0f}")
    print(f"Holdout R²: {results['final_holdout_r2']:.3f}")

One-Command Analysis
--------------------

For the complete analysis with dashboard generation:

.. code-block:: bash

    # Clone the repository first (examples not included in pip install)
    git clone https://github.com/adityapt/deepcausalmmm.git
    cd deepcausalmmm
    pip install -e .
    
    # Run the comprehensive dashboard
    python examples/dashboard_rmse_optimized.py

This will:

* Load and process your data
* Train the model with optimal settings
* Generate comprehensive visualizations
* Create an interactive dashboard
* Save results and model artifacts

Data Format Requirements
------------------------

DeepCausalMMM expects data in NumPy array format with shape **[n_regions, n_weeks, n_channels]**:

* **X_media**: Media spend or impressions [n_regions, n_weeks, n_media_channels]
* **X_control**: Control variables [n_regions, n_weeks, n_control_variables]
* **y**: Target variable [n_regions, n_weeks]

If loading from CSV, you'll need to reshape your data. Your CSV should have this structure:

.. code-block:: text

    date,region,target_variable,media_channel_1,media_channel_2,...,control_var_1,control_var_2,...
    2023-01-01,DMA_1,12500,1000,500,...,0.2,15,...
    2023-01-08,DMA_1,13200,1200,600,...,0.1,18,...
    ...

To convert from CSV to the required format:

.. code-block:: python

    import pandas as pd
    import numpy as np
    
    df = pd.read_csv('mmm_data.csv')
    regions = df['region'].unique()
    weeks = df['week'].unique()
    
    # Reshape to [regions, weeks, channels]
    X_media = np.stack([
        df.pivot(index='region', columns='week', values=col).values
        for col in media_columns
    ], axis=-1)
    
    X_control = np.stack([
        df.pivot(index='region', columns='week', values=col).values
        for col in control_columns
    ], axis=-1)
    
    y = df.pivot(index='region', columns='week', values='target').values

Configuration Customization
----------------------------

Customize the model for your specific use case:

.. code-block:: python

    from deepcausalmmm.core import get_default_config
    from deepcausalmmm.core.trainer import ModelTrainer

    # Get base configuration
    config = get_default_config()

    # Customize for your dataset
    config.update({
        'n_epochs': 2000,           # Adjust training duration
        'learning_rate': 0.005,     # Fine-tune learning rate
        'hidden_dim': 256,          # Model capacity
        'dropout': 0.1,             # Regularization
        'holdout_ratio': 0.15,      # Validation split (default 0.08)
        'burn_in_weeks': 8,         # GRU stabilization
    })

    # Train with custom config (see Basic Usage for full data preparation)
    trainer = ModelTrainer(config)
    model = trainer.create_model(n_media, n_control, n_regions)
    trainer.create_optimizer_and_scheduler()
    results = trainer.train(...)

Model Inference
---------------

Use your trained model for predictions:

.. code-block:: python

    import torch

    # Make predictions directly with the model
    with torch.no_grad():
        y_pred, media_coeffs, _, outputs = model(
            train_tensors['X_media'],
            train_tensors['X_control'],
            train_tensors['R']
        )
    
    # Access predictions and contributions
    predictions = y_pred.cpu().numpy()
    media_contributions = outputs['contributions'].cpu().numpy()
    control_contributions = outputs['control_contributions'].cpu().numpy()
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Total media impact: {media_contributions.sum():.0f}")

Response Curves Analysis
-------------------------

Analyze non-linear saturation effects for each channel:

.. code-block:: python

    from deepcausalmmm.postprocess import ResponseCurveFit
    import pandas as pd
    import torch

    # Get contributions from model
    with torch.no_grad():
        y_pred, _, _, outputs = model(
            train_tensors['X_media'],
            train_tensors['X_control'],
            train_tensors['R']
        )
    
    media_contrib = outputs['contributions'].cpu().numpy()
    
    # Prepare channel data for first channel (example)
    # ResponseCurveFit expects specific columns: 'week_monday', 'spend', 'impressions', 'predicted'
    channel_spend = X_media[:, :, 0].flatten()
    channel_contrib = media_contrib[:, :, 0].flatten()
    
    channel_data = pd.DataFrame({
        'week_monday': pd.date_range('2023-01-01', periods=len(channel_spend), freq='W-MON'),
        'spend': channel_spend,
        'impressions': channel_spend,  # Use same as spend if no separate impression data
        'predicted': channel_contrib
    })

    # Fit response curve
    fitter = ResponseCurveFit(
        data=channel_data,
        bottom_param=False,  # Assume zero response at zero spend
        model_level='Overall',  # 'Overall' or 'DMA'
        date_col='week_monday'
    )

    # Fit the curve and get parameters
    fitted_df = fitter.fit()  # Returns DataFrame with fitted parameters
    
    if fitted_df is not None:
        print(f"Slope: {fitter.slope:.3f}")
        print(f"Half-Saturation Point: {fitter.saturation:,.0f}")
        print(f"Top (Max Response): {fitter.top:,.0f}")
    else:
        print("Curve fitting failed - check data quality")

Response curves help you:

* Identify saturation points for each channel
* Optimize budget allocation across channels
* Understand diminishing returns
* Make data-driven investment decisions

Visualization and Analysis
--------------------------

Generate comprehensive visualizations:

.. code-block:: python

    from deepcausalmmm.postprocess import ComprehensiveAnalyzer

    # Define channel names
    media_cols = [f'Channel_{i+1}' for i in range(5)]
    control_cols = [f'Control_{i+1}' for i in range(3)]

    # Initialize analyzer with required parameters
    analyzer = ComprehensiveAnalyzer(
        model=model,
        media_cols=media_cols,
        control_cols=control_cols,
        output_dir='mmm_results',
        pipeline=pipeline,
        config=config
    )

    # Create full dashboard
    # Note: analyzer.analyze_comprehensive() requires specific data format
    # For complete dashboard, use examples/dashboard_rmse_optimized.py as template

This creates 14+ interactive visualizations including:

* Performance metrics and holdout validation
* Actual vs predicted time series
* Channel effectiveness and coefficient distributions
* Economic contributions and waterfall analysis
* DAG network visualization for causal relationships
* Individual channel analysis and trends
* **Response curves with saturation analysis**

Next Steps
----------

* Read the :doc:`tutorials/index` for detailed examples
* Explore the :doc:`api/index` for complete API reference
* Check out :doc:`examples/index` for advanced use cases
* See :doc:`contributing` to contribute to the project

Common Issues
-------------

**Memory Issues**
    Reduce batch size or use gradient checkpointing for large datasets.

**Convergence Problems**
    Try adjusting learning rate, regularization, or number of epochs.

**Poor Performance**
    Check data quality, feature engineering, and hyperparameter tuning.

**GPU Issues**
    Ensure CUDA-compatible PyTorch installation and sufficient GPU memory.
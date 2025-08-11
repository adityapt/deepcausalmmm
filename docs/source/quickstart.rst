Quick Start Guide
=================

This guide will get you up and running with DeepCausalMMM in just a few minutes.

Basic Usage
-----------

Here's a complete example of training a DeepCausalMMM model:

.. code-block:: python

    import pandas as pd
    import torch
    from deepcausalmmm import DeepCausalMMM, get_device
    from deepcausalmmm.core import get_default_config
    from deepcausalmmm.core.trainer import ModelTrainer
    from deepcausalmmm.data import UnifiedDataPipeline

    # 1. Load your data
    df = pd.read_csv('your_mmm_data.csv')

    # 2. Get optimized configuration
    config = get_default_config()
    
    # 3. Check device availability
    device = get_device()
    print(f"Using device: {device}")

    # 4. Process data with unified pipeline
    pipeline = UnifiedDataPipeline(config)
    processed_data = pipeline.fit_transform(df)

    # 5. Train with ModelTrainer (recommended approach)
    trainer = ModelTrainer(config)
    model, results = trainer.train(processed_data)

    # 6. View results
    print(f"Training RMSE: {results['train_rmse']:.0f}")
    print(f"Holdout RMSE: {results['holdout_rmse']:.0f}")
    print(f"Holdout R²: {results['holdout_r2']:.3f}")

One-Command Analysis
--------------------

For the complete analysis with dashboard generation:

.. code-block:: bash

    # Run from the project root directory
    python dashboard_rmse_optimized.py

This will:

* Load and process your data
* Train the model with optimal settings
* Generate comprehensive visualizations
* Create an interactive dashboard
* Save results and model artifacts

Data Format Requirements
------------------------

Your MMM dataset should be a CSV file with the following structure:

.. code-block:: text

    date,region,target_variable,media_channel_1,media_channel_2,...,control_var_1,control_var_2,...
    2023-01-01,DMA_1,12500,1000,500,...,0.2,15,...
    2023-01-08,DMA_1,13200,1200,600,...,0.1,18,...
    ...

Required columns:

* **Date column**: Weekly time periods
* **Region column**: Geographic identifiers (DMA, region, etc.)
* **Target variable**: Your KPI (visits, sales, conversions, etc.)
* **Media channels**: Spend or impression data for each channel
* **Control variables**: External factors (weather, events, etc.)

Configuration Customization
----------------------------

Customize the model for your specific use case:

.. code-block:: python

    from deepcausalmmm.core import get_default_config

    # Get base configuration
    config = get_default_config()

    # Customize for your dataset
    config.update({
        'n_epochs': 2000,           # Adjust training duration
        'learning_rate': 0.005,     # Fine-tune learning rate
        'hidden_dim': 256,          # Model capacity
        'dropout': 0.1,             # Regularization
        'holdout_ratio': 0.15,      # Validation split
        'burn_in_weeks': 8,         # GRU stabilization
    })

    # Train with custom config
    trainer = ModelTrainer(config)
    model, results = trainer.train(processed_data)

Model Inference
---------------

Use your trained model for predictions and analysis:

.. code-block:: python

    from deepcausalmmm.core.inference import InferenceManager

    # Initialize inference manager
    inference = InferenceManager(model, pipeline)

    # Make predictions on new data
    predictions = inference.predict(new_media_data, new_control_data)

    # Get channel contributions
    contributions = inference.analyze_contributions(
        media_data, control_data, regions
    )

    # Access detailed outputs
    media_contributions = contributions['media_contributions']
    control_contributions = contributions['control_contributions']

    print(f"Total media impact: {media_contributions.sum():.0f}")

Visualization and Analysis
--------------------------

Generate comprehensive visualizations:

.. code-block:: python

    from deepcausalmmm.postprocess import ComprehensiveAnalyzer

    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer(config)

    # Create full dashboard
    analyzer.analyze_comprehensive(
        model=model,
        processed_data=processed_data,
        results=results
    )

This creates 13 interactive visualizations including:

* Performance metrics and holdout validation
* Actual vs predicted time series
* Channel effectiveness and coefficient distributions
* Economic contributions and waterfall analysis
* DAG network visualization for causal relationships
* Individual channel analysis and trends

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
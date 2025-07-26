#!/usr/bin/env python3
"""
Simple example demonstrating DeepCausalMMM usage.

This script shows how to:
1. Load and preprocess data
2. Train a model
3. Make predictions
4. Analyze results
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Import DeepCausalMMM components
from deepcausalmmm.core.data import load_and_preprocess_data
from deepcausalmmm.core.model import GRUCausalMMM
from deepcausalmmm.core.train import train_model_with_validation, save_training_results
from deepcausalmmm.core.infer import predict, forecast, get_feature_importance
from deepcausalmmm.config import DEFAULT_CONFIG
from deepcausalmmm.utils.metrics import calculate_metrics, plot_results


def create_sample_data(n_samples=100, n_regions=2):
    """Create sample marketing data for demonstration."""
    np.random.seed(42)
    
    # Generate time series data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='W')
    
    data = []
    for region in range(n_regions):
        for i, date in enumerate(dates):
            # Media spending (with some seasonality)
            tv_spend = 1000 + 500 * np.sin(i * 2 * np.pi / 52) + np.random.normal(0, 100)
            digital_spend = 800 + 300 * np.sin(i * 2 * np.pi / 26) + np.random.normal(0, 80)
            radio_spend = 500 + 200 * np.sin(i * 2 * np.pi / 13) + np.random.normal(0, 50)
            
            # Control variables
            price = 10 + 2 * np.sin(i * 2 * np.pi / 52) + np.random.normal(0, 0.5)
            promotion = np.random.choice([0, 1], p=[0.7, 0.3])
            
            # Revenue (target variable) with some lag effects
            base_revenue = 10000
            media_effect = (0.1 * tv_spend + 0.15 * digital_spend + 0.08 * radio_spend)
            control_effect = -50 * price + 2000 * promotion
            noise = np.random.normal(0, 500)
            
            revenue = base_revenue + media_effect + control_effect + noise
            
            data.append({
                'date': date,
                'region': f'Region_{region}',
                'week': i,
                'tv_spend': max(0, tv_spend),
                'digital_spend': max(0, digital_spend),
                'radio_spend': max(0, radio_spend),
                'price': price,
                'promotion': promotion,
                'revenue': max(0, revenue)
            })
    
    return pd.DataFrame(data)


def main():
    """Main example function."""
    print("DeepCausalMMM - Simple Example")
    print("=" * 40)
    
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data(n_samples=104, n_regions=2)  # 2 years of weekly data
    print(f"Created dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Save sample data
    df.to_csv('sample_data.csv', index=False)
    print("Sample data saved to 'sample_data.csv'")
    
    # Configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'marketing_vars': ['tv_spend', 'digital_spend', 'radio_spend'],
        'control_vars': ['price', 'promotion'],
        'dependent_var': 'revenue',
        'region_var': 'region',
        'date_var': 'date',
        'epochs': 1000,  # Reduced for faster execution
        'hidden_size': 32,
        'learning_rate': 1e-3,
        'verbose': True
    })
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    data_dict = load_and_preprocess_data('sample_data.csv', config)
    
    print(f"Data prepared:")
    print(f"  - Media variables: {data_dict['marketing_vars']}")
    print(f"  - Control variables: {data_dict['control_vars']}")
    print(f"  - Regions: {len(data_dict['regions'])}")
    print(f"  - Time steps: {data_dict['X_m'].shape[1]}")
    
    # Initialize model
    print("\nInitializing model...")
    model = GRUCausalMMM(
        A_prior=data_dict['media_adjacency'],
        n_media=len(data_dict['marketing_vars']),
        ctrl_dim=len(data_dict['control_vars']),
        hidden=config['hidden_size'],
        n_regions=len(data_dict['regions']),
        dropout=0.1
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("\nTraining model...")
    results = train_model_with_validation(model, data_dict, config)
    
    print(f"\nTraining completed!")
    print(f"Final R² Score: {results['test_metrics']['r2']:.4f}")
    print(f"RMSE: {results['test_metrics']['rmse']:.2f}")
    print(f"MAPE: {results['test_metrics']['mape']:.2f}%")
    
    # Save results
    print("\nSaving results...")
    output_dir = save_training_results(results, model, config, "example_output")
    print(f"Results saved to {output_dir}/")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    importance = get_feature_importance(
        model, data_dict['X_m'], data_dict['X_c'], data_dict['R'],
        feature_names=data_dict['marketing_vars']
    )
    
    print("Feature Importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.4f}")
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    forecast_results = forecast(
        model, data_dict['X_m'], data_dict['X_c'], data_dict['R'],
        forecast_horizon=12,  # 12 weeks ahead
        scalers={'y_scaler': data_dict['y_scaler']}
    )
    
    print(f"Generated {forecast_results['forecast_horizon']}-period forecasts")
    
    # Save forecasts
    import pandas as pd
    df_forecasts = pd.DataFrame({
        'region': np.repeat(data_dict['regions'], forecast_results['forecast_horizon']),
        'forecast_period': np.tile(range(forecast_results['forecast_horizon']), len(data_dict['regions'])),
        'forecast': forecast_results['forecast_predictions'].flatten()
    })
    df_forecasts.to_csv(f'{output_dir}/forecasts.csv', index=False)
    print(f"Forecasts saved to {output_dir}/forecasts.csv")
    
    print("\nExample completed successfully!")
    print(f"Check the '{output_dir}' directory for detailed results and visualizations.")


if __name__ == "__main__":
    main() 
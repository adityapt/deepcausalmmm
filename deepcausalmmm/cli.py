"""
Command-line interface for DeepCausalMMM.

This module provides a CLI for training and using the model.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any

from .core.data import load_and_preprocess_data
from .core.model import GRUCausalMMM
from .core.train import train_model_with_validation, save_training_results
from .core.infer import predict, forecast, get_feature_importance
from .config import DEFAULT_CONFIG, validate_config, load_config_from_file
from .exceptions import DeepCausalMMMError


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DeepCausalMMM: Deep Learning + Bayesian Networks based causal MMM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model with default settings
  deepcausalmmm train data.csv --output-dir results/
  
  # Train with custom configuration
  deepcausalmmm train data.csv --config config.json --output-dir results/
  
  # Make predictions with trained model
  deepcausalmmm predict model.pth test_data.csv --output predictions.csv
  
  # Generate forecasts
  deepcausalmmm forecast model.pth data.csv --horizon 12 --output forecasts.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('data_file', help='Path to input data file (CSV/Excel)')
    train_parser.add_argument('--config', help='Path to configuration file (JSON/YAML)')
    train_parser.add_argument('--output-dir', default='ml_output', help='Output directory')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--hidden-size', type=int, help='Hidden layer size')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with trained model')
    predict_parser.add_argument('model_file', help='Path to trained model file')
    predict_parser.add_argument('data_file', help='Path to input data file')
    predict_parser.add_argument('--output', default='predictions.csv', help='Output file path')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Generate forecasts')
    forecast_parser.add_argument('model_file', help='Path to trained model file')
    forecast_parser.add_argument('data_file', help='Path to historical data file')
    forecast_parser.add_argument('--horizon', type=int, default=12, help='Forecast horizon')
    forecast_parser.add_argument('--output', default='forecasts.csv', help='Output file path')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze model results')
    analyze_parser.add_argument('model_file', help='Path to trained model file')
    analyze_parser.add_argument('data_file', help='Path to data file')
    analyze_parser.add_argument('--output-dir', default='analysis', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'predict':
            predict_command(args)
        elif args.command == 'forecast':
            forecast_command(args)
        elif args.command == 'analyze':
            analyze_command(args)
    except DeepCausalMMMError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def train_command(args):
    """Handle train command."""
    print("Loading configuration...")
    
    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = DEFAULT_CONFIG.copy()
    
    # Override with command line arguments
    if args.epochs:
        config['epochs'] = args.epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.hidden_size:
        config['hidden_size'] = args.hidden_size
    
    # Validate configuration
    config = validate_config(config)
    
    print("Loading and preprocessing data...")
    
    # Load and preprocess data
    data_dict = load_and_preprocess_data(args.data_file, config)
    
    print("Initializing model...")
    
    # Initialize model
    model = GRUCausalMMM(
        A_prior=data_dict['media_adjacency'],
        n_media=len(data_dict['marketing_vars']),
        ctrl_dim=len(data_dict['control_vars']) if data_dict['control_vars'] else 1,
        hidden=config['hidden_size'],
        n_regions=len(data_dict['regions']),
        dropout=config.get('dropout', 0.1)
    )
    
    print("Training model...")
    
    # Train model
    results = train_model_with_validation(model, data_dict, config)
    
    print("Saving results...")
    
    # Save results
    output_dir = save_training_results(results, model, config, args.output_dir)
    
    print(f"Training completed! Results saved to {output_dir}/")
    print(f"Final R² Score: {results['test_metrics']['r2']:.4f}")
    print(f"RMSE: {results['test_metrics']['rmse']:.2f}")


def predict_command(args):
    """Handle predict command."""
    print("Loading trained model...")
    
    # Load model
    from .core.train import load_trained_model
    model, config, _ = load_trained_model(args.model_file)
    
    print("Loading data...")
    
    # Load data
    data_dict = load_and_preprocess_data(args.data_file, config)
    
    print("Making predictions...")
    
    # Make predictions
    predictions, coefficients, contributions = predict(
        model, data_dict['X_m'], data_dict['X_c'], data_dict['R']
    )
    
    # Inverse transform predictions
    y_scaler = data_dict['y_scaler']
    predictions_original = y_scaler.inverse_transform(
        predictions.cpu().numpy().reshape(-1, 1)
    ).reshape(predictions.shape)
    
    print("Saving predictions...")
    
    # Save predictions
    import pandas as pd
    df_predictions = pd.DataFrame({
        'region': np.repeat(data_dict['regions'], predictions.shape[1]),
        'time_step': np.tile(range(predictions.shape[1]), len(data_dict['regions'])),
        'prediction': predictions_original.flatten()
    })
    
    df_predictions.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


def forecast_command(args):
    """Handle forecast command."""
    print("Loading trained model...")
    
    # Load model
    from .core.train import load_trained_model
    model, config, _ = load_trained_model(args.model_file)
    
    print("Loading data...")
    
    # Load data
    data_dict = load_and_preprocess_data(args.data_file, config)
    
    print(f"Generating {args.horizon}-period forecast...")
    
    # Generate forecast
    from .core.infer import forecast
    forecast_results = forecast(
        model, data_dict['X_m'], data_dict['X_c'], data_dict['R'],
        forecast_horizon=args.horizon,
        scalers={'y_scaler': data_dict['y_scaler']}
    )
    
    print("Saving forecasts...")
    
    # Save forecasts
    import pandas as pd
    df_forecasts = pd.DataFrame({
        'region': np.repeat(data_dict['regions'], args.horizon),
        'forecast_period': np.tile(range(args.horizon), len(data_dict['regions'])),
        'forecast': forecast_results['forecast_predictions'].flatten()
    })
    
    df_forecasts.to_csv(args.output, index=False)
    print(f"Forecasts saved to {args.output}")


def analyze_command(args):
    """Handle analyze command."""
    print("Loading trained model...")
    
    # Load model
    from .core.train import load_trained_model
    model, config, _ = load_trained_model(args.model_file)
    
    print("Loading data...")
    
    # Load data
    data_dict = load_and_preprocess_data(args.data_file, config)
    
    print("Performing analysis...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Feature importance
    from .core.infer import get_feature_importance
    importance = get_feature_importance(
        model, data_dict['X_m'], data_dict['X_c'], data_dict['R'],
        feature_names=data_dict['marketing_vars']
    )
    
    # Contributions analysis
    from .core.infer import get_contributions
    contributions = get_contributions(
        model, data_dict['X_m'], data_dict['X_c'], data_dict['R'],
        feature_names=data_dict['marketing_vars'],
        scalers={'y_scaler': data_dict['y_scaler']}
    )
    
    # Causal effects analysis
    from .core.infer import analyze_causal_effects
    causal_effects = analyze_causal_effects(
        model, data_dict['X_m'], data_dict['X_c'], data_dict['R'],
        feature_names=data_dict['marketing_vars']
    )
    
    # Generate plots
    from .utils.metrics import (
        plot_feature_importance, plot_contributions, plot_causal_effects
    )
    
    plot_feature_importance(importance, args.output_dir)
    plot_contributions(
        contributions['contributions'], 
        contributions['feature_names'], 
        args.output_dir
    )
    plot_causal_effects(causal_effects, args.output_dir)
    
    # Save analysis results
    import pandas as pd
    
    # Feature importance
    df_importance = pd.DataFrame([
        {'feature': k, 'importance': v} for k, v in importance.items()
    ])
    df_importance.to_csv(f'{args.output_dir}/feature_importance.csv', index=False)
    
    # Causal effects
    df_effects = pd.DataFrame([
        {
            'feature': k,
            'average_effect': v['average_effect'],
            'effect_std': v['effect_std']
        } for k, v in causal_effects.items()
    ])
    df_effects.to_csv(f'{args.output_dir}/causal_effects.csv', index=False)
    
    print(f"Analysis completed! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main() 
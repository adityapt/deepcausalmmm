"""
Command-line interface for DeepCausalMMM.

This module provides a CLI for training and using the model.
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .core.unified_model import DeepCausalMMM
from .core.trainer import ModelTrainer
from .core.inference import InferenceManager
from .core.data import UnifiedDataPipeline
from .core.config import get_default_config
from .core.scaling import SimpleGlobalScaler
from .exceptions import DeepCausalMMMError


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DeepCausalMMM: Deep Learning Marketing Mix Model with Causal Structure",
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
    train_parser.add_argument('--enable-dag', action='store_true', help='Enable DAG structure')
    train_parser.add_argument('--enable-interactions', action='store_true', help='Enable channel interactions')
    train_parser.add_argument('--device', help='Device to use (auto/cpu/cuda/cuda:0/etc.)')
    train_parser.add_argument('--mixed-precision', action='store_true', help='Enable mixed precision training')
    train_parser.add_argument('--no-mixed-precision', action='store_false', dest='mixed_precision', help='Disable mixed precision training')
    train_parser.add_argument('--gpu-memory-fraction', type=float, help='Fraction of GPU memory to use')
    train_parser.add_argument('--num-workers', type=int, help='Number of data loading workers')
    train_parser.set_defaults(mixed_precision=True)
    
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
    """
    Train a new DeepCausalMMM model from command line arguments.
    
    Handles data loading, preprocessing, model training, and result saving.
    Supports custom configurations and parameter overrides.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments from argparse
    """    
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
    if args.enable_dag:
        config['use_bayesian_network'] = True
    if args.enable_interactions:
        config['enable_interactions'] = True
    if args.device:
        config['device'] = args.device
    if args.mixed_precision is not None:
        config['mixed_precision'] = args.mixed_precision
    if args.gpu_memory_fraction:
        config['gpu_memory_fraction'] = args.gpu_memory_fraction
    if args.num_workers:
        config['num_workers'] = args.num_workers
    
    # Validate configuration
    config = validate_config(config)
    
    print("Loading and preprocessing data...")
    
    # Load data
    import pandas as pd
    data = pd.read_csv(args.data_file)
    
    # Extract variables based on config
    media_cols = config['media_columns']
    control_cols = config['control_columns']
    target_col = config['target_column']
    region_col = config['region_column']
    
    # Prepare data
    X_media = data[media_cols].values.reshape(-1, 1, len(media_cols))
    X_control = data[control_cols].values.reshape(-1, 1, len(control_cols))
    y = data[target_col].values.reshape(-1, 1)
    region_indices = pd.Categorical(data[region_col]).codes
    
    # Scale data
    scaler = SimpleGlobalScaler()
    X_media_scaled, X_control_scaled, y_scaled = scaler.fit_transform(
        X_media, X_control, y, region_indices
    )
    
    print("Training model with modern ModelTrainer class...")
    
    from .core.trainer import ModelTrainer
    from .core.data import UnifiedDataPipeline
    
    # Convert legacy config to modern config format
    modern_config = {
        'n_epochs': config['epochs'],
        'learning_rate': config['learning_rate'],
        'hidden_dim': config['hidden_size'],
        'dropout': config['dropout'],
        'sparsity_weight': config.get('sparsity_weight', 0.1),
        'enable_dag': config['use_bayesian_network'],
        'enable_interactions': config.get('enable_interactions', True),
        'batch_size': config['batch_size'],
        'l1_weight': config.get('l1_weight', 0.001),
        'l2_weight': config.get('l2_weight', 0.001),
        'coeff_range': config.get('coeff_range', 1.0),
        'burn_in_weeks': config.get('burn_in_weeks', 4),
        'momentum_decay': config.get('momentum_decay', 0.9),
        'warm_start_epochs': config.get('warm_start_epochs', 50),
        'holdout_ratio': config.get('holdout_ratio', 0.2),
    }
    
    # Create and configure data pipeline
    pipeline = UnifiedDataPipeline(modern_config)
    
    # Reconstruct original data arrays for pipeline
    X_media_orig = X_media_scaled  # Assume already processed for CLI
    X_control_orig = X_control_scaled
    y_orig = y_scaled
    
    # Split and process data
    train_data, holdout_data = pipeline.temporal_split(X_media_orig, X_control_orig, y_orig)
    
    # Process training data
    train_tensors = pipeline.fit_and_transform_training(train_data)
    
    # Process holdout data if available
    holdout_tensors = None
    if holdout_data is not None:
        holdout_tensors = pipeline.transform_holdout(holdout_data)
        
    # Create trainer
    trainer = ModelTrainer(modern_config)
    
    # Create model
    n_media = train_tensors['X_media'].shape[2]
    n_control = train_tensors['X_control'].shape[2]
    n_regions = train_tensors['X_media'].shape[0]
    
    model = trainer.create_model(n_media, n_control, n_regions)
    trainer.create_optimizer_and_scheduler()
    
    # Train model
    if holdout_tensors is not None:
        training_results = trainer.train(
            train_tensors['X_media'], train_tensors['X_control'],
            train_tensors['R'], train_tensors['y'],
            holdout_tensors['X_media'], holdout_tensors['X_control'],
            holdout_tensors['R'], holdout_tensors['y'],
            verbose=True
        )
    else:
        training_results = trainer.train(
            train_tensors['X_media'], train_tensors['X_control'],
            train_tensors['R'], train_tensors['y'],
            verbose=True
        )
        
    # Create results dictionary compatible with existing CLI code
    results = {
        'pipeline': pipeline,
        'trainer': trainer,
        'scaler': pipeline.scaler,
        **training_results
    }
    
    print("Saving results...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model with device-agnostic state
    import torch
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'scaler': scaler,
        'feature_names': {
            'media': media_cols,
            'control': control_cols
        }
    }, f'{args.output_dir}/model.pth')
    
    # Save training history
    pd.DataFrame({
        'epoch': range(len(results['train_losses'])),
        'train_loss': results['train_losses'],
        'dag_loss': results['dag_losses']
    }).to_csv(f'{args.output_dir}/training_history.csv', index=False)
    
    print(f"Training completed! Results saved to {args.output_dir}/")
    print(f"Final loss: {results['final_loss']:.4f}")


def predict_command(args):
    """
    Generate predictions using a trained model.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments from argparse
    """
    print("Loading trained model...")
    
    # Load model and config
    checkpoint = torch.load(args.model_file)
    config = checkpoint['config']
    scaler = checkpoint['scaler']
    feature_names = checkpoint['feature_names']
    
    # Initialize model
    model = create_unified_mmm(
        n_media=len(feature_names['media']),
        n_control=len(feature_names['control']),
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        enable_dag=config['use_bayesian_network'],
        enable_interactions=config.get('enable_interactions', True)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Loading data...")
    
    # Load and prepare data
    data = pd.read_csv(args.data_file)
    X_media = data[feature_names['media']].values.reshape(-1, 1, len(feature_names['media']))
    X_control = data[feature_names['control']].values.reshape(-1, 1, len(feature_names['control']))
    region_indices = pd.Categorical(data[config['region_column']]).codes
    
    # Scale data
    X_media_scaled, X_control_scaled, _ = scaler.transform(X_media, X_control, np.zeros_like(X_media[:, :, 0]))
    
    print("Making predictions...")
    
    # Convert to tensors
    X_media_tensor = torch.FloatTensor(X_media_scaled)
    X_control_tensor = torch.FloatTensor(X_control_scaled)
    region_tensor = torch.LongTensor(region_indices)
    
    # Get predictions
    results = predict(model, X_media_tensor, X_control_tensor, region_tensor, scaler)
    
    print("Saving predictions...")
    
    # Create output dataframe
    df_predictions = pd.DataFrame({
        'region': data[config['region_column']],
        'date': data[config['date_column']],
        'prediction': results['unscaled_contributions'].sum(axis=2).flatten()
    })
    
    # Add channel contributions
    for i, channel in enumerate(feature_names['media']):
        df_predictions[f'{channel}_contribution'] = results['unscaled_contributions'][:, :, i].flatten()
    
    df_predictions.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


def forecast_command(args):
    """
    Generate future forecasts using a trained model.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments from argparse
    """
    print("Loading trained model...")
    
    # Load model and config
    checkpoint = torch.load(args.model_file)
    config = checkpoint['config']
    scaler = checkpoint['scaler']
    feature_names = checkpoint['feature_names']
    
    # Initialize model
    model = create_unified_mmm(
        n_media=len(feature_names['media']),
        n_control=len(feature_names['control']),
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        enable_dag=config['use_bayesian_network'],
        enable_interactions=config.get('enable_interactions', True)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Loading data...")
    
    # Load and prepare data
    data = pd.read_csv(args.data_file)
    X_media = data[feature_names['media']].values.reshape(-1, 1, len(feature_names['media']))
    X_control = data[feature_names['control']].values.reshape(-1, 1, len(feature_names['control']))
    region_indices = pd.Categorical(data[config['region_column']]).codes
    
    # Scale data
    X_media_scaled, X_control_scaled, _ = scaler.transform(X_media, X_control, np.zeros_like(X_media[:, :, 0]))
    
    print(f"Generating {args.horizon}-period forecast...")
    
    # Convert to tensors
    X_media_tensor = torch.FloatTensor(X_media_scaled)
    X_control_tensor = torch.FloatTensor(X_control_scaled)
    region_tensor = torch.LongTensor(region_indices)
    
    # Generate forecast
    forecast_results = forecast(
        model, X_media_tensor, X_control_tensor, region_tensor,
        forecast_horizon=args.horizon,
        scaler=scaler
    )
    
    print("Saving forecasts...")
    
    # Create forecast dates
    last_date = pd.to_datetime(data[config['date_column']]).max()
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=args.horizon,
        freq='W-MON'
    )
    
    # Create output dataframe
    df_forecasts = pd.DataFrame({
        'region': np.repeat(data[config['region_column']].unique(), args.horizon),
        'date': np.tile(forecast_dates, len(data[config['region_column']].unique())),
        'forecast': forecast_results['unscaled_forecast_contributions'].sum(axis=2).flatten()
    })
    
    # Add channel contributions
    for i, channel in enumerate(feature_names['media']):
        df_forecasts[f'{channel}_contribution'] = forecast_results['unscaled_forecast_contributions'][:, :, i].flatten()
    
    df_forecasts.to_csv(args.output, index=False)
    print(f"Forecasts saved to {args.output}")


def analyze_command(args):
    """
    Generate comprehensive analysis and visualizations for a trained model.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments from argparse
    """
    print("Loading trained model...")
    
    # Load model and config
    checkpoint = torch.load(args.model_file)
    config = checkpoint['config']
    scaler = checkpoint['scaler']
    feature_names = checkpoint['feature_names']
    
    # Initialize model
    model = create_unified_mmm(
        n_media=len(feature_names['media']),
        n_control=len(feature_names['control']),
        hidden_size=config['hidden_size'],
        dropout=config['dropout'],
        enable_dag=config['use_bayesian_network'],
        enable_interactions=config.get('enable_interactions', True)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Loading data...")
    
    # Load and prepare data
    data = pd.read_csv(args.data_file)
    X_media = data[feature_names['media']].values.reshape(-1, 1, len(feature_names['media']))
    X_control = data[feature_names['control']].values.reshape(-1, 1, len(feature_names['control']))
    region_indices = pd.Categorical(data[config['region_column']]).codes
    
    # Scale data
    X_media_scaled, X_control_scaled, _ = scaler.transform(X_media, X_control, np.zeros_like(X_media[:, :, 0]))
    
    print("Performing analysis...")
    
    # Convert to tensors
    X_media_tensor = torch.FloatTensor(X_media_scaled)
    X_control_tensor = torch.FloatTensor(X_control_scaled)
    region_tensor = torch.LongTensor(region_indices)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Media impact analysis
    impact_analysis = analyze_media_impact(
        model, X_media_tensor, X_control_tensor, region_tensor,
        channel_names=feature_names['media'],
        scaler=scaler
    )
    
    # Generate confidence intervals
    intervals = generate_confidence_intervals(
        model, X_media_tensor, X_control_tensor, region_tensor,
        scaler=scaler
    )
    
    # Save analysis results
    
    # Channel analysis
    df_channels = pd.DataFrame.from_dict(
        impact_analysis['channel_analysis'],
        orient='index'
    )
    df_channels.to_csv(f'{args.output_dir}/channel_analysis.csv')
    
    # Overall metrics
    with open(f'{args.output_dir}/overall_metrics.json', 'w') as f:
        json.dump(impact_analysis['overall_metrics'], f, indent=2)
    
    # Confidence intervals
    df_intervals = pd.DataFrame({
        'region': np.repeat(data[config['region_column']].unique(), X_media.shape[1]),
        'date': np.tile(data[config['date_column']].unique(), len(data[config['region_column']].unique())),
        'mean': intervals['mean_prediction'].flatten(),
        'lower': intervals['prediction_lower'].flatten(),
        'upper': intervals['prediction_upper'].flatten()
    })
    df_intervals.to_csv(f'{args.output_dir}/confidence_intervals.csv', index=False)
    
    # Generate plots
    import plotly.graph_objects as go
    
    # Channel contributions over time
    fig_contrib = go.Figure()
    contributions = impact_analysis['model_outputs']['unscaled_contributions']
    for i, channel in enumerate(feature_names['media']):
        fig_contrib.add_trace(go.Scatter(
            name=channel,
            x=data[config['date_column']].unique(),
            y=contributions[:, :, i].mean(axis=0),
            stackgroup='one'
        ))
    fig_contrib.update_layout(title='Channel Contributions Over Time')
    fig_contrib.write_html(f'{args.output_dir}/contributions.html')
    
    # DAG structure
    if model.enable_dag:
        adjacency = impact_analysis['model_outputs']['adjacency']
        fig_dag = go.Figure(data=go.Heatmap(
            z=adjacency,
            x=feature_names['media'],
            y=feature_names['media'],
            colorscale='RdBu'
        ))
        fig_dag.update_layout(title='Media Channel DAG Structure')
        fig_dag.write_html(f'{args.output_dir}/dag_structure.html')
    
    print(f"Analysis completed! Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main() 

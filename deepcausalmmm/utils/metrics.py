"""
Metrics and visualization utilities for DeepCausalMMM.

This module provides:
- Model evaluation metrics (RMSE, MAPE, R², etc.)
- Visualization functions for results
- Plotting utilities for contributions and forecasts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')
sns.set_palette("husl")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing various metrics
    """
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'r2': np.nan,
            'mape_log': np.nan
        }
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # R² score
    r2 = r2_score(y_true, y_pred)
    
    # Log MAPE (more robust to outliers)
    mape_log = np.mean(np.abs(np.log(y_true + 1e-8) - np.log(y_pred + 1e-8))) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'mape_log': float(mape_log)
    }


def plot_results(results: Dict[str, Any], output_dir: str = "ml_output") -> None:
    """
    Generate comprehensive visualization plots for model results.
    
    Args:
        results: Dictionary containing training results
        output_dir: Directory to save plots
    """
    # Training history
    if 'train_losses' in results:
        plt.figure(figsize=(12, 8))
        
        # Training loss
        plt.subplot(2, 2, 1)
        plt.plot(results['train_losses'], label='Training Loss', color='blue')
        if 'val_losses' in results and results['val_losses'] is not None:
            plt.plot(results['val_losses'], label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        # R² score if available
        if 'test_metrics' in results and 'r2' in results['test_metrics']:
            plt.subplot(2, 2, 2)
            plt.bar(['R² Score'], [results['test_metrics']['r2']], color='green')
            plt.ylabel('R² Score')
            plt.title('Model Performance')
            plt.ylim(0, 1)
        
        # Actual vs Predicted
        if 'test_actual' in results and 'test_predictions' in results:
            plt.subplot(2, 2, 3)
            y_test = results['test_actual'].flatten()
            y_pred = results['test_predictions'].flatten()
            
            plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted')
            plt.grid(True)
            
            # Add R² score to plot
            r2 = results['test_metrics'].get('r2', 0)
            plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residuals
        if 'test_actual' in results and 'test_predictions' in results:
            plt.subplot(2, 2, 4)
            residuals = y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_results.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_contributions(
    contributions: np.ndarray,
    feature_names: List[str],
    output_dir: str = "ml_output",
    title: str = "Media Contributions Over Time"
) -> None:
    """
    Plot media contributions over time.
    
    Args:
        contributions: Contribution array [regions, time_steps, features]
        feature_names: Names of media features
        output_dir: Directory to save plots
        title: Plot title
    """
    # Average across regions
    avg_contributions = contributions.mean(axis=0)  # [time_steps, features]
    
    plt.figure(figsize=(15, 10))
    
    # Time series plot
    plt.subplot(2, 1, 1)
    for i, feature in enumerate(feature_names):
        plt.plot(avg_contributions[:, i], label=feature, linewidth=2)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Contribution')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Stacked area plot
    plt.subplot(2, 1, 2)
    plt.stackplot(range(avg_contributions.shape[0]), 
                  [avg_contributions[:, i] for i in range(avg_contributions.shape[1])],
                  labels=feature_names)
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Contribution')
    plt.title('Cumulative Media Contributions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/media_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_forecasts(
    historical: np.ndarray,
    forecasts: np.ndarray,
    forecast_periods: List[int],
    output_dir: str = "ml_output",
    title: str = "Forecast Results"
) -> None:
    """
    Plot forecast results with historical data.
    
    Args:
        historical: Historical data [regions, time_steps]
        forecasts: Forecast data [regions, forecast_horizon]
        forecast_periods: List of forecast period indices
        output_dir: Directory to save plots
        title: Plot title
    """
    # Average across regions
    avg_historical = historical.mean(axis=0)
    avg_forecasts = forecasts.mean(axis=0)
    
    plt.figure(figsize=(12, 8))
    
    # Historical data
    historical_periods = range(len(avg_historical))
    plt.plot(historical_periods, avg_historical, 'b-', label='Historical', linewidth=2)
    
    # Forecast data
    plt.plot(forecast_periods, avg_forecasts, 'r--', label='Forecast', linewidth=2)
    
    # Add confidence intervals if available
    if 'forecast_intervals' in locals():
        intervals = locals()['forecast_intervals']
        plt.fill_between(forecast_periods, 
                        intervals['lower_bound'].mean(axis=0),
                        intervals['upper_bound'].mean(axis=0),
                        alpha=0.3, color='red', label='Confidence Interval')
    
    plt.xlabel('Time Periods')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Add vertical line at forecast start
    plt.axvline(x=len(historical_periods), color='black', linestyle=':', alpha=0.7)
    plt.text(len(historical_periods), plt.ylim()[1] * 0.9, 'Forecast Start', 
             rotation=90, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecasts.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(
    importance_scores: Dict[str, float],
    output_dir: str = "ml_output",
    title: str = "Feature Importance"
) -> None:
    """
    Plot feature importance scores.
    
    Args:
        importance_scores: Dictionary mapping feature names to importance scores
        output_dir: Directory to save plots
        title: Plot title
    """
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())
    
    # Sort by importance
    sorted_indices = np.argsort(scores)[::-1]
    features = [features[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(features, scores)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.grid(True, axis='x')
    
    # Color bars based on importance
    for i, bar in enumerate(bars):
        if scores[i] > np.mean(scores):
            bar.set_color('green')
        else:
            bar.set_color('orange')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_causal_effects(
    causal_effects: Dict[str, Any],
    output_dir: str = "ml_output",
    title: str = "Causal Effects Analysis"
) -> None:
    """
    Plot causal effects of media interventions.
    
    Args:
        causal_effects: Dictionary containing causal effect analysis
        output_dir: Directory to save plots
        title: Plot title
    """
    features = list(causal_effects.keys())
    effects = [causal_effects[feature]['average_effect'] for feature in features]
    effect_stds = [causal_effects[feature]['effect_std'] for feature in features]
    
    # Sort by effect magnitude
    sorted_indices = np.argsort(np.abs(effects))[::-1]
    features = [features[i] for i in sorted_indices]
    effects = [effects[i] for i in sorted_indices]
    effect_stds = [effect_stds[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    
    # Bar plot with error bars
    bars = plt.barh(features, effects, xerr=effect_stds, capsize=5)
    plt.xlabel('Average Causal Effect')
    plt.title(title)
    plt.grid(True, axis='x')
    
    # Color bars based on effect direction
    for i, bar in enumerate(bars):
        if effects[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # Add vertical line at zero
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/causal_effects.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(
    results: Dict[str, Any],
    output_dir: str = "ml_output"
) -> str:
    """
    Create a comprehensive summary report.
    
    Args:
        results: Dictionary containing all results
        output_dir: Directory to save report
        
    Returns:
        Path to the generated report
    """
    report_path = f"{output_dir}/summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DEEPCAUSALMMM - MODEL SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE:\n")
        f.write("-" * 20 + "\n")
        if 'test_metrics' in results:
            metrics = results['test_metrics']
            f.write(f"R² Score: {metrics.get('r2', 'N/A'):.4f}\n")
            f.write(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}\n")
            f.write(f"MAPE: {metrics.get('mape', 'N/A'):.2f}%\n")
            f.write(f"MAE: {metrics.get('mae', 'N/A'):.4f}\n")
        f.write("\n")
        
        # Training Information
        f.write("TRAINING INFORMATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Epochs Trained: {results.get('epochs_trained', 'N/A')}\n")
        f.write(f"Final Train Loss: {results.get('final_train_loss', 'N/A'):.6f}\n")
        if 'best_val_loss' in results and results['best_val_loss'] is not None:
            f.write(f"Best Validation Loss: {results['best_val_loss']:.6f}\n")
        f.write(f"Early Stopped: {results.get('early_stopped', 'N/A')}\n")
        f.write("\n")
        
        # Data Information
        if 'data_splits' in results:
            f.write("DATA SPLITS:\n")
            f.write("-" * 20 + "\n")
            splits = results['data_splits']
            f.write(f"Total Time Steps: {splits.get('n_time_steps', 'N/A')}\n")
            f.write(f"Train Cut: {splits.get('train_cut', 'N/A')}\n")
            f.write(f"Validation Cut: {splits.get('val_cut', 'N/A')}\n")
            f.write("\n")
        
        # Feature Importance
        if 'feature_importance' in results:
            f.write("TOP FEATURES BY IMPORTANCE:\n")
            f.write("-" * 20 + "\n")
            importance = results['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, score in sorted_features[:10]:  # Top 10
                f.write(f"{feature}: {score:.4f}\n")
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("Report generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 60 + "\n")
    
    print(f"Summary report saved to: {report_path}")
    return report_path 
"""
Inference and forecasting utilities for DeepCausalMMM.

This module handles:
- Model prediction and forecasting
- Feature importance analysis
- Contribution analysis
- Uncertainty quantification
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..exceptions import InferenceError
from .model import GRUCausalMMM


def predict(
    model: GRUCausalMMM,
    X_m: torch.Tensor,
    X_c: torch.Tensor,
    R: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        X_m: Media variables [batch_size, time_steps, n_media]
        X_c: Control variables [batch_size, time_steps, ctrl_dim]
        R: Region indices [batch_size]
        
    Returns:
        Tuple of (predictions, coefficients, contributions)
    """
    model.eval()
    with torch.no_grad():
        predictions, coefficients, contributions = model(X_m, X_c, R)
    
    return predictions, coefficients, contributions


def forecast(
    model: GRUCausalMMM,
    X_m: torch.Tensor,
    X_c: torch.Tensor,
    R: torch.Tensor,
    forecast_horizon: int = 12,
    scalers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate forecasts for future time periods.
    
    Args:
        model: Trained model
        X_m: Historical media variables
        X_c: Historical control variables
        R: Region indices
        forecast_horizon: Number of periods to forecast
        scalers: Dictionary of scalers for inverse transformation
        
    Returns:
        Dictionary containing forecast results
    """
    model.eval()
    
    # Get the last time step for initialization
    B, T, _ = X_m.shape
    
    # Initialize forecast arrays
    X_m_forecast = torch.zeros(B, T + forecast_horizon, X_m.shape[-1])
    X_c_forecast = torch.zeros(B, T + forecast_horizon, X_c.shape[-1])
    
    # Copy historical data
    X_m_forecast[:, :T, :] = X_m
    X_c_forecast[:, :T, :] = X_c
    
    # Generate forecasts
    forecasts = []
    coefficients_forecast = []
    contributions_forecast = []
    
    with torch.no_grad():
        for t in range(forecast_horizon):
            # Use the last available data for forecasting
            X_m_current = X_m_forecast[:, :T+t, :]
            X_c_current = X_c_forecast[:, :T+t, :]
            
            # Make prediction
            pred, coeff, contrib = model(X_m_current, X_c_current, R)
            
            # Get the last prediction
            last_pred = pred[:, -1:]
            last_coeff = coeff[:, -1:, :]
            last_contrib = contrib[:, -1:, :]
            
            forecasts.append(last_pred)
            coefficients_forecast.append(last_coeff)
            contributions_forecast.append(last_contrib)
    
    # Concatenate forecasts
    forecast_predictions = torch.cat(forecasts, dim=1)  # [B, forecast_horizon]
    forecast_coefficients = torch.cat(coefficients_forecast, dim=1)  # [B, forecast_horizon, n_media]
    forecast_contributions = torch.cat(contributions_forecast, dim=1)  # [B, forecast_horizon, n_media]
    
    # Inverse transform if scalers provided
    if scalers and 'y_scaler' in scalers:
        y_scaler = scalers['y_scaler']
        forecast_predictions_original = y_scaler.inverse_transform(
            forecast_predictions.cpu().numpy().reshape(-1, 1)
        ).reshape(forecast_predictions.shape)
    else:
        forecast_predictions_original = forecast_predictions.cpu().numpy()
    
    return {
        'forecast_predictions': forecast_predictions_original,
        'forecast_coefficients': forecast_coefficients.cpu().numpy(),
        'forecast_contributions': forecast_contributions.cpu().numpy(),
        'forecast_horizon': forecast_horizon,
        'forecast_periods': list(range(T, T + forecast_horizon))
    }


def get_feature_importance(
    model: GRUCausalMMM,
    X_m: torch.Tensor,
    X_c: torch.Tensor,
    R: torch.Tensor,
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate feature importance based on learned coefficients.
    
    Args:
        model: Trained model
        X_m: Media variables
        X_c: Control variables
        R: Region indices
        feature_names: Names of media features
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    model.eval()
    with torch.no_grad():
        importance_scores = model.get_feature_importance(X_m, X_c, R)
    
    # Normalize importance scores
    total_importance = importance_scores.sum()
    if total_importance > 0:
        importance_scores = importance_scores / total_importance
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'media_{i}' for i in range(len(importance_scores))]
    
    # Create dictionary
    feature_importance = {}
    for i, name in enumerate(feature_names):
        feature_importance[name] = float(importance_scores[i])
    
    return feature_importance


def get_contributions(
    model: GRUCausalMMM,
    X_m: torch.Tensor,
    X_c: torch.Tensor,
    R: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    scalers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate media contributions over time.
    
    Args:
        model: Trained model
        X_m: Media variables
        X_c: Control variables
        R: Region indices
        feature_names: Names of media features
        scalers: Dictionary of scalers for inverse transformation
        
    Returns:
        Dictionary containing contribution analysis
    """
    model.eval()
    with torch.no_grad():
        predictions, coefficients, contributions = model(X_m, X_c, R)
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'media_{i}' for i in range(contributions.shape[-1])]
    
    # Calculate total contributions
    total_contributions = contributions.sum(dim=-1)  # [B, T]
    
    # Calculate percentage contributions
    percentage_contributions = contributions / (total_contributions.unsqueeze(-1) + 1e-8)
    
    # Inverse transform if scalers provided
    if scalers and 'y_scaler' in scalers:
        y_scaler = scalers['y_scaler']
        contributions_original = y_scaler.inverse_transform(
            contributions.cpu().numpy().reshape(-1, contributions.shape[-1])
        ).reshape(contributions.shape)
    else:
        contributions_original = contributions.cpu().numpy()
    
    # Prepare results
    results = {
        'contributions': contributions_original,
        'percentage_contributions': percentage_contributions.cpu().numpy(),
        'coefficients': coefficients.cpu().numpy(),
        'predictions': predictions.cpu().numpy(),
        'feature_names': feature_names,
        'total_contributions': total_contributions.cpu().numpy()
    }
    
    return results


def analyze_causal_effects(
    model: GRUCausalMMM,
    X_m: torch.Tensor,
    X_c: torch.Tensor,
    R: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    intervention_strength: float = 1.0
) -> Dict[str, Any]:
    """
    Analyze causal effects of media interventions.
    
    Args:
        model: Trained model
        X_m: Media variables
        X_c: Control variables
        R: Region indices
        feature_names: Names of media features
        intervention_strength: Strength of intervention (multiplier)
        
    Returns:
        Dictionary containing causal effect analysis
    """
    model.eval()
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'media_{i}' for i in range(X_m.shape[-1])]
    
    causal_effects = {}
    
    with torch.no_grad():
        # Baseline prediction
        baseline_pred, _, _ = model(X_m, X_c, R)
        
        # Test interventions for each media channel
        for i, feature_name in enumerate(feature_names):
            # Create intervention (increase spending by intervention_strength)
            X_m_intervention = X_m.clone()
            X_m_intervention[:, :, i] = X_m_intervention[:, :, i] * intervention_strength
            
            # Get prediction with intervention
            intervention_pred, _, _ = model(X_m_intervention, X_c, R)
            
            # Calculate causal effect
            causal_effect = intervention_pred - baseline_pred
            
            causal_effects[feature_name] = {
                'baseline_prediction': baseline_pred.cpu().numpy(),
                'intervention_prediction': intervention_pred.cpu().numpy(),
                'causal_effect': causal_effect.cpu().numpy(),
                'average_effect': float(causal_effect.mean()),
                'effect_std': float(causal_effect.std()),
                'intervention_strength': intervention_strength
            }
    
    return causal_effects


def calculate_roas(
    contributions: np.ndarray,
    media_spend: np.ndarray,
    revenue: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate Return on Ad Spend (ROAS) for each media channel.
    
    Args:
        contributions: Media contributions from model
        media_spend: Original media spending data
        revenue: Revenue data
        feature_names: Names of media features
        
    Returns:
        Dictionary mapping feature names to ROAS values
    """
    if feature_names is None:
        feature_names = [f'media_{i}' for i in range(contributions.shape[-1])]
    
    roas_values = {}
    
    for i, feature_name in enumerate(feature_names):
        # Calculate total contribution and spend for this channel
        total_contribution = contributions[:, :, i].sum()
        total_spend = media_spend[:, :, i].sum()
        
        # Calculate ROAS
        if total_spend > 0:
            roas = total_contribution / total_spend
        else:
            roas = 0.0
        
        roas_values[feature_name] = float(roas)
    
    return roas_values


def generate_forecast_intervals(
    model: GRUCausalMMM,
    X_m: torch.Tensor,
    X_c: torch.Tensor,
    R: torch.Tensor,
    forecast_horizon: int = 12,
    n_samples: int = 100,
    confidence_level: float = 0.95
) -> Dict[str, np.ndarray]:
    """
    Generate forecast intervals using Monte Carlo sampling.
    
    Args:
        model: Trained model
        X_m: Historical media variables
        X_c: Historical control variables
        R: Region indices
        forecast_horizon: Number of periods to forecast
        n_samples: Number of Monte Carlo samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary containing forecast intervals
    """
    model.eval()
    
    forecasts = []
    
    # Generate multiple forecasts with noise
    for _ in range(n_samples):
        # Add small noise to inputs for uncertainty
        X_m_noisy = X_m + torch.randn_like(X_m) * 0.01
        X_c_noisy = X_c + torch.randn_like(X_c) * 0.01
        
        # Generate forecast
        forecast_result = forecast(model, X_m_noisy, X_c_noisy, R, forecast_horizon)
        forecasts.append(forecast_result['forecast_predictions'])
    
    # Stack forecasts
    forecasts_array = np.stack(forecasts, axis=0)  # [n_samples, B, forecast_horizon]
    
    # Calculate percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    forecast_intervals = {
        'mean': np.mean(forecasts_array, axis=0),
        'median': np.median(forecasts_array, axis=0),
        'lower_bound': np.percentile(forecasts_array, lower_percentile, axis=0),
        'upper_bound': np.percentile(forecasts_array, upper_percentile, axis=0),
        'std': np.std(forecasts_array, axis=0),
        'confidence_level': confidence_level
    }
    
    return forecast_intervals 
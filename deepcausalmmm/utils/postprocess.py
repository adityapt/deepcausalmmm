"""
Post-processing utilities for DeepCausalMMM coefficient analysis.
Handles burn-in exclusion, coefficient stability analysis, and reliability metrics.

This module prevents common coefficient analysis errors by automatically handling
the burn-in period that occurs during data preprocessing.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings


def calculate_burn_in_length(original_weeks: int, processed_time_steps: int) -> int:
    """
    Calculate burn-in period length automatically.
    
    Args:
        original_weeks: Number of weeks in original dataset per region
        processed_time_steps: Number of time steps after preprocessing
    
    Returns:
        int: Number of burn-in weeks (typically 4)
    """
    return max(0, processed_time_steps - original_weeks)


def exclude_burn_in(
    coefficients: torch.Tensor, 
    burn_in_weeks: int,
    warn_if_no_burn_in: bool = True
) -> torch.Tensor:
    """
    Exclude burn-in period from coefficient analysis.
    
    This function removes the initial time steps that represent burn-in/padding
    rather than real marketing data, preventing inflated coefficient values.
    
    Args:
        coefficients: Raw coefficients [regions, time_steps, channels]
        burn_in_weeks: Number of burn-in weeks to exclude
        warn_if_no_burn_in: Whether to warn if no burn-in detected
    
    Returns:
        torch.Tensor: Clean coefficients [regions, real_time_steps, channels]
    """
    if burn_in_weeks <= 0:
        if warn_if_no_burn_in:
            warnings.warn("No burn-in period detected. Using all time steps.")
        return coefficients
    
    if burn_in_weeks >= coefficients.shape[1]:
        raise ValueError(f"Burn-in period ({burn_in_weeks}) cannot be >= total time steps ({coefficients.shape[1]})")
    
    return coefficients[:, burn_in_weeks:, :]


def calculate_stability_metrics(
    coefficients: torch.Tensor,
    exclude_burn_in_weeks: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive coefficient stability metrics.
    
    Args:
        coefficients: Raw coefficients [regions, time_steps, channels]
        exclude_burn_in_weeks: Burn-in weeks to exclude (None = no exclusion)
    
    Returns:
        Dict with stability metrics including ratios and status
    """
    if exclude_burn_in_weeks is not None and exclude_burn_in_weeks > 0:
        real_coeffs = exclude_burn_in(coefficients, exclude_burn_in_weeks, warn_if_no_burn_in=False)
    else:
        real_coeffs = coefficients
        if exclude_burn_in_weeks is None:
            warnings.warn("No burn-in exclusion applied. Results may be misleading if burn-in period exists.")
    
    # Convert to numpy for easier calculation
    if torch.is_tensor(real_coeffs):
        real_coeffs = real_coeffs.detach().cpu().numpy()
    
    # Calculate stability ratios
    week_1_coeffs = real_coeffs[:, 0, :].flatten()
    week_10_coeffs = real_coeffs[:, min(9, real_coeffs.shape[1]-1), :].flatten()
    week_final_coeffs = real_coeffs[:, -1, :].flatten()
    
    ratios_1_to_10 = []
    ratios_1_to_final = []
    
    for i in range(len(week_1_coeffs)):
        if abs(week_10_coeffs[i]) > 1e-8:
            ratios_1_to_10.append(abs(week_1_coeffs[i] / week_10_coeffs[i]))
        if abs(week_final_coeffs[i]) > 1e-8:
            ratios_1_to_final.append(abs(week_1_coeffs[i] / week_final_coeffs[i]))
    
    avg_1_to_10 = np.mean(ratios_1_to_10) if ratios_1_to_10 else 1.0
    avg_1_to_final = np.mean(ratios_1_to_final) if ratios_1_to_final else 1.0
    
    return {
        'avg_stability_1_to_10': float(avg_1_to_10),
        'max_instability_1_to_10': float(max(ratios_1_to_10)) if ratios_1_to_10 else 1.0,
        'avg_stability_1_to_final': float(avg_1_to_final),
        'max_instability_1_to_final': float(max(ratios_1_to_final)) if ratios_1_to_final 
        else 1.0,
        'coefficient_range': float(np.max(real_coeffs) - np.min(real_coeffs)),
        'coefficient_std': float(np.std(real_coeffs)),
        'coefficient_mean': float(np.mean(real_coeffs)),
        'stability_status': get_stability_status(avg_1_to_10),
        'weeks_analyzed': int(real_coeffs.shape[1])
    }


def get_stability_status(avg_ratio: float) -> str:
    """
    Get human-readable stability status based on coefficient ratio.
    
    Args:
        avg_ratio: Average stability ratio (Week 1 / Week 10)
    
    Returns:
        str: Status description
    """
    if avg_ratio <= 1.1:
        return "EXCELLENT"
    elif avg_ratio <= 1.3:
        return "VERY_GOOD"  
    elif avg_ratio <= 1.8:
        return "GOOD"
    elif avg_ratio <= 2.5:
        return "FAIR"
    else:
        return "NEEDS_IMPROVEMENT"


def analyze_coefficients(
    coefficients: torch.Tensor,
    data_dict: Dict[str, Any],
    exclude_burn_in: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive coefficient analysis with automatic burn-in handling.
    
    This is the main function users should use for coefficient analysis.
    It automatically detects and excludes burn-in periods, preventing
    common analysis errors.
    
    Args:
        coefficients: Raw coefficients from model [regions, time_steps, channels]
        data_dict: Data dictionary from preprocessing (must contain 'regions')
        exclude_burn_in: Whether to automatically exclude burn-in (recommended: True)
    
    Returns:
        Dict with comprehensive analysis results including:
        - stability_metrics: Ratios, status, statistics
        - real_coefficients: Clean coefficients (burn-in excluded)
        - summary: Quick overview for users
    """
    # Calculate burn-in period
    regions = data_dict.get('regions', ['Region_0'])
    
    # Try to get original data length from multiple possible sources
    total_rows = 0
    if 'original_data' in data_dict:
        total_rows = len(data_dict['original_data'])
    elif 'df' in data_dict:
        total_rows = len(data_dict['df'])
    elif 'data' in data_dict:
        total_rows = len(data_dict['data'])
    
    if total_rows > 0:
        original_weeks_per_region = total_rows // len(regions)
    else:
        # Fallback: assume no burn-in if we can't determine original length
        original_weeks_per_region = coefficients.shape[1]
        warnings.warn("Could not determine original data length. Assuming no burn-in period.")
    
    processed_weeks = coefficients.shape[1]
    burn_in_weeks = calculate_burn_in_length(original_weeks_per_region, processed_weeks)
    
    if exclude_burn_in and burn_in_weeks > 0:
        real_coeffs = exclude_burn_in(coefficients, burn_in_weeks, warn_if_no_burn_in=False)
        analysis_note = f"✅ Analysis excludes {burn_in_weeks}-week burn-in period (recommended)"
        exclude_burn_in_weeks = burn_in_weeks
    else:
        real_coeffs = coefficients
        exclude_burn_in_weeks = None
        if burn_in_weeks > 0:
            analysis_note = f"⚠️ WARNING: Analysis includes {burn_in_weeks}-week burn-in period - results may be misleading!"
        else:
            analysis_note = "ℹ️ Analysis includes all time steps (no burn-in detected)"
    
    # Calculate stability metrics
    stability_metrics = calculate_stability_metrics(coefficients, exclude_burn_in_weeks)
    
    # Convert to numpy for return
    real_coeffs_np = real_coeffs.detach().cpu().numpy() if torch.is_tensor(real_coeffs) else real_coeffs
    
    return {
        'stability_metrics': stability_metrics,
        'coefficient_shape': real_coeffs_np.shape,
        'burn_in_weeks_detected': burn_in_weeks,
        'burn_in_excluded': exclude_burn_in and burn_in_weeks > 0,
        'analysis_note': analysis_note,
        'real_coefficients': real_coeffs_np,
        'summary': {
            'stability_ratio': stability_metrics['avg_stability_1_to_10'],
            'status': stability_metrics['stability_status'],
            'weeks_analyzed': stability_metrics['weeks_analyzed'],
            'is_stable': stability_metrics['avg_stability_1_to_10'] <= 1.5,
            'recommendation': 'Use real_coefficients for analysis' if exclude_burn_in else 'Consider excluding burn-in period'
        }
    }


def get_coefficient_summary(analysis: Dict[str, Any]) -> str:
    """
    Get a human-readable summary of coefficient analysis.
    
    Args:
        analysis: Result from analyze_coefficients()
        
    Returns:
        str: Formatted summary for display
    """
    summary = analysis['summary']
    metrics = analysis['stability_metrics']
    
    status_emoji = {
        'EXCELLENT': '🏆',
        'VERY_GOOD': '✅', 
        'GOOD': '📈',
        'FAIR': '⚠️',
        'NEEDS_IMPROVEMENT': '🚨'
    }
    
    emoji = status_emoji.get(summary['status'], '❓')
    
    return f"""
{emoji} COEFFICIENT STABILITY ANALYSIS {emoji}
{'='*50}
Status: {summary['status']} 
Stability Ratio: {summary['stability_ratio']:.3f}x (Week 1 vs Week 10)
Weeks Analyzed: {summary['weeks_analyzed']} (burn-in excluded: {analysis['burn_in_excluded']})
Coefficient Range: {metrics['coefficient_range']:.6f}
Average Coefficient: {metrics['coefficient_mean']:.6f} ± {metrics['coefficient_std']:.6f}

{analysis['analysis_note']}

Recommendation: {summary['recommendation']}
"""


def get_stable_coefficients(
    coefficients: torch.Tensor,
    data_dict: Dict[str, Any]
) -> torch.Tensor:
    """
    Quick function to get clean coefficients with burn-in excluded.
    
    Args:
        coefficients: Raw coefficients from model
        data_dict: Data dictionary from preprocessing
        
    Returns:
        torch.Tensor: Clean coefficients with burn-in excluded
    """
    analysis = analyze_coefficients(coefficients, data_dict, exclude_burn_in=True)
    return torch.tensor(analysis['real_coefficients']) 

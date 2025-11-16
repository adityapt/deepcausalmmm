"""
Utility functions for budget optimization with DeepCausalMMM.

This module provides helper functions to prepare data from DeepCausalMMM
model outputs for budget optimization, including data formatting,
curve parameter extraction, and integration with ResponseCurveFit.
"""

from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

from .response_curves import ResponseCurveFit
from .optimization import BudgetOptimizer, OptimizationResult

logger = logging.getLogger('deepcausalmmm')


def prepare_optimization_data(
    contributions_df: pd.DataFrame,
    media_data: pd.DataFrame,
    *,
    date_col: str = 'week_monday',
    channel_col: str = 'channel',
    contribution_col: str = 'predicted',
    spend_col: str = 'spend',
    impressions_col: str = 'impressions'
) -> pd.DataFrame:
    """
    Prepare data from DeepCausalMMM outputs for response curve fitting and optimization.
    
    This function merges model contribution predictions with media spend/impression
    data to create the required format for ResponseCurveFit.
    
    Parameters
    ----------
    contributions_df : pd.DataFrame
        Model contributions output with columns: date, channel, predicted
    media_data : pd.DataFrame
        Media data with columns: date, channel, spend, impressions
    date_col : str, default='week_monday'
        Name of the date column
    channel_col : str, default='channel'
        Name of the channel column
    contribution_col : str, default='predicted'
        Name of the contribution/prediction column
    spend_col : str, default='spend'
        Name of the spend column
    impressions_col : str, default='impressions'
        Name of the impressions column
        
    Returns
    -------
    pd.DataFrame
        Merged data ready for ResponseCurveFit with columns:
        week_monday, channel, spend, impressions, predicted
        
    Examples
    --------
    >>> # After training DeepCausalMMM model
    >>> contributions = model.get_contributions()  # Your model output
    >>> media_df = pd.read_csv('media_data.csv')
    >>> 
    >>> optimization_data = prepare_optimization_data(
    ...     contributions_df=contributions,
    ...     media_data=media_df
    ... )
    """
    # Ensure date column is datetime
    contributions_df = contributions_df.copy()
    media_data = media_data.copy()
    
    contributions_df[date_col] = pd.to_datetime(contributions_df[date_col])
    media_data[date_col] = pd.to_datetime(media_data[date_col])
    
    # Merge contributions with media data
    merged = pd.merge(
        contributions_df[[date_col, channel_col, contribution_col]],
        media_data[[date_col, channel_col, spend_col, impressions_col]],
        on=[date_col, channel_col],
        how='inner'
    )
    
    # Rename to standard column names for ResponseCurveFit
    merged = merged.rename(columns={
        date_col: 'week_monday',
        channel_col: 'channel',
        contribution_col: 'predicted',
        spend_col: 'spend',
        impressions_col: 'impressions'
    })
    
    # Remove rows with zero or negative spend/impressions (can't fit curves)
    merged = merged[
        (merged['spend'] > 0) & 
        (merged['impressions'] > 0) & 
        (merged['predicted'] >= 0)
    ]
    
    logger.info(
        f"Prepared optimization data: {len(merged)} rows, "
        f"{merged['channel'].nunique()} channels"
    )
    
    return merged


def fit_response_curves_batch(
    data: pd.DataFrame,
    channels: Optional[List[str]] = None,
    *,
    bottom_param: bool = False,
    model_level: str = 'Overall',
    date_col: str = 'week_monday',
    generate_figures: bool = False,
    save_figures: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[Dict[str, Dict], pd.DataFrame]:
    """
    Fit response curves for multiple channels in batch.
    
    This is a convenience wrapper around ResponseCurveFit that processes
    multiple channels and returns both dictionary and DataFrame formats.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data prepared by prepare_optimization_data() with columns:
        week_monday, channel, spend, impressions, predicted
    channels : List[str], optional
        List of channels to fit. If None, fits all channels in data
    bottom_param : bool, default=False
        Whether to fit non-zero intercept
    model_level : str, default='Overall'
        Aggregation level: 'Overall' or 'DMA'
    date_col : str, default='week_monday'
        Name of date column
    generate_figures : bool, default=False
        Whether to generate plots
    save_figures : bool, default=False
        Whether to save plots to files
    output_dir : str, optional
        Directory to save plots (required if save_figures=True)
        
    Returns
    -------
    curves_dict : Dict[str, Dict]
        Response curve parameters by channel
    curves_df : pd.DataFrame
        Response curve parameters as DataFrame
        
    Examples
    --------
    >>> # After preparing data
    >>> curves_dict, curves_df = fit_response_curves_batch(
    ...     data=optimization_data,
    ...     channels=['TV', 'Search', 'Social'],
    ...     generate_figures=True,
    ...     save_figures=True,
    ...     output_dir='./response_curves/'
    ... )
    >>> print(curves_df)
    """
    if channels is None:
        channels = data['channel'].unique().tolist()
    
    if save_figures and not output_dir:
        raise ValueError("output_dir required when save_figures=True")
    
    logger.info(f"Fitting response curves for {len(channels)} channels...")
    
    curves_dict = {}
    curves_list = []
    
    for channel in channels:
        try:
            channel_data = data[data['channel'] == channel].copy()
            
            if len(channel_data) < 10:
                logger.warning(
                    f"Channel '{channel}' has insufficient data ({len(channel_data)} rows), skipping"
                )
                continue
            
            # Create ResponseCurveFit instance
            fitter = ResponseCurveFit(
                data=channel_data,
                bottom_param=bottom_param,
                model_level=model_level,
                date_col=date_col
            )
            
            # Fit the curve
            output_path = None
            if save_figures and output_dir:
                import os
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/{channel}_response_curve.html"
            
            fitter.fit(
                title=f"{channel} Response Curve",
                x_label="Spend ($)",
                y_label="Predicted Response",
                generate_figure=generate_figures,
                save_figure=save_figures,
                output_path=output_path,
                print_r_sqr=True
            )
            
            # Store results if fitting succeeded
            if fitter.fit_flag:
                curves_dict[channel] = {
                    'top': fitter.top,
                    'bottom': fitter.bottom,
                    'saturation': fitter.saturation,
                    'slope': fitter.slope,
                    'r_2': fitter.r_2
                }
                
                curves_list.append({
                    'channel': channel,
                    'top': fitter.top,
                    'bottom': fitter.bottom,
                    'saturation': fitter.saturation,
                    'slope': fitter.slope,
                    'r_2': fitter.r_2
                })
                
                logger.info(
                    f"  {channel}: Slope={fitter.slope:.3f}, "
                    f"Saturation=${fitter.saturation:,.0f}, R²={fitter.r_2:.3f}"
                )
            else:
                logger.warning(f"  {channel}: Fitting failed")
                
        except Exception as e:
            logger.error(f"  {channel}: Error - {e}")
            continue
    
    curves_df = pd.DataFrame(curves_list)
    
    logger.info(f"Successfully fitted {len(curves_dict)}/{len(channels)} curves")
    
    return curves_dict, curves_df


def create_optimizer_from_model_output(
    contributions_df: pd.DataFrame,
    media_data: pd.DataFrame,
    budget: float,
    *,
    channels: Optional[List[str]] = None,
    num_weeks: int = 52,
    constraints: Optional[Dict[str, Dict[str, float]]] = None,
    method: str = 'trust-constr',
    generate_figures: bool = False,
    save_figures: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[BudgetOptimizer, pd.DataFrame]:
    """
    End-to-end: Create optimizer from DeepCausalMMM model outputs.
    
    This function handles the complete workflow:
    1. Prepare data from model outputs
    2. Fit response curves for all channels
    3. Create and configure BudgetOptimizer
    
    Parameters
    ----------
    contributions_df : pd.DataFrame
        Model contribution predictions
    media_data : pd.DataFrame
        Media spend and impression data
    budget : float
        Total budget to optimize
    channels : List[str], optional
        Channels to include. If None, uses all channels
    num_weeks : int, default=52
        Planning horizon in weeks
    constraints : Dict[str, Dict[str, float]], optional
        Channel spend constraints
    method : str, default='trust-constr'
        Optimization method
    generate_figures : bool, default=False
        Whether to generate response curve plots
    save_figures : bool, default=False
        Whether to save plots
    output_dir : str, optional
        Directory for plots
        
    Returns
    -------
    optimizer : BudgetOptimizer
        Configured optimizer ready to run
    curves_df : pd.DataFrame
        Response curve parameters
        
    Examples
    --------
    >>> # Complete workflow from model outputs to optimizer
    >>> optimizer, curves = create_optimizer_from_model_output(
    ...     contributions_df=model_contributions,
    ...     media_data=media_df,
    ...     budget=1000000,
    ...     constraints={'TV': {'lower': 100000, 'upper': 600000}},
    ...     generate_figures=True,
    ...     save_figures=True,
    ...     output_dir='./optimization_results/'
    ... )
    >>> 
    >>> # Run optimization
    >>> result = optimizer.optimize()
    >>> print(result.allocation)
    """
    # Step 1: Prepare data
    logger.info("Step 1: Preparing optimization data...")
    opt_data = prepare_optimization_data(contributions_df, media_data)
    
    # Step 2: Fit response curves
    logger.info("Step 2: Fitting response curves...")
    curves_dict, curves_df = fit_response_curves_batch(
        data=opt_data,
        channels=channels,
        generate_figures=generate_figures,
        save_figures=save_figures,
        output_dir=output_dir
    )
    
    if not curves_dict:
        raise ValueError("No response curves successfully fitted")
    
    # Step 3: Create optimizer
    logger.info("Step 3: Creating optimizer...")
    fitted_channels = list(curves_dict.keys())
    
    optimizer = BudgetOptimizer(
        budget=budget,
        channels=fitted_channels,
        response_curves=curves_dict,
        num_weeks=num_weeks,
        method=method
    )
    
    # Set constraints if provided
    if constraints:
        # Filter constraints to only include channels we have curves for
        valid_constraints = {
            ch: c for ch, c in constraints.items() 
            if ch in fitted_channels
        }
        if valid_constraints:
            optimizer.set_constraints(valid_constraints)
    
    logger.info(f"Optimizer ready: {len(fitted_channels)} channels, ${budget:,.0f} budget")
    
    return optimizer, curves_df


def compare_current_vs_optimal(
    current_allocation: Dict[str, float],
    optimal_result: OptimizationResult,
    *,
    metric_name: str = "Response"
) -> pd.DataFrame:
    """
    Compare current budget allocation vs optimized allocation.
    
    Parameters
    ----------
    current_allocation : Dict[str, float]
        Current spend by channel
    optimal_result : OptimizationResult
        Result from optimizer.optimize()
    metric_name : str, default='Response'
        Name of the metric being optimized
        
    Returns
    -------
    pd.DataFrame
        Comparison table with current, optimal, and deltas
        
    Examples
    --------
    >>> current = {'TV': 400000, 'Search': 350000, 'Social': 250000}
    >>> result = optimizer.optimize()
    >>> 
    >>> comparison = compare_current_vs_optimal(current, result)
    >>> print(comparison)
    """
    # Get optimal allocation
    optimal_allocation = optimal_result.allocation
    
    # Ensure same channels
    channels = sorted(set(current_allocation.keys()) | set(optimal_allocation.keys()))
    
    comparison = []
    for channel in channels:
        current_spend = current_allocation.get(channel, 0)
        optimal_spend = optimal_allocation.get(channel, 0)
        
        # Get response from optimal result
        channel_row = optimal_result.by_channel[
            optimal_result.by_channel['channel'] == channel
        ]
        
        if len(channel_row) > 0:
            optimal_response = channel_row['total_response'].iloc[0]
            optimal_roi = channel_row['roi'].iloc[0]
        else:
            optimal_response = 0
            optimal_roi = 0
        
        comparison.append({
            'channel': channel,
            'current_spend': current_spend,
            'optimal_spend': optimal_spend,
            'spend_delta': optimal_spend - current_spend,
            'spend_delta_pct': ((optimal_spend - current_spend) / current_spend * 100) if current_spend > 0 else 0,
            f'optimal_{metric_name.lower()}': optimal_response,
            'optimal_roi': optimal_roi
        })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('optimal_spend', ascending=False).reset_index(drop=True)
    
    return df


def generate_optimization_report(
    result: OptimizationResult,
    curves_df: pd.DataFrame,
    current_allocation: Optional[Dict[str, float]] = None,
    *,
    output_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive text report of optimization results.
    
    Parameters
    ----------
    result : OptimizationResult
        Optimization result
    curves_df : pd.DataFrame
        Response curve parameters
    current_allocation : Dict[str, float], optional
        Current allocation for comparison
    output_path : str, optional
        Path to save report (if not provided, returns as string)
        
    Returns
    -------
    str
        Formatted report text
        
    Examples
    --------
    >>> report = generate_optimization_report(
    ...     result=result,
    ...     curves_df=curves,
    ...     current_allocation={'TV': 400000, 'Search': 350000, 'Social': 250000},
    ...     output_path='optimization_report.txt'
    ... )
    >>> print(report)
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BUDGET OPTIMIZATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
    lines.append(f"Method: {result.method}")
    lines.append(f"Total Budget: ${sum(result.allocation.values()):,.0f}")
    lines.append(f"Predicted Response: {result.predicted_response:,.0f}")
    lines.append(f"Overall ROI: {result.predicted_response / sum(result.allocation.values()):.3f}")
    lines.append("")
    
    # Optimal Allocation
    lines.append("OPTIMAL ALLOCATION")
    lines.append("-" * 80)
    for _, row in result.by_channel.iterrows():
        lines.append(
            f"{row['channel']:20s} | "
            f"${row['total_spend']:>12,.0f} ({row['spend_pct']:>5.1f}%) | "
            f"Response: {row['total_response']:>12,.0f} ({row['response_pct']:>5.1f}%) | "
            f"ROI: {row['roi']:>6.3f}"
        )
    lines.append("")
    
    # Response Curves
    lines.append("RESPONSE CURVE PARAMETERS")
    lines.append("-" * 80)
    for _, row in curves_df.iterrows():
        lines.append(
            f"{row['channel']:20s} | "
            f"Saturation: ${row['saturation']:>12,.0f} | "
            f"Slope: {row['slope']:>5.3f} | "
            f"R²: {row['r_2']:>5.3f}"
        )
    lines.append("")
    
    # Comparison with current
    if current_allocation:
        lines.append("COMPARISON: CURRENT vs OPTIMAL")
        lines.append("-" * 80)
        comparison = compare_current_vs_optimal(current_allocation, result)
        for _, row in comparison.iterrows():
            delta_str = f"{row['spend_delta']:+,.0f} ({row['spend_delta_pct']:+.1f}%)"
            lines.append(
                f"{row['channel']:20s} | "
                f"Current: ${row['current_spend']:>12,.0f} | "
                f"Optimal: ${row['optimal_spend']:>12,.0f} | "
                f"Delta: {delta_str:>25s}"
            )
        lines.append("")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {output_path}")
    
    return report


#!/usr/bin/env python3
"""
Example script demonstrating ResponseCurveFitter usage with DeepCausalMMM.

This script shows how to:
1. Train a real DeepCausalMMM model
2. Extract channel-level predictions for each DMA-week
3. Fit response curves to actual model outputs
4. Visualize and analyze the results
"""

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from typing import Optional
from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline


class ResponseCurveFit(object):
    """Response curve fitting using Hill equation (user's original implementation)"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        bottom_param: bool = False,
        Modellevel: str = 'Overall',
        Datecol = 'week_monday'
    ) -> None:
        self.data = data
        self.bottom_param = bottom_param
        self.Modellevel = Modellevel
        self.Datecol = Datecol

    def Hill(self, X: np.ndarray, *params) -> np.ndarray:
        self.top = params[0]
        self.bottom = params[1] if self.bottom_param else 0
        self.saturation = params[2]
        self.slope = params[3]

        return self.bottom + (self.top - self.bottom) * X**self.slope / (
            self.saturation**self.slope + X**self.slope
        )
    
    def get_param(self, curve_fit_kws: dict):
        min_data = np.amin(self.y_data)
        max_data = np.amax(self.y_data)

        h = abs(max_data - min_data)
        param_initial = [max_data, min_data, 0.5 * (self.X_data[-1] - self.X_data[0]), 1]
        param_bounds = (
            [max_data - 0.5 * h, min_data - 0.5 * h, self.X_data[0] * 0.1, 0.01],
            [max_data + 0.5 * h, min_data + 0.5 * h, self.X_data[-1] * 10, 100],
        )
        curve_fit_kws.setdefault("p0", param_initial)
        curve_fit_kws.setdefault("bounds", param_bounds)
        popt, _ = curve_fit(self.Hill, self.X_data, self.y_data, **curve_fit_kws)
        if not self.bottom_param:
            popt[1] = 0
        return [float(param) for param in popt]

    def regression(
        self,
        x_fit,
        y_fit,
        x_label,
        y_label,
        title,
        sigfigs,
        log_x,
        print_r_sqr,
        generate_figure,
        view_figure,
        *params,
    ) -> None:
        corrected_y_data = self.Hill(self.X_data, *params)
        self.r_2 = r2_score(self.y_data, corrected_y_data)

        if generate_figure:
            self.figure = go.Figure()
            
            self.figure.add_trace(go.Scatter(
                x=self.X_data, y=self.y_data,
                name='Observed Data',
                mode='markers'
            ))
            
            self.figure.add_trace(go.Scatter(
                x=x_fit, y=y_fit,
                name='Fitted Model',
                mode='lines'
            ))
            
            self.figure.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                legend_title="Legend",
                width=1500,
                height=900,
                yaxis_zeroline=False,
                xaxis_zeroline=False
            )
        
            if print_r_sqr:
                print(f"   R² Score: {self.r_2:.4f}")

            if view_figure:
                self.figure.show()

    def fit_model(
        self,
        x_label: str = "x",
        y_label: str = "y",
        title: str = "Fitted Hill equation",
        sigfigs: int = 6,
        log_x: bool = False,
        print_r_sqr: bool = True,
        generate_figure: bool = True,
        view_figure: bool = False,
        save_figure: bool = False,
        output_path: Optional[str] = None,
        curve_fit_kws: Optional[dict] = None,
    ):
        if self.Modellevel == 'Overall':
            cpi = self.data['spend'].sum() / self.data['impressions'].sum()
            
            self.data_agg = self.data[[self.Datecol, 'impressions', 'predicted']].groupby(self.Datecol).sum()
            self.data_agg['spend'] = self.data_agg['impressions'] * cpi
            self.data_agg.sort_values(by='spend', inplace=True)
            
            self.X_data = np.array(self.data_agg['spend'])
            self.y_data = np.array(self.data_agg['predicted'])
        
            if self.X_data[0] > self.X_data[-1]:
                raise ValueError(
                    f"The first point {self.X_data[0]} and the last point {self.X_data[-1]} are not amenable with the scipy.curvefit function."
                )

            if curve_fit_kws is None:
                curve_fit_kws = {}
            
            self.generate_figure = generate_figure
            self.x_fit = np.logspace(
                np.log10(self.X_data[0]), np.log10(self.X_data[-1]), len(self.y_data)
            )
            
            self.fit_flag = False
            
            try:
                params = self.get_param(curve_fit_kws)
                
                self.y_fit = self.Hill(self.x_fit, *params)
                self.equation = f"{np.round(self.bottom, sigfigs)} + ({np.round(self.top, sigfigs)}-{np.round(self.bottom, sigfigs)})*x**{(np.round(self.slope, sigfigs))} / ({np.round(self.saturation, sigfigs)}**{(np.round(self.slope, sigfigs))} + x**{(np.round(self.slope, sigfigs))})"

                self.regression(
                    self.x_fit,
                    self.y_fit,
                    x_label,
                    y_label,
                    title,
                    sigfigs,
                    log_x,
                    print_r_sqr,
                    generate_figure,
                    view_figure,
                    *params,
                )
                
                if save_figure and output_path and generate_figure:
                    self.figure.write_html(output_path)
                    print(f"   Figure saved to: {output_path}")
                
                self.fit_flag = True
                
            except RuntimeError as re:
                self.r_2 = 0
                print(f"    Fitting failed: {re}")
    
    def get_summary(self):
        """Get summary of fitted parameters"""
        return {
            'params': {
                'top': self.top,
                'bottom': self.bottom,
                'saturation': self.saturation,
                'slope': self.slope
            },
            'r2': self.r_2,
            'equation': self.equation if hasattr(self, 'equation') else None
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.fit_flag:
            return self.bottom + (self.top - self.bottom) * X**self.slope / (
                self.saturation**self.slope + X**self.slope
            )


def load_real_mmm_data(filepath="data/MMM Data.csv"):
    """Load and process the real MMM Data.csv"""
    print(f" Loading Real MMM Data from: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Identify columns
    impression_cols = [col for col in df.columns if 'impressions_' in col]
    value_cols = [col for col in df.columns if 'value_' in col and col != 'value_visits_visits']
    target_col = 'value_visits_visits'
    region_col = 'dmacode'
    time_col = 'weekid'
    
    # Clean channel names
    media_names = []
    for col in impression_cols:
        clean_name = col.replace('impressions_', '').split('_delayed')[0].split('_exponential')[0].split('_geometric')[0]
        clean_name = clean_name.replace('_', ' ')
        media_names.append(clean_name)
    
    control_names = []
    for col in value_cols:
        clean_name = col.replace('value_', '').replace('econmetricsmsa_', '').replace('mortgagemetrics_', '').replace('moodys_', '')
        clean_name = clean_name.replace('_sm', '').replace('_', ' ').title()
        control_names.append(clean_name)
    
    # Get unique regions and weeks
    regions = sorted(df[region_col].unique())
    weeks = sorted(df[time_col].unique())
    n_regions = len(regions)
    n_weeks = len(weeks)
    
    # Create complete grid
    complete_index = pd.MultiIndex.from_product([regions, weeks], names=[region_col, time_col])
    complete_df = pd.DataFrame(index=complete_index).reset_index()
    df_complete = complete_df.merge(df, on=[region_col, time_col], how='left')
    
    # Handle missing values
    for col in impression_cols:
        df_complete[col] = df_complete[col].fillna(0)
    
    for col in value_cols + [target_col]:
        df_complete[col] = df_complete.groupby(region_col)[col].fillna(method='ffill').fillna(method='bfill')
        if df_complete[col].isna().any():
            df_complete[col] = df_complete[col].fillna(df_complete[col].mean())
    
    # Create mappings
    region_map = {region: i for i, region in enumerate(regions)}
    week_map = {week: i for i, week in enumerate(weeks)}
    df_complete['region_idx'] = df_complete[region_col].map(region_map)
    df_complete['week_idx'] = df_complete[time_col].map(week_map)
    df_complete = df_complete.sort_values(['region_idx', 'week_idx'])
    
    # Extract arrays
    X_media_list = []
    X_control_list = []
    y_list = []
    
    for region_idx in range(n_regions):
        region_data = df_complete[df_complete['region_idx'] == region_idx].sort_values('week_idx')
        X_media_list.append(region_data[impression_cols].values.astype(np.float32))
        X_control_list.append(region_data[value_cols].values.astype(np.float32))
        y_list.append(region_data[target_col].values.astype(np.float32))
    
    X_media = np.stack(X_media_list, axis=0)
    X_control = np.stack(X_control_list, axis=0)
    y = np.stack(y_list, axis=0)
    
    print(f"    Loaded: {n_regions} regions × {n_weeks} weeks, {len(media_names)} channels")
    
    return X_media, X_control, y, media_names, control_names, df_complete, impression_cols


def train_model_and_get_predictions():
    """Train DeepCausalMMM model and get channel-level predictions using package postprocessing"""
    print("\n" + "=" * 80)
    print("STEP 1: Training DeepCausalMMM Model")
    print("=" * 80)
    
    # Load data
    X_media, X_control, y, media_names, control_names, df_complete, impression_cols = load_real_mmm_data()
    
    # Load config
    config = get_default_config()
    config['n_epochs'] = 2500  # Train for better convergence
    
    # Create pipeline
    pipeline = UnifiedDataPipeline(config)
    
    # Split data
    train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
    train_tensors = pipeline.fit_and_transform_training(train_data)
    
    # Create model
    trainer = ModelTrainer(config)
    n_media = train_tensors['X_media'].shape[2]
    n_control = train_tensors['X_control'].shape[2]
    n_regions = train_tensors['X_media'].shape[0]
    
    model = trainer.create_model(n_media, n_control, n_regions)
    trainer.create_optimizer_and_scheduler()
    
    # Train
    print("\n Training model...")
    full_data = {'X_media': X_media, 'X_control': X_control, 'y': y}
    full_tensors = pipeline.fit_and_transform_training(full_data)
    
    training_results = trainer.train(
        train_tensors['X_media'], train_tensors['X_control'],
        train_tensors['R'], train_tensors['y'],
        y_full_for_baseline=full_tensors['y'],
        verbose=True
    )
    
    print(f"\n Training complete!")
    print(f"   Training R²: {training_results['final_train_r2']:.3f}")
    
    # Print Hill parameters from the trained model
    print("\n" + "=" * 80)
    print("MODEL'S HILL PARAMETERS (from DeepCausalMMM)")
    print("=" * 80)
    print(f"{'Channel':<40} {'Slope (a)':<12} {'Shape (g)':<12}")
    print("-" * 80)
    
    with torch.no_grad():
        for ch_idx, ch_name in enumerate(media_names):
            hill_a_raw = model.hill_a[ch_idx].item()
            hill_g_raw = model.hill_g[ch_idx].item()
            
            # Apply transformations (same as in model)
            hill_a = torch.nn.functional.softplus(torch.tensor(hill_a_raw)).item()
            hill_a = min(max(hill_a, 2.0), 5.0)  # Clamped to [2.0, 5.0]
            
            hill_g = torch.nn.functional.softplus(torch.tensor(hill_g_raw)).item()
            
            print(f"{ch_name:<40} {hill_a:<12.4f} {hill_g:<12.6f}")
    
    print("=" * 80)
    
    # Get channel-level predictions using PACKAGE POSTPROCESSING
    print("\n" + "=" * 80)
    print("STEP 2: Extracting Channel-Level Predictions (Using Package Postprocessing)")
    print("=" * 80)
    
    # Use the pipeline's predict_and_postprocess method - handles everything correctly!
    postprocess_results = pipeline.predict_and_postprocess(
        model=model,
        X_media=X_media,
        X_control=X_control,
        channel_names=media_names,
        control_names=control_names,
        combine_with_holdout=True
    )
    
    # Extract predictions in original scale
    predictions_orig = postprocess_results['predictions']  # Already in original scale
    
    # Extract ALL components from model (in log-space)
    media_contributions_log = postprocess_results['media_contributions']  # [n_regions, n_weeks, n_channels]
    control_contributions_log = postprocess_results['control_contributions']  # [n_regions, n_weeks, n_controls]
    
    # Get model outputs which include baseline and seasonality
    print("    Getting baseline and seasonality from model...")
    
    # Re-run model to get all components
    full_data = {'X_media': X_media, 'X_control': X_control, 'y': y}
    full_tensors = pipeline.fit_and_transform_training(full_data)
    
    model.eval()
    with torch.no_grad():
        predictions_log, media_contrib_log, control_contrib_log, outputs = model(
            full_tensors['X_media'], 
            full_tensors['X_control'], 
            full_tensors['R']
        )
    
    # Extract baseline and seasonality from outputs
    baseline_log = outputs.get('baseline', torch.zeros_like(predictions_log))
    seasonal_log = outputs.get('seasonal_contributions', torch.zeros_like(predictions_log))
    
    # Remove burn-in from all components
    burn_in = pipeline.padding_weeks
    if burn_in > 0:
        media_contrib_log = media_contrib_log[:, burn_in:, :]
        control_contrib_log = control_contrib_log[:, burn_in:, :]
        baseline_log = baseline_log[:, burn_in:]
        seasonal_log = seasonal_log[:, burn_in:]
        predictions_log = predictions_log[:, burn_in:]
    
    print(f"    Extracted all components (after burn-in removal):")
    print(f"      Media: {media_contrib_log.shape}")
    print(f"      Control: {control_contrib_log.shape}")
    print(f"      Baseline: {baseline_log.shape}")
    print(f"      Seasonal: {seasonal_log.shape}")
    print(f"      Predictions (log): {predictions_log.shape}")
    
    # Convert predictions to original scale using the same method as dashboard
    predictions_orig_tensor = torch.expm1(torch.clamp(predictions_log, max=20.0))
    
    # USE PROPORTIONAL ALLOCATION METHOD (same as dashboard)
    print("\n    Using proportional allocation method to get contributions in original scale...")
    
    # Get the scaler from pipeline
    scaler = pipeline.get_scaler()
    
    # Use the package's inverse_transform_contributions method
    contrib_results = scaler.inverse_transform_contributions(
        media_contributions=media_contrib_log,
        y_pred_orig=predictions_orig_tensor,
        baseline=baseline_log,
        control_contributions=control_contrib_log,
        seasonal_contributions=seasonal_log
    )
    
    # Extract proportionally allocated media contributions
    media_contributions_orig = contrib_results['media']  # [n_regions, n_weeks, n_channels] in original scale
    
    print(f"    Proportionally allocated contributions computed:")
    print(f"      Media contrib (log): [{media_contrib_log.min():.4f}, {media_contrib_log.max():.4f}]")
    print(f"      Media contrib (orig): [{media_contributions_orig.min():.0f}, {media_contributions_orig.max():.0f}]")
    print()
    print("    Per-channel contribution ranges (original scale):")
    
    # Show range for each channel
    for ch_idx, ch_name in enumerate(media_names):
        ch_contrib = media_contributions_orig[:, :, ch_idx]
        print(f"      {ch_name}: [{ch_contrib.min():.0f}, {ch_contrib.max():.0f}], mean={ch_contrib.mean():.0f}")
    
    return media_contributions_orig, X_media, df_complete, media_names, impression_cols


def create_sample_data(n_weeks: int = 52, n_regions: int = 5) -> pd.DataFrame:
    """
    Create sample data for demonstration purposes.
    
    In a real scenario, this data would come from:
    - Historical spend/impressions data
    - Model predictions from a trained DeepCausalMMM model
    
    Parameters
    ----------
    n_weeks : int
        Number of weeks of data
    n_regions : int
        Number of regions/DMAs
    
    Returns
    -------
    pd.DataFrame
        Sample data with spend, impressions, and predicted values
    """
    np.random.seed(42)
    
    data_list = []
    
    for region in range(1, n_regions + 1):
        for week in range(n_weeks):
            # Simulate varying spend levels
            base_spend = np.random.uniform(5000, 50000)
            impressions = base_spend * np.random.uniform(80, 120)  # Variable CPI
            
            # Simulate saturation effect (Hill curve)
            saturation = 30000
            slope = 2.0
            top = 100000 * region  # Regions have different scales
            predicted = top * base_spend**slope / (saturation**slope + base_spend**slope)
            predicted += np.random.normal(0, predicted * 0.1)  # Add noise
            
            data_list.append({
                'week_monday': pd.Timestamp('2023-01-01') + pd.Timedelta(weeks=week),
                'dmacode': f'DMA_{region:03d}',
                'spend': base_spend,
                'impressions': impressions,
                'predicted': max(0, predicted)  # Ensure non-negative
            })
    
    return pd.DataFrame(data_list)


def example_overall_curve():
    """Example 1: Fit overall (aggregated) response curve."""
    print("=" * 80)
    print("EXAMPLE 1: Overall Response Curve")
    print("=" * 80)
    
    # Create sample data
    data = create_sample_data(n_weeks=52, n_regions=5)
    print(f"\n Created sample data: {len(data)} rows, {data['dmacode'].nunique()} regions")
    
    # Initialize fitter for overall level
    fitter = ResponseCurveFitter(
        data,
        model_level='Overall',
        date_col='week_monday',
        fit_intercept=False  # Response at zero spend = 0
    )
    
    # Fit the curve
    fitter.fit(
        title="Overall Response Curve: Spend vs Predicted Outcome",
        x_label="Marketing Spend ($)",
        y_label="Predicted Outcome",
        print_r2=True,
        generate_figure=True,
        view_figure=False,  # Set to True to open in browser
        save_figure=True,
        output_path="response_curve_overall_example.html"
    )
    
    # Get summary
    summary = fitter.get_summary()
    print(f"\n Fitted Parameters:")
    print(f"   Top (saturation level): ${summary['params']['top']:,.2f}")
    print(f"   Saturation point: ${summary['params']['saturation']:,.2f}")
    print(f"   Slope (steepness): {summary['params']['slope']:.3f}")
    print(f"   R² Score: {summary['r2']:.4f}")
    
    # Make predictions on new spend levels
    new_spends = np.array([10000, 25000, 50000, 75000, 100000])
    predictions = fitter.predict(new_spends)
    
    print(f"\n Predictions for new spend levels:")
    for spend, pred in zip(new_spends, predictions):
        print(f"   Spend ${spend:>7,} → Predicted: {pred:>10,.2f}")
    
    # Calculate marginal ROI
    print(f"\n Marginal ROI Analysis:")
    for i in range(len(new_spends) - 1):
        delta_spend = new_spends[i + 1] - new_spends[i]
        delta_response = predictions[i + 1] - predictions[i]
        marginal_roi = delta_response / delta_spend
        print(f"   ${new_spends[i]:>7,} → ${new_spends[i+1]:>7,}: "
              f"Marginal ROI = {marginal_roi:.4f}")
    
    return fitter


def example_region_curves():
    """Example 2: Fit region-level response curves."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Region-Level Response Curves")
    print("=" * 80)
    
    # Create sample data
    data = create_sample_data(n_weeks=52, n_regions=10)
    print(f"\n Created sample data: {len(data)} rows, {data['dmacode'].nunique()} regions")
    
    # Initialize fitter for region level
    fitter = ResponseCurveFitter(
        data,
        model_level='Region',
        date_col='week_monday',
        region_col='dmacode',
        fit_intercept=False
    )
    
    # Fit curves for all regions
    region_params = fitter.fit()
    
    # Display results
    print(f"\n Region Parameters (Top 5 by R²):")
    top_regions = region_params.nlargest(5, 'r2')
    print(top_regions.to_string(index=False))
    
    # Create comparison plot
    fig = fitter.plot_region_comparison(
        top_n=10,
        metric='r2',
        save_path='response_curve_region_comparison.html'
    )
    print(f"\n Region comparison plot saved to: response_curve_region_comparison.html")
    
    # Identify best and worst performing regions
    best_region = region_params.loc[region_params['r2'].idxmax()]
    worst_region = region_params.loc[region_params['r2'].idxmin()]
    
    print(f"\n Best Performing Region:")
    print(f"   Region: {best_region['dmacode']}")
    print(f"   R²: {best_region['r2']:.4f}")
    print(f"   Top: {best_region['top']:,.2f}")
    print(f"   Saturation: {best_region['saturation']:,.2f}")
    
    print(f"\n  Worst Performing Region:")
    print(f"   Region: {worst_region['dmacode']}")
    print(f"   R²: {worst_region['r2']:.4f}")
    
    return fitter, region_params


def example_with_real_model_predictions():
    """
    Example 3: Integration with DeepCausalMMM model predictions.
    
    This example shows the typical workflow:
    1. Train a DeepCausalMMM model
    2. Get predictions for each channel
    3. Fit response curves to understand saturation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Integration with DeepCausalMMM")
    print("=" * 80)
    
    print("""
    Typical workflow for using ResponseCurveFitter with DeepCausalMMM:
    
    1. Train your model:
       ```python
       from deepcausalmmm import DeepCausalMMM, ModelTrainer
       
       model = DeepCausalMMM(...)
       trainer = ModelTrainer(model, ...)
       trainer.train(X_media, X_control, y, regions)
       ```
    
    2. Get predictions and contributions:
       ```python
       from deepcausalmmm import InferenceManager
       
       inference = InferenceManager(model, ...)
       results = inference.predict(X_media, X_control, regions)
       channel_contributions = results['contributions']  # [n_regions, n_weeks, n_channels]
       ```
    
    3. Prepare data for response curves:
       For each channel, create a DataFrame with:
       - Historical spend data
       - Impressions (if available)
       - Model predictions (channel contributions)
       
       ```python
       # Example for a single channel (e.g., 'TV')
       channel_data = pd.DataFrame({
           'week_monday': dates,
           'dmacode': regions,
           'spend': tv_spend,
           'impressions': tv_impressions,
           'predicted': channel_contributions[:, :, channel_idx].flatten()
       })
       ```
    
    4. Fit response curves:
       ```python
       from deepcausalmmm import ResponseCurveFitter
       
       fitter = ResponseCurveFitter(
           channel_data,
           model_level='Overall',  # or 'Region'
           fit_intercept=False
       )
       
       fitter.fit(
           title=f"Response Curve: TV Channel",
           generate_figure=True,
           save_figure=True,
           output_path='tv_response_curve.html'
       )
       ```
    
    5. Use for optimization:
       ```python
       # Find optimal spend allocation
       current_spend = 50000
       current_response = fitter.predict(np.array([current_spend]))[0]
       
       # Test different spend levels
       test_spends = np.linspace(10000, 100000, 50)
       test_responses = fitter.predict(test_spends)
       
       # Calculate ROI for each level
       rois = test_responses / test_spends
       optimal_idx = np.argmax(rois)
       optimal_spend = test_spends[optimal_idx]
       
       print(f"Optimal spend: ${optimal_spend:,.2f}")
       print(f"Expected ROI: {rois[optimal_idx]:.4f}")
       ```
    
    Benefits:
    - Understand saturation points for each channel
    - Optimize budget allocation across channels
    - Identify diminishing returns
    - Support scenario planning and what-if analysis
    """)


def fit_response_curves_to_real_data():
    """Fit response curves to real DeepCausalMMM predictions using proportional allocation"""
    print("\n" + "=" * 80)
    print("STEP 3: Fitting Response Curves to Real Model Predictions")
    print("=" * 80)
    
    # Get real model predictions (using proportional allocation method)
    media_contributions, X_media, df_complete, media_names, impression_cols = train_model_and_get_predictions()
    
    # Convert to numpy (already in original scale from proportional allocation)
    if isinstance(media_contributions, torch.Tensor):
        media_contributions_np = media_contributions.detach().cpu().numpy()
    else:
        media_contributions_np = media_contributions
    n_regions, n_weeks, n_channels = media_contributions_np.shape
    
    print(f"    Using PROPORTIONALLY ALLOCATED contributions (original scale)")
    print(f"    This correctly distributes total predicted visits across all components")
    print(f"    Method: contribution_orig = (contrib_log / total_log) × y_pred_orig")
    
    print(f"\n Processing {n_channels} channels across {n_regions} regions and {n_weeks} weeks")
    
    # Process each channel
    all_results = {}
    
    for ch_idx, (channel_name, impression_col) in enumerate(zip(media_names, impression_cols)):
        print(f"\n{'='*60}")
        print(f"Channel {ch_idx+1}/{n_channels}: {channel_name}")
        print(f"{'='*60}")
        
        # AGGREGATE TO NATIONAL WEEKLY LEVEL (same as dashboard)
        # Sum impressions and contributions across all DMAs for each week
        print(f"    Aggregating to national weekly level...")
        
        weekly_impressions = []
        weekly_contributions = []
        
        for week_idx in range(n_weeks):
            # Sum across all regions for this week
            week_impressions = X_media[:, week_idx, ch_idx].sum()
            week_contributions = media_contributions_np[:, week_idx, ch_idx].sum()
            
            weekly_impressions.append(week_impressions)
            weekly_contributions.append(week_contributions)
        
        # Create DataFrame with weekly aggregated data
        channel_df = pd.DataFrame({
            'week_monday': [pd.Timestamp('2023-01-01') + pd.Timedelta(weeks=i) for i in range(n_weeks)],
            'week': list(range(1, n_weeks + 1)),
            'spend': weekly_impressions,
            'impressions': weekly_impressions,
            'predicted': weekly_contributions
        })
        
        # Remove zero spend rows (can't fit curves to zero spend)
        channel_df = channel_df[channel_df['spend'] > 0]
        
        if len(channel_df) < 10:
            print(f"     Skipping {channel_name}: insufficient non-zero data points ({len(channel_df)})")
            continue
        
        print(f"    Data points: {len(channel_df)}")
        print(f"    Spend range: ${channel_df['spend'].min():,.0f} - ${channel_df['spend'].max():,.0f}")
        print(f"    Prediction range: {channel_df['predicted'].min():,.0f} - {channel_df['predicted'].max():,.0f}")
        
        # Fit overall response curve using user's original ResponseCurveFit class
        try:
            fitter = ResponseCurveFit(
                channel_df,
                bottom_param=False,
                Modellevel='Overall',
                Datecol='week_monday'
            )
            
            fitter.fit_model(
                title=f"Response Curve: {channel_name}",
                x_label="Impressions",
                y_label="Predicted Visits",
                print_r_sqr=True,
                generate_figure=True,
                save_figure=True,
                output_path=f"response_curve_{channel_name.replace(' ', '_')}.html"
            )
            
            summary = fitter.get_summary()
            all_results[channel_name] = summary
            
            print(f"    Fitted successfully!")
            print(f"      Top: {summary['params']['top']:,.0f}")
            print(f"      Saturation: {summary['params']['saturation']:,.0f}")
            print(f"      Slope: {summary['params']['slope']:.3f}")
            print(f"      R²: {summary['r2']:.4f}")
            
        except Exception as e:
            print(f"    Failed to fit: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Response Curve Fitting Results")
    print("=" * 80)
    
    if all_results:
        results_df = pd.DataFrame([
            {
                'Channel': ch,
                'Top': res['params']['top'],
                'Saturation': res['params']['saturation'],
                'Slope': res['params']['slope'],
                'R²': res['r2']
            }
            for ch, res in all_results.items()
        ])
        
        results_df = results_df.sort_values('R²', ascending=False)
        print(results_df.to_string(index=False))
        
        print(f"\n Successfully fitted {len(all_results)}/{n_channels} channels")
        print(f" Generated {len(all_results)} HTML response curve files")
    else:
        print(" No channels were successfully fitted")
    
    return all_results


def main():
    """Run all examples."""
    print("\n" + "" * 40)
    print("ResponseCurveFitter with Real DeepCausalMMM Data")
    print("" * 40)
    
    # Fit response curves to real model predictions
    results = fit_response_curves_to_real_data()
    
    print("\n" + "=" * 80)
    print(" Response curve fitting completed!")
    print("=" * 80)
    print("\nGenerated files:")
    for channel_name in results.keys():
        filename = f"response_curve_{channel_name.replace(' ', '_')}.html"
        print(f"  - {filename}")
    print("\nOpen these files in your browser to view the interactive plots.")


if __name__ == "__main__":
    main()

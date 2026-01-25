#!/usr/bin/env python3
"""
Dashboard with Linear Scaling (y/y_mean per DMA)
================================================
Testing linear scaling approach for proper attribution:
- Scale: y_scaled = y / y_mean_per_dma
- Inverse: y_orig = y_scaled * y_mean_per_dma
- Attribution: Uses PROPORTIONAL ALLOCATION (same as scaling.py):
  * Calculate component ratios in scaled space
  * Apply ratios to original-space prediction
  * Guarantees components sum to 100%!
"""

import numpy as np
import torch
import torch.nn as nn
from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.core.unified_model import DeepCausalMMM

def load_real_mmm_data(filepath="examples/data/MMM Data.csv"):
    """Load and process the real MMM Data.csv"""
    import pandas as pd
    
    print(f" Loading Real MMM Data from: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"    Loaded data shape: {df.shape}")
    
    # Identify columns
    impression_cols = [col for col in df.columns if 'impressions_' in col]
    
    if 'target_visits' in df.columns:
        target_col = 'target_visits'
        value_cols = [col for col in df.columns if col.startswith('control_')]
    else:
        target_col = 'value_visits_visits'
        value_cols = [col for col in df.columns if 'value_' in col and col != 'value_visits_visits']
    
    region_col = 'dmacode'
    time_col = 'weekid'
    
    # Clean names
    media_names = [col.replace('impressions_', '').split('_delayed')[0].split('_exponential')[0].split('_geometric')[0].replace('_', ' ') for col in impression_cols]
    control_names = [col.replace('control_', 'Control ').replace('value_', '').replace('econmetricsmsa_', '').replace('mortgagemetrics_', '').replace('moodys_', '').replace('_sm', '').replace('_', ' ').title() for col in value_cols]
    
    print(f"    Media channels ({len(impression_cols)}): {media_names}")
    print(f"    Control variables ({len(value_cols)}): {control_names}")
    
    # Get regions and weeks
    regions = sorted(df[region_col].unique())
    weeks = sorted(df[time_col].unique())
    n_regions = len(regions)
    n_weeks = len(weeks)
    
    print(f"    Data structure: {n_regions} regions × {n_weeks} weeks")
    
    # Create complete grid and fill missing
    complete_index = pd.MultiIndex.from_product([regions, weeks], names=[region_col, time_col])
    complete_df = pd.DataFrame(index=complete_index).reset_index()
    df_complete = complete_df.merge(df, on=[region_col, time_col], how='left')
    
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
    X_media_list, X_control_list, y_list = [], [], []
    
    for region_idx in range(n_regions):
        region_data = df_complete[df_complete['region_idx'] == region_idx].sort_values('week_idx')
        X_media_list.append(region_data[impression_cols].values.astype(np.float32))
        X_control_list.append(region_data[value_cols].values.astype(np.float32))
        y_list.append(region_data[target_col].values.astype(np.float32))
    
    X_media = np.stack(X_media_list, axis=0)
    X_control = np.stack(X_control_list, axis=0)
    y = np.stack(y_list, axis=0)
    
    print(f"    Visits range: {y.min():,.0f} - {y.max():,.0f}")
    print(f"    Real MMM data successfully loaded!")
    
    return X_media, X_control, y, media_names, control_names

def custom_linear_scaling(X_media, X_control, y, holdout_ratio=0.08):
    """
    Custom linear scaling: y_scaled = y / y_mean_per_dma
    This preserves additivity for proper attribution!
    """
    print("\n CUSTOM LINEAR SCALING (y/y_mean per DMA)...")
    
    n_regions, n_weeks, n_media = X_media.shape
    n_control = X_control.shape[2]
    
    # 1. Temporal split
    holdout_weeks = int(n_weeks * holdout_ratio)
    train_weeks = n_weeks - holdout_weeks
    
    print(f"    Train: {train_weeks} weeks, Holdout: {holdout_weeks} weeks")
    
    X_media_train = X_media[:, :train_weeks, :]
    X_control_train = X_control[:, :train_weeks, :]
    y_train = y[:, :train_weeks]
    
    X_media_holdout = X_media[:, train_weeks:, :]
    X_control_holdout = X_control[:, train_weeks:, :]
    y_holdout = y[:, train_weeks:]
    
    # 2. Calculate y_mean per DMA (using ONLY training data to avoid leakage)
    y_mean_per_dma = y_train.mean(axis=1)  # [n_regions]
    print(f"    DMA means range: [{y_mean_per_dma.min():,.0f}, {y_mean_per_dma.max():,.0f}]")
    
    # 3. Scale target: y / y_mean per DMA
    y_train_scaled = y_train / y_mean_per_dma[:, None]  # [regions, weeks]
    y_holdout_scaled = y_holdout / y_mean_per_dma[:, None]
    
    print(f"    Scaled train range: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
    print(f"    Scaled holdout range: [{y_holdout_scaled.min():.3f}, {y_holdout_scaled.max():.3f}]")
    
    # 4. Media: Share-of-Voice (same as before)
    def scale_media_sov(X):
        total = X.sum(axis=2, keepdims=True)
        X_scaled = np.where(total > 0, X / total, 1.0 / n_media)
        return X_scaled
    
    X_media_train_scaled = scale_media_sov(X_media_train)
    X_media_holdout_scaled = scale_media_sov(X_media_holdout)
    
    # 5. Controls: Z-score (fit on train only)
    X_control_mean = X_control_train.mean(axis=(0, 1))
    X_control_std = X_control_train.std(axis=(0, 1)) + 1e-8
    
    X_control_train_scaled = (X_control_train - X_control_mean) / X_control_std
    X_control_holdout_scaled = (X_control_holdout - X_control_mean) / X_control_std
    
    # Clip controls
    X_control_train_scaled = np.clip(X_control_train_scaled, -5, 5)
    X_control_holdout_scaled = np.clip(X_control_holdout_scaled, -5, 5)
    
    print(f"    Linear scaling complete! (No log transformation)")
    
    return {
        'X_media_train': X_media_train_scaled,
        'X_control_train': X_control_train_scaled,
        'y_train': y_train_scaled,
        'X_media_holdout': X_media_holdout_scaled,
        'X_control_holdout': X_control_holdout_scaled,
        'y_holdout': y_holdout_scaled,
        'y_mean_per_dma': y_mean_per_dma,  # CRITICAL: Save for inverse transform
        'train_weeks': train_weeks,
        'holdout_weeks': holdout_weeks
    }

def add_padding(X_media, X_control, y, burn_in_weeks=6):
    """Add burn-in padding"""
    n_regions, n_weeks, n_media = X_media.shape
    n_control = X_control.shape[2]
    
    # Zero padding
    X_media_pad = np.concatenate([np.zeros((n_regions, burn_in_weeks, n_media)), X_media], axis=1)
    X_control_pad = np.concatenate([np.zeros((n_regions, burn_in_weeks, n_control)), X_control], axis=1)
    y_pad = np.concatenate([np.zeros((n_regions, burn_in_weeks)), y], axis=1)
    
    return torch.FloatTensor(X_media_pad), torch.FloatTensor(X_control_pad), torch.FloatTensor(y_pad)

def train_model_custom_scaling():
    """Train model with custom linear scaling"""
    
    print(" LINEAR SCALING MMM TEST")
    print("=" * 60)
    
    # 1. Load data
    config = get_default_config()
    X_media, X_control, y, media_names, control_names = load_real_mmm_data()
    
    n_regions, n_weeks, n_media = X_media.shape
    n_control = X_control.shape[2]
    
    # 2. Custom linear scaling
    scaled_data = custom_linear_scaling(X_media, X_control, y, holdout_ratio=config['holdout_ratio'])
    
    # 3. Add padding
    burn_in = config['burn_in_weeks']
    X_media_train, X_control_train, y_train = add_padding(
        scaled_data['X_media_train'], 
        scaled_data['X_control_train'], 
        scaled_data['y_train'],
        burn_in
    )
    
    X_media_holdout, X_control_holdout, y_holdout = add_padding(
        scaled_data['X_media_holdout'],
        scaled_data['X_control_holdout'],
        scaled_data['y_holdout'],
        burn_in
    )
    
    # 4. Create region tensors
    R_train = torch.arange(n_regions).unsqueeze(1).expand(-1, X_media_train.shape[1]).long()
    R_holdout = torch.arange(n_regions).unsqueeze(1).expand(-1, X_media_holdout.shape[1]).long()
    
    # 5. Create model
    print("\n Creating Model...")
    model = DeepCausalMMM(
        n_media=n_media,
        ctrl_dim=n_control,
        n_regions=n_regions,
        hidden=config.get('hidden_dim', 64),
        dropout=config.get('dropout', 0.1),
        l1_weight=config.get('l1_weight', 0.001),
        l2_weight=config.get('l2_weight', 0.001),
        burn_in_weeks=burn_in,
        use_coefficient_momentum=True,
        momentum_decay=config.get('momentum_decay', 0.9),
        use_warm_start=True,
        warm_start_epochs=config.get('warm_start_epochs', 50),
        stabilization_method=config.get('stabilization_method', 'exponential'),
        gru_layers=config.get('gru_layers', 1),
        ctrl_hidden_ratio=config.get('ctrl_hidden_ratio', 0.5)
    )
    
    # 6. Initialize baseline and seasonality (CRITICAL!)
    print("\n Initializing Baseline and Seasonality...")
    model.initialize_baseline(y_train)
    print(f"    Seasonality initialized: {model.seasonal_components is not None}")
    
    # 7. Train model
    print("\n Training Model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Warm-start
    print(f"    Warm-start: {config['warm_start_epochs']} epochs")
    model.warm_start_training(X_media_train, X_control_train, R_train, y_train, optimizer)
    
    # Main training - use FULL epochs from config
    n_epochs = config['n_epochs']
    print(f"    Main training: {n_epochs} epochs (FULL training)")
    model.train()
    
    # Use tqdm for progress tracking
    from tqdm import tqdm
    for epoch in tqdm(range(n_epochs), desc="Training"):
        optimizer.zero_grad()
        
        predictions, _, _, _ = model(X_media_train, X_control_train, R_train)
        loss = nn.MSELoss()(predictions, y_train)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 500 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                pred_holdout, _, _, _ = model(X_media_holdout, X_control_holdout, R_holdout)
                holdout_loss = nn.MSELoss()(pred_holdout, y_holdout).item()
                print(f"\n    Epoch {epoch}: Train Loss={loss.item():.4f}, Holdout Loss={holdout_loss:.4f}")
    
    # 7. Evaluate
    print("\n Evaluating...")
    model.eval()
    with torch.no_grad():
        pred_train, _, _, outputs_train = model(X_media_train, X_control_train, R_train)
        pred_holdout, _, _, outputs_holdout = model(X_media_holdout, X_control_holdout, R_holdout)
    
    # Remove padding
    pred_train = pred_train[:, burn_in:]
    pred_holdout = pred_holdout[:, burn_in:]
    y_train_eval = y_train[:, burn_in:]
    y_holdout_eval = y_holdout[:, burn_in:]
    
    # Calculate metrics in SCALED space
    from sklearn.metrics import r2_score, mean_squared_error
    train_rmse_scaled = np.sqrt(mean_squared_error(y_train_eval.numpy().flatten(), pred_train.numpy().flatten()))
    train_r2_scaled = r2_score(y_train_eval.numpy().flatten(), pred_train.numpy().flatten())
    
    holdout_rmse_scaled = np.sqrt(mean_squared_error(y_holdout_eval.numpy().flatten(), pred_holdout.numpy().flatten()))
    holdout_r2_scaled = r2_score(y_holdout_eval.numpy().flatten(), pred_holdout.numpy().flatten())
    
    print(f"\n RESULTS (Scaled Space):")
    print(f"    Train RMSE: {train_rmse_scaled:.4f}, R²: {train_r2_scaled:.3f}")
    print(f"    Holdout RMSE: {holdout_rmse_scaled:.4f}, R²: {holdout_r2_scaled:.3f}")
    
    # Inverse transform to original space
    y_mean_per_dma = torch.FloatTensor(scaled_data['y_mean_per_dma'])
    
    pred_train_orig = pred_train * y_mean_per_dma[:, None]
    pred_holdout_orig = pred_holdout * y_mean_per_dma[:, None]
    
    y_train_orig = y_train_eval * y_mean_per_dma[:, None]
    y_holdout_orig = y_holdout_eval * y_mean_per_dma[:, None]
    
    train_rmse_orig = np.sqrt(mean_squared_error(y_train_orig.numpy().flatten(), pred_train_orig.numpy().flatten()))
    train_r2_orig = r2_score(y_train_orig.numpy().flatten(), pred_train_orig.numpy().flatten())
    
    holdout_rmse_orig = np.sqrt(mean_squared_error(y_holdout_orig.numpy().flatten(), pred_holdout_orig.numpy().flatten()))
    holdout_r2_orig = r2_score(y_holdout_orig.numpy().flatten(), pred_holdout_orig.numpy().flatten())
    
    print(f"\n RESULTS (Original Space):")
    print(f"    Train RMSE: {train_rmse_orig:,.0f} visits, R²: {train_r2_orig:.3f}")
    print(f"    Holdout RMSE: {holdout_rmse_orig:,.0f} visits, R²: {holdout_r2_orig:.3f}")
    
    # 8. ATTRIBUTION TEST - The whole point!
    print(f"\n" + "=" * 80)
    print(f"ATTRIBUTION TEST WITH LINEAR SCALING")
    print(f"=" * 80)
    
    # Helper to scale media
    def scale_media_sov(X):
        total = X.sum(axis=2, keepdims=True)
        n_ch = X.shape[2]
        X_scaled = np.where(total > 0, X / total, 1.0 / n_ch)
        return X_scaled
    
    # Scale full dataset
    X_control_mean = scaled_data['X_control_train'].mean(axis=(0, 1))
    X_control_std = scaled_data['X_control_train'].std(axis=(0, 1)) + 1e-8
    
    # Get components from full dataset
    X_media_full, X_control_full, y_full = add_padding(
        scale_media_sov(X_media),
        np.clip((X_control - X_control_mean) / X_control_std, -5, 5),
        y / y_mean_per_dma.numpy()[:, None],
        burn_in
    )
    R_full = torch.arange(n_regions).unsqueeze(1).expand(-1, X_media_full.shape[1]).long()
    
    with torch.no_grad():
        pred_full, _, media_contrib, outputs_full = model(X_media_full, X_control_full, R_full)
    
    # Remove padding
    pred_full = pred_full[:, burn_in:]
    media_contrib = media_contrib[:, burn_in:, :]
    baseline = outputs_full['baseline'][:, burn_in:]
    control_contrib = outputs_full['control_contributions'][:, burn_in:, :]
    seasonal = outputs_full['seasonal_contribution'][:, burn_in:]
    trend = outputs_full['trend_contribution'][:, burn_in:]
    prediction_scale = outputs_full['prediction_scale']  # ← CRITICAL: Get the scale factor!
    
    # PROPORTIONAL ALLOCATION (same as scaling.py logic)
    # This ensures components sum to 100% by construction!
    
    # CRITICAL: The model's baseline ALREADY includes seasonal!
    # raw_prediction = media + controls + baseline_with_seasonal + trend
    # So we should NOT treat seasonal as a separate component!
    
    # 1. Get the raw prediction (before prediction_scale)
    raw_pred_scaled = outputs_full['raw_prediction'][:, burn_in:]  # [regions, weeks]
    
    # 2. Inverse transform prediction to original space
    pred_full_orig = pred_full * y_mean_per_dma[:, None]
    
    # 3. Calculate component ratios in SCALED space
    # Note: baseline already includes seasonal, so we use it as-is
    baseline_ratio = baseline / (raw_pred_scaled + 1e-8)
    trend_ratio = trend / (raw_pred_scaled + 1e-8)
    
    # For multi-dimensional components (media, controls), calculate per-channel ratios
    media_ratios = media_contrib / (raw_pred_scaled.unsqueeze(-1) + 1e-8)
    control_ratios = control_contrib / (raw_pred_scaled.unsqueeze(-1) + 1e-8)
    
    # 4. Apply ratios to ORIGINAL-SPACE prediction
    baseline_orig = pred_full_orig * baseline_ratio  # Includes seasonal
    trend_orig = pred_full_orig * trend_ratio
    media_contrib_orig = pred_full_orig.unsqueeze(-1) * media_ratios
    control_contrib_orig = pred_full_orig.unsqueeze(-1) * control_ratios
    
    # 5. Extract seasonal separately for reporting (but don't add to total!)
    seasonal_ratio = seasonal / (raw_pred_scaled + 1e-8)
    seasonal_orig = pred_full_orig * seasonal_ratio
    
    # Sum check
    total_pred = pred_full_orig.sum().item()
    total_baseline = baseline_orig.sum().item()
    total_media = media_contrib_orig.sum().item()
    total_control = control_contrib_orig.sum().item()
    total_seasonal = seasonal_orig.sum().item()
    total_trend = trend_orig.sum().item()
    
    # Components sum: baseline (includes seasonal) + media + controls + trend
    components_sum = total_baseline + total_media + total_control + total_trend
    
    print(f"\nComponents (Original Space - Visits):")
    print(f"    Baseline (incl. seasonal): {total_baseline:15,.0f} ({total_baseline/total_pred*100:5.1f}%)")
    print(f"      - of which seasonal:     {total_seasonal:15,.0f} ({total_seasonal/total_pred*100:5.1f}%)")
    print(f"    Media:                     {total_media:15,.0f} ({total_media/total_pred*100:5.1f}%)")
    print(f"    Controls:                  {total_control:15,.0f} ({total_control/total_pred*100:5.1f}%)")
    print(f"    Trend:                     {total_trend:15,.0f} ({total_trend/total_pred*100:5.1f}%)")
    print(f"    " + "-" * 76)
    print(f"    TOTAL:       {total_pred:15,.0f} (100.0%)")
    print(f"\n    Components Sum: {components_sum:,.0f}")
    print(f"    Prediction:     {total_pred:,.0f}")
    print(f"    Difference:     {abs(components_sum - total_pred):,.0f}")
    print(f"    Match:          {abs(components_sum - total_pred) < 1000} (Should be TRUE!)")
    
    print(f"\n" + "=" * 80)
    print(f"LINEAR SCALING TEST COMPLETE!")
    print(f"=" * 80)
    
    if abs(components_sum - total_pred) < 1000:
        print(f" SUCCESS: Components sum correctly with linear scaling!")
        print(f" Media attribution: {total_media/total_pred*100:.1f}%")
    else:
        print(f" WARNING: Components don't sum correctly")
        print(f" Difference: {abs(components_sum - total_pred):,.0f}")

if __name__ == "__main__":
    train_model_custom_scaling()


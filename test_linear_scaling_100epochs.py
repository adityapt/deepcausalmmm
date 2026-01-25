#!/usr/bin/env python3
"""
Test script for linear scaling (y/y_mean) implementation
Run with 100 epochs to quickly verify the changes work correctly
"""

import sys
import os

# Ensure we're using the local package
sys.path.insert(0, '/Users/adityapu/Documents/GitHub/deepcausalmmm')

print("="*80)
print("TESTING LINEAR SCALING IMPLEMENTATION (y/mean)")
print("="*80)
print(f"Running quick 100-epoch test to verify architecture changes")
print("")

# Import after path is set
from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline
import numpy as np
import pandas as pd

# Load data
print("Loading data...")
df = pd.read_csv('/Users/adityapu/Documents/GitHub/deepcausalmmm/examples/data/MMM Data.csv')

# Extract data
impression_cols = [col for col in df.columns if 'impressions_' in col]
if 'target_visits' in df.columns:
    target_col = 'target_visits'
    value_cols = [col for col in df.columns if col.startswith('control_')]
else:
    target_col = 'value_visits_visits'
    value_cols = [col for col in df.columns if 'value_' in col and col != 'value_visits_visits']

region_col = 'dmacode'
time_col = 'weekid'

regions = sorted(df[region_col].unique())
weeks = sorted(df[time_col].unique())
n_regions = len(regions)
n_weeks = len(weeks)

print(f"Data: {n_regions} regions × {n_weeks} weeks")
print(f"Media channels: {len(impression_cols)}")
print(f"Control variables: {len(value_cols)}")

# Create complete grid and fill missing values
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

print(f"Visits range: {y.min():,.0f} - {y.max():,.0f}")
print(f"Visits mean: {y.mean():,.0f}")

# Create config with 100 epochs for testing
print("\nCreating configuration (100 epochs for quick test)...")
config = get_default_config()
config['n_epochs'] = 100  # Quick test
print(f"Epochs: {config['n_epochs']}")

# Create pipeline and split data
print("\nProcessing data with UnifiedDataPipeline...")
pipeline = UnifiedDataPipeline(config)
train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)

print(f"Train weeks: {pipeline.train_weeks}")
print(f"Holdout weeks: {train_data['y'].shape[1] if holdout_data else 0}")

# Process training and holdout data
train_tensors = pipeline.fit_and_transform_training(train_data)
holdout_tensors = pipeline.transform_holdout(holdout_data) if holdout_data else None

print(f"Processed shapes:")
print(f"  X_media: {train_tensors['X_media'].shape}")
print(f"  X_control: {train_tensors['X_control'].shape}")
print(f"  y: {train_tensors['y'].shape}")

# Check scaling constants
scaler = pipeline.get_scaler()
if 'y_mean_per_region' in scaler.scaling_constants:
    y_means = scaler.scaling_constants['y_mean_per_region']
    print(f"\nLinear scaling active:")
    print(f"  y_mean_per_region shape: {y_means.shape}")
    print(f"  y_mean_per_region range: [{y_means.min():.0f}, {y_means.max():.0f}]")
    print(f"  y_mean_per_region mean: {y_means.mean():.0f}")
else:
    print("\nWARNING: y_mean_per_region not found! Still using log scaling?")

# Create trainer and model
print("\nCreating model...")
trainer = ModelTrainer(config)
model = trainer.create_model(
    n_media=train_tensors['X_media'].shape[2],
    n_control=train_tensors['X_control'].shape[2],
    n_regions=train_tensors['X_media'].shape[0]
)
trainer.create_optimizer_and_scheduler()

print(f"Model created with {model.hidden_size} hidden units")

# Train
print(f"\nTraining for {config['n_epochs']} epochs...")
print("="*80)

if holdout_tensors:
    results = trainer.train(
        train_tensors['X_media'], train_tensors['X_control'],
        train_tensors['R'], train_tensors['y'],
        holdout_tensors['X_media'], holdout_tensors['X_control'],
        holdout_tensors['R'], holdout_tensors['y'],
        pipeline=pipeline,
        verbose=True
    )
else:
    results = trainer.train(
        train_tensors['X_media'], train_tensors['X_control'],
        train_tensors['R'], train_tensors['y'],
        pipeline=pipeline,
        verbose=True
    )

print("="*80)
print("\nTEST RESULTS:")
print("="*80)
print(f"Training R²: {results['final_train_r2']:.4f}")
print(f"Training RMSE (orig): {results['final_train_rmse']:,.0f}")

if 'final_holdout_r2' in results:
    print(f"Holdout R²: {results['final_holdout_r2']:.4f}")
    print(f"Holdout RMSE (orig): {results['final_holdout_rmse']:,.0f}")
    print(f"R² Gap: {results['final_train_r2'] - results['final_holdout_r2']:.4f}")

# Test attribution
print("\n" + "="*80)
print("TESTING ATTRIBUTION (Simple Additivity Check)")
print("="*80)

import torch
model.eval()
with torch.no_grad():
    # Get predictions and components
    y_pred_scaled, _, _, outputs = model(
        train_tensors['X_media'], 
        train_tensors['X_control'], 
        train_tensors['R']
    )
    
    # Get components in scaled space
    baseline_scaled = outputs.get('baseline', torch.zeros_like(y_pred_scaled))
    media_scaled = outputs.get('contributions', torch.zeros((*y_pred_scaled.shape, len(impression_cols))))
    ctrl_scaled = outputs.get('control_contributions', torch.zeros((*y_pred_scaled.shape, len(value_cols))))
    seasonal_scaled = outputs.get('seasonal_contribution', torch.zeros_like(y_pred_scaled))
    trend_scaled = outputs.get('trend_contribution', torch.zeros_like(y_pred_scaled))
    pred_scale = outputs.get('prediction_scale', torch.tensor(1.0))
    
    print(f"\nComponent shapes (scaled space):")
    print(f"  Predictions: {y_pred_scaled.shape}")
    print(f"  Baseline: {baseline_scaled.shape}")
    print(f"  Media: {media_scaled.shape}")
    print(f"  Controls: {ctrl_scaled.shape}")
    print(f"  Seasonal: {seasonal_scaled.shape}")
    print(f"  Trend: {trend_scaled.shape}")
    print(f"  Prediction scale: {pred_scale.item():.4f}")
    
    # Inverse transform using the new method
    contrib_results = scaler.inverse_transform_contributions(
        media_contributions=media_scaled,
        baseline=baseline_scaled,
        control_contributions=ctrl_scaled,
        seasonal_contributions=seasonal_scaled,
        trend_contributions=trend_scaled,
        prediction_scale=pred_scale
    )
    
    # Inverse transform predictions
    y_pred_orig = scaler.inverse_transform_target(y_pred_scaled)
    
    # Calculate component sums
    baseline_orig = contrib_results.get('baseline', torch.zeros_like(y_pred_orig))
    media_orig = contrib_results.get('media', torch.zeros((*y_pred_orig.shape, len(impression_cols))))
    ctrl_orig = contrib_results.get('control', torch.zeros((*y_pred_orig.shape, len(value_cols))))
    seasonal_orig = contrib_results.get('seasonal', torch.zeros_like(y_pred_orig))
    trend_orig = contrib_results.get('trend', torch.zeros_like(y_pred_orig))
    
    components_sum = (baseline_orig + media_orig.sum(dim=2) + 
                      ctrl_orig.sum(dim=2) + seasonal_orig + trend_orig)
    
    # Check if they match
    diff = torch.abs(y_pred_orig - components_sum).mean().item()
    max_diff = torch.abs(y_pred_orig - components_sum).max().item()
    
    print(f"\nAdditivity Check (Original Scale):")
    print(f"  Mean prediction: {y_pred_orig.mean().item():,.0f}")
    print(f"  Mean components sum: {components_sum.mean().item():,.0f}")
    print(f"  Mean absolute difference: {diff:.2f}")
    print(f"  Max absolute difference: {max_diff:.2f}")
    print(f"  Relative error: {(diff / y_pred_orig.mean().item()) * 100:.4f}%")
    
    if diff < 1.0:  # Less than 1 visit difference
        print("  ✓ PASS: Components sum to predictions (additive!)")
    else:
        print("  ✗ FAIL: Components don't sum correctly")
    
    # Calculate attribution percentages
    total_pred = y_pred_orig.sum().item()
    media_pct = (media_orig.sum().item() / total_pred) * 100
    baseline_pct = (baseline_orig.sum().item() / total_pred) * 100
    ctrl_pct = (ctrl_orig.sum().item() / total_pred) * 100
    seasonal_pct = (seasonal_orig.sum().item() / total_pred) * 100
    trend_pct = (trend_orig.sum().item() / total_pred) * 100
    
    print(f"\nAttribution (% of total):")
    print(f"  Media: {media_pct:.1f}%")
    print(f"  Baseline: {baseline_pct:.1f}%")
    print(f"  Controls: {ctrl_pct:.1f}%")
    print(f"  Seasonality: {seasonal_pct:.1f}%")
    print(f"  Trend: {trend_pct:.1f}%")
    print(f"  Total: {media_pct + baseline_pct + ctrl_pct + seasonal_pct + trend_pct:.1f}%")
    
    # Check if media attribution is reasonable (should be ~40%)
    if 30 <= media_pct <= 50:
        print(f"  ✓ Media attribution looks reasonable ({media_pct:.1f}% is close to expected ~40%)")
    else:
        print(f"  ? Media attribution is {media_pct:.1f}% (expected ~40%)")

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. If test passes, run full 2500 epochs")
print("2. Verify attribution stays ~40% with full training")
print("3. Compare with log-space results")
print("="*80)


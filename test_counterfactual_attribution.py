#!/usr/bin/env python3
"""
Counterfactual Test: Understanding True Attribution
===================================================
This tests what the model predicts with different components zeroed out.
"""

import numpy as np
import pandas as pd
import torch
from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline

print("=" * 80)
print("COUNTERFACTUAL ATTRIBUTION TEST")
print("=" * 80)

# Load data
print("\n1. Loading Data...")
df = pd.read_csv('examples/data/MMM Data.csv')
impression_cols = [col for col in df.columns if 'impressions_' in col]
target_col = 'target_visits'
value_cols = [col for col in df.columns if col.startswith('control_')]
region_col = 'dmacode'
time_col = 'weekid'

regions = sorted(df[region_col].unique())
weeks = sorted(df[time_col].unique())
n_regions, n_weeks = len(regions), len(weeks)

# Prepare data
complete_index = pd.MultiIndex.from_product([regions, weeks], names=[region_col, time_col])
complete_df = pd.DataFrame(index=complete_index).reset_index()
df_complete = complete_df.merge(df, on=[region_col, time_col], how='left')

for col in impression_cols:
    df_complete[col] = df_complete[col].fillna(0)
for col in value_cols + [target_col]:
    df_complete[col] = df_complete.groupby(region_col)[col].fillna(method='ffill').fillna(method='bfill')
    if df_complete[col].isna().any():
        df_complete[col] = df_complete[col].fillna(df_complete[col].mean())

region_map = {region: i for i, region in enumerate(regions)}
df_complete['region_idx'] = df_complete[region_col].map(region_map)
df_complete = df_complete.sort_values(['region_idx', time_col])

X_media = np.zeros((n_regions, n_weeks, len(impression_cols)), dtype=np.float32)
X_control = np.zeros((n_regions, n_weeks, len(value_cols)), dtype=np.float32)
y = np.zeros((n_regions, n_weeks), dtype=np.float32)

for r in range(n_regions):
    region_data = df_complete[df_complete['region_idx'] == r].sort_values(time_col)
    X_media[r] = region_data[impression_cols].values[:n_weeks]
    X_control[r] = region_data[value_cols].values[:n_weeks]
    y[r] = region_data[target_col].values[:n_weeks]

total_actual_visits = y.sum()
print(f"   Total Actual Visits: {total_actual_visits:,.0f}")

# Load trained model
print("\n2. Loading Trained Model...")
config = get_default_config()
pipeline = UnifiedDataPipeline(config)

# Split and process
train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
train_tensors = pipeline.fit_and_transform_training(train_data)

# Create and train model (quick training just to get weights)
trainer = ModelTrainer(config)
n_media_channels = train_tensors['X_media'].shape[2]
n_control_vars = train_tensors['X_control'].shape[2]
n_regions_model = train_tensors['X_media'].shape[0]

model = trainer.create_model(n_media_channels, n_control_vars, n_regions_model)
trainer.create_optimizer_and_scheduler()

print("   Training model (this will take a moment)...")
if holdout_data is not None:
    holdout_tensors = pipeline.transform_holdout(holdout_data)
    training_results = trainer.train(
        train_tensors['X_media'], train_tensors['X_control'],
        train_tensors['R'], train_tensors['y'],
        holdout_tensors['X_media'], holdout_tensors['X_control'],
        holdout_tensors['R'], holdout_tensors['y'],
        verbose=False
    )
else:
    training_results = trainer.train(
        train_tensors['X_media'], train_tensors['X_control'],
        train_tensors['R'], train_tensors['y'],
        verbose=False
    )

print(f"   Model trained: R² = {training_results['final_train_r2']:.3f}")

# Now run counterfactual tests
print("\n" + "=" * 80)
print("COUNTERFACTUAL TESTS")
print("=" * 80)

# Process full dataset for predictions
full_data = {'X_media': X_media, 'X_control': X_control, 'y': y}
full_tensors = pipeline.fit_and_transform_training(full_data)

def predict_and_inverse(X_media_input, X_control_input):
    """Helper function to predict and inverse transform"""
    # Transform
    X_media_scaled, X_control_scaled, _ = pipeline.scaler.transform(
        X_media_input, X_control_input, np.zeros_like(X_media_input[:, :, 0])
    )
    X_media_padded, X_control_padded, _ = pipeline._add_padding(
        X_media_scaled, X_control_scaled, torch.zeros(X_media_scaled.shape[0], X_media_scaled.shape[1])
    )
    
    # Region tensor
    R = torch.arange(n_regions, dtype=torch.long)
    
    # Predict
    model.eval()
    with torch.no_grad():
        y_pred_log, _, _, _ = model(X_media_padded, X_control_padded, R)
    
    # Remove padding and inverse transform
    burn_in = config['burn_in_weeks']
    y_pred_log_eval = y_pred_log[:, burn_in:burn_in+n_weeks]
    y_pred_orig = torch.expm1(torch.clamp(y_pred_log_eval, max=20.0))
    
    return y_pred_orig.sum().item()

print("\nTest 1: Full Model (All Components)")
print("-" * 80)
total_full = predict_and_inverse(X_media, X_control)
print(f"   Prediction with ALL components: {total_full:,.0f} visits")
print(f"   Actual visits:                  {total_actual_visits:,.0f} visits")
print(f"   Prediction accuracy:            {(total_full / total_actual_visits * 100):.1f}%")

print("\nTest 2: ZERO Media (Baseline + Controls Only)")
print("-" * 80)
X_media_zero = np.zeros_like(X_media)
total_no_media = predict_and_inverse(X_media_zero, X_control)
media_incremental = total_full - total_no_media
print(f"   Prediction WITHOUT media:       {total_no_media:,.0f} visits")
print(f"   Media incremental contribution: {media_incremental:,.0f} visits")
print(f"   Media as % of prediction:       {(media_incremental / total_full * 100):.1f}%")
print(f"   Media as % of actual:           {(media_incremental / total_actual_visits * 100):.1f}%")

print("\nTest 3: ZERO Controls (Baseline + Media Only)")
print("-" * 80)
X_control_zero = np.zeros_like(X_control)
total_no_controls = predict_and_inverse(X_media, X_control_zero)
control_incremental = total_full - total_no_controls
print(f"   Prediction WITHOUT controls:    {total_no_controls:,.0f} visits")
print(f"   Control incremental:            {control_incremental:,.0f} visits")
print(f"   Controls as % of prediction:    {(control_incremental / total_full * 100):.1f}%")

print("\nTest 4: Baseline ONLY (No Media, No Controls)")
print("-" * 80)
total_baseline_only = predict_and_inverse(X_media_zero, X_control_zero)
print(f"   Prediction with ONLY baseline:  {total_baseline_only:,.0f} visits")
print(f"   Baseline as % of prediction:    {(total_baseline_only / total_full * 100):.1f}%")
print(f"   Baseline as % of actual:        {(total_baseline_only / total_actual_visits * 100):.1f}%")

print("\n" + "=" * 80)
print("ATTRIBUTION SUMMARY (Counterfactual Method)")
print("=" * 80)
print(f"   Baseline (organic):             {total_baseline_only:,.0f} visits ({(total_baseline_only / total_full * 100):.1f}%)")
print(f"   Controls incremental:           {control_incremental:,.0f} visits ({(control_incremental / total_full * 100):.1f}%)")
print(f"   Media incremental:              {media_incremental:,.0f} visits ({(media_incremental / total_full * 100):.1f}%)")
print(f"   TOTAL:                          {total_full:,.0f} visits (100.0%)")

# Check if components sum correctly
components_sum = total_baseline_only + control_incremental + media_incremental
print(f"\n   Components sum:                 {components_sum:,.0f} visits")
print(f"   Matches prediction:             {abs(components_sum - total_full) < 1000}")

print("\n" + "=" * 80)
print("COMPARISON WITH PROPORTIONAL ALLOCATION")
print("=" * 80)
print(f"   Dashboard method (proportional): Media = 93.6%")
print(f"   Counterfactual method:           Media = {(media_incremental / total_full * 100):.1f}%")
print(f"   Difference:                      {93.6 - (media_incremental / total_full * 100):.1f} percentage points")

if abs(93.6 - (media_incremental / total_full * 100)) > 20:
    print("\n   ⚠️  LARGE DISCREPANCY DETECTED!")
    print("   The proportional allocation method is giving very different")
    print("   results from the counterfactual method.")
else:
    print("\n   ✅ Methods are reasonably aligned.")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if media_incremental / total_full > 0.8:
    print("   🚨 CONCERN: Media drives >80% of visits in counterfactual test")
    print("   This suggests:")
    print("   - Model genuinely believes media is responsible for most visits")
    print("   - Baseline/organic traffic is very small")
    print("   - This may indicate model overfitting to media variables")
elif media_incremental / total_full > 0.5:
    print("   ⚠️  WARNING: Media drives >50% of visits in counterfactual test")
    print("   This is higher than typical for MMM but may be valid if:")
    print("   - Business is highly media-dependent")
    print("   - Baseline represents minimal brand equity")
elif media_incremental / total_full > 0.2:
    print("   ✅ REASONABLE: Media drives 20-50% of visits")
    print("   This is within typical MMM ranges")
else:
    print("   ℹ️  LOW: Media drives <20% of visits")
    print("   Controls and baseline dominate")

print("\n" + "=" * 80)


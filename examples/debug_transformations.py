#!/usr/bin/env python3
"""
Debug script to print all intermediate values and weights in the model
"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '/Users/adityapu/Documents/GitHub/deepcausalmmm')

from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline

def load_real_mmm_data(filepath="data/MMM Data.csv"):
    """Load the real MMM data"""
    df = pd.read_csv(filepath)
    
    impression_cols = [col for col in df.columns if 'impressions_' in col]
    value_cols = [col for col in df.columns if 'value_' in col and col != 'value_visits_visits']
    target_col = 'value_visits_visits'
    region_col = 'dmacode'
    time_col = 'weekid'
    
    media_names = [col.replace('impressions_', '').split('_delayed')[0].split('_exponential')[0].split('_geometric')[0].replace('_', ' ') for col in impression_cols]
    control_names = [col.replace('value_', '').replace('econmetricsmsa_', '').replace('mortgagemetrics_', '').replace('moodys_', '').replace('_sm', '').replace('_', ' ').title() for col in value_cols]
    
    regions = sorted(df[region_col].unique())
    weeks = sorted(df[time_col].unique())
    n_regions = len(regions)
    n_weeks = len(weeks)
    
    complete_index = pd.MultiIndex.from_product([regions, weeks], names=[region_col, time_col])
    df_complete = df.set_index([region_col, time_col]).reindex(complete_index).reset_index()
    
    for col in impression_cols + value_cols + [target_col]:
        df_complete[col] = df_complete.groupby(region_col)[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    X_media = np.zeros((n_regions, n_weeks, len(impression_cols)))
    X_control = np.zeros((n_regions, n_weeks, len(value_cols)))
    y = np.zeros((n_regions, n_weeks))
    
    for i, region in enumerate(regions):
        region_data = df_complete[df_complete[region_col] == region].sort_values(time_col)
        X_media[i] = region_data[impression_cols].values
        X_control[i] = region_data[value_cols].values
        y[i] = region_data[target_col].values
    
    return X_media, X_control, y, media_names, control_names

print("=" * 80)
print("DEBUG: Model Intermediate Values and Weights")
print("=" * 80)

# Load data
print("\n1. Loading data...")
X_media, X_control, y, media_names, control_names = load_real_mmm_data()
print(f"   Data shape: {X_media.shape[0]} regions × {X_media.shape[1]} weeks × {X_media.shape[2]} channels")

# Find SEM - Google Search channel
push_idx = None
for i, name in enumerate(media_names):
    if 'Google Search' in name or 'google search' in name.lower():
        push_idx = i
        print(f"   SEM - Google Search channel found at index {push_idx}: '{name}'")
        break

if push_idx is None:
    print("   ERROR: SEM - Google Search channel not found!")
    print(f"   Available channels: {media_names}")
    sys.exit(1)

# Load config and create pipeline
print("\n2. Setting up model...")
config = get_default_config()
config['n_epochs'] = 100  # Quick training for debug
config['warm_start_epochs'] = 10

pipeline = UnifiedDataPipeline(config)
train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
train_tensors = pipeline.fit_and_transform_training(train_data)

# Create and train model
trainer = ModelTrainer(config)
n_media = train_tensors['X_media'].shape[2]
n_control = train_tensors['X_control'].shape[2]
n_regions = train_tensors['X_media'].shape[0]

model = trainer.create_model(n_media, n_control, n_regions)
trainer.create_optimizer_and_scheduler()

print("\n3. Training model (100 epochs)...")
full_data = {'X_media': X_media, 'X_control': X_control, 'y': y}
full_tensors = pipeline.fit_and_transform_training(full_data)
y_full_scaled = full_tensors['y']

training_results = trainer.train(
    train_tensors['X_media'], train_tensors['X_control'],
    train_tensors['R'], train_tensors['y'],
    y_full_for_baseline=y_full_scaled,
    verbose=False
)

print(f"   Training complete! Final R²: {training_results.get('best_r2', 0):.3f}")

# Now inspect the Push channel transformations
print("\n" + "=" * 80)
print("SEM - GOOGLE SEARCH CHANNEL TRANSFORMATION ANALYSIS")
print("=" * 80)

# Get Push data
push_impressions_orig = X_media[:, :, push_idx]  # Original impressions
print(f"\n📊 Original Impressions:")
print(f"   Range: [{push_impressions_orig.min():.2f}, {push_impressions_orig.max():.2f}]")
print(f"   Mean: {push_impressions_orig.mean():.2f}, Std: {push_impressions_orig.std():.2f}")

# Get SOV-scaled version
push_scaled = train_tensors['X_media'][:, :, push_idx].cpu().numpy()
print(f"\n📏 After SOV Scaling:")
print(f"   Range: [{push_scaled.min():.8f}, {push_scaled.max():.8f}]")
print(f"   Mean: {push_scaled.mean():.8f}, Std: {push_scaled.std():.8f}")

# Pass through model with full forward pass to get intermediate outputs
model.eval()
with torch.no_grad():
    # Get all media channels scaled
    Xm = train_tensors['X_media']  # [regions, weeks, channels]
    
    print(f"\n🔄 Model Input (SOV-scaled) - Push Channel:")
    print(f"   Range: [{Xm[:, :, push_idx].min():.8f}, {Xm[:, :, push_idx].max():.8f}]")
    print(f"   Mean: {Xm[:, :, push_idx].mean():.8f}, Std: {Xm[:, :, push_idx].std():.8f}")
    
    # Get adstock and Hill parameters
    alpha_raw = model.alpha[push_idx].item()
    alpha = torch.sigmoid(torch.tensor(alpha_raw)).item()
    print(f"\n🔄 Adstock Parameter:")
    print(f"   Raw alpha: {alpha_raw:.6f}")
    print(f"   Sigmoid(alpha): {alpha:.6f} (clamped to [0, 0.8])")
    alpha_clamped = min(alpha, 0.8)
    print(f"   Effective alpha: {alpha_clamped:.6f}")
    
    # Manually compute adstock for Push to see intermediate values
    push_data = Xm[:, :, push_idx]  # [regions, weeks]
    B, T = push_data.shape
    push_adstock_manual = torch.zeros_like(push_data)
    push_adstock_manual[:, 0] = push_data[:, 0]
    for t in range(1, T):
        push_adstock_manual[:, t] = push_data[:, t] + alpha_clamped * push_adstock_manual[:, t-1]
        push_adstock_manual[:, t] = torch.clamp(push_adstock_manual[:, t], 0, 10)
    
    print(f"\n🔄 After Adstock:")
    print(f"   Range: [{push_adstock_manual.min():.8f}, {push_adstock_manual.max():.8f}]")
    print(f"   Mean: {push_adstock_manual.mean():.8f}, Std: {push_adstock_manual.std():.8f}")
    print(f"   ⚠️  Cap at 10: Max value is {push_adstock_manual.max():.8f} (hitting cap: {push_adstock_manual.max() >= 9.9})")
    print(f"   📊 % of values > 5: {(push_adstock_manual > 5).float().mean() * 100:.2f}%")
    print(f"   📊 % of values > 1: {(push_adstock_manual > 1).float().mean() * 100:.2f}%")
    
    # Hill transformation
    hill_a_raw = model.hill_a[push_idx].item()
    hill_g_raw = model.hill_g[push_idx].item()
    hill_a = torch.nn.functional.softplus(torch.tensor(hill_a_raw)).item()
    hill_g = torch.nn.functional.softplus(torch.tensor(hill_g_raw)).item()
    hill_a_clamped = max(0.1, min(2.0, hill_a))
    hill_g_clamped = max(0.01, min(1.0, hill_g))
    
    print(f"\n📈 Hill Parameters:")
    print(f"   Raw a: {hill_a_raw:.6f}, Softplus(a): {hill_a:.6f}, Clamped: {hill_a_clamped:.6f}")
    print(f"   Raw g: {hill_g_raw:.6f}, Softplus(g): {hill_g:.6f}, Clamped: {hill_g_clamped:.6f}")
    
    # Manually compute Hill
    x_safe = torch.nn.functional.relu(push_adstock_manual) + 1e-8
    num = torch.pow(x_safe, hill_a_clamped)
    denom = num + torch.pow(torch.tensor(hill_g_clamped), hill_a_clamped)
    push_hill_manual = num / (denom + 1e-8)
    push_hill_manual = torch.clamp(push_hill_manual, 0, 1)
    
    print(f"\n📈 After Hill Transform:")
    print(f"   Range: [{push_hill_manual.min():.8f}, {push_hill_manual.max():.8f}]")
    print(f"   Mean: {push_hill_manual.mean():.8f}, Std: {push_hill_manual.std():.8f}")
    print(f"   ⚠️  Cap at 1: Max value is {push_hill_manual.max():.8f} (hitting cap: {push_hill_manual.max() >= 0.99})")
    print(f"   📊 % of values > 0.9: {(push_hill_manual > 0.9).float().mean() * 100:.2f}%")
    print(f"   📊 % of values > 0.5: {(push_hill_manual > 0.5).float().mean() * 100:.2f}%")
    print(f"   📊 % of values > 0.1: {(push_hill_manual > 0.1).float().mean() * 100:.2f}%")
    
    # Full forward pass to get coefficients
    predictions, baseline, seasonal, outputs = model(
        train_tensors['X_media'],
        train_tensors['X_control'],
        train_tensors['R']
    )
    
    # Check what keys are available and get coefficients
    print(f"\n⚖️  Available output keys: {list(outputs.keys())}")
    
    if 'coefficients' in outputs:
        media_coeffs = outputs['coefficients']  # This is the media coefficients
        push_coeffs = media_coeffs[:, :, push_idx]
        print(f"\n⚖️  Learned Coefficients (time-varying from GRU):")
        print(f"   Shape: {push_coeffs.shape}")
        print(f"   Range: [{push_coeffs.min():.8f}, {push_coeffs.max():.8f}]")
        print(f"   Mean: {push_coeffs.mean():.8f}, Std: {push_coeffs.std():.8f}")
        
        # Check variation across time
        coeff_variation_per_region = push_coeffs.std(dim=1).mean()
        print(f"   📊 Average std across weeks per region: {coeff_variation_per_region:.8f}")
        
        # Show first region's coefficients over time
        print(f"\n   📊 First region coefficients over first 10 weeks:")
        for t in range(min(10, push_coeffs.shape[1])):
            print(f"      Week {t+1}: {push_coeffs[0, t].item():.8f}")
    else:
        print(f"\n⚖️  Coefficients key not found!")
    
    # Media contributions (log-space)
    if 'contributions' in outputs:
        media_contribs = outputs['contributions']  # This is media contributions
        push_contrib_log = media_contribs[:, :, push_idx]
    else:
        print(f"\n❌ Contributions not found in outputs!")
        sys.exit(1)
    
    print(f"\n📊 Media Contributions (log-space):")
    print(f"   Range: [{push_contrib_log.min():.8f}, {push_contrib_log.max():.8f}]")
    print(f"   Mean: {push_contrib_log.mean():.8f}, Std: {push_contrib_log.std():.8f}")
    
    # After expm1
    push_contrib_orig = torch.expm1(torch.clamp(push_contrib_log, max=20.0))
    
    print(f"\n📊 After expm1 (original scale):")
    print(f"   Range: [{push_contrib_orig.min():.2f}, {push_contrib_orig.max():.2f}]")
    print(f"   Mean: {push_contrib_orig.mean():.2f}, Std: {push_contrib_orig.std():.2f}")
    print(f"   Total contribution: {push_contrib_orig.sum():.2f} visits")

print("\n" + "=" * 80)
print("IMPRESSIONS vs CONTRIBUTIONS ANALYSIS")
print("=" * 80)

# Create scatter plot data
import matplotlib.pyplot as plt

# Flatten data for scatter plot
# Use only training weeks to match contributions
n_train_weeks = push_contrib_orig.shape[1]
impressions_train = push_impressions_orig[:, :n_train_weeks]  # Match training weeks
impressions_flat = impressions_train.flatten()
contributions_flat = push_contrib_orig.detach().cpu().numpy().flatten()

print(f"\n📊 Scatter Plot Data:")
print(f"   Total data points: {len(impressions_flat)}")
print(f"   Impressions range: [{impressions_flat.min():.2f}, {impressions_flat.max():.2f}]")
print(f"   Contributions range: [{contributions_flat.min():.2f}, {contributions_flat.max():.2f}]")

# Check correlation
import numpy as np
correlation = np.corrcoef(impressions_flat, contributions_flat)[0, 1]
print(f"   Correlation: {correlation:.4f}")

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(impressions_flat, contributions_flat, alpha=0.3, s=10)
plt.xlabel('Original Impressions')
plt.ylabel('Contributions (Original Scale, visits)')
plt.title('SEM - Google Search: Impressions vs Contributions')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/sem_google_search_scatter.png', dpi=150)
print(f"\n✅ Scatter plot saved to: /tmp/sem_google_search_scatter.png")

# Show binned statistics to see the relationship
n_bins = 20
impression_bins = np.linspace(impressions_flat.min(), impressions_flat.max(), n_bins + 1)
bin_indices = np.digitize(impressions_flat, impression_bins)

print(f"\n📊 Binned Analysis (20 bins):")
print(f"   {'Impressions Range':<30} {'Mean Contribution':<20} {'Std':<15} {'Count'}")
print(f"   {'-'*30} {'-'*20} {'-'*15} {'-'*10}")

for i in range(1, n_bins + 1):
    mask = bin_indices == i
    if mask.sum() > 0:
        bin_start = impression_bins[i-1]
        bin_end = impression_bins[i]
        mean_contrib = contributions_flat[mask].mean()
        std_contrib = contributions_flat[mask].std()
        count = mask.sum()
        print(f"   [{bin_start:>8.0f}, {bin_end:>8.0f}]  {mean_contrib:>15.4f}  {std_contrib:>12.4f}  {count:>8}")

print("\n" + "=" * 80)
print("SUMMARY: Check if contributions vary with impressions")
print("=" * 80)

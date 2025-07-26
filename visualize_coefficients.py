#!/usr/bin/env python3
"""
Visualize Marketing Variable Coefficients Over Time
Shows how the effectiveness of each marketing channel changes over time.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the package to path
sys.path.insert(0, os.path.abspath('.'))

def visualize_coefficients():
    """Visualize how marketing coefficients change over time."""
    print("=" * 60)
    print("COEFFICIENT VISUALIZATION")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('your_data.csv')
    df['region'] = 'Main_Region'
    total_revenue = df['revenue'].sum()
    
    print(f"📊 Data: {len(df)} weeks, Total Revenue: ${total_revenue:,.0f}")
    
    # Import and configure
    from deepcausalmmm import GRUCausalMMM, DEFAULT_CONFIG
    from deepcausalmmm.core.data import prepare_data_for_training, create_media_adjacency
    from deepcausalmmm.core.train import train_model
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        'marketing_vars': ['tv_spend', 'digital_spend', 'radio_spend', 'other_spend'],
        'control_vars': ['conversions', 'impressions', 'clicks'],
        'dependent_var': 'revenue',
        'region_var': 'region',
        'date_var': 'date',
        'epochs': 100,
        'hidden_size': 32,
        'learning_rate': 1e-3,
        'verbose': False
    })
    
    # Prepare data
    data_dict = prepare_data_for_training(df, config)
    media_adjacency = create_media_adjacency(data_dict['marketing_vars'])
    data_dict['media_adjacency'] = media_adjacency
    
    # Train model
    model = GRUCausalMMM(
        A_prior=data_dict['media_adjacency'],
        n_media=len(data_dict['marketing_vars']),
        ctrl_dim=len(data_dict['control_vars']),
        hidden=config['hidden_size'],
        n_regions=len(data_dict['regions']),
        dropout=0.1
    )
    
    train_data = {
        'X_m': data_dict['X_m'],
        'X_c': data_dict['X_c'],
        'y': data_dict['y'],
        'R': data_dict['R']
    }
    
    print("🚀 Training model...")
    results = train_model(model, train_data, config)
    
    # Generate predictions and get coefficients
    model.eval()
    with torch.no_grad():
        predictions, coefficients, contributions = model(
            data_dict['X_m'], 
            data_dict['X_c'], 
            data_dict['R']
        )
    
    # Convert to numpy
    coefficients_np = coefficients.numpy()
    contributions_np = contributions.numpy()
    
    # Handle the burn-in padding - exclude first 4 weeks
    original_length = len(df)  # 105 weeks
    coefficients_clean = coefficients_np[:, 4:4+original_length, :]  # Remove burn-in
    contributions_clean = contributions_np[:, 4:4+original_length, :]  # Remove burn-in
    
    print(f"✅ Model trained and coefficients extracted")
    print(f"   - Coefficients shape: {coefficients_clean.shape}")
    print(f"   - Time periods: {coefficients_clean.shape[1]}")
    print(f"   - Marketing variables: {len(data_dict['marketing_vars'])}")
    
    # Create visualizations
    print("\n📊 Creating coefficient visualizations...")
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs('coefficient_analysis', exist_ok=True)
    
    # 1. Individual coefficient plots over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(data_dict['marketing_vars']):
        ax = axes[i]
        time_periods = range(coefficients_clean.shape[1])
        
        # Plot coefficient over time
        ax.plot(time_periods, coefficients_clean[0, :, i], linewidth=2, label=f'{var} Coefficient')
        
        # Add trend line
        z = np.polyfit(time_periods, coefficients_clean[0, :, i], 1)
        p = np.poly1d(z)
        ax.plot(time_periods, p(time_periods), "--", alpha=0.7, label='Trend')
        
        ax.set_title(f'{var} Coefficient Over Time')
        ax.set_xlabel('Time Period (Weeks)')
        ax.set_ylabel('Coefficient Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_coef = np.mean(coefficients_clean[0, :, i])
        std_coef = np.std(coefficients_clean[0, :, i])
        ax.text(0.02, 0.98, f'Mean: {mean_coef:.4f}\nStd: {std_coef:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('coefficient_analysis/individual_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. All coefficients on same plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = sns.color_palette("husl", len(data_dict['marketing_vars']))
    for i, var in enumerate(data_dict['marketing_vars']):
        ax.plot(time_periods, coefficients_clean[0, :, i], 
                linewidth=2, label=var, color=colors[i])
    
    ax.set_title('Marketing Variable Coefficients Over Time')
    ax.set_xlabel('Time Period (Weeks)')
    ax.set_ylabel('Coefficient Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coefficient_analysis/all_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Coefficient heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap data
    heatmap_data = coefficients_clean[0, :, :].T  # Transpose to get variables as rows
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r')
    
    # Set labels
    ax.set_yticks(range(len(data_dict['marketing_vars'])))
    ax.set_yticklabels(data_dict['marketing_vars'])
    
    # Set x-axis labels (every 10 weeks)
    x_ticks = np.arange(0, coefficients_clean.shape[1], 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'Week {i}' for i in x_ticks])
    
    ax.set_title('Coefficient Heatmap Over Time')
    ax.set_xlabel('Time Period (Weeks)')
    ax.set_ylabel('Marketing Variables')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coefficient Value')
    
    plt.tight_layout()
    plt.savefig('coefficient_analysis/coefficient_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Coefficient statistics and trends
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Coefficient statistics
    mean_coefficients = np.mean(coefficients_clean[0, :, :], axis=0)
    std_coefficients = np.std(coefficients_clean[0, :, :], axis=0)
    
    x_pos = np.arange(len(data_dict['marketing_vars']))
    bars = ax1.bar(x_pos, mean_coefficients, yerr=std_coefficients, 
                   capsize=5, color=colors, alpha=0.7)
    
    ax1.set_title('Average Coefficient Values with Standard Deviation')
    ax1.set_xlabel('Marketing Variables')
    ax1.set_ylabel('Coefficient Value')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(data_dict['marketing_vars'], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, mean_coefficients, std_coefficients):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.001,
                f'{mean_val:.4f}\n±{std_val:.4f}', ha='center', va='bottom')
    
    # Coefficient trends (slope of trend line)
    trends = []
    for i in range(len(data_dict['marketing_vars'])):
        z = np.polyfit(time_periods, coefficients_clean[0, :, i], 1)
        trends.append(z[0])  # Slope
    
    bars2 = ax2.bar(x_pos, trends, color=colors, alpha=0.7)
    ax2.set_title('Coefficient Trends (Slope)')
    ax2.set_xlabel('Marketing Variables')
    ax2.set_ylabel('Trend Slope')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(data_dict['marketing_vars'], rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, trend in zip(bars2, trends):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if trend >= 0 else -0.001),
                f'{trend:.4f}', ha='center', va='bottom' if trend >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('coefficient_analysis/coefficient_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Coefficient vs Contribution correlation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(data_dict['marketing_vars']):
        ax = axes[i]
        
        # Get coefficients and contributions for this variable
        coef_values = coefficients_clean[0, :, i]
        contrib_values = contributions_clean[0, :, i]
        
        # Calculate correlation
        correlation = np.corrcoef(coef_values, contrib_values)[0, 1]
        
        # Scatter plot
        ax.scatter(coef_values, contrib_values, alpha=0.6)
        ax.set_xlabel('Coefficient Value')
        ax.set_ylabel('Contribution Value')
        ax.set_title(f'{var}: Coefficient vs Contribution\nCorrelation: {correlation:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coefficient_analysis/coefficient_vs_contribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save coefficient data
    coefficient_df = pd.DataFrame({
        'week': range(coefficients_clean.shape[1])
    })
    
    for i, var in enumerate(data_dict['marketing_vars']):
        coefficient_df[f'{var}_coefficient'] = coefficients_clean[0, :, i]
        coefficient_df[f'{var}_contribution'] = contributions_clean[0, :, i]
    
    coefficient_df.to_csv('coefficient_analysis/coefficient_data.csv', index=False)
    
    # Print summary statistics
    print("\n📊 COEFFICIENT SUMMARY STATISTICS")
    print("=" * 50)
    
    for i, var in enumerate(data_dict['marketing_vars']):
        coef_values = coefficients_clean[0, :, i]
        mean_coef = np.mean(coef_values)
        std_coef = np.std(coef_values)
        min_coef = np.min(coef_values)
        max_coef = np.max(coef_values)
        
        # Calculate trend
        z = np.polyfit(time_periods, coef_values, 1)
        trend = z[0]
        
        print(f"\n{var}:")
        print(f"   - Mean: {mean_coef:.4f}")
        print(f"   - Std:  {std_coef:.4f}")
        print(f"   - Min:  {min_coef:.4f}")
        print(f"   - Max:  {max_coef:.4f}")
        print(f"   - Trend: {trend:.4f} {'(increasing)' if trend > 0 else '(decreasing)' if trend < 0 else '(stable)'}")
    
    print(f"\n📁 Visualizations saved in 'coefficient_analysis/' directory:")
    print("   - individual_coefficients.png")
    print("   - all_coefficients.png")
    print("   - coefficient_heatmap.png")
    print("   - coefficient_statistics.png")
    print("   - coefficient_vs_contribution.png")
    print("   - coefficient_data.csv")
    
    print("\n" + "=" * 60)
    print("✅ COEFFICIENT VISUALIZATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    visualize_coefficients() 
#!/usr/bin/env python3
"""
Complete DeepCausalMMM Analysis with Sample Data
Shows model statistics, predictions, and variable contributions over time.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the package to path
sys.path.insert(0, os.path.abspath('.'))

def run_complete_analysis():
    """Run complete analysis with sample data."""
    print("=" * 80)
    print("DEEPCAUSALMMM - COMPLETE ANALYSIS WITH SAMPLE DATA")
    print("=" * 80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: Load or generate sample data
    print("\n1. 📊 LOADING SAMPLE DATA")
    print("-" * 50)
    
    if os.path.exists('sample_marketing_data.csv'):
        df = pd.read_csv('sample_marketing_data.csv')
        print(f"✅ Loaded existing sample data: {len(df)} rows")
    else:
        print("❌ Sample data not found. Please run the test first.")
        return
    
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Regions: {df['region'].nunique()}")
    print(f"   Revenue range: ${df['revenue'].min():,.0f} - ${df['revenue'].max():,.0f}")
    print(f"   Media variables: {[col for col in df.columns if 'spend' in col]}")
    
    # Step 2: Import and configure model
    print("\n2. 🤖 CONFIGURING MODEL")
    print("-" * 50)
    
    try:
        from deepcausalmmm import GRUCausalMMM, DEFAULT_CONFIG
        from deepcausalmmm.core.data import prepare_data_for_training, create_media_adjacency
        from deepcausalmmm.core.train import train_model
        from deepcausalmmm.utils.metrics import calculate_metrics
        print("✅ Package imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return
    
    # Configure model parameters
    config = DEFAULT_CONFIG.copy()
    config.update({
        'marketing_vars': ['tv_spend', 'digital_spend', 'radio_spend', 'print_spend'],
        'control_vars': ['price', 'promotion', 'competition', 'economic_indicator'],
        'dependent_var': 'revenue',
        'region_var': 'region',
        'date_var': 'date',
        'epochs': 500,
        'hidden_size': 32,
        'learning_rate': 1e-3,
        'batch_size': 16,
        'dropout': 0.1,
        'verbose': True,
        'early_stopping_patience': 20
    })
    
    print(f"   Model configuration:")
    print(f"   - Hidden size: {config['hidden_size']}")
    print(f"   - Learning rate: {config['learning_rate']}")
    print(f"   - Epochs: {config['epochs']}")
    print(f"   - Marketing variables: {config['marketing_vars']}")
    print(f"   - Control variables: {config['control_vars']}")
    
    # Step 3: Prepare data
    print("\n3. 🔧 PREPARING DATA")
    print("-" * 50)
    
    try:
        data_dict = prepare_data_for_training(df, config)
        
        # Create media adjacency matrix
        media_adjacency = create_media_adjacency(data_dict['marketing_vars'])
        data_dict['media_adjacency'] = media_adjacency
        
        print("✅ Data preparation successful")
        print(f"   - Regions: {len(data_dict['regions'])}")
        print(f"   - Time steps: {data_dict['X_m'].shape[1]}")
        print(f"   - Media variables: {len(data_dict['marketing_vars'])}")
        print(f"   - Control variables: {len(data_dict['control_vars'])}")
        print(f"   - Data shapes: X_m={data_dict['X_m'].shape}, X_c={data_dict['X_c'].shape}, y={data_dict['y'].shape}")
        
    except Exception as e:
        print(f"❌ Data preparation error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Initialize and train model
    print("\n4. 🚀 TRAINING MODEL")
    print("-" * 50)
    
    try:
        model = GRUCausalMMM(
            A_prior=data_dict['media_adjacency'],
            n_media=len(data_dict['marketing_vars']),
            ctrl_dim=len(data_dict['control_vars']),
            hidden=config['hidden_size'],
            n_regions=len(data_dict['regions']),
            dropout=config['dropout']
        )
        
        print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Prepare training data
        train_data = {
            'X_m': data_dict['X_m'],
            'X_c': data_dict['X_c'],
            'y': data_dict['y'],
            'R': data_dict['R']
        }
        
        # Train model
        print("   Training model...")
        results = train_model(model, train_data, config)
        
        print("✅ Training completed")
        print(f"   - Final loss: {results['final_train_loss']:.6f}")
        print(f"   - Epochs trained: {results['epochs_trained']}")
        print(f"   - Training time: {results.get('training_time', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Generate predictions and contributions
    print("\n5. 📈 GENERATING PREDICTIONS & CONTRIBUTIONS")
    print("-" * 50)
    
    try:
        model.eval()
        with torch.no_grad():
            predictions, coefficients, contributions = model(
                data_dict['X_m'], 
                data_dict['X_c'], 
                data_dict['R']
            )
        
        # Convert to numpy for analysis
        predictions_np = predictions.numpy()
        coefficients_np = coefficients.numpy()
        contributions_np = contributions.numpy()
        actual_np = data_dict['y'].numpy()
        
        print("✅ Predictions and contributions generated")
        print(f"   - Predictions shape: {predictions_np.shape}")
        print(f"   - Coefficients shape: {coefficients_np.shape}")
        print(f"   - Contributions shape: {contributions_np.shape}")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Calculate model statistics
    print("\n6. 📊 MODEL STATISTICS")
    print("-" * 50)
    
    try:
        # Calculate metrics manually
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from sklearn.metrics import mean_absolute_percentage_error
        
        actual_flat = actual_np.flatten()
        pred_flat = predictions_np.flatten()
        
        metrics = {
            'r2_score': r2_score(actual_flat, pred_flat),
            'rmse': np.sqrt(mean_squared_error(actual_flat, pred_flat)),
            'mae': mean_absolute_error(actual_flat, pred_flat),
            'mape': mean_absolute_percentage_error(actual_flat, pred_flat) * 100,
            'mse': mean_squared_error(actual_flat, pred_flat)
        }
        
        print("✅ Model Performance Metrics:")
        print(f"   - R² Score: {metrics['r2_score']:.4f}")
        print(f"   - RMSE: {metrics['rmse']:.2f}")
        print(f"   - MAE: {metrics['mae']:.2f}")
        print(f"   - MAPE: {metrics['mape']:.2f}%")
        print(f"   - MSE: {metrics['mse']:.2f}")
        
        # Calculate overall statistics
        total_revenue = actual_np.sum()
        predicted_revenue = predictions_np.sum()
        revenue_accuracy = (predicted_revenue / total_revenue - 1) * 100
        
        print(f"\n   Overall Statistics:")
        print(f"   - Total Actual Revenue: ${total_revenue:,.0f}")
        print(f"   - Total Predicted Revenue: ${predicted_revenue:,.0f}")
        print(f"   - Revenue Accuracy: {revenue_accuracy:+.2f}%")
        
    except Exception as e:
        print(f"❌ Metrics calculation error: {e}")
        return
    
    # Step 7: Analyze variable contributions
    print("\n7. 🎯 VARIABLE CONTRIBUTIONS ANALYSIS")
    print("-" * 50)
    
    try:
        # Calculate average contributions by variable
        avg_contributions = np.mean(contributions_np, axis=(0, 1))  # Average across regions and time
        
        print("✅ Variable Contributions (Average):")
        for i, var in enumerate(data_dict['marketing_vars']):
            contrib = avg_contributions[i]
            contrib_pct = (contrib / np.mean(actual_np)) * 100
            print(f"   - {var}: ${contrib:,.0f} ({contrib_pct:+.2f}%)")
        
        # Calculate total contributions by variable
        total_contributions = np.sum(contributions_np, axis=(0, 1))
        print(f"\n   Total Contributions by Variable:")
        for i, var in enumerate(data_dict['marketing_vars']):
            contrib = total_contributions[i]
            contrib_pct = (contrib / total_revenue) * 100
            print(f"   - {var}: ${contrib:,.0f} ({contrib_pct:+.2f}%)")
        
    except Exception as e:
        print(f"❌ Contributions analysis error: {e}")
        return
    
    # Step 8: Create visualizations
    print("\n8. 📊 CREATING VISUALIZATIONS")
    print("-" * 50)
    
    try:
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        os.makedirs('analysis_output', exist_ok=True)
        
        # 1. Actual vs Predicted
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Time series comparison
        for r in range(min(3, len(data_dict['regions']))):  # Show first 3 regions
            region_name = data_dict['regions'][r]
            ax1.plot(actual_np[r, :], label=f'Actual - {region_name}', alpha=0.7)
            ax1.plot(predictions_np[r, :], '--', label=f'Predicted - {region_name}', alpha=0.7)
        
        ax1.set_title('Actual vs Predicted Revenue Over Time')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Revenue ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        ax2.scatter(actual_np.flatten(), predictions_np.flatten(), alpha=0.6)
        ax2.plot([actual_np.min(), actual_np.max()], [actual_np.min(), actual_np.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Revenue ($)')
        ax2.set_ylabel('Predicted Revenue ($)')
        ax2.set_title(f'Actual vs Predicted (R² = {metrics["r2_score"]:.3f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis_output/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Variable Contributions Over Time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, var in enumerate(data_dict['marketing_vars']):
            if i < 4:  # Only plot first 4 variables
                ax = axes[i]
                for r in range(min(2, len(data_dict['regions']))):  # Show first 2 regions
                    region_name = data_dict['regions'][r]
                    ax.plot(contributions_np[r, :, i], label=f'{region_name}', alpha=0.7)
                
                ax.set_title(f'{var} Contribution Over Time')
                ax.set_xlabel('Time Period')
                ax.set_ylabel('Contribution ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis_output/variable_contributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Variable Importance Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        variables = data_dict['marketing_vars']
        avg_contribs = np.mean(contributions_np, axis=(0, 1))
        
        bars = ax.bar(variables, avg_contribs, color=sns.color_palette("husl", len(variables)))
        ax.set_title('Average Variable Contributions')
        ax.set_xlabel('Marketing Variables')
        ax.set_ylabel('Average Contribution ($)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, contrib in zip(bars, avg_contribs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'${contrib:,.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('analysis_output/variable_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations created:")
        print("   - actual_vs_predicted.png")
        print("   - variable_contributions.png")
        print("   - variable_importance.png")
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 9: Save detailed results
    print("\n9. 💾 SAVING DETAILED RESULTS")
    print("-" * 50)
    
    try:
        # Create results summary
        results_summary = {
            'model_performance': metrics,
            'training_info': {
                'epochs_trained': results['epochs_trained'],
                'final_loss': results['final_train_loss'],
                'total_parameters': sum(p.numel() for p in model.parameters())
            },
            'data_info': {
                'regions': data_dict['regions'],
                'time_periods': data_dict['X_m'].shape[1],
                'marketing_vars': data_dict['marketing_vars'],
                'control_vars': data_dict['control_vars']
            },
            'variable_contributions': {
                var: {
                    'avg_contribution': float(avg_contributions[i]),
                    'total_contribution': float(total_contributions[i]),
                    'contribution_pct': float((total_contributions[i] / total_revenue) * 100)
                }
                for i, var in enumerate(data_dict['marketing_vars'])
            }
        }
        
        # Save to JSON
        import json
        with open('analysis_output/results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save predictions and contributions to CSV
        predictions_df = pd.DataFrame({
            'region': np.repeat(data_dict['regions'], data_dict['X_m'].shape[1]),
            'time_period': np.tile(range(data_dict['X_m'].shape[1]), len(data_dict['regions'])),
            'actual_revenue': actual_np.flatten(),
            'predicted_revenue': predictions_np.flatten(),
            'prediction_error': (predictions_np.flatten() - actual_np.flatten())
        })
        
        # Add contribution columns
        for i, var in enumerate(data_dict['marketing_vars']):
            predictions_df[f'{var}_contribution'] = contributions_np[:, :, i].flatten()
        
        predictions_df.to_csv('analysis_output/predictions_and_contributions.csv', index=False)
        
        print("✅ Results saved:")
        print("   - results_summary.json")
        print("   - predictions_and_contributions.csv")
        
    except Exception as e:
        print(f"❌ Results saving error: {e}")
        return
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print("📁 Output files created in 'analysis_output/' directory:")
    print("   - actual_vs_predicted.png")
    print("   - variable_contributions.png") 
    print("   - variable_importance.png")
    print("   - results_summary.json")
    print("   - predictions_and_contributions.csv")
    print("\n📊 Key Results:")
    print(f"   - Model R² Score: {metrics['r2_score']:.3f}")
    print(f"   - Revenue Accuracy: {revenue_accuracy:+.2f}%")
    print(f"   - Top Contributing Variable: {data_dict['marketing_vars'][np.argmax(avg_contributions)]}")
    print("=" * 80)

if __name__ == "__main__":
    run_complete_analysis() 
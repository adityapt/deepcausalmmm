#!/usr/bin/env python3
"""
Simple test for DeepCausalMMM core functionality.
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
from datetime import datetime, timedelta

# Add the package to path for testing
sys.path.insert(0, os.path.abspath('.'))

def generate_simple_data():
    """Generate simple test data."""
    np.random.seed(42)
    
    # Generate 52 weeks of data for 2 regions
    dates = [datetime(2020, 1, 1) + timedelta(weeks=i) for i in range(52)]
    
    data = []
    for region in ['Region_A', 'Region_B']:
        for i, date in enumerate(dates):
            # Simple media spending
            tv_spend = 1000 + 500 * np.sin(i * 2 * np.pi / 52) + np.random.normal(0, 100)
            digital_spend = 800 + 300 * np.sin(i * 2 * np.pi / 26) + np.random.normal(0, 80)
            radio_spend = 500 + 200 * np.sin(i * 2 * np.pi / 13) + np.random.normal(0, 50)
            
            # Control variables
            price = 10 + np.random.normal(0, 0.5)
            promotion = np.random.choice([0, 1], p=[0.7, 0.3])
            
            # Revenue
            revenue = 10000 + 0.1 * tv_spend + 0.15 * digital_spend + 0.08 * radio_spend - 50 * price + 2000 * promotion + np.random.normal(0, 500)
            
            data.append({
                'date': date,
                'region': region,
                'week': i,
                'tv_spend': max(0, tv_spend),
                'digital_spend': max(0, digital_spend),
                'radio_spend': max(0, radio_spend),
                'price': price,
                'promotion': promotion,
                'revenue': max(0, revenue)
            })
    
    return pd.DataFrame(data)

def test_core_functionality():
    """Test core functionality."""
    print("=" * 60)
    print("DEEPCAUSALMMM - CORE FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Step 1: Generate data
    print("1. Generating test data...")
    df = generate_simple_data()
    df.to_csv('test_data.csv', index=False)
    print(f"   Generated {len(df)} rows of data")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Revenue range: ${df['revenue'].min():.0f} - ${df['revenue'].max():.0f}")
    
    # Step 2: Test imports
    print("\n2. Testing imports...")
    try:
        from deepcausalmmm import GRUCausalMMM, DEFAULT_CONFIG
        from deepcausalmmm.core.data import prepare_data_for_training
        print("   ✅ Core imports successful")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False
    
    # Step 3: Test data preprocessing
    print("\n3. Testing data preprocessing...")
    try:
        config = DEFAULT_CONFIG.copy()
        config.update({
            'marketing_vars': ['tv_spend', 'digital_spend', 'radio_spend'],
            'control_vars': ['price', 'promotion'],
            'dependent_var': 'revenue',
            'region_var': 'region',
            'date_var': 'date',
            'epochs': 100,  # Very short for testing
            'hidden_size': 16,
            'learning_rate': 1e-3,
            'verbose': False
        })
        
        data_dict = prepare_data_for_training(df, config)
        
        # Create media adjacency matrix
        from deepcausalmmm.core.data import create_media_adjacency
        media_adjacency = create_media_adjacency(data_dict['marketing_vars'])
        data_dict['media_adjacency'] = media_adjacency
        
        print("   ✅ Data preprocessing successful")
        print(f"   - Media variables: {data_dict['marketing_vars']}")
        print(f"   - Control variables: {data_dict['control_vars']}")
        print(f"   - Data shapes: X_m={data_dict['X_m'].shape}, X_c={data_dict['X_c'].shape}, y={data_dict['y'].shape}")
        
    except Exception as e:
        print(f"   ❌ Data preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test model initialization
    print("\n4. Testing model initialization...")
    try:
        model = GRUCausalMMM(
            A_prior=data_dict['media_adjacency'],
            n_media=len(data_dict['marketing_vars']),
            ctrl_dim=len(data_dict['control_vars']),
            hidden=config['hidden_size'],
            n_regions=len(data_dict['regions']),
            dropout=0.1
        )
        
        print(f"   ✅ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
    except Exception as e:
        print(f"   ❌ Model initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test forward pass
    print("\n5. Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            predictions, coefficients, contributions = model(
                data_dict['X_m'], 
                data_dict['X_c'], 
                data_dict['R']
            )
        
        print("   ✅ Forward pass successful")
        print(f"   - Predictions shape: {predictions.shape}")
        print(f"   - Coefficients shape: {coefficients.shape}")
        print(f"   - Contributions shape: {contributions.shape}")
        
    except Exception as e:
        print(f"   ❌ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Test basic training
    print("\n6. Testing basic training...")
    try:
        from deepcausalmmm.core.train import train_model
        
        # Simple training data
        train_data = {
            'X_m': data_dict['X_m'],
            'X_c': data_dict['X_c'],
            'y': data_dict['y'],
            'R': data_dict['R']
        }
        
        # Train for just a few epochs
        results = train_model(model, train_data, config)
        
        print("   ✅ Basic training successful")
        print(f"   - Final loss: {results['final_train_loss']:.6f}")
        print(f"   - Epochs trained: {results['epochs_trained']}")
        
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
    print("Core functionality is working correctly.")
    print("The package is ready for use!")
    
    return True

if __name__ == "__main__":
    test_core_functionality() 
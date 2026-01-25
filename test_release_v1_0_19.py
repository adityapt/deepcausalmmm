#!/usr/bin/env python
"""
Comprehensive Release Test for DeepCausalMMM v1.0.19
====================================================

Tests all critical functionality before production release.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all core imports."""
    print("\n" + "="*80)
    print("TEST 1: Package Imports")
    print("="*80)
    
    try:
        import deepcausalmmm
        from deepcausalmmm import DeepCausalMMM, get_device
        from deepcausalmmm.core import get_default_config, update_config
        from deepcausalmmm.core.trainer import ModelTrainer
        from deepcausalmmm.core.data import UnifiedDataPipeline
        from deepcausalmmm.core.scaling import SimpleGlobalScaler
        from deepcausalmmm.postprocess import (
            ComprehensiveAnalyzer,
            ResponseCurveFit,
            BudgetOptimizer,
            optimize_budget_from_curves
        )
        from deepcausalmmm.utils.data_generator import ConfigurableDataGenerator
        
        print(f"   Package version: {deepcausalmmm.__version__}")
        print(f"   Device: {get_device()}")
        print("   SUCCESS: All imports successful")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()
        return False

def test_version():
    """Test version number is 1.0.19."""
    print("\n" + "="*80)
    print("TEST 2: Version Number")
    print("="*80)
    
    try:
        import deepcausalmmm
        version = deepcausalmmm.__version__
        
        if version == "1.0.19":
            print(f"   Version: {version}")
            print("   SUCCESS: Version is 1.0.19")
            return True
        else:
            print(f"   FAILED: Version is {version}, expected 1.0.19")
            return False
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()
        return False

def test_data_generation():
    """Test synthetic data generation."""
    print("\n" + "="*80)
    print("TEST 3: Data Generation")
    print("="*80)
    
    try:
        import numpy as np
        from deepcausalmmm.core import get_default_config
        from deepcausalmmm.utils.data_generator import ConfigurableDataGenerator
        
        config = get_default_config()
        config['random_seed'] = 42
        
        generator = ConfigurableDataGenerator(config)
        X_media, X_control, y = generator.generate_mmm_dataset(
            n_regions=10,
            n_weeks=52,
            n_media_channels=5,
            n_control_channels=3
        )
        
        assert X_media.shape == (10, 52, 5), f"Wrong X_media shape: {X_media.shape}"
        assert X_control.shape == (10, 52, 3), f"Wrong X_control shape: {X_control.shape}"
        assert y.shape == (10, 52), f"Wrong y shape: {y.shape}"
        
        print(f"   X_media: {X_media.shape}")
        print(f"   X_control: {X_control.shape}")
        print(f"   y: {y.shape}")
        print("   SUCCESS: Data generation works")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()
        return False

def test_pipeline():
    """Test UnifiedDataPipeline."""
    print("\n" + "="*80)
    print("TEST 4: Data Pipeline")
    print("="*80)
    
    try:
        import numpy as np
        from deepcausalmmm.core import get_default_config
        from deepcausalmmm.core.data import UnifiedDataPipeline
        from deepcausalmmm.utils.data_generator import ConfigurableDataGenerator
        
        config = get_default_config()
        config['random_seed'] = 42
        
        generator = ConfigurableDataGenerator(config)
        X_media, X_control, y = generator.generate_mmm_dataset(
            n_regions=10, n_weeks=52, n_media_channels=5, n_control_channels=3
        )
        
        pipeline = UnifiedDataPipeline(config)
        train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
        train_tensors = pipeline.fit_and_transform_training(train_data)
        
        assert 'X_media' in train_tensors
        assert 'X_control' in train_tensors
        assert 'y' in train_tensors
        assert 'R' in train_tensors
        
        print(f"   Training tensors keys: {list(train_tensors.keys())}")
        print(f"   X_media shape: {train_tensors['X_media'].shape}")
        print("   SUCCESS: Pipeline works")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation and initialization."""
    print("\n" + "="*80)
    print("TEST 5: Model Creation")
    print("="*80)
    
    try:
        from deepcausalmmm.core import get_default_config
        from deepcausalmmm.core.trainer import ModelTrainer
        
        config = get_default_config()
        trainer = ModelTrainer(config)
        
        model = trainer.create_model(n_media=5, n_control=3, n_regions=10)
        
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'hill_a')
        assert hasattr(model, 'hill_g')
        
        print(f"   Model type: {type(model).__name__}")
        print(f"   Media channels: 5")
        print(f"   Control variables: 3")
        print(f"   Regions: 10")
        print("   SUCCESS: Model creation works")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()
        return False

def test_training():
    """Test quick training (5 epochs)."""
    print("\n" + "="*80)
    print("TEST 6: Model Training (5 epochs)")
    print("="*80)
    
    try:
        import numpy as np
        from deepcausalmmm.core import get_default_config
        from deepcausalmmm.core.trainer import ModelTrainer
        from deepcausalmmm.core.data import UnifiedDataPipeline
        from deepcausalmmm.utils.data_generator import ConfigurableDataGenerator
        
        config = get_default_config()
        config['random_seed'] = 42
        config['n_epochs'] = 5  # Quick test
        
        generator = ConfigurableDataGenerator(config)
        X_media, X_control, y = generator.generate_mmm_dataset(
            n_regions=10, n_weeks=52, n_media_channels=5, n_control_channels=3
        )
        
        pipeline = UnifiedDataPipeline(config)
        train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
        train_tensors = pipeline.fit_and_transform_training(train_data)
        
        trainer = ModelTrainer(config)
        model = trainer.create_model(
            n_media=train_tensors['X_media'].shape[2],
            n_control=train_tensors['X_control'].shape[2],
            n_regions=train_tensors['X_media'].shape[0]
        )
        trainer.create_optimizer_and_scheduler()
        
        results = trainer.train(
            train_tensors['X_media'], train_tensors['X_control'],
            train_tensors['R'], train_tensors['y'],
            pipeline=pipeline,
            verbose=False
        )
        
        assert 'final_train_r2' in results
        assert 'final_train_rmse' in results
        
        print(f"   Final Training R²: {results['final_train_r2']:.4f}")
        print(f"   Final Training RMSE: {results['final_train_rmse']:,.0f}")
        print("   SUCCESS: Training works")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()
        return False

def test_attribution_additivity():
    """Test that components sum to 100%."""
    print("\n" + "="*80)
    print("TEST 7: Attribution Additivity")
    print("="*80)
    
    try:
        import numpy as np
        import torch
        from deepcausalmmm.core import get_default_config
        from deepcausalmmm.core.trainer import ModelTrainer
        from deepcausalmmm.core.data import UnifiedDataPipeline
        from deepcausalmmm.core.scaling import SimpleGlobalScaler
        from deepcausalmmm.utils.data_generator import ConfigurableDataGenerator
        
        config = get_default_config()
        config['random_seed'] = 42
        config['n_epochs'] = 5
        
        generator = ConfigurableDataGenerator(config)
        X_media, X_control, y = generator.generate_mmm_dataset(
            n_regions=10, n_weeks=52, n_media_channels=5, n_control_channels=3
        )
        
        pipeline = UnifiedDataPipeline(config)
        train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
        train_tensors = pipeline.fit_and_transform_training(train_data)
        
        trainer = ModelTrainer(config)
        model = trainer.create_model(
            n_media=train_tensors['X_media'].shape[2],
            n_control=train_tensors['X_control'].shape[2],
            n_regions=train_tensors['X_media'].shape[0]
        )
        trainer.create_optimizer_and_scheduler()
        
        results = trainer.train(
            train_tensors['X_media'], train_tensors['X_control'],
            train_tensors['R'], train_tensors['y'],
            pipeline=pipeline,
            verbose=False
        )
        
        # Test forward pass and additivity
        model.eval()
        with torch.no_grad():
            y_pred, media_contrib, ctrl_contrib, outputs = model(
                train_tensors['X_media'],
                train_tensors['X_control'],
                train_tensors['R']
            )
        
        # Inverse transform
        scaler = pipeline.scaler
        y_pred_orig = scaler.inverse_transform_target(y_pred)
        
        contrib_results = scaler.inverse_transform_contributions(
            media_contributions=media_contrib,
            baseline=outputs.get('baseline'),
            control_contributions=ctrl_contrib,
            seasonal_contributions=outputs.get('seasonal_contribution'),
            trend_contributions=outputs.get('trend_contribution'),
            prediction_scale=outputs.get('prediction_scale')
        )
        
        # Calculate total contributions
        baseline_total = contrib_results['baseline'].sum().item()
        seasonal_total = contrib_results.get('seasonal', torch.tensor(0.0)).sum().item()
        media_total = contrib_results['media'].sum().item()
        control_total = contrib_results.get('control', torch.tensor(0.0)).sum().item()
        
        components_sum = baseline_total + seasonal_total + media_total + control_total
        predictions_total = y_pred_orig.sum().item()
        
        relative_error = abs(components_sum - predictions_total) / predictions_total * 100
        
        print(f"   Predictions total: {predictions_total:,.0f}")
        print(f"   Components sum: {components_sum:,.0f}")
        print(f"   Relative error: {relative_error:.3f}%")
        
        if relative_error < 5.0:
            print("   SUCCESS: Attribution sums correctly (<5% error for quick test)")
            return True
        else:
            print(f"   FAILED: Attribution error {relative_error:.3f}% > 5%")
            return False
            
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()
        return False

def test_examples():
    """Test example scripts are executable."""
    print("\n" + "="*80)
    print("TEST 8: Example Scripts")
    print("="*80)
    
    try:
        import subprocess
        import os
        
        os.chdir(Path(__file__).parent)
        
        # Test budget optimization example
        result = subprocess.run(
            ["python", "examples/example_budget_optimization.py"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("   example_budget_optimization.py: SUCCESS")
        else:
            print(f"   example_budget_optimization.py: FAILED")
            print(f"   Error: {result.stderr}")
            return False
        
        # Test response curves data loading
        result = subprocess.run(
            ["python", "-c", 
             "from examples.example_response_curves import load_real_mmm_data; "
             "X_media, X_control, y, media_names, control_names, df, cols = load_real_mmm_data(); "
             "print(f'SUCCESS: {X_media.shape}')"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print("   example_response_curves.py data loading: SUCCESS")
        else:
            print(f"   example_response_curves.py data loading: FAILED")
            print(f"   Error: {result.stderr}")
            return False
        
        print("   SUCCESS: All example scripts executable")
        return True
        
    except Exception as e:
        print(f"   FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n")
    print("="*80)
    print("DEEPCAUSALMMM v1.0.19 RELEASE TEST SUITE")
    print("="*80)
    
    tests = [
        test_imports,
        test_version,
        test_data_generation,
        test_pipeline,
        test_model_creation,
        test_training,
        test_attribution_additivity,
        test_examples,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n   CRITICAL FAILURE in {test.__name__}: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if all(results):
        print("\n" + "="*80)
        print("ALL TESTS PASSED - READY FOR PRODUCTION RELEASE v1.0.19")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("SOME TESTS FAILED - FIX ISSUES BEFORE RELEASE")
        print("="*80)
        return 1

if __name__ == "__main__":
    sys.exit(main())


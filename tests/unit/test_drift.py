import numpy as np, pandas as pd
from mmm_monitor.drift import psi, ks_pvalue, DriftDetector

def test_no_drift():
    """Test that no drift is detected between similar distributions"""
    # Use consistent random seeding
    np.random.seed(42)
    base = pd.Series(np.random.normal(0, 1, 1000))
    cur = pd.Series(np.random.normal(0, 1, 500))  # Same distribution
    
    # More reasonable thresholds for no-drift case
    psi_val = psi(cur, base)
    ks_val = ks_pvalue(cur, base)
    
    # PSI should be relatively low for similar distributions
    assert psi_val < 0.2, f"PSI too high: {psi_val}"
    # KS p-value should indicate no significant difference (> 0.01 for loose test)
    assert ks_val > 0.01, f"KS p-value too low: {ks_val}"

def test_detect_drift():
    """Test that drift is detected between different distributions"""
    np.random.seed(42)
    base = pd.Series(np.random.normal(0, 1, 1000))
    cur = pd.Series(np.random.normal(2, 1, 500))  # Shifted mean - should detect drift
    
    psi_val = psi(cur, base)
    ks_val = ks_pvalue(cur, base)
    
    # PSI should be high for different distributions
    assert psi_val > 0.2, f"PSI too low for drift case: {psi_val}"
    # KS p-value should indicate significant difference
    assert ks_val < 0.05, f"KS p-value too high for drift case: {ks_val}"

def test_drift_detector():
    """Test the DriftDetector class functionality"""
    np.random.seed(42)
    baseline_data = pd.DataFrame({'residual': np.random.normal(0, 1, 1000)})
    current_data = pd.Series(np.random.normal(0, 1, 500), name='residual')
    
    detector = DriftDetector(baseline_data, window=50, psi_threshold=0.2, ks_threshold=0.05)
    result = detector.detect_residual_drift(current_data)
    
    # Check that result has expected keys
    expected_keys = {'psi', 'ks_pvalue', 'psi_alert', 'ks_alert'}
    assert set(result.keys()) == expected_keys
    
    # Check that values are reasonable
    assert isinstance(result['psi'], float)
    assert isinstance(result['ks_pvalue'], float)
    assert isinstance(result['psi_alert'], (bool, np.bool_))
    assert isinstance(result['ks_alert'], (bool, np.bool_))

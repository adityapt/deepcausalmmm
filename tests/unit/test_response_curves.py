"""
Unit tests for response curves module.
"""
import pytest
import pandas as pd
import numpy as np
from deepcausalmmm.postprocess.response_curves import ResponseCurveFit, ResponseCurveFitter


class TestResponseCurveFit:
    """Test suite for ResponseCurveFit class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_points = 100
        
        # Generate impressions and spend
        impressions = np.linspace(1, 10000, n_points)
        spend = impressions * 0.5  # CPI = 0.5
        
        # Generate contributions with Hill saturation curve
        # y = x^a / (x^a + g^a) where a=2.5, g=5000
        a, g = 2.5, 2500  # Using spend, so g is lower
        predicted = (spend ** a) / (spend ** a + g ** a)
        predicted = predicted * 1000000  # Scale to millions
        
        # Add some noise
        predicted += np.random.normal(0, 10000, n_points)
        
        # Create DataFrame with required columns
        df = pd.DataFrame({
            'week_monday': pd.date_range('2023-01-01', periods=n_points, freq='W'),
            'impressions': impressions,
            'spend': spend,
            'predicted': predicted
        })
        
        return df
    
    def test_initialization(self, sample_data):
        """Test ResponseCurveFit initialization."""
        fitter = ResponseCurveFit(
            data=sample_data,
            model_level='Overall',
            date_col='week_monday'
        )
        
        assert fitter.model_level == 'Overall'
        assert fitter.date_col == 'week_monday'
        assert not fitter.bottom_param
        assert len(fitter.data) == len(sample_data)
    
    def test_backward_compatibility_params(self, sample_data):
        """Test backward compatibility with old parameter names."""
        fitter = ResponseCurveFit(
            data=sample_data,
            model_level='Overall',
            date_col='week_monday'
        )
        
        # Old attribute names should still work
        assert fitter.Modellevel == 'Overall'
        assert fitter.Datecol == 'week_monday'
    
    def test_hill_equation(self, sample_data):
        """Test Hill equation calculation."""
        fitter = ResponseCurveFit(
            data=sample_data,
            model_level='Overall'
        )
        
        x = np.array([100, 1000, 5000, 10000])
        # params: top, bottom, saturation, slope
        params = (1000000, 0, 5000, 2.5)
        
        result = fitter._hill_equation(x, *params)
        
        # Check that result is reasonable
        assert np.all(result >= 0)
        assert np.all(result <= params[0])  # Should not exceed top
        
        # Check that it's monotonically increasing
        assert np.all(np.diff(result) >= 0)
    
    def test_fit_overall(self, sample_data):
        """Test fitting at overall level."""
        fitter = ResponseCurveFit(
            data=sample_data,
            model_level='Overall'
        )
        
        # Fit without generating figure
        result = fitter.fit(
            generate_figure=False,
            print_r_sqr=False
        )
        
        # Check that parameters are reasonable
        assert hasattr(fitter, 'slope')
        assert hasattr(fitter, 'saturation')
        assert hasattr(fitter, 'r_2')
        
        assert fitter.slope > 0, "Slope should be positive"
        assert fitter.saturation > 0, "Saturation should be positive"
        assert fitter.r_2 >= 0, "R² should be non-negative"
    
    def test_backward_compatibility_methods(self, sample_data):
        """Test backward compatibility with old method names."""
        fitter = ResponseCurveFit(
            data=sample_data,
            model_level='Overall'
        )
        
        # Test Hill (old name for _hill_equation)
        x = np.array([1000, 5000])
        params = (1000000, 0, 5000, 2.5)
        
        result1 = fitter.Hill(x, *params)
        result2 = fitter._hill_equation(x, *params)
        assert np.allclose(result1, result2)
        
        # Test fit_model (old name for fit)
        result = fitter.fit_model(
            generate_figure=False,
            print_r_sqr=False
        )
        
        # Should have fitted parameters
        assert hasattr(fitter, 'slope')
        assert hasattr(fitter, 'saturation')
    
    def test_predict(self, sample_data):
        """Test prediction on new data."""
        fitter = ResponseCurveFit(
            data=sample_data,
            model_level='Overall'
        )
        
        # First fit the model
        fitter.fit(generate_figure=False, print_r_sqr=False)
        
        # Predict on new spend levels
        new_spend = np.array([1000, 3000, 5000])
        predictions = fitter.predict(new_spend)
        
        # Check predictions are reasonable
        assert len(predictions) == len(new_spend)
        assert np.all(predictions >= 0)
        assert np.all(np.diff(predictions) >= 0)  # Monotonic
    
    def test_get_summary(self, sample_data):
        """Test get_summary method."""
        fitter = ResponseCurveFit(
            data=sample_data,
            model_level='Overall'
        )
        
        # Fit first
        fitter.fit(generate_figure=False, print_r_sqr=False)
        
        # Get summary
        summary = fitter.get_summary()
        
        assert 'params' in summary
        assert 'r2' in summary
        assert 'equation' in summary
        
        assert 'slope' in summary['params']
        assert 'saturation' in summary['params']
        assert 'top' in summary['params']
        assert 'bottom' in summary['params']
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with minimal data
        df_small = pd.DataFrame({
            'week_monday': pd.date_range('2023-01-01', periods=3, freq='W'),
            'impressions': [100, 1000, 5000],
            'spend': [50, 500, 2500],
            'predicted': [100, 500, 800]
        })
        
        fitter = ResponseCurveFit(
            data=df_small,
            model_level='Overall'
        )
        
        # Should still work with minimal data
        try:
            fitter.fit(generate_figure=False, print_r_sqr=False)
            assert fitter.slope > 0
            assert fitter.saturation > 0
        except (RuntimeError, ValueError):
            # It's acceptable to fail on very sparse data
            pass
    
    def test_monotonic_data(self):
        """Test with perfectly monotonic data."""
        df_mono = pd.DataFrame({
            'week_monday': pd.date_range('2023-01-01', periods=6, freq='W'),
            'impressions': [100, 200, 300, 400, 500, 600],
            'spend': [50, 100, 150, 200, 250, 300],
            'predicted': [100, 200, 300, 400, 500, 600]
        })
        
        fitter = ResponseCurveFit(
            data=df_mono,
            model_level='Overall'
        )
        
        fitter.fit(generate_figure=False, print_r_sqr=False)
        
        # Should fit reasonably well to monotonic data
        assert fitter.r_2 > 0.5, "Should fit reasonably to monotonic data"


class TestResponseCurveFitter:
    """Test backward compatibility alias."""
    
    def test_alias_exists(self):
        """Test that ResponseCurveFitter alias exists."""
        assert ResponseCurveFitter is ResponseCurveFit
    
    def test_alias_works(self):
        """Test that alias can be instantiated."""
        df = pd.DataFrame({
            'week_monday': pd.date_range('2023-01-01', periods=3, freq='W'),
            'impressions': [100, 1000, 5000],
            'spend': [50, 500, 2500],
            'predicted': [100, 500, 800]
        })
        
        fitter = ResponseCurveFitter(
            data=df,
            model_level='Overall'
        )
        
        assert isinstance(fitter, ResponseCurveFit)
        fitter.fit(generate_figure=False, print_r_sqr=False)
        assert fitter.slope > 0
        assert fitter.saturation > 0


class TestIntegration:
    """Integration tests for response curves."""
    
    def test_full_workflow(self):
        """Test complete workflow from data to fit."""
        np.random.seed(42)
        
        # Create realistic data
        impressions = np.linspace(100, 100000, 50)
        spend = impressions * 0.5  # CPI = 0.5
        a, g = 3.0, 25000
        predicted = (spend ** a) / (spend ** a + g ** a) * 5000000
        predicted += np.random.normal(0, 100000, 50)
        
        df = pd.DataFrame({
            'week_monday': pd.date_range('2023-01-01', periods=50, freq='W'),
            'impressions': impressions,
            'spend': spend,
            'predicted': predicted
        })
        
        # Initialize fitter
        fitter = ResponseCurveFit(
            data=df,
            model_level='Overall',
            date_col='week_monday'
        )
        
        # Fit curve
        fitter.fit(generate_figure=False, print_r_sqr=False)
        
        # Verify results
        assert fitter.slope > 0
        assert fitter.saturation > 0
        assert fitter.r_2 > 0.5, "Should achieve reasonable fit on synthetic data"
        
        # Verify slope is close to true value (with some tolerance)
        assert 2.0 < fitter.slope < 5.0, "Slope should be in reasonable range"
        
        # Verify saturation is in reasonable range
        assert 1000 < fitter.saturation < 100000, "Saturation should be in reasonable range"
        
        # Test prediction
        new_spend = np.array([10000, 30000, 50000])
        predictions = fitter.predict(new_spend)
        assert len(predictions) == 3
        assert np.all(predictions > 0)
        
        # Test summary
        summary = fitter.get_summary()
        assert summary['r2'] == fitter.r_2
        assert summary['params']['slope'] == fitter.slope


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
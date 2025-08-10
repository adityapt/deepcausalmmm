import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import datetime as dt

from mmm_monitor.mlflow import (
    is_available,
    get_historical_metrics,
    get_residuals_for_drift,
    detect_drift_from_mlflow,
    log_drift_detection_result
)


class TestMlflowIntegration:
    
    def test_is_available_with_mlflow(self):
        """Test is_available when mlflow is installed"""
        with patch('mmm_monitor.mlflow.mlflow') as mock_mlflow:
            mock_mlflow.__bool__ = Mock(return_value=True)
            assert is_available() is True
    
    def test_is_available_without_mlflow(self):
        """Test is_available when mlflow is not installed"""
        with patch('mmm_monitor.mlflow.mlflow', None):
            assert is_available() is False
    
    @patch('mmm_monitor.mlflow.get_client')
    def test_get_historical_metrics(self, mock_get_client):
        """Test getting historical metrics from MLflow"""
        # Mock the client and runs
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # Mock experiment
        mock_experiment = Mock()
        mock_experiment.experiment_id = "test_exp_id"
        mock_client.get_experiment_by_name.return_value = mock_experiment
        
        # Mock runs
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_run.info.start_time = 1640995200000  # 2022-01-01 timestamp in ms
        mock_run.info.status = "FINISHED"
        mock_run.data.tags = {"mlflow.runName": "test_run"}
        mock_run.data.metrics = {"residual": 0.1, "mape": 5.0}
        
        mock_client.search_runs.return_value = [mock_run]
        
        # Test the function
        result = get_historical_metrics("test_experiment", ["residual", "mape"])
        
        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["run_id"] == "test_run_id"
        assert result.iloc[0]["residual"] == 0.1
        assert result.iloc[0]["mape"] == 5.0
    
    @patch('mmm_monitor.mlflow.get_historical_metrics')
    def test_get_residuals_for_drift(self, mock_get_metrics):
        """Test getting residuals for drift detection"""
        # Mock historical metrics
        mock_df = pd.DataFrame({
            'run_id': ['run1', 'run2', 'run3', 'run4'],
            'start_time': pd.date_range('2023-01-01', periods=4, freq='D'),
            'residual': [0.1, 0.2, 0.3, 0.4],
            'y_true': [100, 110, 120, 130],
            'y_pred': [99.9, 109.8, 119.7, 129.6]
        })
        mock_get_metrics.return_value = mock_df
        
        # Test the function
        baseline, current = get_residuals_for_drift("test_experiment", baseline_days=30)
        
        # Verify results
        assert isinstance(baseline, pd.Series)
        assert isinstance(current, pd.Series)
        assert len(baseline) == 2  # First half
        assert len(current) == 2   # Second half
        assert baseline.iloc[0] == 0.1
        assert current.iloc[0] == 0.3
    
    @patch('mmm_monitor.mlflow.get_residuals_for_drift')
    def test_detect_drift_from_mlflow_no_drift(self, mock_get_residuals):
        """Test drift detection when no drift is detected"""
        # Mock residuals with no drift
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 100))
        current = pd.Series(np.random.normal(0, 1, 50))
        
        mock_get_residuals.return_value = (baseline, current)
        
        # Test the function
        result = detect_drift_from_mlflow("test_experiment")
        
        # Verify results
        assert "error" not in result
        assert "psi" in result
        assert "ks_pvalue" in result
        assert "psi_alert" in result
        assert "ks_alert" in result
        assert result["experiment_name"] == "test_experiment"
        assert result["baseline_size"] == 100
        assert result["current_size"] == 50
    
    @patch('mmm_monitor.mlflow.get_residuals_for_drift')
    def test_detect_drift_from_mlflow_with_drift(self, mock_get_residuals):
        """Test drift detection when drift is detected"""
        # Mock residuals with drift
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 100))
        current = pd.Series(np.random.normal(2, 1, 50))  # Shifted mean
        
        mock_get_residuals.return_value = (baseline, current)
        
        # Test the function
        result = detect_drift_from_mlflow("test_experiment")
        
        # Verify results
        assert "error" not in result
        assert result["psi_alert"] is True  # Should detect drift
        assert result["ks_alert"] is True   # Should detect drift
    
    @patch('mmm_monitor.mlflow.get_residuals_for_drift')
    def test_detect_drift_from_mlflow_error(self, mock_get_residuals):
        """Test drift detection error handling"""
        # Mock an error
        mock_get_residuals.side_effect = ValueError("Test error")
        
        # Test the function
        result = detect_drift_from_mlflow("test_experiment")
        
        # Verify error handling
        assert "error" in result
        assert result["error"] == "Test error"
        assert result["experiment_name"] == "test_experiment"
    
    @patch('mmm_monitor.mlflow.log_report')
    def test_log_drift_detection_result(self, mock_log_report):
        """Test logging drift detection results"""
        # Mock drift result
        drift_result = {
            "psi": 0.15,
            "ks_pvalue": 0.02,
            "psi_alert": False,
            "ks_alert": True,
            "experiment_name": "test_experiment"
        }
        
        # Test the function
        log_drift_detection_result(drift_result)
        
        # Verify log_report was called
        mock_log_report.assert_called_once()
        args, kwargs = mock_log_report.call_args
        
        assert args[0] == drift_result  # First argument should be the drift result
        assert "drift_check_" in args[1]  # Second argument should be run name
        assert kwargs["tags"]["drift_detection"] == "true"
        assert kwargs["tags"]["source_experiment"] == "test_experiment"


class TestConnectorIntegration:
    
    def test_pymc_connector_basic_functionality(self):
        """Test basic PyMC connector functionality"""
        # Create mock xarray Dataset
        mock_posterior = Mock()
        mock_posterior.data_vars = ['sales']
        
        # Mock sales data
        mock_sales = Mock()
        mock_sales.mean.return_value.values = [100, 110, 120, 130]
        mock_posterior.__getitem__.return_value = mock_sales
        
        # Mock coordinates
        mock_coords = Mock()
        mock_coords.__contains__ = Mock(return_value=True)
        mock_coords.__getitem__ = Mock(return_value=Mock(values=pd.date_range('2023-01-01', periods=4)))
        mock_posterior.coords = mock_coords
        
        # Test data
        true_sales = pd.Series([99, 109, 119, 129])
        
        # Create connector
        from mmm_monitor.connectors.pymc_marketing import PyMCConnector
        connector = PyMCConnector(mock_posterior, true_sales)
        
        # Test iter_events
        events = list(connector.iter_events())
        assert len(events) == 4
        assert events[0]["y_true"] == 99
        assert events[0]["y_pred"] == 100
        assert events[0]["residual"] == -1.0
        assert events[0]["meta"]["library"] == "PyMC-Marketing"
    
    def test_base_connector_drift_detection(self):
        """Test base connector drift detection"""
        from mmm_monitor.connectors.base import BaseConnector
        
        # Create a simple test connector
        class TestConnector(BaseConnector):
            name = "test"
            
            def __init__(self, data, **kwargs):
                super().__init__(**kwargs)
                self.data = data
            
            def iter_events(self):
                return iter(self.data)
        
        # Test data with no drift
        np.random.seed(42)
        data = []
        for i in range(100):
            residual = np.random.normal(0, 1)
            data.append({
                "timestamp": f"2023-01-{i+1:02d}",
                "y_true": 100 + residual,
                "y_pred": 100,
                "residual": residual
            })
        
        connector = TestConnector(data)
        result = connector.detect_drift()
        
        # Verify results
        assert "error" not in result
        assert "psi" in result
        assert "ks_pvalue" in result
        assert result["connector"] == "test"
        assert result["total_events"] == 100


if __name__ == "__main__":
    pytest.main([__file__]) 
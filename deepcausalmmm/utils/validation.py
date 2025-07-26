"""
Data validation utilities for DeepCausalMMM.

This module provides functions for validating inputs and checking data quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from ..exceptions import ValidationError, DataError


def validate_inputs(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> None:
    """
    Validate input data and configuration.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    # Check if dataframe is not empty
    if df.empty:
        raise ValidationError("Dataframe is empty")
    
    # Check required columns
    required_columns = ['date', 'revenue']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValidationError(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            raise ValidationError("'date' column must be convertible to datetime")
    
    if not pd.api.types.is_numeric_dtype(df['revenue']):
        raise ValidationError("'revenue' column must be numeric")
    
    # Check for negative revenue
    if (df['revenue'] < 0).any():
        raise ValidationError("Revenue values cannot be negative")
    
    # Check marketing variables if specified
    marketing_vars = config.get('marketing_vars', [])
    if marketing_vars:
        missing_marketing = [var for var in marketing_vars if var not in df.columns]
        if missing_marketing:
            raise ValidationError(f"Missing marketing variables: {missing_marketing}")
        
        # Check that marketing variables are numeric and non-negative
        for var in marketing_vars:
            if not pd.api.types.is_numeric_dtype(df[var]):
                raise ValidationError(f"Marketing variable '{var}' must be numeric")
            if (df[var] < 0).any():
                raise ValidationError(f"Marketing variable '{var}' cannot have negative values")
    
    # Check control variables if specified
    control_vars = config.get('control_vars', [])
    if control_vars:
        missing_control = [var for var in control_vars if var not in df.columns]
        if missing_control:
            raise ValidationError(f"Missing control variables: {missing_control}")
        
        # Check that control variables are numeric
        for var in control_vars:
            if not pd.api.types.is_numeric_dtype(df[var]):
                raise ValidationError(f"Control variable '{var}' must be numeric")
    
    # Check region variable if specified
    region_var = config.get('region_var')
    if region_var and region_var not in df.columns:
        raise ValidationError(f"Region variable '{region_var}' not found in dataframe")


def check_data_quality(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check data quality and return quality metrics.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        
    Returns:
        Dictionary containing quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicates': 0,
        'outliers': {},
        'data_range': {},
        'quality_score': 0.0
    }
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    quality_report['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    # Check for duplicates
    quality_report['duplicates'] = df.duplicated().sum()
    
    # Check for outliers in numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            quality_report['outliers'][col] = outliers
    
    # Check data ranges
    for col in numeric_columns:
        if col in df.columns:
            quality_report['data_range'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
    
    # Calculate quality score
    total_cells = len(df) * len(df.columns)
    missing_cells = sum(quality_report['missing_values'].values())
    duplicate_rows = quality_report['duplicates']
    
    quality_score = 1.0 - (missing_cells / total_cells) - (duplicate_rows / len(df))
    quality_report['quality_score'] = max(0.0, quality_score)
    
    return quality_report


def validate_config(
    config: Dict[str, Any]
) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValidationError: If configuration is invalid
    """
    # Check required configuration keys
    required_keys = ['hidden_size', 'learning_rate', 'epochs']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValidationError(f"Missing required configuration keys: {missing_keys}")
    
    # Validate parameter ranges
    if config['hidden_size'] <= 0:
        raise ValidationError("hidden_size must be positive")
    
    if not (0 < config['learning_rate'] < 1):
        raise ValidationError("learning_rate must be between 0 and 1")
    
    if config['epochs'] <= 0:
        raise ValidationError("epochs must be positive")
    
    # Validate optional parameters if present
    if 'dropout' in config:
        if not (0 <= config['dropout'] < 1):
            raise ValidationError("dropout must be between 0 and 1")
    
    if 'test_size' in config:
        if not (0 < config['test_size'] < 1):
            raise ValidationError("test_size must be between 0 and 1")
    
    if 'validation_size' in config:
        if not (0 < config['validation_size'] < 1):
            raise ValidationError("validation_size must be between 0 and 1")


def check_model_compatibility(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any]
) -> None:
    """
    Check if model configuration is compatible with data configuration.
    
    Args:
        model_config: Model configuration
        data_config: Data configuration
        
    Raises:
        ValidationError: If configurations are incompatible
    """
    # Check if number of media variables matches
    if 'n_media' in model_config and 'marketing_vars' in data_config:
        expected_media = len(data_config['marketing_vars'])
        if model_config['n_media'] != expected_media:
            raise ValidationError(
                f"Model expects {model_config['n_media']} media variables, "
                f"but data has {expected_media}"
            )
    
    # Check if number of control variables matches
    if 'ctrl_dim' in model_config and 'control_vars' in data_config:
        expected_control = len(data_config['control_vars'])
        if model_config['ctrl_dim'] != expected_control:
            raise ValidationError(
                f"Model expects {model_config['ctrl_dim']} control variables, "
                f"but data has {expected_control}"
            )


def validate_training_data(
    X_m: np.ndarray,
    X_c: np.ndarray,
    y: np.ndarray,
    R: np.ndarray
) -> None:
    """
    Validate training data arrays.
    
    Args:
        X_m: Media variables array
        X_c: Control variables array
        y: Target variable array
        R: Region indices array
        
    Raises:
        ValidationError: If data is invalid
    """
    # Check shapes
    if len(X_m.shape) != 3:
        raise ValidationError("X_m must be 3-dimensional (regions, time_steps, features)")
    
    if len(X_c.shape) != 3:
        raise ValidationError("X_c must be 3-dimensional (regions, time_steps, features)")
    
    if len(y.shape) != 2:
        raise ValidationError("y must be 2-dimensional (regions, time_steps)")
    
    if len(R.shape) != 1:
        raise ValidationError("R must be 1-dimensional (regions)")
    
    # Check consistent dimensions
    n_regions, n_time_steps, n_media = X_m.shape
    _, _, n_control = X_c.shape
    
    if X_c.shape[0] != n_regions or X_c.shape[1] != n_time_steps:
        raise ValidationError("X_c dimensions must match X_m")
    
    if y.shape[0] != n_regions or y.shape[1] != n_time_steps:
        raise ValidationError("y dimensions must match X_m")
    
    if R.shape[0] != n_regions:
        raise ValidationError("R must have same number of regions as data")
    
    # Check for NaN values
    if np.isnan(X_m).any():
        raise ValidationError("X_m contains NaN values")
    
    if np.isnan(X_c).any():
        raise ValidationError("X_c contains NaN values")
    
    if np.isnan(y).any():
        raise ValidationError("y contains NaN values")
    
    if np.isnan(R).any():
        raise ValidationError("R contains NaN values")
    
    # Check for infinite values
    if np.isinf(X_m).any():
        raise ValidationError("X_m contains infinite values")
    
    if np.isinf(X_c).any():
        raise ValidationError("X_c contains infinite values")
    
    if np.isinf(y).any():
        raise ValidationError("y contains infinite values")
    
    if np.isinf(R).any():
        raise ValidationError("R contains infinite values") 
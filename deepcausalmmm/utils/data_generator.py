"""
Config-driven synthetic data generator for DeepCausalMMM.
Replaces hardcoded data generation with configurable parameters.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from deepcausalmmm.core.config import get_default_config


class ConfigurableDataGenerator:
    """
    Generate synthetic MMM data using configuration parameters.
    
    All data generation parameters are driven by configuration to ensure
    consistency and reproducibility across examples and tests.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data generator.
        
        Args:
            config: Configuration dictionary. If None, uses default config.
        """
        self.config = config or get_default_config()
        self.data_config = self.config.get('synthetic_data', {})
        
        # Set random seed for reproducibility
        seed = self.config.get('random_seed', 42)
        np.random.seed(seed)
        
    def generate_mmm_dataset(self, 
                           n_regions: int = 2,
                           n_weeks: int = 104,
                           n_media_channels: int = 5,
                           n_control_channels: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a complete MMM dataset with realistic patterns.
        
        Args:
            n_regions: Number of regions
            n_weeks: Number of weeks
            n_media_channels: Number of media channels
            n_control_channels: Number of control variables
            
        Returns:
            Tuple of (X_media, X_control, y) arrays
        """
        # Generate media data with realistic spend patterns
        X_media = self._generate_media_data(n_regions, n_weeks, n_media_channels)
        
        # Generate control variables
        X_control = self._generate_control_data(n_regions, n_weeks, n_control_channels)
        
        # Generate target variable with realistic MMM relationships
        y = self._generate_target_variable(X_media, X_control)
        
        return X_media, X_control, y
        
    def _generate_media_data(self, n_regions: int, n_weeks: int, n_channels: int) -> np.ndarray:
        """Generate realistic media spend data."""
        # Base spend levels (different for each region and channel)
        base_spend_range = self.data_config.get('base_spend_range', (10000, 50000))
        base_spend = np.random.uniform(
            base_spend_range[0], 
            base_spend_range[1], 
            (n_regions, n_channels)
        )
        
        # Seasonal pattern
        seasonality_strength = self.data_config.get('seasonality_strength', 0.3)
        weeks = np.arange(n_weeks)
        seasonal_pattern = 1 + seasonality_strength * np.sin(2 * np.pi * weeks / 52)
        
        # Generate spend data
        X_media = np.zeros((n_regions, n_weeks, n_channels))
        
        for region in range(n_regions):
            for channel in range(n_channels):
                # Base pattern with seasonality
                channel_spend = base_spend[region, channel] * seasonal_pattern
                
                # Add some randomness
                noise_level = self.data_config.get('media_noise_level', 0.2)
                noise = np.random.normal(0, noise_level, n_weeks)
                channel_spend *= (1 + noise)
                
                # Ensure non-negative
                channel_spend = np.maximum(channel_spend, 0)
                
                X_media[region, :, channel] = channel_spend
                
        return X_media
        
    def _generate_control_data(self, n_regions: int, n_weeks: int, n_controls: int) -> np.ndarray:
        """Generate control variables (economic indicators, etc.)."""
        control_range = self.data_config.get('control_range', (-2, 2))
        
        X_control = np.random.uniform(
            control_range[0], 
            control_range[1], 
            (n_regions, n_weeks, n_controls)
        )
        
        # Add some temporal correlation for realism
        correlation_strength = self.data_config.get('control_correlation', 0.7)
        
        for region in range(n_regions):
            for control in range(n_controls):
                # Apply some smoothing to create temporal correlation
                for week in range(1, n_weeks):
                    X_control[region, week, control] = (
                        correlation_strength * X_control[region, week-1, control] +
                        (1 - correlation_strength) * X_control[region, week, control]
                    )
                    
        return X_control
        
    def _generate_target_variable(self, X_media: np.ndarray, X_control: np.ndarray) -> np.ndarray:
        """Generate target variable with realistic MMM relationships."""
        n_regions, n_weeks, n_media = X_media.shape
        n_controls = X_control.shape[2]
        
        # Media coefficients (decreasing effectiveness)
        media_coeff_range = self.data_config.get('media_coeff_range', (0.1, 0.8))
        media_coeffs = np.random.uniform(
            media_coeff_range[0], 
            media_coeff_range[1], 
            n_media
        )
        # Sort in descending order for realism
        media_coeffs = np.sort(media_coeffs)[::-1]
        
        # Control coefficients
        control_coeff_range = self.data_config.get('control_coeff_range', (-0.5, 0.5))
        control_coeffs = np.random.uniform(
            control_coeff_range[0], 
            control_coeff_range[1], 
            n_controls
        )
        
        # Base levels for each region
        base_level_range = self.data_config.get('base_level_range', (40000, 60000))
        base_levels = np.random.uniform(
            base_level_range[0], 
            base_level_range[1], 
            n_regions
        ).reshape(-1, 1)
        
        # Generate target
        y = np.zeros((n_regions, n_weeks))
        
        for region in range(n_regions):
            # Base level
            y[region, :] = base_levels[region]
            
            # Media contributions (with diminishing returns)
            for channel in range(n_media):
                # Apply adstock transformation for realism
                adstock_rate = self.data_config.get('adstock_rate', 0.5)
                adstocked_media = self._apply_adstock(
                    X_media[region, :, channel], 
                    adstock_rate
                )
                
                # Apply saturation curve
                saturation_param = self.data_config.get('saturation_param', 0.5)
                saturated_media = self._apply_saturation(
                    adstocked_media, 
                    saturation_param
                )
                
                y[region, :] += media_coeffs[channel] * saturated_media
                
            # Control contributions
            for control in range(n_controls):
                y[region, :] += control_coeffs[control] * X_control[region, :, control] * 1000
                
        # Add noise
        noise_level = self.data_config.get('target_noise_level', 0.05)
        noise = np.random.normal(0, noise_level * np.mean(y), y.shape)
        y += noise
        
        # Ensure non-negative
        y = np.maximum(y, 0)
        
        return y
        
    def _apply_adstock(self, x: np.ndarray, rate: float) -> np.ndarray:
        """Apply adstock transformation to media data."""
        adstocked = np.zeros_like(x)
        adstocked[0] = x[0]
        
        for i in range(1, len(x)):
            adstocked[i] = x[i] + rate * adstocked[i-1]
            
        return adstocked
        
    def _apply_saturation(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """Apply saturation curve to media data."""
        return x / (x + alpha * np.max(x))


def get_synthetic_data_config() -> Dict[str, Any]:
    """
    Get default synthetic data configuration.
    
    Returns:
        Dictionary with synthetic data parameters
    """
    return {
        'base_spend_range': (10000, 50000),
        'seasonality_strength': 0.3,
        'media_noise_level': 0.2,
        'control_range': (-2, 2),
        'control_correlation': 0.7,
        'media_coeff_range': (0.1, 0.8),
        'control_coeff_range': (-0.5, 0.5),
        'base_level_range': (40000, 60000),
        'adstock_rate': 0.5,
        'saturation_param': 0.5,
        'target_noise_level': 0.05,
    }


def update_config_with_synthetic_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with synthetic data parameters.
    
    Args:
        config: Base configuration
        
    Returns:
        Updated configuration with synthetic data settings
    """
    if 'synthetic_data' not in config:
        config['synthetic_data'] = get_synthetic_data_config()
    
    return config
import pandas as pd
import numpy as np
import torch
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Tuple, Optional

class DetectSeasonality:
    """
    Seasonality detection and decomposition for time series data.
    
    Provides methods for extracting seasonal patterns from time series,
    with support for multi-region analysis in marketing mix modeling.
    """
    
    def __init__(self):
        """Initialize the seasonality detector."""
        pass
    
    def decompose(self, X, period=52):
        """
        Perform seasonal decomposition using multiplicative model.
        
        Args:
            X: Time series data
            period: Seasonal period (default 52 for weekly data = annual seasonality)
            
        Returns:
            Seasonal decomposition result
        """
        # Use additive model to handle zero/negative values better
        decomposition = seasonal_decompose(X, model='additive', period=period)
        return decomposition

    def extract_seasonal_components_per_region(
        self, 
        y_data: np.ndarray, 
        start_week: int = 0
    ) -> torch.Tensor:
        """
        Extract seasonal components for each region separately.
        
        Args:
            y_data: Target data [n_regions, n_weeks]
            start_week: Starting week index for proper alignment
            
        Returns:
            Seasonal components [n_regions, n_weeks] as torch.Tensor
        """
        n_regions, n_weeks = y_data.shape
        seasonal_components = []
        
        print(f"🌊 Extracting seasonal components per region...")
        print(f"   📊 Processing {n_regions} regions × {n_weeks} weeks")
        
        for region_idx in range(n_regions):
            region_data = y_data[region_idx, :]
            
            # Convert to pandas Series for statsmodels
            region_series = pd.Series(region_data)
            
            try:
                # Determine appropriate period based on data length
                period = min(52, n_weeks // 2)  # Annual cycle or half the data length
                
                # Perform additive seasonal decomposition with explicit period
                decomposition = self.decompose(region_series, period=period)
                seasonal_component = decomposition.seasonal.values
                
                # Handle any NaN values (can occur at edges)
                if np.isnan(seasonal_component).any():
                    # Fill NaN with mean of non-NaN values
                    seasonal_mean = np.nanmean(seasonal_component)
                    seasonal_component = np.nan_to_num(seasonal_component, nan=seasonal_mean)
                
                seasonal_components.append(seasonal_component)
                
            except Exception as e:
                print(f"   ⚠️ Region {region_idx}: Seasonal decomposition failed, using mean: {e}")
                # Fallback: use mean seasonal pattern (flat)
                seasonal_component = np.ones(n_weeks)
                seasonal_components.append(seasonal_component)
        
        # Stack all regions and convert to tensor
        seasonal_tensor = torch.tensor(np.stack(seasonal_components, axis=0), dtype=torch.float32)
        
        print(f"   ✅ Seasonal components extracted:")
        print(f"      Range: [{seasonal_tensor.min():.3f}, {seasonal_tensor.max():.3f}]")
        print(f"      Mean: {seasonal_tensor.mean():.3f}")
        print(f"      Shape: {seasonal_tensor.shape}")
        
        return seasonal_tensor

    def get_seasonal_contribution_for_inference(
        self, 
        seasonal_components: torch.Tensor,
        weeks_slice: Optional[slice] = None
    ) -> torch.Tensor:
        """
        Get seasonal components for inference, with optional time slicing.
        
        Args:
            seasonal_components: Pre-computed seasonal components [n_regions, n_weeks]
            weeks_slice: Optional slice for specific weeks
            
        Returns:
            Seasonal components for the specified time period
        """
        if weeks_slice is not None:
            return seasonal_components[:, weeks_slice]
        return seasonal_components

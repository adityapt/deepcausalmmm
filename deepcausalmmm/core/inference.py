"""
Modern InferenceManager class for DeepCausalMMM model inference.
Provides a clean, reusable interface for model predictions and analysis.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings

from deepcausalmmm.core.unified_model import DeepCausalMMM
from deepcausalmmm.core.data import UnifiedDataPipeline
from deepcausalmmm.core.scaling import SimpleGlobalScaler
from deepcausalmmm.core.config import get_default_config
from deepcausalmmm.utils.device import get_device


class InferenceManager:
    """
    Modern class-based interface for DeepCausalMMM model inference.
    
    Handles:
    - Model predictions on new data
    - Contribution analysis (media, control, baseline)
    - Coefficient extraction
    - Data preprocessing for inference
    - Inverse transformations for interpretable results
    """
    
    def __init__(self, 
        model: DeepCausalMMM,
                 pipeline: Optional[UnifiedDataPipeline] = None,
                 scaler: Optional[SimpleGlobalScaler] = None,
                 config: Optional[Dict[str, Any]] = None,
        channel_names: Optional[List[str]] = None,
                 control_names: Optional[List[str]] = None):
        """
        Initialize the inference manager.
        
        Args:
            model: Trained DeepCausalMMM model
            pipeline: UnifiedDataPipeline used for training (preferred)
            scaler: SimpleGlobalScaler used for training (legacy support)
            config: Configuration dictionary
            channel_names: List of media channel names
            control_names: List of control variable names
        """
        self.model = model
        self.pipeline = pipeline
        self.scaler = scaler or (pipeline.scaler if pipeline else None)
        self.config = config or get_default_config()
        self.channel_names = channel_names or []
        self.control_names = control_names or []
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Validate inputs
        if self.pipeline is None and self.scaler is None:
            warnings.warn(
                "Neither pipeline nor scaler provided. "
                "Inference may not work correctly without proper data preprocessing.",
                UserWarning
            )
            
    def predict(self, 
                X_media: np.ndarray,
                X_control: np.ndarray,
                return_contributions: bool = True,
                remove_padding: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X_media: Media data [n_regions, n_weeks, n_media_channels]
            X_control: Control data [n_regions, n_weeks, n_control_vars]
            return_contributions: Whether to return contribution breakdowns
            remove_padding: Whether to remove burn-in padding from results
            
        Returns:
            Dictionary containing predictions and optionally contributions
        """
        # Preprocess data
        if self.pipeline is not None:
            # Use modern pipeline approach
            X_media_processed, X_control_processed = self._preprocess_with_pipeline(
                X_media, X_control
            )
        elif self.scaler is not None:
            # Use legacy scaler approach
            X_media_processed, X_control_processed = self._preprocess_with_scaler(
                X_media, X_control
            )
        else:
            # No preprocessing - assume data is already processed
            X_media_processed = torch.FloatTensor(X_media).to(self.device)
            X_control_processed = torch.FloatTensor(X_control).to(self.device)
            
        # Create region tensor
        n_regions = X_media_processed.shape[0]
        R = torch.zeros(n_regions, dtype=torch.long).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            if return_contributions:
                predictions, media_contributions, control_contributions = self.model(
                    X_media_processed, X_control_processed, R
                )
            else:
                predictions = self.model(X_media_processed, X_control_processed, R)
                media_contributions = None
                control_contributions = None
                
        # Convert to numpy and move to CPU
            results = {
            'predictions': predictions.cpu().numpy()
        }
        
        if return_contributions:
            results['media_contributions'] = media_contributions.cpu().numpy()
            results['control_contributions'] = control_contributions.cpu().numpy()
            
        # Remove padding if requested
        if remove_padding and hasattr(self.model, 'burn_in_weeks'):
            burn_in = self.model.burn_in_weeks
            if burn_in > 0:
                for key in results:
                    if results[key] is not None and results[key].shape[1] > burn_in:
                        results[key] = results[key][:, burn_in:, ...]
        
        return results
    
    def predict_and_inverse_transform(self,
                                    X_media: np.ndarray,
                                    X_control: np.ndarray,
                                    return_contributions: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions and apply inverse transformations for interpretable results.
        
        Args:
            X_media: Media data [n_regions, n_weeks, n_media_channels]
            X_control: Control data [n_regions, n_weeks, n_control_vars]
            return_contributions: Whether to return contribution breakdowns
            
        Returns:
            Dictionary containing predictions and contributions in original scale
        """
        # Get predictions in transformed space
        results = self.predict(X_media, X_control, return_contributions, remove_padding=False)
        
        # Apply inverse transformations
        if self.pipeline is not None:
            # Use pipeline for inverse transformation
            inverse_results = self.pipeline.inverse_transform_predictions(
                results['predictions'],
                results.get('media_contributions'),
                results.get('control_contributions'),
                remove_padding=True
            )
            
            return {
                'predictions': inverse_results['predictions'],
                'media_contributions': inverse_results.get('media_contributions'),
                'control_contributions': inverse_results.get('control_contributions'),
                'baseline': inverse_results.get('baseline')
            }
            
        elif self.scaler is not None:
            # Use scaler for inverse transformation
            predictions_orig = self.scaler.inverse_transform_target(results['predictions'])
            
            result_dict = {'predictions': predictions_orig}
            
            if return_contributions:
                # Note: Contribution inverse transformation with scaler is more complex
                # For now, return contributions in transformed space with a warning
                warnings.warn(
                    "Contribution inverse transformation with legacy scaler is not fully supported. "
                    "Consider using UnifiedDataPipeline for complete functionality.",
                    UserWarning
                )
                result_dict['media_contributions'] = results.get('media_contributions')
                result_dict['control_contributions'] = results.get('control_contributions')
                
            return result_dict
            
        else:
            # No inverse transformation available
            warnings.warn("No scaler or pipeline available for inverse transformation.", UserWarning)
            return results
            
    def get_coefficients(self) -> Dict[str, np.ndarray]:
        """
        Extract model coefficients.
        
        Returns:
            Dictionary containing media and control coefficients
        """
        coefficients = {}
        
        # Extract media coefficients
        if hasattr(self.model, 'coeff_gen'):
            with torch.no_grad():
                # Get current coefficients
                media_coeffs = self.model.coeff_gen(
                    torch.zeros(1, 1, self.model.n_media).to(self.device)
                )
                coefficients['media'] = media_coeffs.cpu().numpy()
                
        # Extract control coefficients
        if hasattr(self.model, 'ctrl_coeff_gen'):
            with torch.no_grad():
                control_coeffs = self.model.ctrl_coeff_gen(
                    torch.zeros(1, 1, self.model.ctrl_dim).to(self.device)
                )
                coefficients['control'] = control_coeffs.cpu().numpy()
                
        # Extract stable coefficients if available
        if hasattr(self.model, 'stable_media_coeff'):
            coefficients['stable_media'] = self.model.stable_media_coeff.detach().cpu().numpy()
            
        if hasattr(self.model, 'stable_ctrl_coeff'):
            coefficients['stable_control'] = self.model.stable_ctrl_coeff.detach().cpu().numpy()
            
        return coefficients
        
    def get_dag_adjacency(self) -> Optional[np.ndarray]:
        """
        Extract DAG adjacency matrix if available.
        
        Returns:
            Adjacency matrix or None if DAG is not enabled
        """
        if hasattr(self.model, 'adj_logits') and self.model.enable_dag:
            with torch.no_grad():
                # Apply triangular mask and sigmoid
                adj_matrix = torch.sigmoid(self.model.adj_logits)
                if hasattr(self.model, 'tri_mask'):
                    adj_matrix = adj_matrix * self.model.tri_mask
                return adj_matrix.cpu().numpy()
        return None
        
    def analyze_contributions(self,
                            X_media: np.ndarray,
                            X_control: np.ndarray,
                            aggregate_regions: bool = True,
                            aggregate_time: bool = False) -> Dict[str, Any]:
        """
        Comprehensive contribution analysis.
        
        Args:
            X_media: Media data
            X_control: Control data
            aggregate_regions: Whether to aggregate across regions
            aggregate_time: Whether to aggregate across time
            
        Returns:
            Dictionary with detailed contribution analysis
        """
        # Get predictions and contributions
        results = self.predict_and_inverse_transform(X_media, X_control, return_contributions=True)
        
        analysis = {
            'total_predictions': results['predictions'],
            'media_contributions': results.get('media_contributions'),
            'control_contributions': results.get('control_contributions'),
            'baseline': results.get('baseline', 0)
        }
        
        # Aggregate if requested
        if aggregate_regions:
            for key in ['total_predictions', 'media_contributions', 'control_contributions']:
                if analysis[key] is not None:
                    analysis[f'{key}_by_region'] = np.mean(analysis[key], axis=0)
                    
        if aggregate_time:
            for key in ['total_predictions', 'media_contributions', 'control_contributions']:
                if analysis[key] is not None:
                    analysis[f'{key}_by_time'] = np.mean(analysis[key], axis=1)
                    
        # Calculate contribution percentages
        if analysis['media_contributions'] is not None:
            total_media = np.sum(analysis['media_contributions'], axis=-1, keepdims=True)
            total_pred = analysis['total_predictions']
            
            analysis['media_contribution_pct'] = (
                analysis['media_contributions'] / (total_pred[..., None] + 1e-8) * 100
            )
            
        return analysis
        
    def _preprocess_with_pipeline(self, X_media: np.ndarray, X_control: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess data using UnifiedDataPipeline."""
        # Add seasonality features
        X_control_with_seasonality = self.pipeline._add_seasonality_features(
            torch.tensor(X_control), start_week=0
        )
        
        # Scale data
        X_media_scaled, X_control_scaled, _ = self.pipeline.scaler.transform(
            X_media, X_control_with_seasonality.numpy(), np.zeros_like(X_media[:, :, 0])
        )
        
        # Add padding
        X_media_padded, X_control_padded, _ = self.pipeline._add_padding(
            X_media_scaled, X_control_scaled, torch.zeros(X_media_scaled.shape[0], X_media_scaled.shape[1])
        )
        
        return X_media_padded.to(self.device), X_control_padded.to(self.device)
        
    def _preprocess_with_scaler(self, X_media: np.ndarray, X_control: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess data using legacy SimpleGlobalScaler."""
        # Scale data
        X_media_scaled, X_control_scaled, _ = self.scaler.transform(
            X_media, X_control, np.zeros_like(X_media[:, :, 0])
        )
        
        # Convert to tensors
        X_media_tensor = torch.FloatTensor(X_media_scaled).to(self.device)
        X_control_tensor = torch.FloatTensor(X_control_scaled).to(self.device)
        
        return X_media_tensor, X_control_tensor


# Legacy compatibility class
class ModelInference(InferenceManager):
    """
    Legacy compatibility wrapper for InferenceManager.
    
    This class provides backward compatibility with existing code
    that uses the old ModelInference interface.
    """
    
    def __init__(self, model, scaler, channel_names=None, control_names=None, **kwargs):
        """Initialize with legacy interface."""
        warnings.warn(
            "ModelInference is deprecated. Please use InferenceManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        super().__init__(
            model=model,
            scaler=scaler,
            channel_names=channel_names,
            control_names=control_names,
            **kwargs
        )